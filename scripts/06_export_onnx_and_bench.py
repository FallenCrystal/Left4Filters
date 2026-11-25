#!/usr/bin/env python3
"""
Export trained Left4FiltersModel (Visual Encoder + Multimodal Fusion Head) to
portable ONNX models (FP32 + INT8 quantized), verify numerical & metric fidelity
on the locked Test split + Synthetic Red-Text Stress Test, and benchmark CPU
latency and file sizes.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import open_clip
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from timm.utils.model import reparameterize_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.l4filter_features import (
    LEXICAL_PRIOR_DIM,
    TEXT_VOCAB_DIM as TEXT_HASH_DIM,
    VISUAL_PRIOR_DIM,
    extract_text_features,
    extract_visual_priors,
)
from src.l4filter_model import MultimodalFusionHead, VisualEncoderWrapper


def preprocess_pil_256(img: Image.Image) -> np.ndarray:
    """Matches MobileCLIP2-S0 exact eval transform: Resize shortest side 256 -> CenterCrop 256 -> [0,1] CHW."""
    im = img.convert("RGB")
    w, h = im.size
    if w < h:
        new_w = 256
        new_h = max(256, int(round(h * 256.0 / w)))
    else:
        new_h = 256
        new_w = max(256, int(round(w * 256.0 / h)))
    im = im.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = (new_w - 256) // 2
    top = (new_h - 256) // 2
    im = im.crop((left, top, left + 256, top + 256))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))  # (3, 256, 256)


def export_onnx_pair(
    visual_backbone: torch.nn.Module,
    fusion_head: MultimodalFusionHead,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_wrap = VisualEncoderWrapper(visual_backbone).eval()
    fusion_head.eval()

    vis_fp32 = out_dir / "vision_encoder_fp32.onnx"
    head_fp32 = out_dir / "fusion_head_fp32.onnx"

    # 1. Export Vision Encoder (FP32)
    dummy_px = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        vis_wrap,
        (dummy_px,),
        str(vis_fp32),
        input_names=["pixel_values"],
        output_names=["img_emb"],
        dynamic_axes={"pixel_values": {0: "batch"}, "img_emb": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(vis_fp32))

    # 2. Export Fusion Head (FP32)
    dummy_head_inputs = (
        torch.randn(1, 512),
        torch.randn(1, VISUAL_PRIOR_DIM),
        torch.randn(1, TEXT_HASH_DIM),
        torch.randn(1, LEXICAL_PRIOR_DIM),
        torch.ones(1, 1),
        torch.ones(1, 1),
    )
    torch.onnx.export(
        fusion_head,
        dummy_head_inputs,
        str(head_fp32),
        input_names=["img_emb", "img_priors", "txt_hash", "txt_lex", "has_img", "has_txt"],
        output_names=["probs"],
        dynamic_axes={
            "img_emb": {0: "batch"},
            "img_priors": {0: "batch"},
            "txt_hash": {0: "batch"},
            "txt_lex": {0: "batch"},
            "has_img": {0: "batch"},
            "has_txt": {0: "batch"},
            "probs": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx.checker.check_model(str(head_fp32))

    # 3. Dynamic INT8 quantization (weight compression + fast CPU MatMul/Conv)
    vis_int8 = out_dir / "vision_encoder_int8.onnx"
    head_int8 = out_dir / "fusion_head_int8.onnx"

    quantize_dynamic(
        model_input=str(vis_fp32),
        model_output=str(vis_int8),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gemm", "Conv"],
    )
    quantize_dynamic(
        model_input=str(head_fp32),
        model_output=str(head_int8),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )

    # Also test MatMul-only INT8 for vision (often zero accuracy loss + tiny size!)
    vis_mm_int8 = out_dir / "vision_encoder_mm_int8.onnx"
    quantize_dynamic(
        model_input=str(vis_fp32),
        model_output=str(vis_mm_int8),
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
    )

    paths = {
        "vis_fp32": vis_fp32,
        "vis_int8": vis_int8,
        "vis_mm_int8": vis_mm_int8,
        "head_fp32": head_fp32,
        "head_int8": head_int8,
    }
    for k, p in paths.items():
        print(f"  [export] {k:12s}: {p.stat().st_size / (1024 * 1024):6.2f} MB  ({p.name})", flush=True)
    return paths


def run_ort_eval(
    vis_path: Path,
    head_path: Path,
    test_records: list[dict[str, Any]],
    img_dir: Path,
    stress_dir: Path,
    num_threads: int = 4,
) -> dict[str, Any]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    vis_sess = ort.InferenceSession(str(vis_path), sess_options=opts, providers=["CPUExecutionProvider"])
    head_sess = ort.InferenceSession(str(head_path), sess_options=opts, providers=["CPUExecutionProvider"])

    probs_list = []
    img_latencies = []
    head_latencies = []
    y_ov, rt, at = [], [], []

    for i, rec in enumerate(test_records):
        with Image.open(img_dir / f"{rec['wid']}.jpg") as im:
            im_rgb = im.convert("RGB")
            px = preprocess_pil_256(im_rgb)[None, ...]
            pri = extract_visual_priors(im_rgb)[None, ...]
        th, tl = extract_text_features(rec["title"], rec["desc"], rec["tags"])
        th = th[None, ...]
        tl = tl[None, ...]

        t0 = time.perf_counter()
        emb = vis_sess.run(["img_emb"], {"pixel_values": px})[0]
        t1 = time.perf_counter()
        pr = head_sess.run(
            ["probs"],
            {
                "img_emb": emb,
                "img_priors": pri,
                "txt_hash": th,
                "txt_lex": tl,
                "has_img": np.ones((1, 1), dtype=np.float32),
                "has_txt": np.ones((1, 1), dtype=np.float32),
            },
        )[0][0]
        t2 = time.perf_counter()

        if i >= 5:  # skip warmup
            img_latencies.append((t1 - t0) * 1000.0)
            head_latencies.append((t2 - t1) * 1000.0)
        probs_list.append(pr)
        y_ov.append(rec["ov"])
        rt.append(rec["rt"])
        at.append(rec["at"])

    probs = np.asarray(probs_list, dtype=np.float32)
    y_ov_arr = np.asarray(y_ov)
    rt_arr = np.asarray(rt)
    at_arr = np.asarray(at)

    p_ov = probs[:, 0]
    m02 = (y_ov_arr == 0) | (y_ov_arr == 2)
    auc_2v0 = float(roc_auc_score((y_ov_arr[m02] == 2).astype(int), p_ov[m02]))
    auc_any = float(roc_auc_score((y_ov_arr >= 1).astype(int), p_ov))
    ap_2v0 = float(average_precision_score((y_ov_arr[m02] == 2).astype(int), p_ov[m02]))

    m_rt = (rt_arr == 1) & (y_ov_arr == 0)
    m_at = (at_arr == 1) & (y_ov_arr == 0)

    # Stress red text
    stress_manifest = json.loads((stress_dir / "manifest.json").read_text(encoding="utf-8"))
    stress_p = []
    for rec in stress_manifest:
        with Image.open(rec["image"]) as im:
            im_rgb = im.convert("RGB")
            px = preprocess_pil_256(im_rgb)[None, ...]
            pri = extract_visual_priors(im_rgb)[None, ...]
        th, tl = extract_text_features(rec["title"], rec["description"], rec["tags"])
        emb = vis_sess.run(["img_emb"], {"pixel_values": px})[0]
        pr = head_sess.run(
            ["probs"],
            {
                "img_emb": emb,
                "img_priors": pri,
                "txt_hash": th[None, ...],
                "txt_lex": tl[None, ...],
                "has_img": np.ones((1, 1), dtype=np.float32),
                "has_txt": np.ones((1, 1), dtype=np.float32),
            },
        )[0][0]
        stress_p.append(float(pr[0]))
    stress_p_arr = np.asarray(stress_p)

    return {
        "vis_mb": round(vis_path.stat().st_size / (1024 * 1024), 2),
        "head_mb": round(head_path.stat().st_size / (1024 * 1024), 2),
        "total_mb": round((vis_path.stat().st_size + head_path.stat().st_size) / (1024 * 1024), 2),
        "vis_ms_median": round(float(np.median(img_latencies)), 2),
        "head_ms_median": round(float(np.median(head_latencies)), 3),
        "total_ms_median": round(float(np.median(img_latencies) + np.median(head_latencies)), 2),
        "auc_clear_2v0": round(auc_2v0, 4),
        "ap_clear_2v0": round(ap_2v0, 4),
        "auc_any_12v0": round(auc_any, 4),
        "red_trap_mean_p": round(float(p_ov[m_rt].mean()), 4) if m_rt.sum() else 0.0,
        "red_trap_fpr_05": round(float((p_ov[m_rt] >= 0.5).mean()), 4) if m_rt.sum() else 0.0,
        "anime_trap_mean_p": round(float(p_ov[m_at].mean()), 4) if m_at.sum() else 0.0,
        "stress_red_mean_p": round(float(stress_p_arr.mean()), 4),
        "stress_red_max_p": round(float(stress_p_arr.max()), 4),
        "stress_red_fpr_05": round(float((stress_p_arr >= 0.5).mean()), 4),
        "probs": probs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Checkpoint (.pt) from training")
    ap.add_argument("--mode", choices=["frozen", "finetuned"], default="finetuned")
    ap.add_argument("--catalog", default="data/raw/catalog_items.jsonl")
    ap.add_argument("--labels", default="data/annotations/merged_final.jsonl")
    ap.add_argument("--split", default="data/annotations/split_manifest.json")
    ap.add_argument("--img-dir", default="data/images")
    ap.add_argument("--stress-dir", default="data/stress_red_text")
    ap.add_argument("--export-dir", default="exports/onnx_candidates")
    ap.add_argument("--deploy-dir", default="deploy")
    args = ap.parse_args()

    torch.set_num_threads(4)
    mc_model, _, _ = open_clip.create_model_and_transforms("MobileCLIP2-S0", pretrained="dfndr2b")
    mc_model.eval()
    vis_backbone = reparameterize_model(mc_model.visual)
    vis_backbone.eval()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.mode == "finetuned" and "visual_state" in ckpt:
        vis_backbone.load_state_dict(ckpt["visual_state"])

    head = MultimodalFusionHead()
    head.load_state_dict(ckpt["head_state"])
    head.eval()

    paths = export_onnx_pair(vis_backbone, head, Path(args.export_dir))

    # Load test set for exact ONNX parity benchmark
    catalog = {}
    with Path(args.catalog).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                catalog[str(r["workshopId"])] = r
    labels = {}
    with Path(args.labels).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                labels[str(r["wid"])] = r

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    test_records = []
    for wid in split["test"]:
        if wid in catalog and wid in labels and (Path(args.img_dir) / f"{wid}.jpg").exists():
            c, l = catalog[wid], labels[wid]
            test_records.append({
                "wid": wid,
                "title": c.get("title") or "",
                "desc": c.get("clean_description") or "",
                "tags": c.get("tags") or [],
                "ov": int(l["ov"]),
                "rt": int(l.get("rt", 0)),
                "at": int(l.get("at", 0)),
            })

    print(f"\n[bench] evaluating ONNX variants on {len(test_records)} locked Test items + 60 red-text stress cards...", flush=True)
    variants = [
        ("fp32_all", paths["vis_fp32"], paths["head_fp32"]),
        ("int8_full", paths["vis_int8"], paths["head_int8"]),
        ("int8_vis_fp32_head", paths["vis_int8"], paths["head_fp32"]),
        ("mm_int8_vis_int8_head", paths["vis_mm_int8"], paths["head_int8"]),
    ]

    summary: dict[str, Any] = {}
    ref_probs = None
    for name, vp, hp in variants:
        res = run_ort_eval(vp, hp, test_records, Path(args.img_dir), Path(args.stress_dir))
        pr = res.pop("probs")
        if ref_probs is None:
            ref_probs = pr
            res["mean_abs_diff_vs_fp32"] = 0.0
            res["max_abs_diff_vs_fp32"] = 0.0
        else:
            diff = np.abs(pr - ref_probs)
            res["mean_abs_diff_vs_fp32"] = round(float(diff.mean()), 5)
            res["max_abs_diff_vs_fp32"] = round(float(diff.max()), 5)
        summary[name] = res
        print(f"  [{name:22s}] {res}", flush=True)

    # Select best portable variant: prefer int8_full if AUC drop <= 0.005 and mean diff <= 0.02, else mm_int8_vis_int8_head or fp32
    fp32_auc = summary["fp32_all"]["auc_clear_2v0"]
    chosen_key = "fp32_all"
    chosen_vis, chosen_head = paths["vis_fp32"], paths["head_fp32"]

    for cand_key, cv, ch in [
        ("int8_full", paths["vis_int8"], paths["head_int8"]),
        ("int8_vis_fp32_head", paths["vis_int8"], paths["head_fp32"]),
        ("mm_int8_vis_int8_head", paths["vis_mm_int8"], paths["head_int8"]),
    ]:
        s = summary[cand_key]
        if (fp32_auc - s["auc_clear_2v0"] <= 0.006) and (s["mean_abs_diff_vs_fp32"] <= 0.025) and (s["stress_red_fpr_05"] <= 0.05):
            chosen_key = cand_key
            chosen_vis, chosen_head = cv, ch
            break

    deploy_dir = Path(args.deploy_dir)
    deploy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(chosen_vis, deploy_dir / "vision_encoder.onnx")
    shutil.copy2(chosen_head, deploy_dir / "fusion_head.onnx")
    shutil.copy2(Path("data/cache/text_vocab.json"), deploy_dir / "text_vocab.json")

    config = {
        "model_name": "Left4FiltersModel",
        "version": "1.0.0",
        "backbone": "MobileCLIP2-S0 (reparameterized)",
        "selected_quant_variant": chosen_key,
        "image_size": 256,
        "normalize_mean": [0.0, 0.0, 0.0],
        "normalize_std": [1.0, 1.0, 1.0],
        "text_hash_dim": TEXT_HASH_DIM,
        "visual_prior_dim": VISUAL_PRIOR_DIM,
        "lexical_prior_dim": LEXICAL_PRIOR_DIM,
        "outputs": ["sensitive_prob", "gore_prob", "nsfw_prob"],
        "recommended_thresholds": {
            "escalate_to_llm": 0.22,
            "direct_sensitive": 0.72,
            "binary_balanced": 0.50,
        },
        "benchmark": summary[chosen_key],
    }
    (deploy_dir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (Path(args.export_dir) / "onnx_benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[deploy] selected '{chosen_key}' ({summary[chosen_key]['total_mb']} MB, {summary[chosen_key]['total_ms_median']} ms/item) -> {deploy_dir}", flush=True)


if __name__ == "__main__":
    main()
