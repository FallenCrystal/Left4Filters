#!/usr/bin/env python3
"""Retrain the fusion head on the deployed encoder and compare on the locked split."""
from __future__ import annotations

import argparse
import importlib
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.l4filter import extract_visual_priors, preprocess_image
from scripts.synth_red_text import TRAIN_PHRASES, render_red_text_card
from src.l4filter_model import MultimodalFusionHead

training = importlib.import_module("scripts.04_train_and_compare")
export = importlib.import_module("scripts.06_export_onnx_and_bench")
THRESHOLDS = (0.20, 0.72)
MODES = ("both", "image_only", "text_only")
HEAD_INPUTS = ("img_emb", "img_priors", "txt_hash", "txt_lex", "has_img", "has_txt")


def session(path: Path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def encode_images(
    vision: ort.InferenceSession, images, image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings, priors, batch = [], [], []
    for img in images:
        priors.append(extract_visual_priors(img))
        batch.append(preprocess_image(img, image_size))
        if len(batch) == 12:
            embeddings.append(vision.run(["img_emb"], {"pixel_values": np.stack(batch)})[0])
            batch.clear()
    if batch:
        embeddings.append(vision.run(["img_emb"], {"pixel_values": np.stack(batch)})[0])
    return np.concatenate(embeddings), np.stack(priors)


def load_images(paths):
    for path in paths:
        with Image.open(path) as im:
            yield im.convert("RGB")


def head_inputs(arr: dict[str, np.ndarray], mode: str) -> tuple[np.ndarray, ...]:
    n = len(arr["img_emb"])
    return (
        arr["img_emb"], arr["img_pri"], arr["txt_hash"], arr["txt_lex"],
        np.full((n, 1), mode != "text_only", np.float32),
        np.full((n, 1), mode != "image_only", np.float32),
    )


def onnx_predict(head: ort.InferenceSession, arr: dict[str, np.ndarray], mode: str) -> np.ndarray:
    return head.run(["probs"], dict(zip(HEAD_INPUTS, head_inputs(arr, mode))))[0]


def torch_predict(head: nn.Module, arr: dict[str, np.ndarray], mode: str) -> np.ndarray:
    head.eval()
    with torch.inference_mode():
        return head(*(torch.from_numpy(a) for a in head_inputs(arr, mode))).numpy()


class MeanFusionHead(nn.Module):
    def __init__(self, first: nn.Module, second: nn.Module) -> None:
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, img_emb, img_priors, txt_hash, txt_lex, has_img, has_txt):
        inputs = (img_emb, img_priors, txt_hash, txt_lex, has_img, has_txt)
        return (self.first(*inputs) + self.second(*inputs)) * 0.5


def metrics(predict, arr: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        mode: training.evaluate_predictions(predict(arr, mode), arr, mode, THRESHOLDS)
        for mode in MODES
    }


def selection_score(result: dict[str, dict[str, float]]) -> float:
    both = result["both"]
    return (
        both["auc_clear_2v0"] + both["auc_any_12v0"] - both["brier_soft"]
        - 0.45 * both["gate_false_block_safe_rate"]
        - 0.25 * both["gate_missed_strong_rate"]
        - 0.10 * both["gate_llm_frac"]
        + 0.2 * (result["image_only"]["auc_any_12v0"] + result["text_only"]["auc_any_12v0"])
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", type=Path, default=Path("deploy"))
    ap.add_argument("--catalog", type=Path, default=Path("data/raw/catalog_items.jsonl"))
    ap.add_argument("--labels", type=Path, default=Path("data/annotations/merged_final.jsonl"))
    ap.add_argument("--split", type=Path, default=Path("data/annotations/split_manifest.json"))
    ap.add_argument("--image-dir", type=Path, default=Path("data/images"))
    ap.add_argument("--stress-dir", type=Path, default=Path("data/stress_red_text"))
    ap.add_argument("--cache", type=Path, help="Optional embeddings computed with this base vision encoder")
    ap.add_argument("--baseline-checkpoint", type=Path, help="Optional PyTorch checkpoint for warm starts")
    ap.add_argument("--out-dir", type=Path, default=Path("exports/retrained"))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.base_dir / "model_config.json").read_text())
    vocab = json.loads((args.base_dir / "text_vocab.json").read_text())
    image_size = cfg["image_size"]
    vision = session(args.base_dir / "vision_encoder.onnx", args.threads)
    base_head = session(args.base_dir / "fusion_head.onnx", args.threads)

    stress_files = sorted(args.stress_dir.glob("*.jpg"))
    paths = {f"stress:{p.name}": p for p in stress_files}
    with args.labels.open(encoding="utf-8") as labels:
        for line in labels:
            if line.strip():
                wid = str(json.loads(line)["wid"])
                path = args.image_dir / f"{wid}.jpg"
                if path.is_file():
                    paths[wid] = path
    cache = args.cache or args.out_dir / "fusion_features.npz"
    if args.cache is None or not cache.exists():
        emb, pri = encode_images(vision, load_images(paths.values()), image_size)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, wids=np.array(list(paths)), emb=emb, priors=pri)
    samples, indexes, emb, pri = training.load_data(args.catalog, args.labels, cache)
    if missing := paths.keys() - indexes.keys():
        raise ValueError(f"The feature cache is missing {len(missing)} requested images. Re-run without --cache.")
    train, val, test = training.make_or_load_creator_split(samples, args.split)
    saved_split = json.loads(args.split.read_text())
    for name, rows in (("val", val), ("test", test)):
        if set(saved_split[name]) != {s["wid"] for s in rows}:
            raise ValueError(f"Missing image features or labels in the locked {name} split")

    def arrays(rows):
        return training.build_feature_arrays(rows, emb, pri, vocab)

    tr, va, te = (arrays(rows) for rows in (train, val, test))
    rng = random.Random(1234)
    phrases = [rng.choice(TRAIN_PHRASES) for _ in range(120)]
    red_emb, red_pri = encode_images(vision, (render_red_text_card(p, rng) for p in phrases), image_size)
    red = training.build_feature_arrays([
        dict(cache_idx=i, title=f"{p} Red Font", desc="Safe text or UI mod", tags=["UI"],
             ig=0, tg=0, tn=0, ov=0, rt=1, at=0, **{"in": 0})
        for i, p in enumerate(phrases)
    ], red_emb, red_pri, vocab)
    tr = training.concat_arrays(tr, red)

    def base_predict(arr, mode):
        return onnx_predict(base_head, arr, mode)

    base_val = metrics(base_predict, va)
    scores = {"baseline": selection_score(base_val)}
    best_name, best = "baseline", None
    print(f"[val] baseline = {scores['baseline']:.6f}", flush=True)

    def consider(name, candidate):
        nonlocal best_name, best
        result = metrics(lambda arr, mode: torch_predict(candidate, arr, mode), va)
        scores[name] = selection_score(result)
        both = result["both"]
        print(f"[val] {name} = {scores[name]:.6f} clear={both['auc_clear_2v0']:.4f} "
              f"any={both['auc_any_12v0']:.4f} miss={both['gate_missed_strong_rate']:.4f}", flush=True)
        if scores[name] > scores[best_name]:
            best_name, best = name, candidate

    warm_state = None
    if args.baseline_checkpoint:
        warm_state = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=True)["head_state"]
        previous = MultimodalFusionHead()
        previous.load_state_dict(warm_state)
    recipes = [("fresh_42", 42, 50, 1.2e-3, 0.02, None),
               ("fresh_101", 101, 50, 1.2e-3, 0.012, None)]
    if warm_state is not None:
        recipes += [("warm_2026", 2026, 22, 2.5e-4, 0.012, warm_state),
                    ("warm_777", 777, 26, 4.0e-4, 0.020, warm_state)]
    for name, seed, epochs, lr, noise, state in recipes:
        candidate, _ = training.train_fusion_head(
            tr, va, epochs=epochs, lr=lr, seed=seed, embedding_noise=noise, initial_state=state,
        )
        consider(name, candidate)
        if warm_state is not None:
            consider(f"{name}_blend", MeanFusionHead(previous, candidate))

    out_head = args.out_dir / "fusion_head.onnx"
    if best is not None:
        _, quantized = export.export_fusion_head(best, args.out_dir)
        shutil.copyfile(quantized, out_head)
    elif out_head.resolve() != (args.base_dir / "fusion_head.onnx").resolve():
        shutil.copyfile(args.base_dir / "fusion_head.onnx", out_head)
    selected_head = session(out_head, args.threads)

    def selected_predict(arr, mode):
        return onnx_predict(selected_head, arr, mode)

    def stress(predict):
        if not stress_files:
            return {}
        arr = arrays([
            dict(cache_idx=indexes[f"stress:{p.name}"], title="", desc="", tags=[],
                 ig=0, tg=0, tn=0, ov=0, rt=1, at=0, **{"in": 0})
            for p in stress_files
        ])
        probabilities = predict(arr, "image_only")[:, 0]
        return {"n": len(probabilities), "max": float(probabilities.max()),
                "mean": float(probabilities.mean()), "at_review": int((probabilities >= THRESHOLDS[0]).sum())}

    results = {
        "selected": best_name, "validation_scores": scores,
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "baseline_validation": base_val, "selected_validation": metrics(selected_predict, va),
        "baseline_test": metrics(base_predict, te), "selected_test": metrics(selected_predict, te),
        "baseline_red_stress": stress(base_predict), "selected_red_stress": stress(selected_predict),
    }
    (args.out_dir / "fusion_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"[onnx] selected {best_name}, evaluated on {len(test)} locked test items", flush=True)


if __name__ == "__main__":
    main()
