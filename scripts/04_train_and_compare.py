#!/usr/bin/env python3
"""
Train, compare, calibrate, and evaluate lightweight multimodal sensitivity models
for Left 4 Dead 2 Steam Workshop items.

Experiments compared on an identical creator-disjoint Test split:
  1. baseline_shallow       : Text Hashing + Lexical + 24-dim visual color/edge priors (no deep vision)
  2. mc2s0_frozen_fusion    : MobileCLIP2-S0 (frozen visual) + MultimodalFusionHead + Modality Dropout + Calibration
  3. mc2s0_finetuned_fusion : MobileCLIP2-S0 (unfrozen final visual stage + head fine-tuning) + MultimodalFusionHead + Calibration
  + Modality ablations (Image-only, Text-only)
  + Special trap evaluations (Real Red-Text Traps, 60-image Synthetic Red-Text Stress Test, Safe-Anime Traps)
  + Two-stage LLM-routing threshold analysis (Safe / Escalate to LLM / Direct Sensitive)
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from timm.utils.model import reparameterize_model
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.synth_red_text import TRAIN_PHRASES, render_red_text_card
from src.l4filter_features import (
    LEXICAL_PRIOR_DIM,
    TEXT_VOCAB_DIM as TEXT_HASH_DIM,
    VISUAL_PRIOR_DIM,
    extract_text_features,
    extract_visual_priors,
)
from src.l4filter_model import MultimodalFusionHead, VisualEncoderWrapper

SOFT_SUB = {0: 0.0, 1: 0.42, 2: 1.0}
SOFT_OV = {0: 0.0, 1: 0.48, 2: 1.0}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(
    catalog_path: Path,
    labels_path: Path,
    cache_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, int], np.ndarray, np.ndarray]:
    catalog: dict[str, dict[str, Any]] = {}
    with catalog_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                catalog[str(r["workshopId"])] = r

    cache = np.load(cache_path, allow_pickle=True)
    wids_cache = [str(x) for x in cache["wids"]]
    wid_to_idx = {w: i for i, w in enumerate(wids_cache)}
    embs = cache["emb"].astype(np.float32)
    priors = cache["priors"].astype(np.float32)

    samples: list[dict[str, Any]] = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            lbl = json.loads(line)
            wid = str(lbl["wid"])
            if wid not in catalog or wid not in wid_to_idx:
                continue
            cat = catalog[wid]
            samples.append({
                "wid": wid,
                "creator": cat.get("creatorSteamId") or wid,
                "title": cat.get("title") or "",
                "desc": cat.get("description") or cat.get("clean_description") or "",
                "tags": cat.get("tags") or [],
                "ig": int(lbl["ig"]),
                "in": int(lbl["in"]),
                "tg": int(lbl["tg"]),
                "tn": int(lbl["tn"]),
                "ov": int(lbl["ov"]),
                "rt": int(lbl.get("rt", 0)),
                "at": int(lbl.get("at", 0)),
                "cache_idx": wid_to_idx[wid],
            })
    print(f"[data] loaded {len(samples)} labeled samples matched with catalog & image cache", flush=True)
    return samples, wid_to_idx, embs, priors


def make_or_load_creator_split(
    samples: list[dict[str, Any]],
    split_path: Path,
    val_frac: float = 0.14,
    test_frac: float = 0.14,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Creator-disjoint stratified split.
    If split_path exists (e.g. Round 2 active learning), preserves locked val/test wids
    and adds all newly annotated items strictly into train (excluding any creators already in val/test)!
    """
    if split_path.exists():
        saved = json.loads(split_path.read_text(encoding="utf-8"))
        val_wids = set(saved["val"])
        test_wids = set(saved["test"])
        val_creators = set(saved.get("val_creators", []))
        test_creators = set(saved.get("test_creators", []))

        train_s, val_s, test_s = [], [], []
        for s in samples:
            if s["wid"] in test_wids:
                test_s.append(s)
            elif s["wid"] in val_wids:
                val_s.append(s)
            elif s["creator"] in test_creators:
                # Keep creator isolation strict: route extra items from test creators out of train
                continue
            elif s["creator"] in val_creators:
                continue
            else:
                train_s.append(s)
        print(
            f"[split:locked] train={len(train_s)} val={len(val_s)} test={len(test_s)} (strict creator-disjoint)",
            flush=True,
        )
        return train_s, val_s, test_s

    rng = random.Random(seed)
    by_creator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_creator[s["creator"]].append(s)

    # Stratify creator groups by their dominant risk type
    groups_by_key: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for cid, grp in by_creator.items():
        has_gore = any(max(x["ig"], x["tg"]) == 2 for x in grp)
        has_nsfw = any(max(x["in"], x["tn"]) == 2 for x in grp)
        has_rt = any(x["rt"] == 1 and x["ig"] == 0 for x in grp)
        has_at = any(x["at"] == 1 and x["in"] == 0 for x in grp)
        if has_gore:
            k = "gore"
        elif has_nsfw:
            k = "nsfw"
        elif has_rt:
            k = "red_trap"
        elif has_at:
            k = "anime_trap"
        else:
            k = "other"
        groups_by_key[k].append((cid, grp))

    train_s, val_s, test_s = [], [], []
    val_creators, test_creators = set(), set()

    for k, grps in groups_by_key.items():
        rng.shuffle(grps)
        total_k = sum(len(g) for _, g in grps)
        t_target = int(round(total_k * test_frac))
        v_target = int(round(total_k * val_frac))
        t_cnt = v_cnt = 0
        for cid, grp in grps:
            if t_cnt < t_target:
                test_s.extend(grp)
                test_creators.add(cid)
                t_cnt += len(grp)
            elif v_cnt < v_target:
                val_s.extend(grp)
                val_creators.add(cid)
                v_cnt += len(grp)
            else:
                train_s.extend(grp)

    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(
            {
                "train": [s["wid"] for s in train_s],
                "val": [s["wid"] for s in val_s],
                "test": [s["wid"] for s in test_s],
                "val_creators": sorted(val_creators),
                "test_creators": sorted(test_creators),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[split:new] train={len(train_s)} val={len(val_s)} test={len(test_s)} -> {split_path}", flush=True)
    return train_s, val_s, test_s


def build_feature_arrays(
    samples: list[dict[str, Any]],
    embs: np.ndarray,
    priors: np.ndarray,
    vocab_data: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    n = len(samples)
    img_emb = np.zeros((n, embs.shape[1]), dtype=np.float32)
    img_pri = np.zeros((n, VISUAL_PRIOR_DIM), dtype=np.float32)
    txt_hash = np.zeros((n, TEXT_HASH_DIM), dtype=np.float32)
    txt_lex = np.zeros((n, LEXICAL_PRIOR_DIM), dtype=np.float32)

    y_ov = np.zeros(n, dtype=np.float32)
    y_ig = np.zeros(n, dtype=np.float32)
    y_in = np.zeros(n, dtype=np.float32)
    y_tg = np.zeros(n, dtype=np.float32)
    y_tn = np.zeros(n, dtype=np.float32)
    y_ov_int = np.zeros(n, dtype=np.int64)
    y_gore_int = np.zeros(n, dtype=np.int64)
    y_nsfw_int = np.zeros(n, dtype=np.int64)
    rt = np.zeros(n, dtype=np.int64)
    at = np.zeros(n, dtype=np.int64)
    weights = np.ones(n, dtype=np.float32)

    for i, s in enumerate(samples):
        idx = s["cache_idx"]
        img_emb[i] = embs[idx]
        img_pri[i] = priors[idx]
        h, l = extract_text_features(s["title"], s["desc"], s["tags"], vocab_data)
        txt_hash[i] = h
        txt_lex[i] = l

        y_ov[i] = SOFT_OV[s["ov"]]
        y_ig[i] = SOFT_SUB[s["ig"]]
        y_in[i] = SOFT_SUB[s["in"]]
        y_tg[i] = SOFT_SUB[s["tg"]]
        y_tn[i] = SOFT_SUB[s["tn"]]
        y_ov_int[i] = s["ov"]
        y_gore_int[i] = max(s["ig"], s["tg"])
        y_nsfw_int[i] = max(s["in"], s["tn"])
        rt[i] = s["rt"]
        at[i] = s["at"]

        # Upweight hard-negative traps & clear positive examples slightly
        if (s["rt"] == 1 and s["ig"] == 0) or (s["at"] == 1 and s["in"] == 0):
            weights[i] = 1.65
        elif s["ov"] == 2:
            weights[i] = 1.25

    return {
        "img_emb": img_emb,
        "img_pri": img_pri,
        "txt_hash": txt_hash,
        "txt_lex": txt_lex,
        "y_ov": y_ov,
        "y_ig": y_ig,
        "y_in": y_in,
        "y_tg": y_tg,
        "y_tn": y_tn,
        "y_ov_int": y_ov_int,
        "y_gore_int": y_gore_int,
        "y_nsfw_int": y_nsfw_int,
        "rt": rt,
        "at": at,
        "weights": weights,
    }


def generate_synth_red_train_features(
    visual_model: nn.Module,
    preprocess,
    count: int = 120,
    seed: int = 1234,
) -> dict[str, np.ndarray]:
    """Generate synthetic SAFE red-text negative examples (train-only phrases!)"""
    rng = random.Random(seed)
    img_emb = np.zeros((count, 512), dtype=np.float32)
    img_pri = np.zeros((count, VISUAL_PRIOR_DIM), dtype=np.float32)
    txt_hash = np.zeros((count, TEXT_HASH_DIM), dtype=np.float32)
    txt_lex = np.zeros((count, LEXICAL_PRIOR_DIM), dtype=np.float32)

    tensors = []
    for i in range(count):
        phrase = rng.choice(TRAIN_PHRASES)
        im = render_red_text_card(phrase, rng)
        img_pri[i] = extract_visual_priors(im)
        tensors.append(preprocess(im))
        # Mix normal title and L4D red/blood-harvest style safe titles
        title = phrase if rng.random() < 0.7 else f"{phrase} - Red Font UI"
        h, l = extract_text_features(title, "Simple text/font/UI mod for Left 4 Dead 2", ["UI", "Textures"])
        txt_hash[i] = h
        txt_lex[i] = l

    visual_model.eval()
    with torch.no_grad():
        for start in range(0, count, 32):
            bx = torch.stack(tensors[start : start + 32], dim=0)
            feat = visual_model(bx)
            feat = F.normalize(feat, dim=-1)
            img_emb[start : start + len(bx)] = feat.cpu().numpy()

    zeros = np.zeros(count, dtype=np.float32)
    zint = np.zeros(count, dtype=np.int64)
    return {
        "img_emb": img_emb,
        "img_pri": img_pri,
        "txt_hash": txt_hash,
        "txt_lex": txt_lex,
        "y_ov": zeros,
        "y_ig": zeros,
        "y_in": zeros,
        "y_tg": zeros,
        "y_tn": zeros,
        "y_ov_int": zint,
        "y_gore_int": zint,
        "y_nsfw_int": zint,
        "rt": np.ones(count, dtype=np.int64),
        "at": zint,
        "weights": np.full(count, 1.25, dtype=np.float32),
    }


def concat_arrays(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([a[k], b[k]], axis=0) for k in a}


def compute_multitask_loss(
    out: dict[str, torch.Tensor],
    y_ov: torch.Tensor,
    y_ig: torch.Tensor,
    y_in: torch.Tensor,
    y_tg: torch.Tensor,
    y_tn: torch.Tensor,
    has_img: torch.Tensor,
    has_txt: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    # Target for overall/gore/nsfw adapts dynamically when one modality is dropped!
    both = (has_img > 0.5) & (has_txt > 0.5)
    only_img = (has_img > 0.5) & (has_txt < 0.5)

    tgt_gore_both = torch.maximum(y_ig, y_tg)
    tgt_nsfw_both = torch.maximum(y_in, y_tn)
    tgt_gore = torch.where(both.squeeze(-1), tgt_gore_both, torch.where(only_img.squeeze(-1), y_ig, y_tg))
    tgt_nsfw = torch.where(both.squeeze(-1), tgt_nsfw_both, torch.where(only_img.squeeze(-1), y_in, y_tn))
    tgt_ov = torch.where(
        both.squeeze(-1),
        y_ov,
        torch.where(only_img.squeeze(-1), torch.maximum(y_ig, y_in), torch.maximum(y_tg, y_tn)),
    )

    def wbce(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(logit.squeeze(-1), target, reduction="none")
        w = weights
        if mask is not None:
            w = w * mask.squeeze(-1)
        return (loss * w).sum() / (w.sum() + 1e-6)

    l_ov = wbce(out["overall_logit"], tgt_ov)
    l_gore = wbce(out["gore_logit"], tgt_gore)
    l_nsfw = wbce(out["nsfw_logit"], tgt_nsfw)

    l_ig = wbce(out["vis_logits"][:, 0:1], y_ig, has_img)
    l_in = wbce(out["vis_logits"][:, 1:2], y_in, has_img)
    l_tg = wbce(out["txt_logits"][:, 0:1], y_tg, has_txt)
    l_tn = wbce(out["txt_logits"][:, 1:2], y_tn, has_txt)

    return 1.0 * l_ov + 0.55 * l_gore + 0.55 * l_nsfw + 0.45 * (l_ig + l_in) + 0.45 * (l_tg + l_tn)


def fit_calibration(
    model: MultimodalFusionHead,
    val_arr: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit post-hoc Platt scaling (scale, bias) on validation set logits for [overall, gore, nsfw]."""
    model.eval()
    with torch.no_grad():
        n = len(val_arr["y_ov"])
        out = model.forward_logits(
            torch.from_numpy(val_arr["img_emb"]),
            torch.from_numpy(val_arr["img_pri"]),
            torch.from_numpy(val_arr["txt_hash"]),
            torch.from_numpy(val_arr["txt_lex"]),
            torch.ones(n, 1),
            torch.ones(n, 1),
        )
        raw = torch.cat([out["overall_logit"], out["gore_logit"], out["nsfw_logit"]], dim=-1).numpy()

    targets = [
        val_arr["y_ov"],
        np.maximum(val_arr["y_ig"], val_arr["y_tg"]),
        np.maximum(val_arr["y_in"], val_arr["y_tn"]),
    ]
    scales = np.ones(3, dtype=np.float32)
    biases = np.zeros(3, dtype=np.float32)

    for k in range(3):
        logits_k = raw[:, k]
        y_k = targets[k]

        def obj(params: np.ndarray) -> float:
            s, b = params
            z = np.clip(logits_k * s + b, -20.0, 20.0)
            p = 1.0 / (1.0 + np.exp(-z))
            eps = 1e-6
            nll = -np.mean(y_k * np.log(p + eps) + (1.0 - y_k) * np.log(1.0 - p + eps))
            # Mild regularization toward (1, 0) to prevent overfitting on small val sets
            reg = 0.005 * ((s - 1.0) ** 2 + b ** 2)
            return float(nll + reg)

        res = minimize(obj, x0=np.array([1.0, 0.0]), bounds=[(0.35, 3.0), (-2.0, 2.0)], method="L-BFGS-B")
        scales[k] = float(res.x[0])
        biases[k] = float(res.x[1])

    model.set_calibration(torch.from_numpy(scales), torch.from_numpy(biases))
    return scales, biases


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(float(probs[m].mean()) - float(targets[m].mean()))
    return float(ece)


def evaluate_model(
    model: MultimodalFusionHead,
    arr: dict[str, np.ndarray],
    mode: str = "both",
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    n = len(arr["y_ov"])
    has_img = torch.ones(n, 1) if mode in ("both", "image_only") else torch.zeros(n, 1)
    has_txt = torch.ones(n, 1) if mode in ("both", "text_only") else torch.zeros(n, 1)

    with torch.no_grad():
        probs = model(
            torch.from_numpy(arr["img_emb"]),
            torch.from_numpy(arr["img_pri"]),
            torch.from_numpy(arr["txt_hash"]),
            torch.from_numpy(arr["txt_lex"]),
            has_img,
            has_txt,
        ).numpy()

    return evaluate_predictions(probs, arr, mode), probs


def evaluate_predictions(
    probs: np.ndarray,
    arr: dict[str, np.ndarray],
    mode: str = "both",
    thresholds: tuple[float, float] = (0.22, 0.72),
) -> dict[str, float]:
    p_ov, p_gore, p_nsfw = probs[:, 0], probs[:, 1], probs[:, 2]

    if mode == "both":
        y_int = arr["y_ov_int"]
        y_soft = arr["y_ov"]
        y_g_int = arr["y_gore_int"]
        y_n_int = arr["y_nsfw_int"]
    elif mode == "image_only":
        y_ig_i = np.round(arr["y_ig"] * 2).astype(int)
        y_in_i = np.round(arr["y_in"] * 2).astype(int)
        y_int = np.maximum(y_ig_i, y_in_i)
        y_soft = np.maximum(arr["y_ig"], arr["y_in"])
        y_g_int, y_n_int = y_ig_i, y_in_i
    else:
        y_tg_i = np.round(arr["y_tg"] * 2).astype(int)
        y_tn_i = np.round(arr["y_tn"] * 2).astype(int)
        y_int = np.maximum(y_tg_i, y_tn_i)
        y_soft = np.maximum(arr["y_tg"], arr["y_tn"])
        y_g_int, y_n_int = y_tg_i, y_tn_i

    metrics: dict[str, float] = {}
    # 1. Clear positive (2) vs Safe (0)
    m02 = (y_int == 0) | (y_int == 2)
    if (y_int[m02] == 2).sum() > 0 and (y_int[m02] == 0).sum() > 0:
        metrics["auc_clear_2v0"] = float(roc_auc_score((y_int[m02] == 2).astype(int), p_ov[m02]))
        metrics["ap_clear_2v0"] = float(average_precision_score((y_int[m02] == 2).astype(int), p_ov[m02]))

    # 2. Any sensitive (>=1) vs Safe (0)
    if (y_int >= 1).sum() > 0 and (y_int == 0).sum() > 0:
        metrics["auc_any_12v0"] = float(roc_auc_score((y_int >= 1).astype(int), p_ov))
        metrics["ap_any_12v0"] = float(average_precision_score((y_int >= 1).astype(int), p_ov))
        # Best F1 on >=2 vs <2 and >=1 vs 0
        metrics["f1_at_05_any"] = float(f1_score((y_int >= 1).astype(int), (p_ov >= 0.5).astype(int)))

    if (y_int == 2).sum() > 0 and (y_int < 2).sum() > 0:
        metrics["auc_strong_2v01"] = float(roc_auc_score((y_int == 2).astype(int), p_ov))
        metrics["f1_at_05_strong"] = float(f1_score((y_int == 2).astype(int), (p_ov >= 0.5).astype(int)))

    # Gore & NSFW sub-AUCs
    for name, p_sub, y_sub in [("gore", p_gore, y_g_int), ("nsfw", p_nsfw, y_n_int)]:
        ms = (y_sub == 0) | (y_sub == 2)
        if (y_sub[ms] == 2).sum() > 0 and (y_sub[ms] == 0).sum() > 0:
            metrics[f"auc_{name}_2v0"] = float(roc_auc_score((y_sub[ms] == 2).astype(int), p_sub[ms]))

    # Calibration & soft Brier
    metrics["brier_soft"] = float(np.mean((p_ov - y_soft) ** 2))
    metrics["ece_soft"] = compute_ece(p_ov, y_soft)

    # Trap FPRs (specifically for mode == both / image_only)
    m_rt = (arr["rt"] == 1) & (y_int == 0)
    if m_rt.sum() > 0:
        metrics["red_trap_n"] = float(m_rt.sum())
        metrics["red_trap_mean_p"] = float(p_ov[m_rt].mean())
        metrics["red_trap_fpr_05"] = float((p_ov[m_rt] >= 0.5).mean())

    m_at = (arr["at"] == 1) & (y_int == 0)
    if m_at.sum() > 0:
        metrics["anime_trap_n"] = float(m_at.sum())
        metrics["anime_trap_mean_p"] = float(p_ov[m_at].mean())
        metrics["anime_trap_fpr_05"] = float((p_ov[m_at] >= 0.5).mean())

    t_lo, t_hi = thresholds
    safe_zone = p_ov < t_lo
    llm_zone = (p_ov >= t_lo) & (p_ov < t_hi)
    block_zone = p_ov >= t_hi
    metrics["gate_safe_frac"] = float(safe_zone.mean())
    metrics["gate_llm_frac"] = float(llm_zone.mean())
    metrics["gate_block_frac"] = float(block_zone.mean())
    if (y_int == 2).sum() > 0:
        metrics["gate_missed_strong_rate"] = float((safe_zone & (y_int == 2)).sum() / (y_int == 2).sum())
    if (y_int == 0).sum() > 0:
        metrics["gate_false_block_safe_rate"] = float((block_zone & (y_int == 0)).sum() / (y_int == 0).sum())

    return metrics


def eval_synth_red_stress(
    model: MultimodalFusionHead,
    visual_model: nn.Module,
    preprocess,
    stress_dir: Path,
) -> dict[str, float]:
    manifest = json.loads((stress_dir / "manifest.json").read_text(encoding="utf-8"))
    n = len(manifest)
    img_emb = np.zeros((n, 512), dtype=np.float32)
    img_pri = np.zeros((n, VISUAL_PRIOR_DIM), dtype=np.float32)
    txt_hash = np.zeros((n, TEXT_HASH_DIM), dtype=np.float32)
    txt_lex = np.zeros((n, LEXICAL_PRIOR_DIM), dtype=np.float32)

    tensors = []
    for i, rec in enumerate(manifest):
        with Image.open(rec["image"]) as im:
            im_rgb = im.convert("RGB")
            img_pri[i] = extract_visual_priors(im_rgb)
            tensors.append(preprocess(im_rgb))
        h, l = extract_text_features(rec["title"], rec["description"], rec["tags"])
        txt_hash[i] = h
        txt_lex[i] = l

    visual_model.eval()
    with torch.no_grad():
        for st in range(0, n, 32):
            bx = torch.stack(tensors[st : st + 32], dim=0)
            feat = F.normalize(visual_model(bx), dim=-1)
            img_emb[st : st + len(bx)] = feat.cpu().numpy()

    model.eval()
    with torch.no_grad():
        p_both = model(
            torch.from_numpy(img_emb), torch.from_numpy(img_pri),
            torch.from_numpy(txt_hash), torch.from_numpy(txt_lex),
            torch.ones(n, 1), torch.ones(n, 1),
        ).numpy()
        p_img = model(
            torch.from_numpy(img_emb), torch.from_numpy(img_pri),
            torch.zeros(n, TEXT_HASH_DIM), torch.zeros(n, LEXICAL_PRIOR_DIM),
            torch.ones(n, 1), torch.zeros(n, 1),
        ).numpy()

    return {
        "n": float(n),
        "both_mean_p": float(p_both[:, 0].mean()),
        "both_max_p": float(p_both[:, 0].max()),
        "both_fpr_05": float((p_both[:, 0] >= 0.5).mean()),
        "both_fpr_022": float((p_both[:, 0] >= 0.22).mean()),
        "img_only_mean_p": float(p_img[:, 0].mean()),
        "img_only_max_p": float(p_img[:, 0].max()),
        "img_only_fpr_05": float((p_img[:, 0] >= 0.5).mean()),
        "img_only_gore_mean_p": float(p_img[:, 1].mean()),
    }


def train_fusion_head(
    train_arr: dict[str, np.ndarray],
    val_arr: dict[str, np.ndarray],
    zero_img_emb: bool = False,
    epochs: int = 45,
    batch_size: int = 64,
    lr: float = 1.2e-3,
    weight_decay: float = 2e-2,
    seed: int = 42,
    initial_state: dict[str, torch.Tensor] | None = None,
    embedding_noise: float = 0.025,
) -> tuple[MultimodalFusionHead, float]:
    seed_everything(seed)
    tr = {k: v.copy() for k, v in train_arr.items()}
    va = {k: v.copy() for k, v in val_arr.items()}
    if zero_img_emb:
        tr["img_emb"][:] = 0.0
        va["img_emb"][:] = 0.0

    model = MultimodalFusionHead()
    if initial_state is not None:
        model.load_state_dict(initial_state)
        model.set_calibration(torch.ones(3), torch.zeros(3))
    else:
        model.set_feature_stats(
            torch.from_numpy(tr["img_pri"].mean(axis=0)),
            torch.from_numpy(tr["img_pri"].std(axis=0)),
            torch.from_numpy(tr["txt_lex"].mean(axis=0)),
            torch.from_numpy(tr["txt_lex"].std(axis=0)),
        )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    n = len(tr["y_ov"])
    best_score = -1e9
    best_state = copy.deepcopy(model.state_dict())

    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        for st in range(0, n, batch_size):
            idx = perm[st : st + batch_size]
            b = len(idx)

            # Modality dropout: 60% both, 20% img-only, 20% txt-only
            r = np.random.rand(b)
            has_img = torch.from_numpy((r < 0.80).astype(np.float32)).unsqueeze(-1)
            has_txt = torch.from_numpy((r >= 0.20).astype(np.float32)).unsqueeze(-1)

            # Slight feature noise on embeddings for strong generalization
            emb_t = torch.from_numpy(tr["img_emb"][idx])
            if not zero_img_emb:
                emb_t = F.normalize(emb_t + embedding_noise * torch.randn_like(emb_t), dim=-1)

            out = model.forward_logits(
                emb_t,
                torch.from_numpy(tr["img_pri"][idx]),
                torch.from_numpy(tr["txt_hash"][idx]),
                torch.from_numpy(tr["txt_lex"][idx]),
                has_img,
                has_txt,
            )
            loss = compute_multitask_loss(
                out,
                torch.from_numpy(tr["y_ov"][idx]),
                torch.from_numpy(tr["y_ig"][idx]),
                torch.from_numpy(tr["y_in"][idx]),
                torch.from_numpy(tr["y_tg"][idx]),
                torch.from_numpy(tr["y_tn"][idx]),
                has_img,
                has_txt,
                torch.from_numpy(tr["weights"][idx]),
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        # Validate
        vm, _ = evaluate_model(model, va, mode="both")
        # Composite selection score: high AUC + low Brier + low red-trap FPR
        score = (
            vm.get("auc_clear_2v0", 0.5)
            + 0.7 * vm.get("auc_any_12v0", 0.5)
            - 1.2 * vm.get("brier_soft", 0.25)
            - 0.25 * vm.get("red_trap_mean_p", 0.0)
            - 0.15 * vm.get("anime_trap_mean_p", 0.0)
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    fit_calibration(model, va)
    return model, best_score


class FinetuneDataset(Dataset):
    def __init__(
        self,
        samples: list[dict[str, Any]],
        img_dir: Path,
        arr: dict[str, np.ndarray],
        is_train: bool = True,
        synth_red_count: int = 80,
    ) -> None:
        self.samples = samples
        self.img_dir = img_dir
        self.arr = arr
        self.is_train = is_train
        self.synth_red_count = synth_red_count if is_train else 0
        self.rng = random.Random(777)

        if is_train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(256, scale=(0.82, 1.0), ratio=(0.85, 1.18)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.015),
                transforms.ToTensor(),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.samples) + self.synth_red_count

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx >= len(self.samples):
            # Dynamically rendered safe red-text sample
            phrase = self.rng.choice(TRAIN_PHRASES)
            im = render_red_text_card(phrase, self.rng)
            pri = extract_visual_priors(im)
            px = self.tf(im)
            h, l = extract_text_features(f"{phrase} Red Font", "Safe red text/UI addon", ["UI"])
            return {
                "px": px,
                "img_pri": torch.from_numpy(pri),
                "txt_hash": torch.from_numpy(h),
                "txt_lex": torch.from_numpy(l),
                "y_ov": torch.tensor(0.0),
                "y_ig": torch.tensor(0.0),
                "y_in": torch.tensor(0.0),
                "y_tg": torch.tensor(0.0),
                "y_tn": torch.tensor(0.0),
                "w": torch.tensor(1.25),
            }

        s = self.samples[idx]
        p = self.img_dir / f"{s['wid']}.jpg"
        with Image.open(p) as im:
            im_rgb = im.convert("RGB")
            px = self.tf(im_rgb)
        return {
            "px": px,
            "img_pri": torch.from_numpy(self.arr["img_pri"][idx]),
            "txt_hash": torch.from_numpy(self.arr["txt_hash"][idx]),
            "txt_lex": torch.from_numpy(self.arr["txt_lex"][idx]),
            "y_ov": torch.tensor(self.arr["y_ov"][idx]),
            "y_ig": torch.tensor(self.arr["y_ig"][idx]),
            "y_in": torch.tensor(self.arr["y_in"][idx]),
            "y_tg": torch.tensor(self.arr["y_tg"][idx]),
            "y_tn": torch.tensor(self.arr["y_tn"][idx]),
            "w": torch.tensor(self.arr["weights"][idx]),
        }


def extract_embs_for_samples(
    visual: nn.Module,
    samples: list[dict[str, Any]],
    img_dir: Path,
    preprocess,
    batch_size: int = 32,
) -> np.ndarray:
    visual.eval()
    n = len(samples)
    out = np.zeros((n, 512), dtype=np.float32)
    buf, idxs = [], []
    with torch.no_grad():
        for i, s in enumerate(samples):
            with Image.open(img_dir / f"{s['wid']}.jpg") as im:
                buf.append(preprocess(im.convert("RGB")))
                idxs.append(i)
            if len(buf) >= batch_size or i == n - 1:
                x = torch.stack(buf, dim=0)
                f = F.normalize(visual(x), dim=-1).cpu().numpy()
                out[idxs] = f
                buf.clear()
                idxs.clear()
    return out


def finetune_mobileclip_tail(
    base_visual: nn.Module,
    init_head: MultimodalFusionHead,
    train_s: list[dict[str, Any]],
    val_s: list[dict[str, Any]],
    train_arr: dict[str, np.ndarray],
    val_arr: dict[str, np.ndarray],
    img_dir: Path,
    preprocess,
    epochs: int = 4,
    batch_size: int = 32,
) -> tuple[nn.Module, MultimodalFusionHead]:
    """
    CPU-efficient tail fine-tuning:
    Freezes early conv stages of MobileCLIP2-S0 and unfreezes only the top semantic
    stage (stages.3 + head) with gentle LR (5e-5) for 4 epochs.
    """
    visual = copy.deepcopy(base_visual)
    head = copy.deepcopy(init_head)

    # Freeze everything in visual first, then unfreeze last semantic stage & projection head
    for p in visual.parameters():
        p.requires_grad = False
    unfrozen = 0
    for name, p in visual.named_parameters():
        if any(k in name for k in ("stages.3", "conv_exp", "head")):
            p.requires_grad = True
            unfrozen += p.numel()
    print(f"[finetune] unfroze {unfrozen/1e6:.2f}M params in MobileCLIP2-S0 visual tail", flush=True)

    ds = FinetuneDataset(train_s, img_dir, train_arr, is_train=True, synth_red_count=80)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)

    opt = torch.optim.AdamW(
        [
            {"params": [p for p in visual.parameters() if p.requires_grad], "lr": 2.5e-5, "weight_decay": 2e-2},
            {"params": head.parameters(), "lr": 3e-4, "weight_decay": 2.5e-2},
        ]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(dl), eta_min=5e-6)

    best_score = -1e9
    best_vis = copy.deepcopy(visual.state_dict())
    best_head = copy.deepcopy(head.state_dict())

    for ep in range(1, epochs + 1):
        t0 = time.time()
        visual.train()
        # Keep frozen BN/Norm layers in eval mode
        for name, mod in visual.named_modules():
            if not any(k in name for k in ("stages.3", "conv_exp", "head")):
                if isinstance(mod, (nn.BatchNorm2d, nn.LayerNorm)):
                    mod.eval()
        head.train()
        tot_loss = 0.0
        steps = 0

        for batch in dl:
            b = batch["px"].size(0)
            r = np.random.rand(b)
            has_img = torch.from_numpy((r < 0.85).astype(np.float32)).unsqueeze(-1)
            has_txt = torch.from_numpy((r >= 0.20).astype(np.float32)).unsqueeze(-1)

            feat = F.normalize(visual(batch["px"]), dim=-1)
            out = head.forward_logits(
                feat,
                batch["img_pri"],
                batch["txt_hash"],
                batch["txt_lex"],
                has_img,
                has_txt,
            )
            loss = compute_multitask_loss(
                out,
                batch["y_ov"], batch["y_ig"], batch["y_in"], batch["y_tg"], batch["y_tn"],
                has_img, has_txt, batch["w"],
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(visual.parameters()) + list(head.parameters()), 1.5)
            opt.step()
            sched.step()
            tot_loss += float(loss.item())
            steps += 1

        # Re-extract val embeddings with updated visual tail
        val_embs_ft = extract_embs_for_samples(visual, val_s, img_dir, preprocess)
        va_ft = {**val_arr, "img_emb": val_embs_ft}
        vm, _ = evaluate_model(head, va_ft, mode="both")
        score = (
            vm.get("auc_clear_2v0", 0.5)
            + 0.7 * vm.get("auc_any_12v0", 0.5)
            - 1.2 * vm.get("brier_soft", 0.25)
            - 0.25 * vm.get("red_trap_mean_p", 0.0)
            - 0.15 * vm.get("anime_trap_mean_p", 0.0)
        )
        print(
            f"  [ft ep {ep}/{epochs}] loss={tot_loss/max(steps,1):.4f} "
            f"val_auc2v0={vm.get('auc_clear_2v0',0):.4f} val_aucAny={vm.get('auc_any_12v0',0):.4f} "
            f"redP={vm.get('red_trap_mean_p',0):.3f} ({time.time()-t0:.1f}s)",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_vis = copy.deepcopy(visual.state_dict())
            best_head = copy.deepcopy(head.state_dict())

    visual.load_state_dict(best_vis)
    head.load_state_dict(best_head)
    val_embs_ft = extract_embs_for_samples(visual, val_s, img_dir, preprocess)
    fit_calibration(head, {**val_arr, "img_emb": val_embs_ft})
    return visual, head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/raw/catalog_items.jsonl")
    ap.add_argument("--labels", default="data/annotations/merged_labels.jsonl")
    ap.add_argument("--cache", default="data/cache/img_features_mc2s0.npz")
    ap.add_argument("--img-dir", default="data/images")
    ap.add_argument("--stress-dir", default="data/stress_red_text")
    ap.add_argument("--split", default="data/annotations/split_manifest.json")
    ap.add_argument("--out-dir", default="checkpoints/exp_r1")
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--ft-epochs", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(4)
    seed_everything(42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, _, embs, priors = load_data(Path(args.catalog), Path(args.labels), Path(args.cache))
    train_s, val_s, test_s = make_or_load_creator_split(samples, Path(args.split))

    tr_arr = build_feature_arrays(train_s, embs, priors)
    va_arr = build_feature_arrays(val_s, embs, priors)
    te_arr = build_feature_arrays(test_s, embs, priors)

    # Load MobileCLIP2-S0 reparameterized visual model
    mc_model, _, preprocess = open_clip.create_model_and_transforms("MobileCLIP2-S0", pretrained="dfndr2b")
    mc_model.eval()
    vis_reparam = reparameterize_model(mc_model.visual)
    vis_reparam.eval()

    # Add synthetic red-text negative cards ONLY to training array (train-only phrases!)
    synth_tr = generate_synth_red_train_features(vis_reparam, preprocess, count=120, seed=1234)
    tr_arr_aug = concat_arrays(tr_arr, synth_tr)

    results: dict[str, Any] = {}

    # === Experiment 1: Shallow Baseline (Text + 24-dim visual color/edge priors, NO deep vision) ===
    print("\n=== Exp 1: Shallow Baseline (No deep vision) ===", flush=True)
    m_shallow, _ = train_fusion_head(tr_arr_aug, va_arr, zero_img_emb=True, epochs=40, seed=42)
    te_zero = {**te_arr, "img_emb": np.zeros_like(te_arr["img_emb"])}
    m1_test, _ = evaluate_model(m_shallow, te_zero, mode="both")
    results["baseline_shallow"] = {"test_both": m1_test}
    print("  Test:", {k: round(v, 4) for k, v in m1_test.items()}, flush=True)

    # === Experiment 2: MobileCLIP2-S0 Frozen + Multimodal Fusion Head (5-seed ensemble check + best single) ===
    print("\n=== Exp 2: MobileCLIP2-S0 (Frozen Visual) + MultimodalFusionHead ===", flush=True)
    best_frz_model = None
    best_frz_sc = -1e9
    for sd in (42, 101, 2026, 777, 999):
        m_cand, sc = train_fusion_head(tr_arr_aug, va_arr, zero_img_emb=False, epochs=55, seed=sd)
        if sc > best_frz_sc:
            best_frz_sc = sc
            best_frz_model = m_cand
    assert best_frz_model is not None

    m2_both, p2_test = evaluate_model(best_frz_model, te_arr, mode="both")
    m2_img, _ = evaluate_model(best_frz_model, te_arr, mode="image_only")
    m2_txt, _ = evaluate_model(best_frz_model, te_arr, mode="text_only")
    m2_stress = eval_synth_red_stress(best_frz_model, vis_reparam, preprocess, Path(args.stress_dir))
    results["mc2s0_frozen_fusion"] = {
        "test_both": m2_both,
        "test_image_only": m2_img,
        "test_text_only": m2_txt,
        "stress_red_text": m2_stress,
    }
    print("  Test (Both):", {k: round(v, 4) for k, v in m2_both.items()}, flush=True)
    print("  Test (ImgOnly):", {k: round(v, 4) for k, v in m2_img.items()}, flush=True)
    print("  Test (TxtOnly):", {k: round(v, 4) for k, v in m2_txt.items()}, flush=True)
    print("  Stress Red-Text:", {k: round(v, 4) for k, v in m2_stress.items()}, flush=True)

    torch.save(
        {"head_state": best_frz_model.state_dict(), "val_score": best_frz_sc},
        out_dir / "mc2s0_frozen_head.pt",
    )

    # === Experiment 3: MobileCLIP2-S0 Partial Visual Tail Fine-Tuning ===
    if not args.skip_finetune:
        print("\n=== Exp 3: MobileCLIP2-S0 (Tail Fine-Tuned Visual) + MultimodalFusionHead ===", flush=True)
        ft_vis, ft_head = finetune_mobileclip_tail(
            vis_reparam, best_frz_model, train_s, val_s, tr_arr, va_arr,
            Path(args.img_dir), preprocess, epochs=args.ft_epochs, batch_size=32,
        )
        te_embs_ft = extract_embs_for_samples(ft_vis, test_s, Path(args.img_dir), preprocess)
        te_arr_ft = {**te_arr, "img_emb": te_embs_ft}

        m3_both, _ = evaluate_model(ft_head, te_arr_ft, mode="both")
        m3_img, _ = evaluate_model(ft_head, te_arr_ft, mode="image_only")
        m3_txt, _ = evaluate_model(ft_head, te_arr_ft, mode="text_only")
        m3_stress = eval_synth_red_stress(ft_head, ft_vis, preprocess, Path(args.stress_dir))
        results["mc2s0_finetuned_fusion"] = {
            "test_both": m3_both,
            "test_image_only": m3_img,
            "test_text_only": m3_txt,
            "stress_red_text": m3_stress,
        }
        print("  Test (Both):", {k: round(v, 4) for k, v in m3_both.items()}, flush=True)
        print("  Test (ImgOnly):", {k: round(v, 4) for k, v in m3_img.items()}, flush=True)
        print("  Stress Red-Text:", {k: round(v, 4) for k, v in m3_stress.items()}, flush=True)

        torch.save(
            {"visual_state": ft_vis.state_dict(), "head_state": ft_head.state_dict()},
            out_dir / "mc2s0_finetuned.pt",
        )

    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[done] saved evaluation summary -> {out_dir / 'eval_results.json'}", flush=True)


if __name__ == "__main__":
    main()
