#!/usr/bin/env python3
"""
Round 2 Active Learning & Targeted Gap-Filling Miner:
Mines 480 highest-value unlabeled items (strictly creator-disjoint from locked Val/Test):
  1. Gore-Enrichment (140): high visual blood/gore/wounds/gibs/bloody-hand candidates
  2. Safe-CJK & Multilingual Balance (90): Chinese/Japanese non-NSFW & mixed L4D2 mods
  3. Red-Text / Non-Gore Red FP Traps (90): high red / flat-bg / red-backdrop items where R1 prob is elevated
  4. Uncertainty & Cross-Modal Disagreement (100): boundary prob ~0.5 or |p_img - p_txt| large
  5. Hard NSFW / Revealing Boundary (60): borderline NSFW & description-NSFW candidates

Renders 30 crisp 4x4 contact sheets into data/batches_r2/ for the EXISTING 3 subagents.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib.util
_spec = importlib.util.spec_from_file_location("prescreen_mod", Path(__file__).resolve().parent / "02_prescreen_and_sample.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
render_contact_sheets = _mod.render_contact_sheets
from src.l4filter_features import extract_text_features
from src.l4filter_model import MultimodalFusionHead

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
GORE_HINT_RE = re.compile(
    r"blood|gore|gib|flesh|wound|brutal|visceral|corpse|slaughter|bleed|mutilat|dismember|grotesque|hand|scary|horror|血|恐怖|尸|断肢|内脏|猎奇|重口",
    re.IGNORECASE,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/raw/catalog_items.jsonl")
    ap.add_argument("--cache", default="data/cache/img_features_mc2s0.npz")
    ap.add_argument("--prescreen", default="data/cache/prescreen_scores.json")
    ap.add_argument("--labels", default="data/annotations/merged_r1.jsonl")
    ap.add_argument("--split", default="data/annotations/split_manifest.json")
    ap.add_argument("--ckpt", default="checkpoints/exp_r1/mc2s0_frozen_head.pt")
    ap.add_argument("--img-dir", default="data/images")
    ap.add_argument("--out-dir", default="data/batches_r2")
    ap.add_argument("--target", type=int, default=480)
    args = ap.parse_args()

    catalog: dict[str, dict[str, Any]] = {}
    with Path(args.catalog).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                catalog[str(r["workshopId"])] = r

    labeled_wids: set[str] = set()
    with Path(args.labels).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                labeled_wids.add(str(json.loads(line)["wid"]))

    prescreen = json.loads(Path(args.prescreen).read_text(encoding="utf-8"))
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    banned_creators = set(split.get("val_creators", [])) | set(split.get("test_creators", []))

    cache = np.load(args.cache, allow_pickle=True)
    wids_all = [str(x) for x in cache["wids"]]
    embs_all = cache["emb"].astype(np.float32)
    priors_all = cache["priors"].astype(np.float32)

    pool_idx: list[int] = []
    pool_wids: list[str] = []
    for i, w in enumerate(wids_all):
        if w in labeled_wids or w not in catalog:
            continue
        cid = catalog[w].get("creatorSteamId") or w
        if cid in banned_creators:
            continue
        pool_idx.append(i)
        pool_wids.append(w)

    print(f"[AL-R2] scanning {len(pool_wids)} unlabeled items (creator-isolated from Val/Test)", flush=True)

    n = len(pool_wids)
    img_emb = embs_all[pool_idx]
    img_pri = priors_all[pool_idx]
    txt_hash = np.zeros((n, 4096), dtype=np.float32)
    txt_lex = np.zeros((n, 16), dtype=np.float32)
    is_cjk = np.zeros(n, dtype=np.float32)
    has_gore_kw = np.zeros(n, dtype=np.float32)
    ps_gore = np.zeros(n, dtype=np.float32)
    ps_red_trap = np.zeros(n, dtype=np.float32)

    for j, w in enumerate(pool_wids):
        r = catalog[w]
        title = r.get("title", "")
        desc = r.get("clean_description", "")
        h, l = extract_text_features(title, desc, r.get("tags", []))
        txt_hash[j] = h
        txt_lex[j] = l
        is_cjk[j] = 1.0 if CJK_RE.search(title) else 0.0
        has_gore_kw[j] = 1.0 if GORE_HINT_RE.search(title + " " + desc[:250]) else 0.0
        ps = prescreen.get(w, {})
        ps_gore[j] = float(ps.get("gore_vis", 0.0))
        ps_red_trap[j] = float(ps.get("red_trap", 0.0))

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    head = MultimodalFusionHead()
    head.load_state_dict(ckpt["head_state"])
    head.eval()

    with torch.no_grad():
        p_both = head(
            torch.from_numpy(img_emb), torch.from_numpy(img_pri),
            torch.from_numpy(txt_hash), torch.from_numpy(txt_lex),
            torch.ones(n, 1), torch.ones(n, 1),
        ).numpy()
        p_img = head(
            torch.from_numpy(img_emb), torch.from_numpy(img_pri),
            torch.zeros(n, 4096), torch.zeros(n, 16),
            torch.ones(n, 1), torch.zeros(n, 1),
        ).numpy()
        p_txt = head(
            torch.zeros(n, 512), torch.zeros(n, 24),
            torch.from_numpy(txt_hash), torch.from_numpy(txt_lex),
            torch.zeros(n, 1), torch.ones(n, 1),
        ).numpy()

    # 1. Visual gore enrichment (prioritize high predicted gore OR high CLIP gore + blood pixels)
    gore_enrich = 2.0 * p_img[:, 1] + 0.45 * ps_gore + 3.0 * img_pri[:, 0] + 0.35 * has_gore_kw
    # 2. Safe/Mixed CJK balance (CJK titles with low-to-mid NSFW/gore or elevated false text prob)
    cjk_balance = is_cjk * (1.2 - p_img[:, 2] + 0.8 * np.maximum(p_txt[:, 0] - p_img[:, 0], 0.0))
    # 3. Red-text / Red-theme false-positive traps
    red_fp = (0.35 * ps_red_trap + 2.5 * img_pri[:, 1] + 1.5 * img_pri[:, 2] * (0.4 + img_pri[:, 10] + img_pri[:, 5])) * (0.35 + p_both[:, 0] + p_img[:, 1])
    # 4. Uncertainty & cross-modal disagreement
    uncert_dis = (1.0 - np.abs(p_both[:, 0] - 0.5) * 2.0) + 0.85 * np.abs(p_img[:, 0] - p_txt[:, 0])
    # 5. Hard NSFW / borderline revealing
    nsfw_hard = p_both[:, 2] * (1.15 - np.abs(p_both[:, 2] - 0.55))

    selected: list[tuple[str, str]] = []
    sel_set: set[str] = set()
    sel_embs: list[np.ndarray] = []
    creator_cnt: dict[str, int] = {}

    def take_by_score(score_arr: np.ndarray, name: str, quota: int) -> None:
        order = np.argsort(-score_arr)
        taken = 0
        for j in order:
            w = pool_wids[j]
            if w in sel_set:
                continue
            cid = catalog[w].get("creatorSteamId") or w
            if creator_cnt.get(cid, 0) >= 3:
                continue
            e = img_emb[j]
            if sel_embs and float(np.dot(np.asarray(sel_embs), e).max()) > 0.96:
                continue
            selected.append((w, name))
            sel_set.add(w)
            sel_embs.append(e)
            creator_cnt[cid] = creator_cnt.get(cid, 0) + 1
            taken += 1
            if taken >= quota:
                break
        print(f"  [AL-R2:{name:16s}] mined {taken:3d}/{quota}", flush=True)

    take_by_score(gore_enrich, "al_gore_enrich", 140)
    take_by_score(cjk_balance, "al_cjk_balance", 90)
    take_by_score(red_fp, "al_red_fp", 90)
    take_by_score(uncert_dis, "al_uncert_dis", 100)
    take_by_score(nsfw_hard, "al_nsfw_hard", args.target - len(selected))

    rng = random.Random(2027)
    rng.shuffle(selected)
    render_contact_sheets(selected, catalog, Path(args.img_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
