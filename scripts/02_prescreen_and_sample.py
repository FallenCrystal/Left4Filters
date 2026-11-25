#!/usr/bin/env python3
"""
Step 2:
1. Extract MobileCLIP2-S0 (reparameterized) 512-dim visual embeddings & 24-dim visual priors
   for all downloaded preview images (cached to data/cache/ for fast reuse).
2. Run zero-shot semantic scoring (gore, nsfw, red-text trap, safe anime, safe game)
   purely for stratified candidate selection and hard-negative mining.
3. Select ~1,440 diverse, deduplicated Round-1 items across 7 strata.
4. Render crisp 4x4 contact sheets (1280x1408) + compact metadata JSONs for VLM subagent annotation.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image, ImageDraw, ImageFont
from timm.utils.model import reparameterize_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.l4filter_features import (
    extract_text_features,
    extract_visual_priors,
)

# Zero-shot prompt groups for candidate stratification (NOT ground truth!)
ZS_PROMPTS: dict[str, list[str]] = {
    "gore": [
        "a graphic bloody gore horror image covered in large pools of red blood",
        "a large bloody handprint and blood splatters dripping across the screen",
        "graphic violence with severed body parts, exposed flesh, organs, and heavy blood",
        "disturbing bloody corpse, realistic bleeding wounds, extreme gore splatter",
        "massive amount of red blood pouring and splattering on a zombie body",
    ],
    "nsfw": [
        "a sexually suggestive anime girl wearing a skimpy bikini or revealing lingerie",
        "a lewd erotic illustration showing heavy cleavage, underwear, or near nudity",
        "a provocative sexy female character in a swimsuit, bunny suit, or translucent outfit",
        "nsfw adult hentai ecchi artwork with sexually suggestive pose and exposed skin",
        "a scantily clad 3d female game character in revealing underwear or bikini",
    ],
    "red_text_trap": [
        "plain red text typography words written on a black or white background",
        "a red game logo, red title banner, red warning sign, or red user interface menu",
        "red glowing neon letters, red font preview, or red HUD icon on screen",
        "bright red painted weapon skin, red car, red fire flames, or red armor",
    ],
    "safe_anime": [
        "a cute normally dressed anime character in school uniform or casual clothes",
        "a wholesome anime girl portrait wearing full modest clothing or coat",
        "an anime character holding a weapon in a video game screenshot, fully clothed",
    ],
    "safe_game": [
        "a normal first person shooter video game screenshot with weapons and buildings",
        "a video game map scenery, gun model preview, or standard game menu interface",
        "a normal cartoon or meme thumbnail with no blood and no nudity",
    ],
}


def load_mobileclip2():
    torch.set_num_threads(4)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "MobileCLIP2-S0", pretrained="dfndr2b"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("MobileCLIP2-S0")
    try:
        model.visual = reparameterize_model(model.visual)
    except Exception:
        pass
    return model, preprocess, tokenizer


def encode_prompt_groups(model, tokenizer) -> dict[str, np.ndarray]:
    vecs: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, prompts in ZS_PROMPTS.items():
            toks = tokenizer(prompts)
            emb = model.encode_text(toks)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            mean_emb = emb.mean(dim=0)
            mean_emb = mean_emb / mean_emb.norm()
            vecs[name] = mean_emb.cpu().numpy().astype(np.float32)
    return vecs


def extract_all_image_features(
    items_by_id: dict[str, dict[str, Any]],
    img_dir: Path,
    cache_path: Path,
    batch_size: int = 32,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    img_files = sorted(img_dir.glob("*.jpg"))
    valid_ids = [p.stem for p in img_files if p.stem in items_by_id]
    print(f"[features] found {len(valid_ids)} downloaded images matching catalog", flush=True)

    model, preprocess, tokenizer = load_mobileclip2()
    prompt_vecs = encode_prompt_groups(model, tokenizer)

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        cached_ids = list(data["wids"])
        if len(cached_ids) == len(valid_ids) and cached_ids == valid_ids:
            print(f"[cache hit] loaded {len(cached_ids)} image embeddings & priors from {cache_path}", flush=True)
            return cached_ids, data["emb"], data["priors"], prompt_vecs

    embs = np.zeros((len(valid_ids), 512), dtype=np.float32)
    priors = np.zeros((len(valid_ids), 24), dtype=np.float32)
    good_mask = np.ones(len(valid_ids), dtype=bool)

    t0 = time.time()
    buf_tensors: list[torch.Tensor] = []
    buf_indices: list[int] = []

    def flush_batch() -> None:
        if not buf_tensors:
            return
        x = torch.stack(buf_tensors, dim=0)
        with torch.no_grad():
            feat = model.encode_image(x)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        embs[buf_indices] = feat.cpu().numpy().astype(np.float32)
        buf_tensors.clear()
        buf_indices.clear()

    for i, wid in enumerate(valid_ids):
        p = img_dir / f"{wid}.jpg"
        try:
            with Image.open(p) as im:
                im_rgb = im.convert("RGB")
                priors[i] = extract_visual_priors(im_rgb)
                buf_tensors.append(preprocess(im_rgb))
                buf_indices.append(i)
        except Exception:
            good_mask[i] = False
            continue

        if len(buf_tensors) >= batch_size:
            flush_batch()
        if (i + 1) % 500 == 0 or (i + 1) == len(valid_ids):
            flush_batch()
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.1)
            print(f"[extract] {i+1}/{len(valid_ids)} ({elapsed:.1f}s, {rate:.1f} img/s)", flush=True)

    flush_batch()
    keep_idx = np.where(good_mask)[0]
    valid_ids = [valid_ids[i] for i in keep_idx]
    embs = embs[keep_idx]
    priors = priors[keep_idx]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, wids=np.array(valid_ids), emb=embs, priors=priors)
    print(f"[saved] {len(valid_ids)} embeddings + priors -> {cache_path}", flush=True)
    return valid_ids, embs, priors, prompt_vecs


def compute_prescreen_scores(
    wids: list[str],
    embs: np.ndarray,
    priors: np.ndarray,
    prompt_vecs: dict[str, np.ndarray],
    items_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    # Cosine similarities with zero-shot prompt prototypes
    sim = {k: (embs @ v).astype(np.float32) for k, v in prompt_vecs.items()}
    # Standardize similarities to z-scores for clean relative ranking
    z = {k: (s - s.mean()) / (s.std() + 1e-6) for k, s in sim.items()}

    scores: dict[str, dict[str, float]] = {}
    for i, wid in enumerate(wids):
        rec = items_by_id[wid]
        _, lex = extract_text_features(
            rec.get("title", ""), rec.get("clean_description", ""), rec.get("tags", [])
        )
        # priors indices:
        # 0: blood_ratio, 1: bright_red_ratio, 2: any_red_ratio, 3: center_blood,
        # 5: red_perim_ratio, 10: flat_bg_ratio, 23: bright_red*flat_bg
        blood_r = float(priors[i, 0])
        bright_r = float(priors[i, 1])
        any_red = float(priors[i, 2])
        red_perim = float(priors[i, 5])
        flat_bg = float(priors[i, 10])

        # lex indices:
        # 0: t_gore_s, 1: t_gore_m, 2: d_gore_s, 4: t_nsfw_s, 5: t_nsfw_m, 6: d_nsfw_s, 8: t_safe
        txt_gore = float(lex[0] * 1.5 + lex[1] * 0.7 + lex[2] * 0.8)
        txt_nsfw = float(lex[4] * 1.5 + lex[5] * 0.7 + lex[6] * 0.8)
        txt_safe = float(lex[8] * 1.0 + lex[12] * 0.5)

        gore_vis = float(z["gore"][i] - 0.35 * z["safe_game"][i] - 0.25 * z["red_text_trap"][i] + 2.5 * blood_r)
        nsfw_vis = float(z["nsfw"][i] - 0.45 * z["safe_anime"][i] - 0.30 * z["safe_game"][i])

        gore_comb = gore_vis + 0.85 * txt_gore
        nsfw_comb = nsfw_vis + 0.85 * txt_nsfw

        # Red-text / Red-theme trap score: high red color OR red_text_trap CLIP sim,
        # OR red/UI keywords, while NOT strongly bloody gore
        red_trap = float(
            z["red_text_trap"][i]
            + 3.5 * bright_r
            + 1.8 * any_red * (0.4 + flat_bg + red_perim)
            + 0.6 * txt_safe
            - 0.35 * max(z["gore"][i], 0.0)
        )
        # Also include items that have strong red pixels or "blood/red" title yet low gore vis
        if any_red > 0.06 and z["gore"][i] < 1.2:
            red_trap += 1.2

        # Normal anime trap score: high safe_anime sim, moderate-low nsfw sim
        anime_trap = float(
            z["safe_anime"][i]
            - 0.45 * max(z["nsfw"][i], 0.0)
            - 0.6 * txt_nsfw
        )

        # Cross-modal mismatch candidates (text strong, image weak OR image strong, text weak)
        cross_modal = float(
            max(
                txt_gore - max(gore_vis, 0.0),
                txt_nsfw - max(nsfw_vis, 0.0),
                gore_vis - 1.5 * txt_gore - 0.8,
                nsfw_vis - 1.5 * txt_nsfw - 0.8,
            )
        )

        scores[wid] = {
            "gore_vis": gore_vis,
            "nsfw_vis": nsfw_vis,
            "txt_gore": txt_gore,
            "txt_nsfw": txt_nsfw,
            "gore_comb": gore_comb,
            "nsfw_comb": nsfw_comb,
            "red_trap": red_trap,
            "anime_trap": anime_trap,
            "cross_modal": cross_modal,
            "any_red": any_red,
            "blood_r": blood_r,
            "bright_r": bright_r,
        }
    return scores


def select_round1_candidates(
    wids: list[str],
    embs: np.ndarray,
    scores: dict[str, dict[str, float]],
    items_by_id: dict[str, dict[str, Any]],
    total_target: int = 1440,
) -> list[tuple[str, str]]:
    """
    Stratified selection with near-duplicate visual filtering (cosine > 0.965)
    and creator cap (<= 4 per creator) so every annotation dollar counts!
    """
    wid_to_idx = {w: i for i, w in enumerate(wids)}
    selected: list[tuple[str, str]] = []
    selected_set: set[str] = set()
    selected_embs: list[np.ndarray] = []
    creator_cnt: dict[str, int] = {}

    def try_add(wid: str, stratum: str, max_per_creator: int = 4) -> bool:
        if wid in selected_set:
            return False
        cid = items_by_id[wid].get("creatorSteamId") or wid
        if creator_cnt.get(cid, 0) >= max_per_creator:
            return False
        emb = embs[wid_to_idx[wid]]
        if selected_embs:
            # Fast visual near-duplicate suppression
            sims = np.dot(np.asarray(selected_embs), emb)
            if float(sims.max()) > 0.965:
                return False
        selected.append((wid, stratum))
        selected_set.add(wid)
        selected_embs.append(emb)
        creator_cnt[cid] = creator_cnt.get(cid, 0) + 1
        return True

    def take_ranked(ranked_wids: list[str], stratum: str, quota: int, max_per_creator: int = 4) -> int:
        n = 0
        for w in ranked_wids:
            if try_add(w, stratum, max_per_creator=max_per_creator):
                n += 1
                if n >= quota:
                    break
        print(f"  [sample:{stratum:16s}] took {n:4d} / {quota}", flush=True)
        return n

    # 1. High gore candidates (visual + combined)
    by_gore_vis = sorted(wids, key=lambda w: scores[w]["gore_vis"], reverse=True)
    by_gore_comb = sorted(wids, key=lambda w: scores[w]["gore_comb"], reverse=True)
    take_ranked(by_gore_vis, "high_gore_vis", 150)
    take_ranked(by_gore_comb, "high_gore_comb", 120)

    # 2. High NSFW candidates (visual + combined)
    by_nsfw_vis = sorted(wids, key=lambda w: scores[w]["nsfw_vis"], reverse=True)
    by_nsfw_comb = sorted(wids, key=lambda w: scores[w]["nsfw_comb"], reverse=True)
    take_ranked(by_nsfw_vis, "high_nsfw_vis", 150)
    take_ranked(by_nsfw_comb, "high_nsfw_comb", 120)

    # 3. Red-text / Red-theme Hard Negatives (CRITICAL for user requirement!)
    by_red_trap = sorted(wids, key=lambda w: scores[w]["red_trap"], reverse=True)
    by_bright_red = sorted(wids, key=lambda w: scores[w]["bright_r"] + 0.5 * scores[w]["any_red"], reverse=True)
    take_ranked(by_red_trap, "red_text_trap", 150)
    take_ranked(by_bright_red, "high_red_color", 80)

    # 4. Safe Anime Hard Negatives (CRITICAL so normal anime isn't flagged as NSFW!)
    by_anime_trap = sorted(wids, key=lambda w: scores[w]["anime_trap"], reverse=True)
    take_ranked(by_anime_trap, "anime_trap", 160)

    # 5. Borderline / Mid-range gore & nsfw (decision boundary!)
    mid_gore = [w for w in by_gore_comb if 0.5 <= scores[w]["gore_comb"] <= 2.2]
    mid_nsfw = [w for w in by_nsfw_comb if 0.5 <= scores[w]["nsfw_comb"] <= 2.2]
    rng = random.Random(2026)
    rng.shuffle(mid_gore)
    rng.shuffle(mid_nsfw)
    take_ranked(mid_gore, "border_gore", 110)
    take_ranked(mid_nsfw, "border_nsfw", 110)

    # 6. Cross-modal mismatch (text sensitive / image benign OR vice versa)
    by_cross = sorted(wids, key=lambda w: scores[w]["cross_modal"], reverse=True)
    take_ranked(by_cross, "cross_modal", 90)

    # 7. General random sample across remaining catalog
    remaining = [w for w in wids if w not in selected_set]
    rng.shuffle(remaining)
    need = max(0, total_target - len(selected))
    take_ranked(remaining, "general_random", need, max_per_creator=5)

    rng.shuffle(selected)
    print(f"[sample] total Round-1 selected: {len(selected)} unique items", flush=True)
    return selected


def render_contact_sheets(
    selected: list[tuple[str, str]],
    items_by_id: dict[str, dict[str, Any]],
    img_dir: Path,
    out_dir: Path,
    grid_cols: int = 4,
    grid_rows: int = 4,
    tile_w: int = 320,
    tile_h: int = 320,
    header_h: int = 32,
) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = grid_cols * grid_rows
    cell_h = tile_h + header_h

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    manifest: list[dict[str, Any]] = []
    n_sheets = math.ceil(len(selected) / per_sheet)

    for s_idx in range(n_sheets):
        chunk = selected[s_idx * per_sheet : (s_idx + 1) * per_sheet]
        canvas = Image.new("RGB", (grid_cols * tile_w, grid_rows * cell_h), (24, 24, 28))
        draw = ImageDraw.Draw(canvas)

        sheet_name = f"sheet_{s_idx + 1:03d}"
        sheet_records: list[dict[str, Any]] = []

        for slot_i, (wid, stratum) in enumerate(chunk):
            r = slot_i // grid_cols
            c = slot_i % grid_cols
            x0 = c * tile_w
            y0 = r * cell_h

            # Header bar
            draw.rectangle([x0, y0, x0 + tile_w - 1, y0 + header_h - 1], fill=(15, 23, 42))
            draw.text((x0 + 8, y0 + 6), f"#{slot_i + 1:02d}  ID:{wid}", fill=(250, 204, 21), font=font)

            # Image letterboxed cleanly to preserve aspect ratio (no distortion!)
            img_path = img_dir / f"{wid}.jpg"
            try:
                with Image.open(img_path) as im:
                    im_rgb = im.convert("RGB")
                    im_rgb.thumbnail((tile_w - 4, tile_h - 4), Image.Resampling.LANCZOS)
                    pw, ph = im_rgb.size
                    px = x0 + (tile_w - pw) // 2
                    py = y0 + header_h + (tile_h - ph) // 2
                    canvas.paste(im_rgb, (px, py))
            except Exception:
                draw.text((x0 + 20, y0 + header_h + 100), "IMAGE ERR", fill=(255, 80, 80), font=font)

            # Crisp border between cells
            draw.rectangle([x0, y0, x0 + tile_w - 1, y0 + cell_h - 1], outline=(100, 116, 139), width=2)

            rec = items_by_id[wid]
            sheet_records.append({
                "slot": slot_i + 1,
                "wid": wid,
                "stratum": stratum,
                "title": (rec.get("title") or "")[:140],
                "tags": rec.get("tags") or [],
                "desc": (rec.get("clean_description") or "")[:240],
            })

        sheet_img_path = out_dir / f"{sheet_name}.jpg"
        sheet_meta_path = out_dir / f"{sheet_name}.json"
        canvas.save(sheet_img_path, format="JPEG", quality=88, optimize=True)
        with sheet_meta_path.open("w", encoding="utf-8") as f:
            json.dump({"sheet": sheet_name, "items": sheet_records}, f, ensure_ascii=False, indent=2)

        manifest.append({
            "sheet": sheet_name,
            "image": str(sheet_img_path.resolve()),
            "meta": str(sheet_meta_path.resolve()),
            "count": len(sheet_records),
        })

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[sheets] rendered {len(manifest)} contact sheets ({len(selected)} items) -> {out_dir}", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/raw/catalog_items.jsonl")
    ap.add_argument("--img-dir", default="data/images")
    ap.add_argument("--cache", default="data/cache/img_features_mc2s0.npz")
    ap.add_argument("--scores-out", default="data/cache/prescreen_scores.json")
    ap.add_argument("--out-dir", default="data/batches_r1")
    ap.add_argument("--target", type=int, default=1440)
    args = ap.parse_args()

    items_by_id: dict[str, dict[str, Any]] = {}
    with Path(args.catalog).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                items_by_id[str(rec["workshopId"])] = rec
    print(f"[load] {len(items_by_id)} catalog items", flush=True)

    wids, embs, priors, prompt_vecs = extract_all_image_features(
        items_by_id, Path(args.img_dir), Path(args.cache)
    )
    scores = compute_prescreen_scores(wids, embs, priors, prompt_vecs, items_by_id)
    Path(args.scores_out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.scores_out).open("w", encoding="utf-8") as f:
        json.dump(scores, f)

    selected = select_round1_candidates(wids, embs, scores, items_by_id, total_target=args.target)
    render_contact_sheets(selected, items_by_id, Path(args.img_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
