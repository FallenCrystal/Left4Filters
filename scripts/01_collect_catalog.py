#!/usr/bin/env python3
"""
Fast, server-friendly stratified collector for Left 4 Dead 2 Workshop items
using Left4Addons Catalog APIs ONLY (no Steam Workshop scraping).

Uses verified <1.5s indexed endpoints (fast FTS terms, tag filters, and
multi-sort pagination) and skips any slow query cleanly with a short timeout.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.l4filter_features import (
    GORE_MILD_TERMS,
    GORE_STRONG_TERMS,
    NSFW_MILD_TERMS,
    NSFW_STRONG_TERMS,
    SAFE_TRAP_TERMS,
    clean_workshop_text,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure via environment variables or CLI flags (--api-base / --fallback-api)
# Defaults to local l4a-server address documented in FallenCrystal/Left4Addons.
DEFAULT_PRIMARY_API = os.environ.get("L4A_CATALOG_API", "http://127.0.0.1:8787").rstrip("/")
DEFAULT_SECONDARY_API = os.environ.get("L4A_CATALOG_FALLBACK_API", "").rstrip("/")

# Fast query plan: (bucket, params_base, page_list)
# All queries tested to return in ~0.6-1.4s on indexed columns/high-hit FTS.
FETCH_PLAN: list[tuple[str, dict[str, Any], list[int]]] = [
    # Direct high-hit sensitive queries (fast FTS ~1s)
    ("q_blood", {"q": "blood", "sort": "subscriptions"}, list(range(1, 16)) + [20, 25, 30, 35]),
    ("q_blood_new", {"q": "blood", "sort": "newest"}, list(range(1, 8))),
    ("q_nsfw", {"q": "nsfw", "sort": "subscriptions"}, list(range(1, 16)) + [20, 25, 30]),
    ("q_nsfw_new", {"q": "nsfw", "sort": "newest"}, list(range(1, 8))),
    ("q_r18", {"q": "r18", "sort": "subscriptions"}, list(range(1, 12)) + [20, 35, 50]),
    ("q_anime", {"q": "anime", "sort": "subscriptions"}, list(range(1, 10)) + [20, 40, 80, 120]),
    # Optional fast FTS probes (will auto-skip in 4.5s if slow on server)
    ("q_sexy", {"q": "sexy", "sort": "subscriptions"}, [1, 2, 3, 4]),
    ("q_zombie", {"q": "zombie", "sort": "subscriptions"}, [1, 2, 3, 4, 5]),
    ("q_dead", {"q": "dead", "sort": "subscriptions"}, [1, 2, 3, 4, 5]),
    ("q_red", {"q": "red", "sort": "subscriptions"}, [1, 2, 3, 4, 5]),
    ("q_dark", {"q": "dark", "sort": "subscriptions"}, [1, 2, 3]),
    # Tag-based stratified sampling (all ~1s indexed)
    ("tag_common", {"tag": "Common Infected", "sort": "subscriptions"}, list(range(1, 12)) + [15, 20, 30, 40]),
    ("tag_special", {"tag": "Special Infected", "sort": "subscriptions"}, list(range(1, 10)) + [15, 25]),
    ("tag_witch", {"tag": "Witch", "sort": "subscriptions"}, list(range(1, 12)) + [15, 20, 30]),
    ("tag_textures", {"tag": "Textures", "sort": "subscriptions"}, list(range(1, 10)) + [20, 40, 80, 150]),
    ("tag_ui", {"tag": "UI", "sort": "subscriptions"}, list(range(1, 10)) + [20, 50, 100, 180]),
    ("tag_survivors", {"tag": "Survivors", "sort": "subscriptions"}, list(range(1, 10)) + [25, 60, 120, 200]),
    ("tag_survivors_new", {"tag": "Survivors", "sort": "newest"}, list(range(1, 8)) + [15, 30]),
    ("tag_weapons", {"tag": "Weapons", "sort": "subscriptions"}, [1, 2, 3, 5, 10, 30, 80]),
    ("tag_campaigns", {"tag": "Campaigns", "sort": "subscriptions"}, [1, 2, 3, 5, 15, 40]),
    ("tag_sounds", {"tag": "Sounds", "sort": "subscriptions"}, [1, 2, 5, 20, 60]),
    ("tag_scripts", {"tag": "Scripts", "sort": "subscriptions"}, [1, 2, 5, 20]),
    # Global spread across all 228k workshop items
    ("global_pop", {"sort": "subscriptions"}, list(range(1, 11)) + [20, 40, 80, 150, 300, 600, 1000]),
    ("global_new", {"sort": "newest"}, list(range(1, 11)) + [25, 60, 120, 250, 500, 900]),
    ("global_upd", {"sort": "updated"}, list(range(1, 8)) + [30, 100, 300, 700]),
]

RED_TEXT_LOCAL_RE = re.compile(
    r"\b(red|crimson|scarlet|ruby|fire|flame|font|hud|logo|icon|ui|menu|loading|"
    r"crosshair|sign|poster|text|spray|neon|laser|blood harvest|dead center|left 4 dead)\b|"
    r"(红|红色|字体|图标|界面|火焰|准星|标语|血腥收获)",
    re.IGNORECASE,
)
ANIME_LOCAL_RE = re.compile(
    r"\b(anime|genshin|blue archive|arknights|hololive|vtuber|miku|touhou|"
    r"azur lane|nikke|honkai|fate|fgo|kancolle|girl|waifu|school|uniform)\b|"
    r"(二次元|原神|碧蓝|明日方舟|初音|东方|少女|崩坏|动漫|舰娘|妮姬)",
    re.IGNORECASE,
)


def fetch_page(
    client: httpx.Client,
    params: dict[str, Any],
    api_bases: list[str],
    timeout: float = 5.0,
) -> list[dict[str, Any]] | None:
    req_params = {**params, "pageSize": 100, "status": "active"}
    for base in api_bases:
        if not base:
            continue
        try:
            r = client.get(f"{base}/api/v1/items/search", params=req_params, timeout=timeout)
            if r.status_code == 200:
                return r.json().get("items") or []
        except Exception:
            continue
    return None


def collect_metadata(out_jsonl: Path, api_bases: list[str]) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    if out_jsonl.exists():
        with out_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    items[str(rec["workshopId"])] = rec
        print(f"[resume] loaded {len(items)} existing items from {out_jsonl}", flush=True)

    headers = {"User-Agent": "Left4Filters-Collector/1.0"}
    with httpx.Client(headers=headers, follow_redirects=True) as client:
        for bucket, base_params, pages in FETCH_PLAN:
            added = 0
            consecutive_fail = 0
            for p in pages:
                params = {**base_params, "page": p}
                timeout = 4.5 if "q" in base_params else 8.0
                batch = fetch_page(client, params, api_bases=api_bases, timeout=timeout)
                if batch is None:
                    consecutive_fail += 1
                    print(f"  [skip] {bucket} page={p} timed out/failed", flush=True)
                    if consecutive_fail >= 1 and "q" in base_params:
                        break  # Don't hammer slow FTS queries
                    continue
                consecutive_fail = 0
                if not batch:
                    break
                for it in batch:
                    wid = str(it.get("workshopId") or "")
                    if not wid:
                        continue
                    if wid not in items:
                        desc = it.get("description") or ""
                        items[wid] = {
                            "workshopId": wid,
                            "title": it.get("title") or "",
                            "description": desc[:4000],
                            "clean_description": clean_workshop_text(desc, max_len=800),
                            "creatorSteamId": it.get("creatorSteamId") or "",
                            "creatorName": it.get("creatorName") or "",
                            "previewUrl": it.get("previewUrl") or "",
                            "previews": [
                                pr.get("url")
                                for pr in (it.get("previews") or [])
                                if isinstance(pr, dict) and pr.get("url")
                            ][:4],
                            "tags": it.get("tags") or [],
                            "subscriptions": it.get("subscriptions") or 0,
                            "favorites": it.get("favorites") or 0,
                            "buckets": [bucket],
                        }
                        added += 1
                    else:
                        if bucket not in items[wid]["buckets"]:
                            items[wid]["buckets"].append(bucket)
                if len(batch) < 100:
                    break
                time.sleep(0.05)
            print(f"[bucket:{bucket:18s}] +{added:4d} new | total={len(items)}", flush=True)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for rec in items.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[saved] {len(items)} unique workshop items -> {out_jsonl}", flush=True)
    return items


def classify_local_strata(rec: dict[str, Any]) -> set[str]:
    text = f"{rec.get('title', '')} {' '.join(rec.get('tags', []))} {rec.get('clean_description', '')[:300]}".lower()
    buckets = set(rec.get("buckets", []))
    tags = {t.lower() for t in rec.get("tags", [])}
    strata: set[str] = set()

    if any(k in text for k in GORE_STRONG_TERMS) or "q_blood" in buckets or "q_blood_new" in buckets:
        strata.add("gore_cand")
    elif any(k in text for k in GORE_MILD_TERMS) and bool(tags & {"textures", "common infected", "special infected", "witch", "particles"}):
        strata.add("gore_cand")

    if any(k in text for k in NSFW_STRONG_TERMS) or "q_nsfw" in buckets or "q_nsfw_new" in buckets or "q_r18" in buckets:
        strata.add("nsfw_cand")
    elif any(k in text for k in NSFW_MILD_TERMS):
        strata.add("nsfw_cand")

    if RED_TEXT_LOCAL_RE.search(text) or "tag_ui" in buckets or "ui" in tags:
        strata.add("hard_neg_red_ui")

    if ANIME_LOCAL_RE.search(text) or "q_anime" in buckets:
        strata.add("hard_neg_anime")

    if not strata:
        strata.add("general")
    return strata


def download_one_image(args: tuple[str, str, Path, int]) -> tuple[str, str]:
    wid, url, img_dir, max_side = args
    out_path = img_dir / f"{wid}.jpg"
    if out_path.exists() and out_path.stat().st_size > 500:
        return wid, "cached"
    if not url:
        return wid, "no_url"
    try:
        with httpx.Client(timeout=12.0, follow_redirects=True) as client:
            r = client.get(url, headers={"User-Agent": "Left4FiltersModel-ImgFetch/1.0"})
            if r.status_code != 200 or len(r.content) < 200:
                return wid, f"http_{r.status_code}"
            img = Image.open(io.BytesIO(r.content))
            img = img.convert("RGB")
            if max(img.size) > max_side:
                img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path, format="JPEG", quality=88, optimize=True)
            return wid, "ok"
    except Exception as e:
        return wid, f"err:{type(e).__name__}"


def download_stratified_images(
    items: dict[str, dict[str, Any]],
    img_dir: Path,
    target_total: int = 5600,
    workers: int = 16,
    max_side: int = 512,
) -> None:
    rng = random.Random(42)
    pools: dict[str, list[dict[str, Any]]] = {
        "gore_cand": [],
        "nsfw_cand": [],
        "hard_neg_red_ui": [],
        "hard_neg_anime": [],
        "general": [],
    }
    for rec in items.values():
        if not rec.get("previewUrl"):
            continue
        strata = classify_local_strata(rec)
        rec["local_strata"] = sorted(strata)
        for s in strata:
            pools[s].append(rec)

    print("[local strata pool sizes]", {k: len(v) for k, v in pools.items()}, flush=True)

    # Balanced quotas for image download so every risk & trap category is richly represented
    quotas = {
        "gore_cand": 1400,
        "nsfw_cand": 1400,
        "hard_neg_red_ui": 1000,
        "hard_neg_anime": 900,
        "general": 1200,
    }
    selected: dict[str, dict[str, Any]] = {}
    for k, q in quotas.items():
        lst = pools[k][:]
        rng.shuffle(lst)
        # Encourage creator diversity (cap at 6 items per creator in initial download selection)
        creator_cnt: dict[str, int] = {}
        taken = 0
        for rec in lst:
            cid = rec.get("creatorSteamId") or rec["workshopId"]
            if creator_cnt.get(cid, 0) >= 6:
                continue
            if rec["workshopId"] not in selected:
                selected[rec["workshopId"]] = rec
                creator_cnt[cid] = creator_cnt.get(cid, 0) + 1
                taken += 1
                if taken >= q:
                    break
        print(f"  selected {taken:4d} from {k}", flush=True)

    sel_list = list(selected.values())[:target_total]
    print(f"[images] downloading {len(sel_list)} stratified preview images -> {img_dir}", flush=True)

    tasks = [(rec["workshopId"], rec["previewUrl"], img_dir, max_side) for rec in sel_list]
    counts: dict[str, int] = {}
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (wid, status) in enumerate(ex.map(download_one_image, tasks), 1):
            counts[status] = counts.get(status, 0) + 1
            if i % 500 == 0 or i == len(tasks):
                print(f"[images] {i}/{len(tasks)} ({time.time()-t0:.1f}s) {counts}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default=DEFAULT_PRIMARY_API, help="Primary Left4Addons catalog server URL (or L4A_CATALOG_API env)")
    ap.add_argument("--fallback-api", default=DEFAULT_SECONDARY_API, help="Optional fallback catalog server URL (or L4A_CATALOG_FALLBACK_API env)")
    ap.add_argument("--out", default="data/raw/catalog_items.jsonl")
    ap.add_argument("--img-dir", default="data/images")
    ap.add_argument("--target-images", type=int, default=5600)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()

    api_bases = [u.rstrip("/") for u in (args.api_base, args.fallback_api) if u]
    if not api_bases:
        raise SystemExit("Please specify --api-base or set L4A_CATALOG_API.")

    items = collect_metadata(Path(args.out), api_bases=api_bases)
    if not args.skip_images:
        download_stratified_images(items, Path(args.img_dir), target_total=args.target_images, workers=args.workers)


if __name__ == "__main__":
    main()
