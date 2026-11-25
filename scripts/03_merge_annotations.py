#!/usr/bin/env python3
"""
Merge, validate, and summarize VLM multimodal annotations from subagents.

Expected annotation schema per JSONL record:
{
  "wid": "123456789",
  "ig": 0|1|2,   // image gore: 0=none/normal game/red text, 1=mild/borderline blood/horror, 2=clear disturbing gore/large blood/bloody hand/viscera
  "in": 0|1|2,   // image nsfw: 0=none/normal anime/clothed, 1=mild suggestive/slightly revealing, 2=clear sexual/bikini/lingerie/lewd/nude
  "tg": 0|1|2,   // text gore:  0=none/normal L4D terms, 1=mild blood/horror mention, 2=explicit gore/dismemberment/disturbing horror
  "tn": 0|1|2,   // text nsfw:  0=none/normal anime/char, 1=mild flirty/suggestive, 2=explicit NSFW/R18/lewd/bikini/nude/erotic
  "ov": 0|1|2,   // overall disturbing sensitivity: 0=safe, 1=borderline (ideal for LLM escalation), 2=clearly sensitive
  "rt": 0|1,     // red-text / non-gore red visual trap flag
  "at": 0|1,     // safe-anime visual trap flag (normal clothed anime character)
  "note": "..."  // optional brief reason
}
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any


def clamp_int(val: Any, lo: int, hi: int, default: int = 0) -> int:
    try:
        v = int(round(float(val)))
        return max(lo, min(hi, v))
    except Exception:
        return default


def parse_annotation_files(patterns: list[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    bad_lines = 0
    total_lines = 0

    files: list[Path] = []
    for pat in patterns:
        files.extend(sorted(Path(p) for p in glob.glob(pat)))

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, 1):
                s = raw.strip()
                if not s or s.startswith("#") or s.startswith("```"):
                    continue
                total_lines += 1
                try:
                    obj = json.loads(s)
                except Exception:
                    bad_lines += 1
                    continue
                wid = str(obj.get("wid") or obj.get("workshopId") or "").strip()
                if not wid:
                    bad_lines += 1
                    continue

                ig = clamp_int(obj.get("ig", obj.get("image_gore", 0)), 0, 2)
                in_ = clamp_int(obj.get("in", obj.get("image_nsfw", 0)), 0, 2)
                tg = clamp_int(obj.get("tg", obj.get("text_gore", 0)), 0, 2)
                tn = clamp_int(obj.get("tn", obj.get("text_nsfw", 0)), 0, 2)
                ov_raw = obj.get("ov", obj.get("overall"))
                if ov_raw is None:
                    ov = max(ig, in_, tg, tn)
                else:
                    ov = clamp_int(ov_raw, 0, 2)
                # Enforce logical consistency: overall >= max(ig, in_, tg, tn) when any sub-label is 2
                if max(ig, in_, tg, tn) == 2 and ov == 0:
                    ov = 2
                elif max(ig, in_, tg, tn) >= 1 and ov == 0:
                    ov = 1

                rt = clamp_int(obj.get("rt", obj.get("red_trap", 0)), 0, 1)
                at = clamp_int(obj.get("at", obj.get("anime_trap", 0)), 0, 1)

                records[wid] = {
                    "wid": wid,
                    "ig": ig,
                    "in": in_,
                    "tg": tg,
                    "tn": tn,
                    "ov": ov,
                    "rt": rt,
                    "at": at,
                    "note": str(obj.get("note", ""))[:120],
                    "src": fp.name,
                }

    print(f"[merge] parsed {len(files)} files, {total_lines} lines, {bad_lines} malformed, {len(records)} unique wids")
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--catalog", default="data/raw/catalog_items.jsonl")
    ap.add_argument("--out", default="data/annotations/merged_labels.jsonl")
    args = ap.parse_args()

    valid_wids: set[str] = set()
    if Path(args.catalog).exists():
        with Path(args.catalog).open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    valid_wids.add(str(json.loads(line)["workshopId"]))

    records = parse_annotation_files(args.inputs)
    if valid_wids:
        before = len(records)
        records = {w: r for w, r in records.items() if w in valid_wids}
        if len(records) != before:
            print(f"[filter] dropped {before - len(records)} wids not found in catalog")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for wid in sorted(records.keys()):
            f.write(json.dumps(records[wid], ensure_ascii=False) + "\n")

    print("\n=== Annotation Summary ===")
    for field, name in [
        ("ov", "Overall sensitivity"),
        ("ig", "Image Gore"),
        ("in", "Image NSFW"),
        ("tg", "Text Gore"),
        ("tn", "Text NSFW"),
        ("rt", "Red-Text Trap"),
        ("at", "Safe-Anime Trap"),
    ]:
        c = Counter(r[field] for r in records.values())
        print(f"  {name:22s}: {dict(sorted(c.items()))}")

    # Cross-modal stats
    img_only = sum(1 for r in records.values() if max(r["ig"], r["in"]) >= 2 and max(r["tg"], r["tn"]) == 0)
    txt_only = sum(1 for r in records.values() if max(r["tg"], r["tn"]) >= 2 and max(r["ig"], r["in"]) == 0)
    both_mod = sum(1 for r in records.values() if max(r["ig"], r["in"]) >= 2 and max(r["tg"], r["tn"]) >= 2)
    red_safe = sum(1 for r in records.values() if r["rt"] == 1 and r["ig"] == 0)
    anime_safe = sum(1 for r in records.values() if r["at"] == 1 and r["in"] == 0)
    print(f"\n  Strong Image-only sensitive: {img_only}")
    print(f"  Strong Text-only sensitive : {txt_only}")
    print(f"  Strong Both-modal sensitive: {both_mod}")
    print(f"  Confirmed Red-Text Safe    : {red_safe}")
    print(f"  Confirmed Safe-Anime Safe  : {anime_safe}")
    print(f"[saved] {len(records)} validated records -> {out_path}")


if __name__ == "__main__":
    main()
