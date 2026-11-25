#!/usr/bin/env python3
"""
Build a compact, ZERO-COLLISION multilingual TF-IDF vocabulary (2,048 terms)
unsupervised from all 16,838 L4D2 Workshop catalog items (titles, tags, descriptions).

No label information is used here — purely unsupervised corpus DF/IDF statistics,
plus guaranteed slots for curated L4D2 domain terms so zero hash collisions can ever happen.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

BBCODE_RE = re.compile(r"\[/?[a-zA-Z0-9*]+(?:=[^\]]*)?\]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")
# Extract Latin/Cyrillic/digits words (>=2 chars), single CJK chars, and Kana/Hangul runs
TOKEN_RE = re.compile(
    r"[a-z0-9_+\-]{2,}|[\u4e00-\u9fff]|[\u3040-\u30ff]{2,}|[\uac00-\ud7af]{2,}|[\u0400-\u04ff]{2,}"
)

MUST_INCLUDE = [
    # Gore / Blood / Horror
    "blood", "bloody", "bloodied", "gore", "gory", "gibs", "gib", "splatter", "splatters",
    "dismember", "dismemberment", "visceral", "viscera", "mutilated", "decapitation",
    "intestine", "intestines", "guts", "flesh", "corpse", "wound", "wounds", "bleeding",
    "brutal", "grotesque", "grotesquely", "skinned", "flayed", "slaughter", "bloodbath",
    "horror", "creepy", "scary", "nightmare", "puddle", "puddles", "bloodless",
    "血", "血腥", "鲜血", "断肢", "内脏", "尸体", "猎奇", "重口", "血肉", "血手", "恐怖", "飙血",
    "喋血", "复仇", "グロ", "ホラー", "кровь", "жесть",
    # NSFW / Suggestive
    "nsfw", "r18", "r-18", "r18+", "18+", "+18", "hentai", "nude", "naked", "nudity",
    "topless", "nipple", "nipples", "pussy", "cock", "porn", "xxx", "erotic", "lewd",
    "ecchi", "boobs", "booba", "tits", "oppai", "panty", "panties", "lingerie", "bikini",
    "swimsuit", "stripper", "succubus", "futa", "futanari", "ahegao", "uncensored",
    "sexy", "thicc", "busty", "curvy", "cleavage", "bunny", "maid", "latex", "stockings",
    "fishnet", "underwear", "sensual", "adult", "sex", "sexual", "waifu", "dakimakura",
    "cleavage", "sideboob", "upskirt", "censored", "naughty",
    "清凉", "绅士", "果体", "裸体", "裸", "涩", "涩图", "色情", "色气", "福利", "比基尼",
    "泳装", "内衣", "情趣", "黑丝", "白丝", "丝袜", "大胸", "巨乳", "爆乳", "魅魔",
    "兔女郎", "擦边", "不正经", "性感", "诱惑", "露点", "喷漆", "动图", "抱枕",
    "エロ", "水着", "下着", "セクシー",
    # Safe / Red-text / L4D2 domain anchors
    "red", "crimson", "font", "fonts", "hud", "icon", "icons", "ui", "menu", "logo",
    "loading", "screen", "poster", "banner", "crosshair", "laser", "fire", "flame",
    "sound", "sounds", "music", "voice", "bgm", "ost", "script", "scripts", "vscript",
    "weapon", "weapons", "gun", "rifle", "shotgun", "pistol", "sniper", "melee", "knife",
    "campaign", "campaigns", "map", "maps", "harvest", "mercy", "center", "carnival",
    "parish", "sacrifice", "passing", "reanim", "reanims", "anime", "genshin", "miku",
    "touhou", "hololive", "vtuber", "archive", "arknights", "honkai", "school", "uniform",
    "红", "红色", "红字", "字体", "界面", "图标", "准星", "音效", "音乐", "语音", "脚本",
    "地图", "武器", "替换", "模型", "菜单", "二次元", "原神", "碧蓝", "方舟", "初音",
]


def clean_text(text: str | None, max_len: int = 1000) -> str:
    if not text:
        return ""
    s = BBCODE_RE.sub(" ", text)
    s = URL_RE.sub(" ", s)
    return WS_RE.sub(" ", s).strip()[:max_len]


def tokenize(text: str) -> list[str]:
    low = text.lower()
    raw = TOKEN_RE.findall(low)
    toks: list[str] = []
    for t in raw:
        if len(t) > 24:
            continue
        toks.append(t)
    # Add CJK bigrams (crucial for Chinese terms like 血腥, 泳装, 红字, 字体, 喋血, 复仇)
    for i in range(len(raw) - 1):
        a, b = raw[i], raw[i + 1]
        if len(a) == 1 and len(b) == 1 and ("\u4e00" <= a <= "\u9fff") and ("\u4e00" <= b <= "\u9fff"):
            toks.append(a + b)
    return toks


def main() -> None:
    catalog_path = Path("data/raw/catalog_items.jsonl")
    out_path = Path("data/cache/text_vocab.json")
    vocab_size = 2048

    df: Counter[str] = Counter()
    n_docs = 0
    with catalog_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n_docs += 1
            title = clean_text(r.get("title"), 256)
            desc = clean_text(r.get("clean_description") or r.get("description"), 900)
            tags = [f"tag:{str(t).strip().lower()}" for t in (r.get("tags") or []) if str(t).strip()]
            doc_toks = set(tokenize(title)) | set(tokenize(desc)) | set(tags)
            for tok in doc_toks:
                df[tok] += 1

    # Select guaranteed domain tokens first, then fill up to vocab_size by document frequency (min_df >= 10)
    selected: list[str] = []
    seen: set[str] = set()

    # Add all tags occurring >= 15 times
    for tok, cnt in df.most_common():
        if tok.startswith("tag:") and cnt >= 15 and tok not in seen:
            selected.append(tok)
            seen.add(tok)

    # Add curated domain tokens
    for tok in MUST_INCLUDE:
        if tok not in seen:
            selected.append(tok)
            seen.add(tok)

    # Fill remainder with highest-DF corpus tokens (excluding pure digits and extreme stopwords > 60% DF)
    for tok, cnt in df.most_common():
        if len(selected) >= vocab_size:
            break
        if tok in seen or cnt < 10:
            continue
        if tok.isdigit() and tok not in {"18", "18+"}:
            continue
        if cnt > 0.65 * n_docs:
            continue
        selected.append(tok)
        seen.add(tok)

    vocab = {}
    idfs = []
    for idx, tok in enumerate(selected[:vocab_size]):
        d = df.get(tok, 1)
        idf = math.log((1.0 + n_docs) / (1.0 + d)) + 1.0
        vocab[tok] = idx
        idfs.append(round(idf, 5))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"vocab_size": len(vocab), "n_docs": n_docs, "vocab": vocab, "idf": idfs}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[vocab] built zero-collision multilingual vocab ({len(vocab)} terms from {n_docs} docs, {out_path.stat().st_size/1024:.1f} KB) -> {out_path}")


if __name__ == "__main__":
    main()
