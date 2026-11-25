#!/usr/bin/env python3
"""
Generate realistic SAFE red-text / red-UI / red-logo / red-title-card images
(in English, Chinese, Japanese, Russian) for:
  1. Independent Red-Text Stress Test (verifying near-0% false positive rate on normal red text)
  2. Optional training-time hard-negative augmentation (strictly separated by seed & phrases!)

Matches the user's explicit requirement:
  "如图片大面积展示血 如整个大片血手掌等 但普通红色字不会引起不适"
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont

LATIN_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-BoldItalic.ttf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-BoldItalic.otf",
]
CJK_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
]

TEST_PHRASES = [
    ("RED HUD v2.4", "Simple red color HUD replacement for L4D2", ["UI"]),
    ("LEFT 4 DEAD 2", "Classic red title logo screen", ["UI"]),
    ("BLOOD HARVEST", "Campaign loading poster text banner", ["Campaigns"]),
    ("红色字体美化", "纯红字游戏界面字体替换补丁", ["UI", "Textures"]),
    ("警告：已更新", "模组更新公告红字封面图", ["Miscellaneous"]),
    ("CRITICAL HIT!", "Red damage number font mod", ["UI"]),
    ("DANGER ZONE", "Red warning sign texture pack", ["Textures"]),
    ("生存模式增强", "黑底红字标题封面 仅脚本修改", ["Scripts"]),
    ("RED LASER SIGHT", "Bright crimson laser crosshair mod", ["Weapons", "UI"]),
    ("GAME OVER", "Classic retro red game over text screen", ["UI"]),
    ("NO MERCY HD", "Red lettering poster for campaign", ["Campaigns"]),
    ("特感刷新提示", "红字字幕提示插件", ["Scripts", "UI"]),
    ("FPS BOOST", "Performance config red banner", ["Scripts"]),
    ("UPDATE 3.0", "Changelog red text cover image", ["Miscellaneous"]),
    ("DEAD CENTER", "Red typography title card", ["Campaigns"]),
    ("高伤武器包", "红底白字武器数值修改模组", ["Weapons", "Scripts"]),
    ("PRESS START", "Arcade red font menu screen", ["UI"]),
    ("CUSTOM FONT", "Bold red subtitle font pack", ["UI"]),
    ("BOSS WARNING", "Tank spawn red text alert banner", ["UI", "Tank"]),
    ("红色主题菜单", "简洁红黑配色主菜单背景", ["UI"]),
]

TRAIN_PHRASES = [
    "L4D2 MOD", "VERSION 1.5", "RED THEME", "CUSTOM HUD", "NEW UPDATE",
    "TEXT ONLY", "FONT PACK", "SERVER MOD", "CONFIG V2", "LOADING...",
    "红字封面", "模组说明", "界面美化", "字幕修改", "武器强化",
    "更新日志", "安装教程", "红黑主题", "菜单替换", "准星美化",
    "RED CROSSHAIR", "SCRIPT FIX", "SOUND PACK", "VOICE MOD", "MAP INFO",
    "ALERT!", "READY", "START GAME", "CHAPTER 1", "SURVIVAL",
]

RED_PALETTE = [
    (220, 20, 20),
    (255, 0, 0),
    (190, 15, 25),
    (165, 10, 15),
    (240, 45, 45),
    (140, 0, 0),
    (230, 30, 50),
]

BG_PALETTE = [
    (10, 10, 12),
    (22, 24, 28),
    (245, 245, 245),
    (230, 232, 235),
    (38, 36, 34),
    (18, 22, 32),
]


def _pick_font(text: str, size: int, rng: random.Random) -> ImageFont.FreeTypeFont:
    has_non_ascii = any(ord(c) > 127 for c in text)
    pool = CJK_FONTS if has_non_ascii else (LATIN_FONTS + CJK_FONTS)
    for _ in range(5):
        p = rng.choice(pool)
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def render_red_text_card(
    text: str,
    rng: random.Random,
    bg_img: Image.Image | None = None,
    width: int = 512,
    height: int = 384,
) -> Image.Image:
    style = rng.choice(["solid_red_text", "red_banner_white_text", "glow_red_text", "photo_red_overlay"])

    if style == "photo_red_overlay" and bg_img is not None:
        img = bg_img.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    elif style == "red_banner_white_text":
        bg = rng.choice(RED_PALETTE)
        img = Image.new("RGB", (width, height), bg)
    else:
        bg = rng.choice(BG_PALETTE)
        img = Image.new("RGB", (width, height), bg)
        # Subtle gradient or border lines
        draw_bg = ImageDraw.Draw(img)
        if rng.random() < 0.5:
            border_col = rng.choice(RED_PALETTE)
            draw_bg.rectangle([12, 12, width - 12, height - 12], outline=border_col, width=rng.randint(3, 8))

    draw = ImageDraw.Draw(img)
    font_size = rng.randint(42, 82)
    font = _pick_font(text, font_size, rng)

    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    while tw > width - 40 and font_size > 20:
        font_size = int(font_size * 0.85)
        font = _pick_font(text, font_size, rng)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x = (width - tw) // 2 + rng.randint(-15, 15)
    y = (height - th) // 2 + rng.randint(-30, 30)

    if style == "red_banner_white_text":
        fg = (250, 250, 250)
        draw.text((x + 3, y + 3), text, fill=(60, 0, 0), font=font)
        draw.text((x, y), text, fill=fg, font=font)
    elif style == "glow_red_text":
        glow = Image.new("RGB", (width, height), (0, 0, 0))
        gd = ImageDraw.Draw(glow)
        fg = rng.choice(RED_PALETTE)
        gd.text((x, y), text, fill=fg, font=font)
        glow = glow.filter(ImageFilter.GaussianBlur(radius=4))
        img = Image.blend(img, glow, 0.45)
        draw = ImageDraw.Draw(img)
        draw.text((x, y), text, fill=fg, font=font)
    else:
        fg = rng.choice(RED_PALETTE)
        # Optional dark outline for legibility
        if rng.random() < 0.6:
            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                draw.text((x + dx, y + dy), text, fill=(0, 0, 0), font=font)
        draw.text((x, y), text, fill=fg, font=font)

    # Optional smaller red subtitle
    if rng.random() < 0.55:
        sub = rng.choice(["LEFT 4 DEAD 2 ADDON", "V2.0 RELEASE", "MOD BY AUTHOR", "红字测试", "READ DESCRIPTION"])
        sfont = _pick_font(sub, rng.randint(18, 28), rng)
        sb = draw.textbbox((0, 0), sub, font=sfont)
        sw = sb[2] - sb[0]
        scol = (240, 240, 240) if style == "red_banner_white_text" else rng.choice(RED_PALETTE)
        draw.text(((width - sw) // 2, min(height - 45, y + th + 24)), sub, fill=scol, font=sfont)

    return img


def generate_stress_test_set(out_dir: Path, count: int = 60, seed: int = 9999) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for i in range(count):
        title, desc, tags = TEST_PHRASES[i % len(TEST_PHRASES)]
        img = render_red_text_card(title, rng)
        img_path = out_dir / f"red_text_{i:03d}.jpg"
        img.save(img_path, format="JPEG", quality=90)
        records.append({
            "id": f"red_text_{i:03d}",
            "image": str(img_path),
            "title": title,
            "description": desc,
            "tags": tags,
            "expected_sensitive": 0,
        })
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[synth] generated {len(records)} red-text stress-test images -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/stress_red_text")
    ap.add_argument("--count", type=int, default=60)
    args = ap.parse_args()
    generate_stress_test_set(Path(args.out_dir), count=args.count)
