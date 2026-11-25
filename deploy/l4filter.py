#!/usr/bin/env python3
"""
Left4FiltersModel — Lightweight Portable ONNX Runtime Inference SDK & CLI.

Dependencies: ONLY onnxruntime, numpy, pillow (NO PyTorch / transformers / timm required!).
Supports:
  - Joint text + single image
  - Joint text + multiple preview gallery images (worst-case max aggregation)
  - Image-only
  - Text-only (< 0.4 ms CPU)

Primary output: calibrated probability in [0, 1] for threshold-based sensitivity flagging
or routing uncertain items to an LLM judge.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image

BBCODE_RE = re.compile(r"\[/?[a-zA-Z0-9*]+(?:=[^\]]*)?\]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(
    r"[a-z0-9_+\-]{2,}|[\u4e00-\u9fff]|[\u3040-\u30ff]{2,}|[\uac00-\ud7af]{2,}|[\u0400-\u04ff]{2,}"
)

GORE_STRONG_RE = re.compile(
    r"(?:\b(?:gore|gory|gibs?|dismember(?:ment|ed)?|visceral|viscera|mutilat\w*|decapitat\w*|"
    r"intestines?|entrails|eviscerat\w*|bloodbath|grotesque(?:ly)?|skinned|flayed|"
    r"bloodied|raw\s+flesh|heavy\s+blood|extreme\s+blood|real(?:istic)?\s+blood|"
    r"blood\s+splatter|bloody\s+hand\w*|drenched\s+in\s+blood|bleeding\s+wounds?)\b|"
    r"血腥|断肢|内脏|尸体|猎奇|重口|血肉|血手|飙血|鲜血淋漓|血泊|グロ|リョナ|расчленен|жесть)",
    re.IGNORECASE,
)
GORE_MILD_RE = re.compile(
    r"(?:\b(?:blood|bloody|wounds?|flesh|corpse|bleed(?:ing)?|splatters?|horror|"
    r"scary|creepy|nightmare|slaughter|puddles?)\b|"
    r"血|恐怖|伤口|惊悚|丧尸|ホラー|кровь)",
    re.IGNORECASE,
)
NSFW_STRONG_RE = re.compile(
    r"(?:(?<![a-z0-9])(?:nsfw|r-?18\+?|18\+|\+18|r20|hentai|nude|naked|nudity|topless|"
    r"nipples?|pussy|cock|dick|porn|xxx|erotic|lewd|ecchi|boobs?|booba|tits|oppai|"
    r"pant(?:y|ies)|lingerie|bikini|microbikini|stripper|succubus|futa(?:nari)?|"
    r"ahegao|uncensored|upskirt|sideboob|adult\s+content|sex(?:ual)?)(?![a-z0-9])|"
    r"清凉|绅士|果体|裸体|裸|涩图|涩涩|超涩|涩爆|色情|色气|福利|比基尼|内衣|情趣|"
    r"大胸|巨乳|爆乳|魅魔|擦边|不正经|诱惑|露点|无圣光|エロ|下着|痴女)",
    re.IGNORECASE,
)
NSFW_MILD_RE = re.compile(
    r"(?:\b(?:sexy|swimsuit|thicc|busty|curvy|cleavage|bunny|maid|latex|stockings?|"
    r"fishnet|sensual|underwear|dakimakura|naughty)\b|"
    r"泳装|黑丝|白丝|丝袜|兔女郎|性感|女仆|抱枕|短裙|美腿|水着|セクシー)",
    re.IGNORECASE,
)
SAFE_TRAP_RE = re.compile(
    r"(?:\b(?:blood\s+harvest|back\s*4\s*blood|bloodless|no\s+blood|dead\s+center|"
    r"dark\s+carnival|swamp\s+fever|hard\s+rain|the\s+parish|cold\s+stream|"
    r"no\s+mercy|crash\s+course|death\s+toll|dead\s+air|the\s+sacrifice|the\s+passing|"
    r"left\s*4\s*dead|l4d2?|fonts?|hud|crosshair|icons?|menu|loading\s+screen|"
    r"scripts?|sounds?|music|voice|bgm|ost|reanims?|weapons?|hitmarker)\b|"
    r"喋血复仇|血腥收获|红字|字体|界面|图标|准星|音效|音乐|语音|脚本|菜单|换弹)",
    re.IGNORECASE,
)

VISUAL_PRIOR_DIM = 24
LEXICAL_PRIOR_DIM = 18


def _clean_text(text: str | None, max_len: int = 1600) -> str:
    if not text:
        return ""
    s = BBCODE_RE.sub(" ", text)
    s = URL_RE.sub(" ", s)
    return WS_RE.sub(" ", s).strip()[:max_len]


def _tokenize_vocab(text: str) -> list[str]:
    low = text.lower()
    raw = TOKEN_RE.findall(low)
    toks = [t for t in raw if len(t) <= 24]
    for i in range(len(raw) - 1):
        a, b = raw[i], raw[i + 1]
        if len(a) == 1 and len(b) == 1 and ("\u4e00" <= a <= "\u9fff") and ("\u4e00" <= b <= "\u9fff"):
            toks.append(a + b)
    return toks


def extract_text_features(
    title: str,
    description: str,
    tags: Iterable[str] | None,
    vocab_data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    vocab: dict[str, int] = vocab_data["vocab"]
    idfs: list[float] = vocab_data["idf"]
    dim = int(vocab_data["vocab_size"])

    t_clean = _clean_text(title, max_len=256)
    d_clean = _clean_text(description, max_len=1600)
    tag_list = [str(x).strip() for x in (tags or []) if str(x).strip()]

    counts: dict[int, float] = {}

    def accum(toks: list[str], weight: float) -> None:
        local_c: dict[int, int] = {}
        for tok in toks:
            idx = vocab.get(tok)
            if idx is not None:
                local_c[idx] = local_c.get(idx, 0) + 1
        for idx, c in local_c.items():
            counts[idx] = counts.get(idx, 0.0) + weight * (1.0 + math.log(c)) * idfs[idx]

    accum(_tokenize_vocab(t_clean), 2.5)
    accum([f"tag:{tg.lower()}" for tg in tag_list], 1.2)
    accum(_tokenize_vocab(d_clean[:1000]), 1.0)

    vec = np.zeros(dim, dtype=np.float32)
    if counts:
        for idx, val in counts.items():
            vec[idx] = val
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec /= norm

    t_gore_s = len(GORE_STRONG_RE.findall(t_clean))
    t_gore_m = len(GORE_MILD_RE.findall(t_clean))
    d_gore_s = len(GORE_STRONG_RE.findall(d_clean))
    d_gore_m = len(GORE_MILD_RE.findall(d_clean))

    t_nsfw_s = len(NSFW_STRONG_RE.findall(t_clean))
    t_nsfw_m = len(NSFW_MILD_RE.findall(t_clean))
    d_nsfw_s = len(NSFW_STRONG_RE.findall(d_clean))
    d_nsfw_m = len(NSFW_MILD_RE.findall(d_clean))

    t_safe = len(SAFE_TRAP_RE.findall(t_clean))
    d_safe = len(SAFE_TRAP_RE.findall(d_clean[:600]))

    tag_low_set = {x.lower() for x in tag_list}
    lex = np.array([
        math.log1p(t_gore_s),
        math.log1p(t_gore_m),
        math.log1p(d_gore_s),
        math.log1p(d_gore_m),
        math.log1p(t_nsfw_s),
        math.log1p(t_nsfw_m),
        math.log1p(d_nsfw_s),
        math.log1p(d_nsfw_m),
        math.log1p(t_safe),
        math.log1p(d_safe),
        float((t_gore_s + d_gore_s) > 0),
        float((t_nsfw_s + d_nsfw_s) > 0),
        float((t_nsfw_m + d_nsfw_m) > 0),
        float(bool(tag_low_set & {"survivors", "zoey", "rochelle", "models", "witch"})),
        float(bool(tag_low_set & {"textures", "common infected", "special infected", "particles"})),
        float(bool(tag_low_set & {"ui", "sounds", "scripts", "campaigns", "weapons"})),
        min(len(t_clean), 120) / 120.0,
        min(len(d_clean), 1200) / 1200.0,
    ], dtype=np.float32)
    return vec, lex


def extract_visual_priors(img: Image.Image) -> np.ndarray:
    orig_w, orig_h = img.size
    aspect = math.log(max(orig_w, 1) / max(orig_h, 1))
    thumb = img.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    arr = np.asarray(thumb, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    sat = np.where(cmax > 1e-5, delta / (cmax + 1e-5), 0.0)
    val = cmax

    hue = np.zeros_like(cmax)
    mask_d = delta > 1e-5
    m_r = mask_d & (cmax == r)
    m_g = mask_d & (cmax == g) & (~m_r)
    m_b = mask_d & (~m_r) & (~m_g)
    hue[m_r] = 60.0 * (((g[m_r] - b[m_r]) / (delta[m_r] + 1e-5)) % 6.0)
    hue[m_g] = 60.0 * (((b[m_g] - r[m_g]) / (delta[m_g] + 1e-5)) + 2.0)
    hue[m_b] = 60.0 * (((r[m_b] - g[m_b]) / (delta[m_b] + 1e-5)) + 4.0)

    is_red_hue = (hue <= 16.0) | (hue >= 342.0)
    blood_mask = (
        is_red_hue & (sat >= 0.40) & (val >= 0.12) & (val <= 0.78)
        & (r > g * 1.65) & (r > b * 1.65)
    )
    bright_red_mask = (
        ((hue <= 12.0) | (hue >= 348.0)) & (sat >= 0.72) & (val > 0.75)
        & (r > 0.75) & (g < 0.25) & (b < 0.25)
    )
    any_red_mask = blood_mask | bright_red_mask

    blood_ratio = float(blood_mask.mean())
    bright_red_ratio = float(bright_red_mask.mean())
    any_red_ratio = float(any_red_mask.mean())
    center_blood = float(blood_mask[24:104, 24:104].mean())
    max_quad_blood = float(max(
        blood_mask[:64, :64].mean(), blood_mask[:64, 64:].mean(),
        blood_mask[64:, :64].mean(), blood_mask[64:, 64:].mean(),
    ))

    red_f = any_red_mask.astype(np.float32)
    red_dx = np.abs(red_f[:, 1:] - red_f[:, :-1]).sum()
    red_dy = np.abs(red_f[1:, :] - red_f[:-1, :]).sum()
    red_area = float(red_f.sum())
    red_perim_ratio = float((red_dx + red_dy) / (red_area + 25.0))

    if red_area >= 20:
        red_val_std = float(val[any_red_mask].std())
        red_sat_std = float(sat[any_red_mask].std())
        red_val_mean = float(val[any_red_mask].mean())
    else:
        red_val_std = red_sat_std = red_val_mean = 0.0

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    grad = np.zeros_like(gray)
    grad[:, :-1] += gx
    grad[:-1, :] += gy

    flat_bg_ratio = float((grad < 0.015).mean())
    sharp_edge_ratio = float((grad > 0.25).mean())
    mean_grad = float(grad.mean())
    hv_edge_ratio = float(((gx > 0.20).mean() + (gy > 0.20).mean()) * 0.5)

    dark_ratio = float((val < 0.18).mean())
    very_bright_ratio = float((val > 0.88).mean())
    mean_val = float(val.mean())
    mean_sat = float(sat.mean())

    y_ch = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
    skin_mask = (
        (y_ch > 0.22) & (cb >= 0.30) & (cb <= 0.50) & (cr >= 0.52) & (cr <= 0.68)
        & (r > g) & (g > b * 0.85) & (~any_red_mask)
    )
    skin_ratio = float(skin_mask.mean())
    center_skin_ratio = float(skin_mask[20:108, 20:108].mean())
    anime_skin_mask = (
        (r > 0.72) & (g > 0.52) & (b > 0.42) & (r > g)
        & ((r - b) < 0.38) & ((r - b) > 0.06) & (sat < 0.42)
    )
    anime_skin_ratio = float(anime_skin_mask.mean())
    blood_vs_bright = math.log((blood_ratio + 0.005) / (bright_red_ratio + 0.005))

    return np.array([
        blood_ratio, bright_red_ratio, any_red_ratio, center_blood, max_quad_blood,
        min(red_perim_ratio, 3.0) / 3.0, red_val_std, red_sat_std, red_val_mean,
        blood_vs_bright / 4.0, flat_bg_ratio, sharp_edge_ratio, mean_grad,
        hv_edge_ratio, dark_ratio, very_bright_ratio, mean_val, mean_sat,
        skin_ratio, center_skin_ratio, anime_skin_ratio,
        float(np.clip(aspect, -2.0, 2.0)) / 2.0,
        float(blood_mask.mean() * (1.0 - flat_bg_ratio)),
        float(bright_red_ratio * flat_bg_ratio),
    ], dtype=np.float32)


def preprocess_image(img: Image.Image, size: int = 256) -> np.ndarray:
    im = img.convert("RGB")
    w, h = im.size
    if w < h:
        new_w = size
        new_h = max(size, int(round(h * size / w)))
    else:
        new_h = size
        new_w = max(size, int(round(w * size / h)))
    im = im.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    im = im.crop((left, top, left + size, top + size))
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def _load_pil(img_input: str | Path | bytes | Image.Image) -> Image.Image:
    if isinstance(img_input, Image.Image):
        return img_input.convert("RGB")
    if isinstance(img_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(img_input)).convert("RGB")
    return Image.open(img_input).convert("RGB")


class Left4Filter:
    """Portable CPU sensitivity classifier for L4D2 Workshop text + images."""

    def __init__(self, model_dir: str | Path | None = None, num_threads: int = 4) -> None:
        base = Path(model_dir) if model_dir is not None else Path(__file__).resolve().parent
        cfg_path = base / "model_config.json"
        self.config: dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        self.vocab_data: dict[str, Any] = json.loads((base / "text_vocab.json").read_text(encoding="utf-8"))
        self.txt_dim = int(self.vocab_data["vocab_size"])
        self.img_size = int(self.config.get("image_size", 256))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.vis_sess = ort.InferenceSession(
            str(base / "vision_encoder.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.head_sess = ort.InferenceSession(
            str(base / "fusion_head.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def predict(
        self,
        *,
        title: str = "",
        description: str = "",
        tags: Sequence[str] | None = None,
        image: str | Path | bytes | Image.Image | None = None,
        images: Sequence[str | Path | bytes | Image.Image] | None = None,
    ) -> dict[str, float]:
        """
        Predict calibrated probabilities for an L4D2 Workshop item.
        Supports:
          - text + image
          - text + multiple gallery images (returns max sensitivity across gallery images)
          - image-only
          - text-only

        Returns dict with calibrated probabilities in [0, 1]:
          {"prob": float, "gore_prob": float, "nsfw_prob": float}
        """
        has_txt = bool((title or "").strip() or (description or "").strip() or (tags and len(tags) > 0))
        if has_txt:
            th, tl = extract_text_features(title, description, tags, self.vocab_data)
        else:
            th = np.zeros(self.txt_dim, dtype=np.float32)
            tl = np.zeros(LEXICAL_PRIOR_DIM, dtype=np.float32)

        img_list: list[str | Path | bytes | Image.Image] = []
        if image is not None:
            img_list.append(image)
        if images:
            img_list.extend(images)

        if not img_list:
            if not has_txt:
                return {"prob": 0.0, "gore_prob": 0.0, "nsfw_prob": 0.0}
            out = self.head_sess.run(
                ["probs"],
                {
                    "img_emb": np.zeros((1, 512), dtype=np.float32),
                    "img_priors": np.zeros((1, VISUAL_PRIOR_DIM), dtype=np.float32),
                    "txt_hash": th[None, ...],
                    "txt_lex": tl[None, ...],
                    "has_img": np.zeros((1, 1), dtype=np.float32),
                    "has_txt": np.ones((1, 1), dtype=np.float32),
                },
            )[0][0]
            return {
                "prob": round(float(out[0]), 6),
                "gore_prob": round(float(out[1]), 6),
                "nsfw_prob": round(float(out[2]), 6),
            }

        pils = [_load_pil(im) for im in img_list]
        px_batch = np.stack([preprocess_image(im, self.img_size) for im in pils], axis=0)
        pri_batch = np.stack([extract_visual_priors(im) for im in pils], axis=0)
        emb_batch = self.vis_sess.run(["img_emb"], {"pixel_values": px_batch})[0]

        n = len(pils)
        th_batch = np.repeat(th[None, ...], n, axis=0)
        tl_batch = np.repeat(tl[None, ...], n, axis=0)
        has_img_b = np.ones((n, 1), dtype=np.float32)
        has_txt_b = np.full((n, 1), 1.0 if has_txt else 0.0, dtype=np.float32)

        probs = self.head_sess.run(
            ["probs"],
            {
                "img_emb": emb_batch,
                "img_priors": pri_batch,
                "txt_hash": th_batch,
                "txt_lex": tl_batch,
                "has_img": has_img_b,
                "has_txt": has_txt_b,
            },
        )[0]  # (N, 3)

        return {
            "prob": round(float(probs[:, 0].max()), 6),
            "gore_prob": round(float(probs[:, 1].max()), 6),
            "nsfw_prob": round(float(probs[:, 2].max()), 6),
        }

    def predict_prob(self, **kwargs: Any) -> float:
        """Convenience helper returning ONLY the scalar calibrated probability in [0, 1]."""
        return self.predict(**kwargs)["prob"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Left4FiltersModel lightweight CLI")
    ap.add_argument("--model-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--title", default="")
    ap.add_argument("--desc", default="")
    ap.add_argument("--tags", default="", help="Comma-separated workshop tags")
    ap.add_argument("--image", nargs="*", default=[], help="One or more image file paths")
    ap.add_argument("--prob-only", action="store_true", help="Print only scalar sensitive probability")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    clf = Left4Filter(args.model_dir, num_threads=args.threads)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    res = clf.predict(title=args.title, description=args.desc, tags=tags, images=args.image or None)
    if args.prob_only:
        print(f"{res['prob']:.6f}")
    else:
        print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
