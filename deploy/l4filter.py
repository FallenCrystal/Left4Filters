#!/usr/bin/env python3
"""
Left4FiltersModel — Lightweight Portable ONNX Runtime Inference SDK & CLI.

Dependencies: ONLY onnxruntime, numpy, pillow (NO PyTorch / transformers / timm required!).
Supports:
  - Joint workshop text and additional text entries with one or more preview images
  - Image-only
  - Text-only (< 0.4 ms CPU)

Returns a routing score in [0, 1], category scores and optional evidence for suspicious components.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image

BBCODE_RE = re.compile(
    r"\[/?(?:\*|h[1-6]|b|i|u|s|strike|spoiler|noparse|code|hr|url|img|"
    r"previewyoutube|youtube|video|list|olist|quote|table|tr|th|td|sup|sub|"
    r"center|left|right|color|size|font)(?:=[^\]]*)?\]", re.IGNORECASE,
)
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
BASE_SCORES = ("prob", "gore_prob", "nsfw_prob")
BASE_CATEGORIES = ("sensitive", "gore", "nsfw")
CORPSE_CATEGORIES = ("corpse", "corpse_pile")


class TextComponent(NamedTuple):
    field: str
    index: int | None
    value: str


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
        img_input = io.BytesIO(img_input)
    with Image.open(img_input) as im:
        return im.convert("RGB")


def _text_excerpt(component: TextComponent, reasons: list[dict[str, Any]]) -> str:
    text = _clean_text(component.value, max_len=256 if component.field == "title" else 1600)
    anchor = None
    for reason in reasons:
        category = reason["category"]
        if category == "gore":
            patterns = (GORE_STRONG_RE, GORE_MILD_RE)
            ignored = [m.span() for m in SAFE_TRAP_RE.finditer(text)]
        elif category == "nsfw":
            patterns = (NSFW_STRONG_RE, NSFW_MILD_RE)
            ignored = []
        else:
            continue
        matches = [
            match for pattern in patterns for match in pattern.finditer(text)
            if not any(start <= match.start() and match.end() <= end for start, end in ignored)
        ]
        if matches:
            reason["matches"] = list(dict.fromkeys(m.group()[:80] for m in matches))[:3]
            if anchor is None:
                anchor = matches[0].start()
    start = min(max((anchor or 0) - 48, 0), max(len(text) - 160, 0))
    end = start + 160
    return ("…" if start else "") + text[start:end].strip() + ("…" if end < len(text) else "")


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
        self.review_threshold = float(
            self.config.get("recommended_thresholds", {}).get("escalate_to_llm", 0.20)
        )
        self.corpse_sess = None
        corpse_config = self.config.get("corpse_detection")
        if corpse_config is not None:
            self.corpse_thresholds = np.array(
                [corpse_config["thresholds"][name] for name in CORPSE_CATEGORIES], dtype=np.float32,
            )
            self.corpse_sess = ort.InferenceSession(
                str(base / "corpse_head.onnx"), sess_options=opts, providers=["CPUExecutionProvider"]
            )

    def _encode_text(self, group: Sequence[TextComponent]) -> tuple[np.ndarray, np.ndarray] | None:
        if not group:
            return None
        title = next((part.value for part in group if part.field == "title"), "")
        desc = next((part.value for part in group if part.field in ("description", "texts")), "")
        tags = [part.value for part in group if part.field == "tags"]
        return extract_text_features(title, desc, tags, self.vocab_data)

    def _encode_images(
        self, images: Sequence[str | Path | bytes | Image.Image],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not images:
            return None
        embeddings, priors = [], []
        for start in range(0, len(images), 12):
            pixels = []
            for source in images[start:start + 12]:
                with _load_pil(source) as im:
                    pixels.append(preprocess_image(im, self.img_size))
                    priors.append(extract_visual_priors(im))
            embeddings.append(self.vis_sess.run(["img_emb"], {"pixel_values": np.stack(pixels)})[0])
        return np.concatenate(embeddings), np.stack(priors)

    def _score(
        self, vision: tuple[np.ndarray, np.ndarray] | None, text: tuple[np.ndarray, np.ndarray] | None,
    ) -> np.ndarray:
        if vision is None and text is None:
            return np.zeros((1, 3), dtype=np.float32)
        if vision is None:
            vision_features = (np.zeros((1, 512), np.float32), np.zeros((1, VISUAL_PRIOR_DIM), np.float32))
        else:
            vision_features = vision
        n = len(vision_features[0])
        if text is None:
            text_features = (np.zeros((n, self.txt_dim), np.float32), np.zeros((n, LEXICAL_PRIOR_DIM), np.float32))
        else:
            text_features = tuple(np.repeat(feature[None], n, axis=0) for feature in text)
        return self.head_sess.run(
            ["probs"],
            {
                "img_emb": vision_features[0],
                "img_priors": vision_features[1],
                "txt_hash": text_features[0],
                "txt_lex": text_features[1],
                "has_img": np.full((n, 1), vision is not None, np.float32),
                "has_txt": np.full((n, 1), text is not None, np.float32),
            },
        )[0]

    def _reasons(
        self, candidates: Iterable[tuple[str, np.ndarray, np.ndarray | None]],
    ) -> list[dict[str, Any]]:
        reasons: dict[str, dict[str, Any]] = {}
        for support, scores, without_component in candidates:
            scores = np.atleast_2d(scores)
            delta = None
            if without_component is not None:
                delta = scores - without_component
                crossed = (scores >= self.review_threshold) & (without_component < self.review_threshold)
                scores = np.where((delta >= 0.08) | crossed, scores, 0.0)
            for i, row in enumerate(scores.argmax(axis=0)):
                category = BASE_CATEGORIES[i]
                rounded = round(float(scores[row, i]), 6)
                if rounded >= self.review_threshold and rounded > reasons.get(category, {}).get("score", 0):
                    reason = {"category": category, "score": rounded, "support": support}
                    if delta is not None:
                        reason["delta"] = round(float(delta[row, i]), 6)
                    reasons[category] = reason
        specific = [reason["score"] for name, reason in reasons.items() if name != "sensitive"]
        if "sensitive" in reasons and specific and max(specific) + 0.08 >= reasons["sensitive"]["score"]:
            del reasons["sensitive"]
        return sorted(reasons.values(), key=lambda reason: reason["score"], reverse=True)

    def _evidence(self, refs, vision, groups, features, joint, corpses) -> list[dict[str, Any]]:
        evidence: list[dict[str, Any]] = []
        image_only = self._score(vision, None) if groups else joint[0]
        text_only = [self._score(None, f) for f in features] if vision is not None else joint

        def append(kind, field, index, reasons, **details):
            if reasons:
                entry = {"type": kind, "field": field, "score": max(r["score"] for r in reasons), "reasons": reasons}
                if index is not None:
                    entry["index"] = index
                evidence.append({**entry, **details})

        for i, (field, index) in enumerate(refs):
            candidates = [("standalone", image_only[i], None)]
            candidates.extend(("joint", scores[i], without[0]) for scores, without in zip(joint, text_only))
            reasons = self._reasons(candidates)
            if corpses is not None:
                for category, score, threshold in zip(CORPSE_CATEGORIES, corpses[i], self.corpse_thresholds):
                    if score >= threshold:
                        reasons.append({"category": category, "score": round(float(score), 6), "support": "standalone"})
            append("image", field, index, reasons)

        for g, group in enumerate(groups):
            for i, component in enumerate(group):
                if len(group) == 1:
                    candidates = [("standalone", text_only[g], None)]
                    other_features = None
                else:
                    component_only = self._score(None, self._encode_text([component]))
                    other_features = self._encode_text(group[:i] + group[i + 1:])
                    other_text = self._score(None, other_features)
                    candidates = [("standalone", component_only, None), ("text_context", text_only[g], other_text)]
                if vision is not None:
                    without = self._score(vision, other_features) if other_features is not None else image_only
                    candidates.append(("joint", joint[g], without))
                reasons = self._reasons(candidates)
                if reasons:
                    details = {"text": _text_excerpt(component, reasons)}
                    append("text", component.field, component.index, reasons, **details)
        return sorted(evidence, key=lambda item: item["score"], reverse=True)

    def predict(
        self,
        *,
        title: str = "",
        description: str = "",
        tags: Sequence[str] | None = None,
        image: str | Path | bytes | Image.Image | None = None,
        images: Sequence[str | Path | bytes | Image.Image] | None = None,
        texts: Sequence[str] | None = None,
        explain: bool = False,
    ) -> dict[str, Any]:
        """Score each image with the workshop fields and with every additional text."""
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings")
        structured = [
            TextComponent(field, None, value)
            for field, value in (("title", title), ("description", description))
            if value and value.strip()
        ]
        structured.extend(TextComponent("tags", i, str(tag)) for i, tag in enumerate(tags or ()) if str(tag).strip())
        groups = [structured] if structured else []
        groups.extend([TextComponent("texts", i, value)] for i, value in enumerate(texts or ()) if value.strip())
        features = [self._encode_text(group) for group in groups]

        img_list, refs = [], []
        if image is not None:
            img_list.append(image)
            refs.append(("image", None))
        if images:
            img_list.extend(images)
            refs.extend(("images", i) for i in range(len(images)))
        vision = self._encode_images(img_list)
        joint = [self._score(vision, f) for f in features or [None]]
        max_scores = np.maximum.reduce([scores.max(axis=0) for scores in joint])
        result = {key: round(float(score), 6) for key, score in zip(BASE_SCORES, max_scores)}

        corpses = None
        if self.corpse_sess is not None:
            if vision is not None:
                corpses = self.corpse_sess.run(["corpse_probs"], {"img_emb": vision[0]})[0]
                if np.any(corpses >= self.corpse_thresholds):
                    result["prob"] = max(result["prob"], self.review_threshold)
            for i, name in enumerate(CORPSE_CATEGORIES):
                result[f"{name}_prob"] = round(float(corpses[:, i].max()), 6) if corpses is not None else 0.0
        if explain:
            result["evidence"] = self._evidence(refs, vision, groups, features, joint, corpses)
        return result

    def predict_prob(self, **kwargs: Any) -> float:
        """Return the scalar sensitivity routing score in [0, 1]."""
        return self.predict(**kwargs)["prob"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Left4FiltersModel lightweight CLI")
    ap.add_argument("--model-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--title", default="")
    ap.add_argument("--desc", default="")
    ap.add_argument("--tags", default="", help="Comma-separated workshop tags")
    ap.add_argument("--image", nargs="*", default=[], help="One or more image file paths")
    ap.add_argument("--text", action="append", help="Additional text to score, repeatable")
    ap.add_argument("--explain", action="store_true", help="Include evidence for suspicious image and text components")
    ap.add_argument("--prob-only", action="store_true", help="Print only the scalar routing score")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    clf = Left4Filter(args.model_dir, num_threads=args.threads)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
    res = clf.predict(
        title=args.title, description=args.desc, tags=tags, images=args.image or None,
        texts=args.text, explain=args.explain and not args.prob_only,
    )
    if args.prob_only:
        print(f"{res['prob']:.6f}")
    else:
        print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
