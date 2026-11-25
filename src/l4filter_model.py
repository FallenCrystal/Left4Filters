"""
Lightweight Multimodal Sensitivity Classifier (Left4FiltersModel).
Designed for:
  - Joint Image + Text, Image-only, or Text-only inference via modality masks
  - Modality-specific auxiliary supervision (prevents cross-modal label leakage)
  - Soft-label training (0.0 safe / 0.45-0.50 borderline / 1.0 sensitive) for smooth
    two-stage threshold gating (pass / escalate to LLM / flag sensitive)
  - Built-in post-hoc temperature/Platt calibration baked directly into ONNX export
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusionHead(nn.Module):
    def __init__(
        self,
        img_emb_dim: int = 512,
        img_prior_dim: int = 24,
        txt_vec_dim: int = 2048,
        txt_lex_dim: int = 18,
        hidden_dim: int = 160,
        dropout: float = 0.25,
        txt_dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.img_emb_dim = img_emb_dim
        self.img_prior_dim = img_prior_dim
        self.txt_vec_dim = txt_vec_dim
        self.txt_lex_dim = txt_lex_dim

        # Input standardization buffers (fitted on train split, saved inside ONNX!)
        self.register_buffer("img_prior_mean", torch.zeros(img_prior_dim))
        self.register_buffer("img_prior_std", torch.ones(img_prior_dim))
        self.register_buffer("txt_lex_mean", torch.zeros(txt_lex_dim))
        self.register_buffer("txt_lex_std", torch.ones(txt_lex_dim))

        # Vision branch
        self.vis_sem = nn.Sequential(
            nn.Linear(img_emb_dim, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.vis_pri = nn.Sequential(
            nn.Linear(img_prior_dim, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.vis_proj = nn.Sequential(
            nn.Linear(192 + 48, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.vis_head = nn.Linear(hidden_dim, 2)  # [img_gore, img_nsfw]

        # Text branch (zero-collision 2048-dim multilingual TF-IDF + 18-dim clean boundary lexical priors)
        self.txt_sem = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(txt_vec_dim, 96),
            nn.LayerNorm(96),
            nn.GELU(),
            nn.Dropout(txt_dropout),
        )
        self.txt_pri = nn.Sequential(
            nn.Linear(txt_lex_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.4),
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(96 + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.txt_head = nn.Linear(hidden_dim, 2)  # [txt_gore, txt_nsfw]

        # Cross-modal gated fusion MLP
        fused_in = hidden_dim * 3 + 4 + 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_in, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        # Residual adjustments for [gore, nsfw, overall]
        self.delta_head = nn.Linear(64, 3)

        # Post-hoc Platt/Temperature calibration params (baked into ONNX!)
        # Calibrated logit = (raw_logit / temp) + bias
        self.register_buffer("cal_scale", torch.ones(3))   # [overall, gore, nsfw]
        self.register_buffer("cal_bias", torch.zeros(3))

    def set_feature_stats(
        self,
        img_prior_mean: torch.Tensor,
        img_prior_std: torch.Tensor,
        txt_lex_mean: torch.Tensor,
        txt_lex_std: torch.Tensor,
    ) -> None:
        self.img_prior_mean.copy_(img_prior_mean)
        self.img_prior_std.copy_(torch.clamp(img_prior_std, min=1e-4))
        self.txt_lex_mean.copy_(txt_lex_mean)
        self.txt_lex_std.copy_(torch.clamp(txt_lex_std, min=1e-4))

    def set_calibration(self, scale: torch.Tensor, bias: torch.Tensor) -> None:
        self.cal_scale.copy_(scale)
        self.cal_bias.copy_(bias)

    @staticmethod
    def _noisy_or_logit(logit_a: torch.Tensor, logit_b: torch.Tensor, has_a: torch.Tensor, has_b: torch.Tensor) -> torch.Tensor:
        """
        Smooth probabilistic OR over present modalities:
          p = 1 - (1 - has_a * sigmoid(a)) * (1 - has_b * sigmoid(b))
        Returned in logit space with numerical clamping for ONNX stability.
        """
        pa = torch.sigmoid(logit_a) * has_a
        pb = torch.sigmoid(logit_b) * has_b
        p = 1.0 - (1.0 - pa) * (1.0 - pb)
        p = torch.clamp(p, 1e-5, 1.0 - 1e-5)
        return torch.log(p / (1.0 - p))

    def forward_logits(
        self,
        img_emb: torch.Tensor,
        img_priors: torch.Tensor,
        txt_hash: torch.Tensor,
        txt_lex: torch.Tensor,
        has_img: torch.Tensor,
        has_txt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Standardize priors
        ip = (img_priors - self.img_prior_mean) / self.img_prior_std
        ip = torch.clamp(ip, -8.0, 8.0) * has_img
        tl = (txt_lex - self.txt_lex_mean) / self.txt_lex_std
        tl = torch.clamp(tl, -8.0, 8.0) * has_txt

        # Unimodal representations
        v_h = self.vis_proj(torch.cat([self.vis_sem(img_emb * has_img), self.vis_pri(ip)], dim=-1)) * has_img
        t_h = self.txt_proj(torch.cat([self.txt_sem(txt_hash * has_txt), self.txt_pri(tl)], dim=-1)) * has_txt

        vis_logits = self.vis_head(v_h) * has_img  # [B, 2]: ig, in
        txt_logits = self.txt_head(t_h) * has_txt  # [B, 2]: tg, tn

        # Cross-modal interaction
        has_both = has_img * has_txt
        inter = v_h * t_h * has_both

        fused_in = torch.cat([v_h, t_h, inter, vis_logits, txt_logits, has_img, has_txt], dim=-1)
        deltas = self.delta_head(self.fusion_mlp(fused_in))  # [B, 3]: d_gore, d_nsfw, d_ov

        # Base unimodal-consistent logits combined via probabilistic OR
        base_gore = self._noisy_or_logit(vis_logits[:, 0:1], txt_logits[:, 0:1], has_img, has_txt)
        base_nsfw = self._noisy_or_logit(vis_logits[:, 1:2], txt_logits[:, 1:2], has_img, has_txt)

        gore_logit = base_gore + 0.5 * deltas[:, 0:1]
        nsfw_logit = base_nsfw + 0.5 * deltas[:, 1:2]

        # Base overall is probabilistic OR of gore and nsfw
        p_g = torch.sigmoid(gore_logit)
        p_n = torch.sigmoid(nsfw_logit)
        p_or = torch.clamp(1.0 - (1.0 - p_g) * (1.0 - p_n), 1e-5, 1.0 - 1e-5)
        base_ov = torch.log(p_or / (1.0 - p_or))
        overall_logit = base_ov + 0.5 * deltas[:, 2:3]

        return {
            "overall_logit": overall_logit,
            "gore_logit": gore_logit,
            "nsfw_logit": nsfw_logit,
            "vis_logits": vis_logits,
            "txt_logits": txt_logits,
            "vis_repr": v_h,
            "txt_repr": t_h,
        }

    def forward(
        self,
        img_emb: torch.Tensor,
        img_priors: torch.Tensor,
        txt_hash: torch.Tensor,
        txt_lex: torch.Tensor,
        has_img: torch.Tensor,
        has_txt: torch.Tensor,
    ) -> torch.Tensor:
        """
        ONNX inference forward: returns calibrated probabilities [B, 3] = [sensitive, gore, nsfw].
        """
        out = self.forward_logits(img_emb, img_priors, txt_hash, txt_lex, has_img, has_txt)
        raw = torch.cat([out["overall_logit"], out["gore_logit"], out["nsfw_logit"]], dim=-1)
        cal_logits = raw * self.cal_scale + self.cal_bias
        return torch.sigmoid(cal_logits)


class VisualEncoderWrapper(nn.Module):
    """Wraps a visual backbone to output L2-normalized embeddings for clean ONNX export."""
    def __init__(self, visual_backbone: nn.Module) -> None:
        super().__init__()
        self.visual = visual_backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feat = self.visual(pixel_values)
        return F.normalize(feat, dim=-1)
