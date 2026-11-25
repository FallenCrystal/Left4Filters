#!/usr/bin/env python3
"""
Optional Ultra-Tiny Student Distillation Experiment:
Distills MobileCLIP2-S0's 512-dim visual semantics (across all ~5,500 downloaded L4D2 images!)
into a tiny MobileNetV3-Small (~2.5M params -> ~2.5MB INT8, ~4ms CPU) visual encoder,
then trains & evaluates MultimodalFusionHead on the exact same locked creator-disjoint Test split.

This gives a direct empirical accuracy-vs-size Pareto comparison against MobileCLIP2-S0.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import importlib.util
_spec = importlib.util.spec_from_file_location("train_cmp_mod", Path(__file__).resolve().parent / "04_train_and_compare.py")
_tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tc)
build_feature_arrays = _tc.build_feature_arrays
concat_arrays = _tc.concat_arrays
eval_synth_red_stress = _tc.eval_synth_red_stress
evaluate_model = _tc.evaluate_model
generate_synth_red_train_features = _tc.generate_synth_red_train_features
load_data = _tc.load_data
make_or_load_creator_split = _tc.make_or_load_creator_split
seed_everything = _tc.seed_everything
train_fusion_head = _tc.train_fusion_head


class TinyVisualStudent(nn.Module):
    def __init__(self, backbone_name: str = "mobilenetv3_small_100", out_dim: int = 512) -> None:
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        with torch.no_grad():
            in_dim = int(self.encoder(torch.zeros(1, 3, 224, 224)).shape[-1])
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.encoder(x)), dim=-1)


class DistillImgDataset(Dataset):
    def __init__(self, wids: list[str], embs: np.ndarray, img_dir: Path, is_train: bool = True) -> None:
        self.wids = wids
        self.embs = embs
        self.img_dir = img_dir
        if is_train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.80, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

    def __len__(self) -> int:
        return len(self.wids)

    def __getitem__(self, idx: int):
        with Image.open(self.img_dir / f"{self.wids[idx]}.jpg") as im:
            px = self.tf(im.convert("RGB"))
        return px, torch.from_numpy(self.embs[idx])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/raw/catalog_items.jsonl")
    ap.add_argument("--labels", default="data/annotations/merged_final.jsonl")
    ap.add_argument("--cache", default="data/cache/img_features_mc2s0.npz")
    ap.add_argument("--split", default="data/annotations/split_manifest.json")
    ap.add_argument("--img-dir", default="data/images")
    ap.add_argument("--stress-dir", default="data/stress_red_text")
    ap.add_argument("--out-dir", default="checkpoints/exp_tiny_distill")
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()

    torch.set_num_threads(4)
    seed_everything(42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all cached images (excluding locked test/val creators from distillation!)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    banned_wids = set(split["test"]) | set(split["val"])
    banned_creators = set(split.get("val_creators", [])) | set(split.get("test_creators", []))

    catalog = {}
    with Path(args.catalog).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                catalog[str(r["workshopId"])] = r

    cache = np.load(args.cache, allow_pickle=True)
    all_wids = [str(x) for x in cache["wids"]]
    all_embs = cache["emb"].astype(np.float32)

    distill_wids, distill_idx = [], []
    for i, w in enumerate(all_wids):
        if w in banned_wids:
            continue
        cid = catalog.get(w, {}).get("creatorSteamId") or w
        if cid in banned_creators:
            continue
        distill_wids.append(w)
        distill_idx.append(i)

    print(f"[distill] distilling MobileCLIP2-S0 -> MobileNetV3-Small on {len(distill_wids)} L4D2 images", flush=True)
    student = TinyVisualStudent("mobilenetv3_small_100", out_dim=512)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"[distill] TinyVisualStudent params: {n_params/1e6:.2f}M", flush=True)

    ds = DistillImgDataset(distill_wids, all_embs[distill_idx], Path(args.img_dir), is_train=True)
    dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2)
    opt = torch.optim.AdamW(student.parameters(), lr=1.5e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(dl), eta_min=1e-4)

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        student.train()
        tot_cos = 0.0
        steps = 0
        for px, tgt in dl:
            pred = student(px)
            cos = (pred * tgt).sum(dim=-1).mean()
            loss = 1.0 - cos
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot_cos += float(cos.item())
            steps += 1
        print(f"  [distill ep {ep}/{args.epochs}] mean_cos={tot_cos/max(steps,1):.4f} ({time.time()-t0:.1f}s)", flush=True)

    # Extract student embeddings for labeled samples
    samples, wid_to_idx, _, priors = load_data(Path(args.catalog), Path(args.labels), Path(args.cache))
    train_s, val_s, test_s = make_or_load_creator_split(samples, Path(args.split))

    eval_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    student.eval()
    stu_embs = np.zeros_like(all_embs)
    needed_idx = sorted({s["cache_idx"] for s in (train_s + val_s + test_s)})
    buf, bidx = [], []
    with torch.no_grad():
        for k, ci in enumerate(needed_idx):
            with Image.open(Path(args.img_dir) / f"{all_wids[ci]}.jpg") as im:
                buf.append(eval_tf(im.convert("RGB")))
                bidx.append(ci)
            if len(buf) >= 64 or k == len(needed_idx) - 1:
                feat = student(torch.stack(buf, dim=0)).cpu().numpy()
                stu_embs[bidx] = feat
                buf.clear()
                bidx.clear()

    tr_arr = build_feature_arrays(train_s, stu_embs, priors)
    va_arr = build_feature_arrays(val_s, stu_embs, priors)
    te_arr = build_feature_arrays(test_s, stu_embs, priors)
    synth_tr = generate_synth_red_train_features(student, eval_tf, count=120, seed=1234)
    tr_aug = concat_arrays(tr_arr, synth_tr)

    head, val_sc = train_fusion_head(tr_aug, va_arr, zero_img_emb=False, epochs=50, seed=42)
    m_both, _ = evaluate_model(head, te_arr, mode="both")
    m_img, _ = evaluate_model(head, te_arr, mode="image_only")
    m_stress = eval_synth_red_stress(head, student, eval_tf, Path(args.stress_dir))

    res = {
        "params_m": round(n_params / 1e6, 2),
        "test_both": m_both,
        "test_image_only": m_img,
        "stress_red_text": m_stress,
    }
    print("\n=== Tiny Student (MobileNetV3-Small Distilled) Test Results ===", flush=True)
    print("  Test (Both):", {k: round(v, 4) for k, v in m_both.items()}, flush=True)
    print("  Test (ImgOnly):", {k: round(v, 4) for k, v in m_img.items()}, flush=True)
    print("  Stress Red-Text:", {k: round(v, 4) for k, v in m_stress.items()}, flush=True)

    torch.save({"visual_state": student.state_dict(), "head_state": head.state_dict()}, out_dir / "tiny_student.pt")
    (out_dir / "eval_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
