#!/usr/bin/env python3
"""Train and export a visual classifier for corpses and corpse piles."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
fusion = importlib.import_module("scripts.07_retrain_fusion")
CATEGORIES = ("corpse", "corpse_pile")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def split_by_creator(rows, legacy, locked, previous):
    val_creators = set(map(str, locked["val_creators"]))
    test_creators = set(map(str, locked["test_creators"]))
    train_creators = set(legacy.values()) - val_creators - test_creators
    assignments = {creator: name for name, creators in (
        ("train", train_creators), ("val", val_creators), ("test", test_creators),
    ) for creator in creators}
    creators_by_id = {r["id"]: r["creator"] for r in rows}
    for name, ids in previous.items():
        for sample_id in ids:
            creator = creators_by_id[sample_id]
            if assignments.setdefault(creator, name) != name:
                raise ValueError(f"Conflicting creator assignment for {sample_id}")
    splits = {name: [r for r in rows if assignments.get(r["creator"]) == name]
              for name in ("train", "val", "test")}
    unseen = [r for r in rows if r["creator"] not in assignments]
    labels = [r["label"] for r in unseen]
    groups = [r["creator"] for r in unseen]
    if len(set(groups)) >= 5 and Counter(labels).most_common(1)[0][1] >= 5:
        fold_split = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)
        folds = [list(test) for _, test in fold_split.split(np.zeros(len(unseen)), labels, groups)]
        # Prefer held-out folds with coverage of all three classes.
        folds.sort(key=lambda fold: min(Counter(labels[i] for i in fold).get(c, 0) for c in range(3)), reverse=True)
        for name, fold in zip(("test", "val", "train", "train", "train"), folds):
            splits[name].extend(unseen[i] for i in fold)
    else:
        splits["train"].extend(unseen)
    for name, part in splits.items():
        counts = Counter(r["label"] for r in part)
        if len(counts) != 3:
            raise ValueError(f"The {name} split needs annotations for each of 0, 1, 2")
        print(f"[{name}] n={len(part)} labels={dict(sorted(counts.items()))}", flush=True)
    return splits


class CorpseHead(nn.Module):
    def __init__(self, classifier, temperature: float, anchors: np.ndarray, gamma: float) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("anchors", torch.from_numpy(anchors.T.copy()))
        self.register_buffer("weight", torch.from_numpy(classifier.coef_.T.astype(np.float32)))
        self.register_buffer("bias", torch.from_numpy(classifier.intercept_.astype(np.float32)))
        self.temperature = temperature

    def forward(self, img_emb: torch.Tensor) -> torch.Tensor:
        features = torch.exp(2 * self.gamma * (img_emb @ self.anchors - 1)) if self.gamma else img_emb
        return corpse_probs(features @ self.weight + self.bias, self.temperature)


def corpse_probs(logits, temperature):
    probs = torch.softmax(logits / temperature, dim=1)
    return torch.stack((probs[:, 1] + probs[:, 2], probs[:, 2]), dim=1)


class DenseCorpseHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(dim, 64), nn.LayerNorm(64), nn.GELU(),
                                        nn.Dropout(0.25), nn.Linear(64, 3))
        self.temperature = 1.0

    def forward(self, img_emb):
        return corpse_probs(self.classifier(img_emb), self.temperature)


def fit_temperature(logits, labels):
    def log_loss(temp):
        z = logits / temp
        return (logsumexp(z, axis=1) - z[np.arange(len(z)), labels]).mean() + .01 * (temp - 1) ** 2

    return float(minimize_scalar(log_loss, bounds=(.35, 3), method="bounded").x)


def fit_head(train_x, train_y, val_x, val_y, gamma, regularization, class_weight):
    def features(x):
        return np.exp(2 * gamma * (x @ train_x.T - 1)) if gamma else x

    classifier = LogisticRegression(C=regularization, class_weight=class_weight, max_iter=1000)
    classifier.fit(features(train_x), train_y)
    logits = classifier.decision_function(features(val_x))

    temp = fit_temperature(logits, val_y)
    probs = softmax(logits / temp, axis=1)
    scores = np.stack((1 - probs[:, 0], probs[:, 2]), axis=1)
    return CorpseHead(classifier, temp, train_x, gamma), scores


def train_dense(train_x, train_y, val_x, val_y, seed, balance):
    torch.manual_seed(seed)
    model = DenseCorpseHead(train_x.shape[1])
    x = torch.from_numpy(train_x)
    y = torch.from_numpy(train_y)
    val = torch.from_numpy(val_x)
    weights = (len(y) / (3 * torch.bincount(y).float())) ** balance
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.03)
    best = None
    best_score = -np.inf
    for epoch in range(1, 61):
        model.train()
        for idx in torch.randperm(len(y)).split(64):
            feat = F.normalize(x[idx] + 0.008 * torch.randn_like(x[idx]), dim=1)
            loss = F.cross_entropy(model.classifier(feat), y[idx], weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.inference_mode():
                logits = model.classifier(val)
                model.temperature = fit_temperature(logits.numpy(), val_y)
                probs = corpse_probs(logits, model.temperature).numpy()
            score, _ = validation_score(val_y, probs)
            if score > best_score:
                best_score = score
                best = (copy.deepcopy(model), probs)
    return best


def targets(labels):
    return np.stack((labels > 0, labels == 2), axis=1)


def choose_threshold(y, probs) -> float:
    # Prefer recall for review while limiting each category's validation false positives.
    eligible = []
    for threshold in np.unique(probs):
        threshold = float(np.floor(threshold * 10_000) / 10_000)
        positive = probs >= threshold
        tp = int((positive & y).sum())
        fp = int((positive & ~y).sum())
        if tp > 0 and fp / (~y).sum() <= .10:
            f_score = 3.25 * tp / (2.25 * int(y.sum()) + tp + fp)
            eligible.append((f_score, tp, -fp, threshold))
    return max(eligible)[3] if eligible else 1.0


def evaluate(labels, predictions, thresholds):
    result = {}
    for i, category in enumerate(CATEGORIES):
        y, scores = targets(labels)[:, i], predictions[:, i]
        positive = scores >= thresholds[i]
        tp = int((positive & y).sum())
        fp = int((positive & ~y).sum())
        result[category] = {
            "auc": float(roc_auc_score(y, scores)) if y.any() and (~y).any() else None,
            "average_precision": float(average_precision_score(y, scores)) if y.any() else None,
            "positive": int(y.sum()), "negative": int((~y).sum()),
            "true_positive": tp, "false_positive": fp,
        }
    return result


def validation_score(y, scores):
    labels = targets(y)
    thresholds = [choose_threshold(labels[:, i], scores[:, i]) for i in range(2)]
    result = evaluate(y, scores, thresholds)
    score = sum(
        m["average_precision"] + .45 * m["true_positive"] / m["positive"]
        - .35 * m["false_positive"] / m["negative"]
        for m in result.values()
    )
    return score, thresholds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", type=Path, required=True,
                    help="JSONL with id, creator, local path and label (0 none, 1 body, 2 pile, -1 unsure)")
    ap.add_argument("--base-dir", type=Path, default=Path("deploy"))
    ap.add_argument("--catalog", type=Path, default=Path("data/raw/catalog_items.jsonl"))
    ap.add_argument("--legacy-labels", type=Path, default=Path("data/annotations/merged_final.jsonl"))
    ap.add_argument("--legacy-split", type=Path, default=Path("data/annotations/split_manifest.json"))
    ap.add_argument("--legacy-image-dir", type=Path, default=Path("data/images"))
    ap.add_argument("--legacy-features", type=Path,
                    help="Optional fusion embedding cache computed with this base vision encoder")
    ap.add_argument("--initial-split", type=Path, help="Preserve creator assignments from an earlier run")
    ap.add_argument("--out-dir", type=Path, default=Path("exports/retrained"))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    catalog = {str(r["workshopId"]): str(r["creatorSteamId"] or r["workshopId"]) for r in read_jsonl(args.catalog)}
    old_labels = {str(r["wid"]): r for r in read_jsonl(args.legacy_labels) if str(r["wid"]) in catalog}
    old_creators = {wid: catalog[wid] for wid in old_labels}
    locked = json.loads(args.legacy_split.read_text())
    annotated = read_jsonl(args.annotations)
    if len({r["id"] for r in annotated}) != len(annotated):
        raise ValueError("Duplicate annotation IDs")
    rows = [r for r in annotated if r["label"] in (0, 1, 2)]
    previous = json.loads(args.initial_split.read_text()) if args.initial_split else {}
    splits = split_by_creator(rows, old_creators, locked, previous)
    held_out = {r["creator"] for part in ("val", "test") for r in splits[part]}
    held_out.update(map(str, locked["val_creators"] + locked["test_creators"]))

    # Existing red UI and clothed anime negatives broaden the training distribution.
    annotated_ids = {r["id"] for r in annotated}
    pool = [wid for wid, r in old_labels.items()
            if wid not in annotated_ids and old_creators[wid] not in held_out
            and r["ig"] == r["in"] == 0 and (r["rt"] or r["at"])
            and (args.legacy_image_dir / f"{wid}.jpg").is_file()]
    rng = np.random.default_rng(2026)
    weak = [
        {"id": wid, "path": args.legacy_image_dir / f"{wid}.jpg", "label": 0}
        for wid in rng.permutation(sorted(pool))[:240]
    ]
    print(f"[train] {len(weak)} additional red UI / clothed anime negatives", flush=True)
    cached = {}
    if args.legacy_features:
        with np.load(args.legacy_features, allow_pickle=True) as data:
            cached = dict(zip(map(str, data["wids"]), data["emb"]))
    missing = [r for r in rows + weak if r["id"] not in cached]
    if missing:
        size = json.loads((args.base_dir / "model_config.json").read_text())["image_size"]
        vision = fusion.session(args.base_dir / "vision_encoder.onnx", args.threads)
        images = fusion.load_images(r["path"] for r in missing)
        emb, _ = fusion.encode_images(vision, images, size)
        cached.update(zip((r["id"] for r in missing), emb))

    def arrays(records):
        return np.stack([cached[r["id"]] for r in records]), np.array([r["label"] for r in records])

    train_x, train_y = arrays(splits["train"] + weak)
    val_x, val_y = arrays(splits["val"])
    best, best_thresholds, best_score = None, None, -np.inf

    def consider(head, scores, settings):
        nonlocal best, best_thresholds, best_score, recipe
        score, thresholds = validation_score(val_y, scores)
        if score > best_score:
            best, best_thresholds, best_score, recipe = head, thresholds, score, settings
            print(f"[val] {recipe} score={score:.4f}, thresholds={thresholds}", flush=True)

    recipe = {}
    for gamma in (0.0, 1.0, 3.0):
        for regularization in (3.0, 20.0, 100.0):
            for weight in (None, "balanced"):
                head, scores = fit_head(train_x, train_y, val_x, val_y, gamma, regularization, weight)
                consider(head, scores, {"gamma": gamma, "C": regularization, "class_weight": weight})
    for seed, balance in ((42, 0.5), (101, 1.0), (2026, 0.0)):
        head, scores = train_dense(train_x, train_y, val_x, val_y, seed, balance)
        consider(head, scores, {"model": "mlp", "seed": seed, "class_balance": balance})

    path = args.out_dir / "corpse_head.onnx"
    torch.onnx.export(
        best.eval(), (torch.from_numpy(val_x[:1]),), str(path),
        input_names=["img_emb"], output_names=["corpse_probs"],
        dynamic_axes={"img_emb": {0: "batch"}, "corpse_probs": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    onnx.checker.check_model(str(path))
    exported = fusion.session(path, args.threads)
    test_x, test_y = arrays(splits["test"])
    test_probs = exported.run(None, {"img_emb": test_x})[0]
    results = {
        "recipe": recipe, "thresholds": dict(zip(CATEGORIES, best_thresholds)),
        "annotation_counts": dict(Counter(r["label"] for r in annotated)),
        "train": len(train_y), "train_existing_negatives": len(weak),
        "validation": evaluate(val_y, exported.run(None, {"img_emb": val_x})[0], best_thresholds),
        "test": evaluate(test_y, test_probs, best_thresholds),
    }
    if previous:
        prior_ids = set(previous["test"])
        prior = np.array([r["id"] in prior_ids for r in splits["test"]])
        for mask, part in ((prior, "test_original"), (~prior, "test_added")):
            if mask.any():
                results[part] = evaluate(test_y[mask], test_probs[mask], best_thresholds)
    (args.out_dir / "corpse_results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.out_dir / "corpse_split.json").write_text(json.dumps({
        part: [r["id"] for r in entries] for part, entries in splits.items()
    }, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
