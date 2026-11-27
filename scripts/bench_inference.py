#!/usr/bin/env python3
"""Benchmark the deployed model with an optional accelerator and a CPU reference."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.l4filter import (
    VISION_PROVIDERS,
    Left4Filter,
    VisionProviderError,
    _create_vision_session,
    _session_options,
)


def benchmark(
    clf: Left4Filter, batches: list[list[bytes]], title: str, rounds: int, concurrency: int,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    def predict(images):
        start = perf_counter()
        scores = clf.predict(title=title, images=images)
        return (perf_counter() - start) * 1000, scores

    first_ms, _ = predict(batches[0])
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(predict, [batches[0]] * concurrency))
        start = perf_counter()
        runs = list(pool.map(predict, batches * rounds))
        elapsed = perf_counter() - start
    latencies, scores = zip(*runs)
    stats = {
        "provider": clf.vision_provider,
        "first_prediction_ms": round(first_ms, 3),
        "p50_ms": round(float(np.median(latencies)), 3),
        "p95_ms": round(float(np.percentile(latencies, 95)), 3),
        "items_per_second": round(len(runs) / elapsed, 3),
        "first_item_scores": scores[0],
    }
    return stats, scores


def profile_vision(model_dir: Path, threads: int, provider: str, images: int, image_size: int) -> dict[str, int]:
    with TemporaryDirectory() as directory:
        options = _session_options(threads)
        options.enable_profiling = True
        options.profile_file_prefix = str(Path(directory) / "vision")
        session = _create_vision_session(model_dir / "vision_encoder.onnx", options, provider)
        pixels = np.zeros((images, 3, image_size, image_size), dtype=np.float32)
        try:
            session.run(["img_emb"], {"pixel_values": pixels})
        finally:
            trace_path = session.end_profiling()
        events = json.loads(Path(trace_path).read_text(encoding="utf-8"))

    counts = Counter()
    for event in events:
        args = event.get("args", {})
        if event.get("cat") != "Node" or args.get("op_name", "").startswith("Memcpy"):
            continue
        if args.get("provider"):
            counts[args["provider"]] += 1
    active = session.get_providers()[0]
    if not counts[active]:
        raise VisionProviderError(f"The vision profile recorded no compute kernels on {active}.")
    return dict(counts)


def compare_scores(
    reference: tuple[dict[str, Any], ...], candidate: tuple[dict[str, Any], ...], config: dict[str, Any],
) -> dict[str, Any]:
    pairs = list(zip(reference, candidate, strict=True))
    max_delta = {
        key: round(max(abs(cpu[key] - other[key]) for cpu, other in pairs), 6)
        for key in reference[0]
    }
    settings = config.get("recommended_thresholds", {})
    thresholds = [
        ("prob", name, settings[name]) for name in ("escalate_to_llm", "direct_sensitive") if name in settings
    ]
    corpse_settings = (config.get("corpse_detection") or {}).get("thresholds", {})
    thresholds.extend((f"{name}_prob", name, threshold) for name, threshold in corpse_settings.items())
    return {
        "max_absolute_score_delta": max_delta,
        "threshold_crossings": {
            name: sum((cpu[key] >= threshold) != (other[key] >= threshold) for cpu, other in pairs)
            for key, name, threshold in thresholds
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "deploy")
    parser.add_argument("--vision-provider", choices=VISION_PROVIDERS, default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--images-per-item", type=int, choices=range(1, 13), default=1)
    parser.add_argument("--title", default="", help="Optional title to score with each image batch.")
    parser.add_argument("images", type=Path, nargs="+")
    args = parser.parse_args()

    if args.threads < 0 or args.concurrency < 1 or args.rounds < 1:
        parser.error("Threads must be nonnegative, and concurrency and rounds must be positive.")
    size = args.images_per_item
    if len(args.images) % size:
        parser.error("The number of image files must be a multiple of --images-per-item.")

    try:
        contents = [path.read_bytes() for path in args.images]
    except OSError as exc:
        parser.error(str(exc))
    batches = [contents[start:start + size] for start in range(0, len(contents), size)]
    report = {
        "onnxruntime": ort.__version__,
        "threads": args.threads,
        "concurrency": args.concurrency,
        "items_per_round": len(batches),
        "images_per_item": size,
        "rounds": args.rounds,
        "results": {},
    }
    predictions = {}
    providers = ("cpu",) if args.vision_provider == "cpu" else (args.vision_provider, "cpu")
    try:
        for provider in providers:
            start = perf_counter()
            clf = Left4Filter(args.model_dir, args.threads, vision_provider=provider, vision_cache_size=0)
            startup_ms = (perf_counter() - start) * 1000
            stats, predictions[provider] = benchmark(clf, batches, args.title, args.rounds, args.concurrency)
            stats["startup_ms"] = round(startup_ms, 3)
            if provider != "cpu":
                stats["vision_kernels_by_provider"] = profile_vision(
                    args.model_dir, args.threads, provider, size, clf.img_size,
                )
            report["results"][provider] = stats
        if args.vision_provider != "cpu":
            report["comparison"] = compare_scores(predictions["cpu"], predictions[args.vision_provider], clf.config)
    except VisionProviderError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
