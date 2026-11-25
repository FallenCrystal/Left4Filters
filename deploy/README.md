# Left4Filters — Portable Deployment Bundle

Self-contained CPU inference bundle (~23 MB total, ~226 MB peak RSS, ~40 ms multimodal / ~0.5 ms text-only on a 4-core CPU).

## Install

```bash
pip install -r requirements.txt
```

Only 3 lightweight runtime dependencies: `onnxruntime`, `numpy`, `pillow` (no PyTorch / GPU required).

## Quick Usage

### Python SDK

```python
from l4filter import Left4Filter

clf = Left4Filter(".", num_threads=4)

# Scalar calibrated sensitivity probability in [0, 1]
chance = clf.predict_prob(
    title="Mod Title",
    description="Mod Description",
    tags=["Textures"],
    image="preview.jpg",          # path, bytes, PIL.Image, or omit for text-only
    # images=["p1.jpg", "p2.jpg"] # optional multi-image gallery (worst-case max)
)

# Detailed probabilities
scores = clf.predict(title="Mod Title", image="preview.jpg")
# {"prob": 0.912345, "gore_prob": 0.871234, "nsfw_prob": 0.001234}
```

### CLI

```bash
python l4filter.py --prob-only --title "Mod Title" --desc "..." --tags "UI" --image cover.jpg
```

### HTTP Service (stdlib only)

```bash
python server.py --host 127.0.0.1 --port 8090
```

- `GET /health`
- `POST /predict` with JSON `{"title", "description", "tags", "image_base64", "images_base64"}` → `{"prob", "gore_prob", "nsfw_prob"}`

## Recommended Two-Stage Thresholds

- `prob < 0.20` → Auto-pass (safe)
- `0.20 <= prob < 0.72` → Escalate to LLM/VLM judge (~10% of items)
- `prob >= 0.72` → Direct sensitive flag (0.00% safe false-block rate on locked test set)

## License

Code in this directory is MIT-licensed (`../LICENSE`). `vision_encoder.onnx` is fine-tuned from Apple `MobileCLIP2-S0` and subject to the Apple Machine Learning Research Model License (`apple-amlr`); see `../MODEL_LICENSE_NOTICE.md`.
