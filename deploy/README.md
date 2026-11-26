# Left4Filters — Portable Deployment Bundle

Self-contained ONNX CPU inference bundle (~24 MB total).

## Install

```bash
pip install -r requirements.txt
```

The runtime uses `onnxruntime`, `numpy` and `pillow`.

## Quick Usage

### Python SDK

```python
from l4filter import Left4Filter

clf = Left4Filter(".", num_threads=4)

# Scalar sensitivity routing score in [0, 1]
chance = clf.predict_prob(
    title="Mod Title",
    description="Mod Description",
    tags=["Textures"],
    image="preview.jpg",          # path, bytes, PIL.Image, or omit for text-only
    # images=["p1.jpg", "p2.jpg"] # optional multi-image gallery (worst-case max)
)

# Category scores and evidence from the suspicious components
scores = clf.predict(
    title="Mod Title",
    texts=["Extra notes", "[NSFW] Uncensored version"],
    images=["cover.jpg", "gallery.jpg"],
    explain=True,
)
```

### CLI

```bash
python l4filter.py --title "Mod Title" --image cover.jpg gallery.jpg \
  --text "Extra notes" --text "[NSFW] Uncensored version" --explain

python l4filter.py --prob-only --title "Mod Title" --image cover.jpg
```

### HTTP Service (stdlib only)

```bash
python server.py --host 127.0.0.1 --port 8090
```

- `GET /health`
- `POST /predict` accepts `title`, `description`, `tags`, `texts`, `image_base64`, `images_base64` and `explain` as optional JSON fields. Use an array of strings for `tags`, `texts` and `images_base64`.

## Reusing image features

A long-lived SDK instance or HTTP server caches the embeddings and visual priors of up to 256 image batches. The cache matches the encoded image contents in batch order (up to 12 images per batch) and reuses the features with the text from each request. Batches containing PIL objects use the standard inference path.

Set `vision_cache_size` on `Left4Filter` or pass `--vision-cache-size` to `server.py` to change the capacity. A value of `0` disables the cache.

## Scores and evidence

Each image is scored with the combined title, description and tags, and separately with each entry in `texts`. The response takes the maximum for each score across the request. Requests can also contain only images or only text.

The response contains `prob`, `gore_prob`, `nsfw_prob`, `corpse_prob` and `corpse_pile_prob`. The last two come from the images. `corpse_prob` covers any visible body, including piles, and `corpse_pile_prob` covers heaped or overlapping bodies. They are both zero for text-only requests.

Set `explain` to `true` for `evidence`, ordered by descending score. Each entry identifies an `image` or `text` by its `field` and, for arrays, a zero-based `index` in `images`, `texts` or `tags`. In HTTP responses, `image` refers to `image_base64` and `images` to `images_base64`. Scalar fields such as `title`, `description` and `image` have no index. Text entries include an excerpt and may include matching terms. An empty array means the request had no components meeting the evidence thresholds.

Each reason gives the `category`, `score` and `support`. `standalone` identifies a component scored in isolation. `text_context` uses the other workshop text fields. `joint` uses both image and text. The last two also include `delta`, the score decrease when the component is removed. For example, one entry from `evidence` can look like this.

```json
{
  "type": "text",
  "field": "texts",
  "index": 1,
  "score": 0.998482,
  "text": "[NSFW] [R18] Uncensored bikini version",
  "reasons": [
    {
      "category": "nsfw",
      "score": 0.998482,
      "support": "standalone",
      "matches": ["NSFW", "R18", "Uncensored"]
    }
  ]
}
```

## Recommended Two-Stage Thresholds

- `prob < 0.20` → Auto-pass (safe)
- `0.20 <= prob < 0.72` → Escalate to LLM/VLM judge
- `prob >= 0.72` → Direct sensitive flag

An image that reaches the corpse or corpse-pile threshold in `model_config.json` raises the routing score to at least `0.20`. Direct flags require a fusion score of `0.72` or higher.

## License

Code in this directory is MIT-licensed (`../LICENSE`). `vision_encoder.onnx` is fine-tuned from Apple `MobileCLIP2-S0` and subject to the Apple Machine Learning Research Model License (`apple-amlr`); see `../MODEL_LICENSE_NOTICE.md`.
