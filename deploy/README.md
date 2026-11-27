# Left4Filters — Portable Deployment Bundle

Self-contained ONNX inference bundle (~24 MB total). The vision encoder uses CPU by default and can also run on CUDA or, experimentally, WebGPU.

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

- `GET /health` reports the model and `vision_provider`.
- `POST /predict` accepts `title`, `description`, `tags`, `texts`, `image_base64`, `images_base64` and `explain` as optional JSON fields. Use an array of strings for `tags`, `texts` and `images_base64`.

## Vision providers

For CUDA, use a dedicated virtual environment with a compatible NVIDIA driver. `requirements-cuda.txt` installs the ONNX Runtime GPU distribution and its CUDA and cuDNN runtime wheels. Left4Filters preloads these libraries when CUDA is selected. The GPU distribution includes the CPU provider for comparisons. Windows also requires the Microsoft Visual C++ runtime. [CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) · [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)

```bash
python -m pip install -r requirements-cuda.txt
python l4filter.py --vision-provider cuda --image cover.jpg
python server.py --vision-provider cuda
```

PyPI builds of ONNX Runtime GPU 1.27 and later target CUDA 13.0 or newer. For a driver that supports CUDA 12.8 but lacks CUDA 13 support, install with `python -m pip install -r requirements-cuda.txt "onnxruntime-gpu<1.27"`. Install the CUDA requirements in place of the CPU `requirements.txt` in the GPU environment. [CUDA and cuDNN version table](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)

Use `vision_provider="cuda"` with the Python SDK. The SDK, CLI, and HTTP server also read `LEFT4FILTERS_VISION_PROVIDER` on initialization. An explicit argument takes precedence. The fusion and corpse heads use CPU. CUDA runs the shipped vision graph with TF32 disabled for the initial score comparison. An explicit CUDA request fails with an error if the provider cannot initialize.

The Left4Addons managed runner inherits `LEFT4FILTERS_VISION_PROVIDER`. Launch it with a GPU-enabled Python environment when testing CUDA. The desktop installer's default `requirements.txt` selects the CPU distribution.

`VISION_PROVIDERS` and `_create_vision_session` in `l4filter.py` handle provider registration and session setup.

### WebGPU (experimental)

WebGPU runs the vision encoder on AMD, Intel, NVIDIA and Apple GPUs through D3D12 on Windows, Vulkan on Linux and Metal on macOS. `requirements-webgpu.txt` adds the `onnxruntime-ep-webgpu` plugin to the standard CPU distribution, so it can replace `requirements.txt` without a separate ONNX Runtime build. Linux requires the Vulkan loader (`libvulkan.so.1`) and a Vulkan driver. The plugin is pinned to its 0.6 series because its API is still pre-1.0. [WebGPU plugin](https://pypi.org/project/onnxruntime-ep-webgpu/)

```bash
python -m pip install -r requirements-webgpu.txt
python l4filter.py --vision-provider webgpu --image cover.jpg
python server.py --vision-provider webgpu
```

Left4Filters registers the plugin once per process and binds the vision session to the first WebGPU adapter. The `onnxruntime-webgpu` distribution, which builds the provider into ONNX Runtime, also works. The fusion and corpse heads stay on CPU, as do shape computations and a few small scalar operations in the vision graph. ONNX Runtime reports those placements as warnings through its process-wide logger, so Left4Filters raises that logger to errors while it creates the WebGPU session and then restores the default warning level. Concurrent requests can share the session. On a software Vulkan adapter, WebGPU and CPU scores agreed to within 0.00004 with no threshold crossings. GPU speed has not been measured yet, so compare `scripts/bench_inference.py --vision-provider webgpu` against CPU on the target machine before switching a deployment. The first prediction includes shader compilation, which favors long-running services over one-off CLI calls.

On Windows, the plugin bundles a preview D3D12 Agility SDK that activates only with Developer Mode enabled. Test on a machine with Developer Mode turned off to confirm the behavior that end users get.

DirectML is not supported. Microsoft has moved it to sustained engineering, and its `onnxruntime-directml` distribution is Windows-only, lags the main ONNX Runtime releases and replaces the `onnxruntime` package. WebGPU covers the same Windows GPUs through D3D12. Intel OpenVINO GPU and AMD MIGraphX remain candidates for vendor-specific providers. [DirectML status](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html) · [OpenVINO](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html) · [MIGraphX](https://onnxruntime.ai/docs/execution-providers/MIGraphX-ExecutionProvider.html)

## Inference timings

Set `LEFT4FILTERS_TIMING_LOG` before starting Python. Use an absolute file path to append to a log, or `-` to write to standard error. Logs include model startup, image preprocessing, vision encoding, cache hits, and total SDK prediction time. The Left4Addons fast path calls the vision session directly, so its log includes model startup and vision encoding.

```bash
LEFT4FILTERS_TIMING_LOG="$PWD/inference.log" python l4filter.py --vision-provider cuda --image cover.jpg
```

Run the benchmark from the repository root with the same images and runtime for CPU and CUDA. This example groups the images in pairs. Omit `--images-per-item 2` to measure single-image items.

```bash
python scripts/bench_inference.py --vision-provider cuda --threads 1 --concurrency 4 \
  --images-per-item 2 cover1.jpg gallery1.jpg cover2.jpg gallery2.jpg
```

The benchmark disables the feature cache and reports initialization, first prediction, warm P50/P95, throughput, score differences, and threshold crossings. It profiles the vision graph separately and checks that compute kernels ran on the requested accelerator. The timings cover local SDK prediction. Use representative Workshop images and leave timing logs off for latency comparisons.

Each provider reports the predicted scores for the first item in `first_item_scores`. `max_absolute_score_delta` measures differences between providers across all items and rounds. `threshold_crossings` counts predictions for which the providers landed on different sides of a threshold.

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
