#!/usr/bin/env python3
"""
Zero-framework portable HTTP microservice for Left4FiltersModel.
Uses Python stdlib http.server + l4filter.py (no FastAPI/uvicorn required).

POST /predict
  JSON body:
    {
      "title": "...",
      "description": "...",
      "tags": ["Textures"],
      "texts": ["..."],               // optional additional texts
      "explain": true,                // include component evidence
      "image_base64": "...",           // optional single base64 image
      "images_base64": ["...", "..."]  // optional list of base64 images
    }
  Response:
    {"prob": 0.123456, "gore_prob": 0.081234, "nsfw_prob": 0.021000,
     "corpse_prob": 0.02, "corpse_pile_prob": 0.01, "evidence": []}
"""
from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from l4filter import (
    DEFAULT_VISION_CACHE_SIZE,
    VISION_PROVIDER_ENV,
    VISION_PROVIDERS,
    Left4Filter,
    VisionProviderError,
)


def run_server(
    host: str,
    port: int,
    model_dir: str,
    threads: int,
    *,
    vision_cache_size: int = DEFAULT_VISION_CACHE_SIZE,
    vision_provider: str | None = None,
) -> None:
    clf = Left4Filter(
        model_dir, num_threads=threads, vision_cache_size=vision_cache_size, vision_provider=vision_provider,
    )

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {
                    "status": "ok",
                    "model": clf.config.get("model_name", "Left4FiltersModel"),
                    "vision_provider": clf.vision_provider,
                })
            else:
                self._send(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path != "/predict":
                self._send(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                req = json.loads(raw.decode("utf-8"))
                image = req.get("image_base64")
                texts = req.get("texts")
                explain = req.get("explain", False)
                if texts is not None and (
                    not isinstance(texts, list) or any(not isinstance(value, str) for value in texts)
                ):
                    raise ValueError("texts must be an array of strings")
                if not isinstance(explain, bool):
                    raise ValueError("explain must be a boolean")
                res = clf.predict(
                    title=req.get("title") or "",
                    description=req.get("description") or "",
                    tags=req.get("tags") or None,
                    image=base64.b64decode(image) if image else None,
                    images=[base64.b64decode(value) for value in (req.get("images_base64") or [])],
                    texts=texts,
                    explain=explain,
                )
                self._send(200, res)
            except Exception as e:
                self._send(400, {"error": str(e)})

        def log_message(self, fmt: str, *args) -> None:
            pass

    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"Left4FiltersModel HTTP server listening on http://{host}:{port} (POST /predict, GET /health)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--model-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--vision-provider", choices=VISION_PROVIDERS,
        help=f"Vision encoder provider. Defaults to {VISION_PROVIDER_ENV} or cpu.",
    )
    ap.add_argument(
        "--vision-cache-size", type=int, default=DEFAULT_VISION_CACHE_SIZE,
        help="Maximum cached image batches per process. Defaults to %(default)s. Use 0 to disable caching.",
    )
    args = ap.parse_args()
    try:
        run_server(
            args.host, args.port, args.model_dir, args.threads,
            vision_cache_size=args.vision_cache_size, vision_provider=args.vision_provider,
        )
    except VisionProviderError as exc:
        ap.error(str(exc))
