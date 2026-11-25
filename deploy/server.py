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
      "image_base64": "...",           // optional single base64 image
      "images_base64": ["...", "..."]  // optional list of base64 images
    }
  Response:
    {"prob": 0.123456, "gore_prob": 0.081234, "nsfw_prob": 0.021000}
"""
from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from l4filter import Left4Filter


def run_server(host: str, port: int, model_dir: str, threads: int) -> None:
    clf = Left4Filter(model_dir, num_threads=threads)

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
                self._send(200, {"status": "ok", "model": clf.config.get("model_name", "Left4FiltersModel")})
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
                imgs = []
                if req.get("image_base64"):
                    imgs.append(base64.b64decode(req["image_base64"]))
                for b64 in req.get("images_base64") or []:
                    imgs.append(base64.b64decode(b64))
                res = clf.predict(
                    title=req.get("title") or "",
                    description=req.get("description") or "",
                    tags=req.get("tags") or None,
                    images=imgs or None,
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
    args = ap.parse_args()
    run_server(args.host, args.port, args.model_dir, args.threads)
