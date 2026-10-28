#!/usr/bin/env python3
"""Temporary OpenAI-compatible endpoint for Windows -> WSL connectivity checks."""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class ProbeHandler(BaseHTTPRequestHandler):
    server_version = "WSL-OpenAI-Probe/1.0"

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "service": "wsl-openai-probe"})
            return
        if self.path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "qwen3-1.7b-connectivity-probe",
                            "object": "model",
                            "created": 0,
                            "owned_by": "local-probe",
                        }
                    ],
                },
            )
            return
        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "invalid json"}})
            return
        self._send_json(
            200,
            {
                "id": "chatcmpl-connectivity-probe",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get("model", "qwen3-1.7b-connectivity-probe"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "0"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ProbeHandler)
    print(f"LISTENING http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
