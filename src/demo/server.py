"""Loopback OpenAI-compatible demo service; launch via python -m src.demo.server."""
import argparse
from datetime import datetime, timezone, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import traceback
import uuid

from src.common.io import digest, load, save
from src.demo.backend import Backend, MODELS


class DemoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("configs/paths.local.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--preload", choices=list(MODELS))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    backend = Backend(root, load(args.paths))
    log_lock = threading.Lock()
    status = dict(status="starting", run_id=args.output.name, port=args.port, bind="127.0.0.1", training=False,
                  inference=True, scope="interactive_demo_not_formal_test", source_sha256={str(p.relative_to(root)):digest(p) for p in [Path(__file__), root / "src/demo/backend.py"]})

    def record(value):
        with log_lock, (args.output / "requests.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            print(fmt % values, flush=True)

        def send_json(self, value, code=200):
            raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path in ("/v1/models", "/models"):
                self.send_json(dict(object="list", data=[dict(id=name, object="model", created=0, owned_by="local-course-project") for name in MODELS]))
            elif self.path == "/health":
                self.send_json(dict(status="ready", loaded=backend.loaded, run_id=args.output.name))
            elif self.path == "/demo/cases":
                self.send_json(load(root / "configs/demo/cases.json"))
            elif self.path == "/":
                raw = (root / "src/demo/index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            else:
                self.send_json(dict(error="not found"), 404)

        def do_POST(self):
            request_id = "demo-" + uuid.uuid4().hex
            if self.path not in ("/v1/chat/completions", "/chat/completions"):
                self.send_json(dict(error="not found"), 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1024 * 1024:
                    raise ValueError("请求大小必须在1MB以内")
                body = json.loads(self.rfile.read(length))
                result = backend.predict(body.get("model"), body.get("messages"))
                result.update(request_id=request_id, created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    client_generation_options={k:body[k] for k in ("temperature", "top_p", "max_tokens", "max_completion_tokens", "stream") if k in body},
                    effective_generation_options=dict(do_sample=False, max_new_tokens=8, enable_thinking=False),
                    client_options_override_frozen_protocol=False)
                record(result)
                created = int(time.time())
                if body.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for delta, finish in [(dict(role="assistant", content=result["raw_output"]), None), ({}, "stop")]:
                        chunk = dict(id=request_id, object="chat.completion.chunk", created=created, model=result["model"], choices=[dict(index=0, delta=delta, finish_reason=finish)])
                        self.wfile.write(("data: " + json.dumps(chunk, ensure_ascii=False) + "\n\n").encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    self.close_connection = True
                else:
                    usage = dict(prompt_tokens=result["input_tokens"] or 0, completion_tokens=result["output_tokens"] or 0)
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                    self.send_json(dict(id=request_id, object="chat.completion", created=created, model=result["model"],
                        choices=[dict(index=0, message=dict(role="assistant", content=result["raw_output"]), finish_reason="stop")], usage=usage, demo=result))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                record(dict(request_id=request_id, status="invalid_request", error=str(exc)))
                self.send_json(dict(error=dict(message=str(exc), type="invalid_request_error")), 400)
            except Exception as exc:
                failure = dict(request_id=request_id, status="oom" if "out of memory" in str(exc).lower() else "runtime_error", traceback=traceback.format_exc())
                record(failure)
                self.send_json(dict(error=dict(message="本地模型运行失败，详见演示请求日志", type=failure["status"])), 500)

    try:
        save(args.output / "status.json", status)
        if args.preload:
            with backend.lock:
                backend.ensure(MODELS[args.preload])
        with DemoHTTPServer(("127.0.0.1", args.port), Handler) as server:
            status.update(status="ready", loaded=backend.loaded)
            save(args.output / "status.json", status)
            print(f"DEMO READY http://127.0.0.1:{args.port} ; OpenAI /v1 ; output {args.output}", flush=True)
            server.serve_forever()
    except KeyboardInterrupt:
        status["status"] = "stopped"
    except Exception:
        status.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        backend.unload()
        if status["status"] == "ready":
            status["status"] = "stopped"
        save(args.output / "status.json", status)


if __name__ == "__main__":
    main()
