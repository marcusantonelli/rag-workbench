"""Carry Service calls over stdio (inside the desktop app) or HTTP (browser dev mode).

stdio protocol, one JSON object per line:
    -> {"id": 7, "method": "kb.search", "params": {...}}
    <- {"id": 7, "result": ...}            or {"id": 7, "error": {"code", "message", "hint"}}
    <- {"event": "job", "data": {...}}     (unsolicited: jobs, runtime, kb changes)
The first line the engine writes is {"event": "ready", "data": <app.info>}.

HTTP (127.0.0.1 only): POST /rpc {"method", "params"} -> {"result"} | {"error"};
GET /events streams the same events as Server-Sent Events.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from rag.app.service import Service, ServiceError


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False, allow_nan=False)


def _error(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, ServiceError):
        return exc.to_dict()
    return {"code": "error", "message": f"{type(exc).__name__}: {exc}", "hint": ""}


def serve_stdio() -> int:
    # Keep the protocol stream clean: anything else that prints (libraries, C code) goes to stderr.
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1, newline="\n")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    write_lock = threading.Lock()

    def send(obj: dict[str, Any]) -> None:
        try:
            line = _dumps(obj)
        except ValueError:
            line = json.dumps(json.loads(json.dumps(obj, default=str).replace("NaN", "null")))
        with write_lock:
            proto.write(line + "\n")
            proto.flush()

    service = Service(emit=lambda name, data: send({"event": name, "data": data}))
    pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="rpc")

    def handle(msg: dict[str, Any]) -> None:
        rid = msg.get("id")
        try:
            result = service.call(str(msg.get("method")), msg.get("params") or {})
            send({"id": rid, "result": result})
        except BaseException as exc:  # noqa: BLE001
            send({"id": rid, "error": _error(exc)})

    send({"event": "ready", "data": service.app_info()})
    service.startup()
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                send({"event": "log", "data": {"level": "error", "message": "bad request line"}})
                continue
            if msg.get("method") == "app.shutdown":
                send({"id": msg.get("id"), "result": {"ok": True}})
                break
            pool.submit(handle, msg)
    finally:
        service.shutdown()
        pool.shutdown(wait=False, cancel_futures=True)
    return 0


class _Hub:
    """Fan-out of events to every open /events stream."""

    def __init__(self) -> None:
        self.clients: list[queue.Queue] = []
        self.lock = threading.Lock()

    def publish(self, name: str, data: dict[str, Any]) -> None:
        payload = _dumps({"event": name, "data": data})
        with self.lock:
            for q in self.clients:
                q.put(payload)

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)


def serve_http(port: int = 8765, host: str = "127.0.0.1") -> int:
    hub = _Hub()
    service = Service(emit=hub.publish)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            pass

        def _cors(self) -> None:
            origin = self.headers.get("Origin") or ""
            if origin.startswith(("http://localhost:", "http://127.0.0.1:")):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Headers", "content-type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/rpc":
                self.send_error(404)
                return
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                msg = json.loads(body or b"{}")
                out = {"result": service.call(str(msg.get("method")), msg.get("params") or {})}
            except BaseException as exc:  # noqa: BLE001
                out = {"error": _error(exc)}
            data = _dumps(out).encode("utf-8")
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                data = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self.path != "/events":
                self.send_error(404)
                return
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = hub.subscribe()
            try:
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        payload = q.get(timeout=15)
                        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                hub.unsubscribe(q)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    service.startup()
    print(f"RAG Workbench engine on http://{host}:{port}  (Ctrl+C to stop)", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.shutdown()
        server.server_close()
    return 0
