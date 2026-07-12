#!/usr/bin/env python3
"""HTTP MJPEG server that applies a CUDA (CuPy) image filter to a snapshot URL.

Mirrors packages/blacknode-vision/scripts/cv2_color_stream_server.py's shape
(poll an upstream --source-url, process each frame, re-serve as its own
/stream.mjpg) but the "process" step is a real GPU filter instead of CV2
color tracking. This is what makes CUDAImageFilterStream a genuine live
video pipeline instead of the graph engine's slower cook-in-a-loop path:
this loop never touches the graph at all, it only fetches, filters, and
re-serves -- the same reason the ROS2/CV2 stream servers exist as dedicated
processes rather than graph nodes.

Self-contained on purpose (no import from nodes/cuda.py): subprocess helpers
in this codebase don't depend on Blacknode's package loader having run in
their own process, so the filter math is duplicated here rather than
imported.
"""
from __future__ import annotations

import argparse
import json
import signal
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np

IMAGE_FILTERS = ["grayscale", "invert", "brighten", "threshold", "gaussian_blur", "sharpen", "sobel_edges"]
CONFIG_FIELDS = {
    "source_url",
    "op",
    "amount",
    "max_fps",
    "max_width",
    "jpeg_quality",
    "source_timeout",
}


def wrap_text(text: str, width: int) -> list[str]:
    words = str(text or "").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def status_jpeg(report: str, *, ok: bool) -> bytes:
    canvas = np.full((540, 960, 3), (15, 23, 42), dtype=np.uint8)
    accent = (34, 197, 94) if ok else (68, 68, 239)
    cv2.rectangle(canvas, (0, 0), (960, 72), accent, -1)
    cv2.putText(canvas, "CUDA Image Filter Stream", (32, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "STATUS", (34, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 2, cv2.LINE_AA)
    status = "running" if ok else "not producing frames"
    cv2.putText(canvas, status, (34, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (229, 237, 247), 2, cv2.LINE_AA)
    cv2.putText(canvas, "REPORT", (34, 226), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 2, cv2.LINE_AA)
    y = 264
    for line in wrap_text(report, 78)[:8]:
        cv2.putText(canvas, line, (34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (229, 237, 247), 1, cv2.LINE_AA)
        y += 34
    ok_encoded, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok_encoded:
        return b""
    return encoded.tobytes()


def _img_lum(cp: Any, g: Any) -> Any:
    return 0.299 * g[:, :, 0] + 0.587 * g[:, :, 1] + 0.114 * g[:, :, 2]


def _img_blur(cp: Any, g: Any) -> Any:
    p = cp.pad(g, ((1, 1), (1, 1), (0, 0)), mode="edge")
    h, w = g.shape[:2]
    out = cp.zeros_like(g)
    for di, row in enumerate(((1, 2, 1), (2, 4, 2), (1, 2, 1))):
        for dj, wt in enumerate(row):
            out = out + wt * p[di:di + h, dj:dj + w, :]
    return out / 16.0


def apply_image_filter(cp: Any, op: str, g: Any, amount: float) -> Any:
    """Same math as CUDAImageFilter's node (packages/blacknode-cuda/nodes/cuda.py)."""
    if op == "grayscale":
        lum = _img_lum(cp, g)
        return cp.stack([lum, lum, lum], axis=-1)
    if op == "invert":
        return 1.0 - g
    if op == "brighten":
        return cp.clip(g * amount, 0.0, 1.0)
    if op == "threshold":
        cut = amount if 0.0 < amount < 1.0 else 0.5
        m = (_img_lum(cp, g) > cut).astype(g.dtype)
        return cp.stack([m, m, m], axis=-1)
    if op == "gaussian_blur":
        return _img_blur(cp, g)
    if op == "sharpen":
        return cp.clip(g + amount * (g - _img_blur(cp, g)), 0.0, 1.0)
    if op == "sobel_edges":
        lum = _img_lum(cp, g)
        p = cp.pad(lum, 1, mode="edge")
        h, w = lum.shape

        def at(di: int, dj: int) -> Any:
            return p[di:di + h, dj:dj + w]

        gx = (at(0, 2) + 2 * at(1, 2) + at(2, 2)) - (at(0, 0) + 2 * at(1, 0) + at(2, 0))
        gy = (at(2, 0) + 2 * at(2, 1) + at(2, 2)) - (at(0, 0) + 2 * at(0, 1) + at(0, 2))
        e = cp.clip(cp.sqrt(gx * gx + gy * gy), 0.0, 1.0)
        return cp.stack([e, e, e], axis=-1)
    return g


class SharedState:
    def __init__(self, config: dict[str, Any]) -> None:
        self.lock = threading.Lock()
        report = "waiting for first frame"
        self.jpeg: bytes = status_jpeg(report, ok=False)
        self.status: dict[str, Any] = {"ok": False, "report": report, "updated_at": 0.0}
        self.config = dict(config)
        self.config_version = 0
        self.stop = threading.Event()

    def jpeg_snapshot(self) -> bytes:
        with self.lock:
            return self.jpeg

    def status_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.status)

    def config_snapshot(self) -> tuple[dict[str, Any], int]:
        with self.lock:
            return dict(self.config), self.config_version

    def update_config(self, patch: dict[str, Any]) -> dict[str, Any]:
        clean = {key: value for key, value in patch.items() if key in CONFIG_FIELDS}
        if not clean:
            config, version = self.config_snapshot()
            return {"ok": True, "updated": [], "ignored": sorted(patch), "version": version, "config": config}
        with self.lock:
            self.config.update(clean)
            self.config_version += 1
            config = dict(self.config)
            version = self.config_version
        return {"ok": True, "updated": sorted(clean), "ignored": sorted(set(patch) - set(clean)), "version": version, "config": config}


def initial_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "source_url": args.source_url,
        "op": args.op,
        "amount": args.amount,
        "max_fps": args.max_fps,
        "max_width": args.max_width,
        "jpeg_quality": args.jpeg_quality,
        "source_timeout": args.source_timeout,
    }


def fetch_frame(source_url: str, timeout: float) -> Any:
    req = urllib.request.Request(source_url, headers={"User-Agent": "BlacknodeCUDAFilterStream/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    data = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("source did not return a decodable image")
    return frame


def filter_loop(args: argparse.Namespace, state: SharedState) -> None:
    try:
        import cupy as cp
    except Exception as exc:  # noqa: BLE001
        report = f"CuPy not available ({type(exc).__name__}: {exc})"
        with state.lock:
            state.jpeg = status_jpeg(report, ok=False)
            state.status = {"ok": False, "report": report, "updated_at": time.time()}
        return

    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        device_name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    except Exception as exc:  # noqa: BLE001
        report = f"no CUDA device ({type(exc).__name__}: {exc})"
        with state.lock:
            state.jpeg = status_jpeg(report, ok=False)
            state.status = {"ok": False, "report": report, "updated_at": time.time()}
        return

    while not state.stop.is_set():
        started = time.monotonic()
        config, _version = state.config_snapshot()
        cfg = argparse.Namespace(**config)
        try:
            frame_bgr = fetch_frame(cfg.source_url, cfg.source_timeout)
            if cfg.max_width and frame_bgr.shape[1] > cfg.max_width:
                scale = cfg.max_width / float(frame_bgr.shape[1])
                frame_bgr = cv2.resize(
                    frame_bgr, (cfg.max_width, max(1, int(frame_bgr.shape[0] * scale))), interpolation=cv2.INTER_AREA,
                )
            rgb_float = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0

            g = cp.asarray(rgb_float)
            cp.cuda.Stream.null.synchronize()
            ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
            ev0.record()
            out = apply_image_filter(cp, cfg.op, g, cfg.amount)
            ev1.record()
            ev1.synchronize()
            gpu_ms = cp.cuda.get_elapsed_time(ev0, ev1)
            filtered_rgb = cp.asnumpy(out)

            filtered_bgr = np.clip(filtered_rgb, 0.0, 1.0)[:, :, ::-1]
            filtered_u8 = (filtered_bgr * 255.0 + 0.5).astype(np.uint8)
            ok, encoded = cv2.imencode(".jpg", filtered_u8, [int(cv2.IMWRITE_JPEG_QUALITY), int(cfg.jpeg_quality)])
            if not ok:
                raise RuntimeError("OpenCV JPEG encode failed")

            with state.lock:
                state.jpeg = encoded.tobytes()
                state.status = {
                    "ok": True,
                    "op": cfg.op,
                    "amount": cfg.amount,
                    "device": device_name,
                    "gpu_ms": round(gpu_ms, 4),
                    "report": f"filtering ({cfg.op}) on {device_name}: {round(gpu_ms, 2)}ms/frame",
                    "updated_at": time.time(),
                }
        except Exception as exc:  # noqa: BLE001
            report = f"CUDA filter stream FAILED: {type(exc).__name__}: {exc}"
            with state.lock:
                state.jpeg = status_jpeg(report, ok=False)
                state.status = {"ok": False, "report": report, "updated_at": time.time()}
        elapsed = time.monotonic() - started
        period = 1.0 / max(0.1, float(cfg.max_fps))
        state.stop.wait(max(0.01, period - elapsed))


def make_handler(state: SharedState, *, max_fps: float):
    class Handler(BaseHTTPRequestHandler):
        server_version = "BlacknodeCUDAFilterStream/0.1"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/health.json"):
                self._send_json({"ok": True, **state.status_snapshot()})
                return
            if self.path.startswith("/config.json"):
                config, version = state.config_snapshot()
                self._send_json({"ok": True, "version": version, "config": config})
                return
            if self.path.startswith("/snapshot.jpg"):
                jpeg = state.jpeg_snapshot()
                if not jpeg:
                    self.send_error(503, "no frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if self.path.startswith("/stream.mjpg"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while not state.stop.is_set():
                    jpeg = state.jpeg_snapshot()
                    if not jpeg:
                        time.sleep(0.05)
                        continue
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    config, _version = state.config_snapshot()
                    time.sleep(1.0 / max(0.1, float(config.get("max_fps", max_fps))))
                return
            self.send_error(404, "not found")

        def do_PATCH(self) -> None:  # noqa: N802
            self._handle_config_update()

        def do_POST(self) -> None:  # noqa: N802
            self._handle_config_update()

        def _handle_config_update(self) -> None:
            if not self.path.startswith("/config.json"):
                self.send_error(404, "not found")
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                body = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(body.decode("utf-8") or "{}")
                if not isinstance(payload, dict):
                    raise ValueError("config update must be a JSON object")
            except Exception as exc:  # noqa: BLE001
                self.send_error(400, f"invalid config update: {type(exc).__name__}: {exc}")
                return
            self._send_json(state.update_config(payload))

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", required=True, help="upstream snapshot.jpg URL to poll each frame")
    parser.add_argument("--op", choices=IMAGE_FILTERS, default="grayscale")
    parser.add_argument("--amount", type=float, default=1.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-fps", type=float, default=10.0)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--source-timeout", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    shared = SharedState(initial_config(args))
    signal.signal(signal.SIGTERM, lambda _sig, _frame: shared.stop.set())
    thread = threading.Thread(target=filter_loop, args=(args, shared), daemon=True)
    thread.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(shared, max_fps=args.max_fps))
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        shared.stop.set()
