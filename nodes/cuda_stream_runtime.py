"""Start/stop the CUDA image-filter live-stream helper process.

Mirrors packages/blacknode-ros2/nodes/ros2_runtime.py's start_image_stream /
stop_image_stream exactly: launch a detached subprocess, wait for its HTTP
port to open, track it in a registry keyed by stream_id so a later stop can
find and terminate it. Deliberately a separate module (not folded into
cuda.py) so the plain-subprocess-management code stays easy to find, same
split as ros2_runtime.py vs ros2.py.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

_streams: dict[str, dict[str, Any]] = {}


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _port_open(host: str, port: int, timeout: float = 0.15) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _post_json(url: str, payload: dict[str, Any], timeout: float = 1.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "BlacknodeCUDARuntime/0.1"},
        method="PATCH",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _terminate_process(proc: subprocess.Popen) -> bool:
    if proc.poll() is not None:
        return False
    import os
    import signal

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            proc.kill()
    return True


def _stream_script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "cuda_filter_stream_server.py"


def start_filter_stream(
    *,
    stream_id: str,
    source_url: str,
    op: str,
    amount: float,
    host: str,
    port: int,
    max_fps: float,
    max_width: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    """Start a CUDA-filtered MJPEG relay reading from source_url. Returns URL/report data."""
    if not source_url:
        return {"ok": False, "error": "source_url is required (wire in an upstream stream's snapshot_url)"}
    script = _stream_script()
    if not script.exists():
        return {"ok": False, "error": f"stream helper not found: {script}"}

    existing = _streams.get(stream_id)
    if existing and existing.get("proc") is not None and existing["proc"].poll() is None:
        update_result = update_filter_stream_config(stream_id, {
            "source_url": source_url,
            "op": op,
            "amount": amount,
            "max_fps": max_fps,
            "max_width": max_width,
            "jpeg_quality": jpeg_quality,
        })
        return {
            "ok": bool(update_result.get("ok", True)),
            "stream_id": stream_id,
            "stream_url": existing.get("url", ""),
            "snapshot_url": existing.get("snapshot_url", ""),
            "health_url": existing.get("health_url", ""),
            "updated": update_result.get("updated", []),
        }

    stop_filter_stream(stream_id)
    selected_port = int(port) if int(port) > 0 else _free_port(host)
    args = [
        sys.executable,
        str(script),
        "--source-url", source_url,
        "--op", op,
        "--amount", str(amount),
        "--host", host,
        "--port", str(selected_port),
        "--max-fps", str(max_fps),
        "--max-width", str(max_width),
        "--jpeg-quality", str(jpeg_quality),
    ]
    try:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return {"ok": False, "error": "stream helper exited before opening its HTTP port"}
        if _port_open(host, selected_port):
            break
        time.sleep(0.05)
    else:
        _terminate_process(proc)
        return {"ok": False, "error": f"stream helper did not open http://{host}:{selected_port}"}

    url = f"http://{host}:{selected_port}/stream.mjpg"
    _streams[stream_id] = {
        "proc": proc,
        "url": url,
        "snapshot_url": f"http://{host}:{selected_port}/snapshot.jpg",
        "health_url": f"http://{host}:{selected_port}/health.json",
        "config_url": f"http://{host}:{selected_port}/config.json",
        "source_url": source_url,
        "op": op,
    }
    return {
        "ok": True,
        "stream_id": stream_id,
        "stream_url": url,
        "snapshot_url": _streams[stream_id]["snapshot_url"],
        "health_url": _streams[stream_id]["health_url"],
        "port": selected_port,
    }


def update_filter_stream_config(stream_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    item = _streams.get(stream_id)
    if not item:
        return {"ok": True, "active": False, "updated": [], "report": f"CUDA filter stream '{stream_id}' is not running"}
    proc = item.get("proc")
    if proc is None or proc.poll() is not None:
        _streams.pop(stream_id, None)
        return {"ok": True, "active": False, "updated": [], "report": f"CUDA filter stream '{stream_id}' has stopped"}
    config_url = str(item.get("config_url") or "")
    if not config_url:
        return {"ok": False, "active": True, "error": f"CUDA filter stream '{stream_id}' has no config endpoint"}
    try:
        result = _post_json(config_url, updates, timeout=1.0)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "active": True, "error": f"{type(exc).__name__}: {exc}"}
    if "source_url" in updates:
        item["source_url"] = str(updates.get("source_url") or "")
    if "op" in updates:
        item["op"] = str(updates.get("op") or "")
    return {"ok": True, "active": True, **result}


def stop_filter_stream(stream_id: str = "") -> dict[str, Any]:
    """Stop one filter stream by id, or all filter streams when stream_id is empty."""
    ids = [stream_id] if stream_id else list(_streams)
    stopped = 0
    for sid in ids:
        item = _streams.pop(sid, None)
        if not item:
            continue
        if _terminate_process(item["proc"]):
            stopped += 1
    return {"ok": True, "stopped": stopped}


def runtime_status() -> dict[str, Any]:
    """Return active CUDA filter streams for the editor runtime controls."""
    streams: list[dict[str, Any]] = []
    for stream_id, item in list(_streams.items()):
        proc = item.get("proc")
        if proc is None or proc.poll() is not None:
            _streams.pop(stream_id, None)
            continue
        streams.append({
            "stream_id": stream_id,
            "stream_url": item.get("url", ""),
            "snapshot_url": item.get("snapshot_url", ""),
            "health_url": item.get("health_url", ""),
            "op": item.get("op", ""),
        })
    return {
        "ok": True,
        "active": bool(streams),
        "streams": streams,
        "report": f"{len(streams)} CUDA filter stream(s) active" if streams else "no CUDA filter streams active",
    }


def stop_runtime_services() -> dict[str, Any]:
    result = stop_filter_stream("")
    stopped = int(result.get("stopped") or 0)
    return {
        "ok": True,
        "stopped": {"streams": stopped, "managed_runs": 0, "detached": 0, "cv2_streams": 0, "reasoning_streams": 0},
        "report": f"stopped {stopped} CUDA filter stream(s)",
    }


def stream_running(stream_id: str) -> bool:
    item = _streams.get(stream_id)
    if not item:
        return False
    return item["proc"].poll() is None
