"""Managed local processes for Warp LiDAR development viewers."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any


_VIEWERS: dict[str, dict[str, Any]] = {}


def _safe_id(value: str) -> str:
    return "".join(ch for ch in str(value or "") if ch.isalnum() or ch in "_-")[:80]


def _stop_process(proc: subprocess.Popen) -> bool:
    if proc.poll() is not None:
        return False
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=3.0)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except Exception:
            return False
    return True


def stop_viewer(viewer_id: str = "") -> dict[str, Any]:
    clean_id = _safe_id(viewer_id)
    keys = [clean_id] if clean_id else list(_VIEWERS)
    stopped = 0
    for key in keys:
        item = _VIEWERS.pop(key, None)
        if item:
            if _stop_process(item["proc"]):
                stopped += 1
            for path_key in ("scan_path", "config_path"):
                handoff_path = Path(str(item.get(path_key) or ""))
                if handoff_path.is_file():
                    handoff_path.unlink(missing_ok=True)
    return {"ok": True, "stopped": stopped}


def start_viewer(*, viewer_id: str, scan: dict, options: dict[str, Any]) -> dict[str, Any]:
    clean_id = _safe_id(viewer_id)
    if not clean_id:
        return {"ok": False, "error": "viewer_id is required"}
    if scan.get("kind") != "blacknode.laser-scan-stream":
        return {"ok": False, "error": "laser_scan must be a blacknode.laser-scan-stream"}
    device = str(options.get("device") or "cuda:0").strip()
    if device == "cpu":
        return {
            "ok": False,
            "error": (
                "WarpLiDARViewer requires a CUDA device for its native OpenGL window; "
                "use device=cuda:0. WarpLaserScanFilter still supports device=cpu."
            ),
        }
    stop_viewer(clean_id)
    script = Path(__file__).resolve().parents[1] / "scripts" / "warp_lidar_viewer.py"
    if not script.is_file():
        return {"ok": False, "error": f"Warp viewer helper not found: {script}"}
    handoff = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".laser-scan.json",
        prefix=f"blacknode-{clean_id}-",
        delete=False,
    )
    try:
        with handoff:
            json.dump(scan, handoff, allow_nan=True, separators=(",", ":"))
    except Exception:
        Path(handoff.name).unlink(missing_ok=True)
        raise
    scan_path = Path(handoff.name)
    arguments = [
        sys.executable,
        str(script),
        "--scan-file", str(scan_path),
        "--device", device,
        "--filter-min", str(options.get("filter_min_m", 0.1)),
        "--filter-max", str(options.get("filter_max_m", 12.0)),
        "--stride", str(options.get("stride", 1)),
        "--sensor-x", str(options.get("sensor_x_m", 0.0)),
        "--sensor-y", str(options.get("sensor_y_m", 0.0)),
        "--sensor-yaw", str(options.get("sensor_yaw_rad", 0.0)),
        "--point-radius", str(options.get("point_radius_m", 0.025)),
        "--fps", str(options.get("fps", 30)),
        "--scan-hz", str(options.get("scan_hz", 0.25)),
        "--ray-trail-count", str(options.get("ray_trail_count", 96)),
    ]
    if options.get("show_raw", True):
        arguments.append("--show-raw")
    if options.get("show_filtered", True):
        arguments.append("--show-filtered")
    if options.get("animate_scan", True):
        arguments.append("--animate-scan")
    if options.get("show_rays", True):
        arguments.append("--show-rays")
    if options.get("accumulate_hits", True):
        arguments.append("--accumulate-hits")
    if options.get("compare_numpy", False):
        arguments.append("--compare-numpy")
    if options.get("live", False):
        arguments.append("--watch")
    log_dir = Path(tempfile.gettempdir()) / "blacknode-warp-viewers"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{clean_id}.log"
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(arguments, stdout=log, stderr=subprocess.STDOUT, **kwargs)
    except Exception as exc:
        scan_path.unlink(missing_ok=True)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    _VIEWERS[clean_id] = {
        "proc": proc,
        "log_path": str(log_path),
        "scan_path": str(scan_path),
    }
    time.sleep(0.4)
    if proc.poll() is not None:
        detail = log_path.read_text(encoding="utf-8", errors="replace").strip()
        _VIEWERS.pop(clean_id, None)
        scan_path.unlink(missing_ok=True)
        return {
            "ok": False,
            "error": "Warp viewer exited during startup" + (f": {detail}" if detail else ""),
        }
    return {"ok": True, "running": True, "viewer_id": clean_id, "log_path": str(log_path)}


def update_viewer_scan(viewer_id: str, scan: dict) -> dict[str, Any]:
    """Atomically replace the handoff read by a live native Viewer worker."""
    clean_id = _safe_id(viewer_id)
    item = _VIEWERS.get(clean_id)
    if not item:
        return {"ok": False, "error": f"Viewer {clean_id or viewer_id!r} is not running"}
    proc = item.get("proc")
    if proc is None or proc.poll() is not None:
        _VIEWERS.pop(clean_id, None)
        return {"ok": False, "error": f"Viewer {clean_id!r} has stopped"}
    raw_scan_path = str(item.get("scan_path") or "")
    if not raw_scan_path:
        return {"ok": False, "error": f"Viewer {clean_id!r} has no scan handoff"}
    scan_path = Path(raw_scan_path)
    temporary = scan_path.with_suffix(f"{scan_path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(scan, allow_nan=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, scan_path)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "updated": True, "viewer_id": clean_id}


def start_slam_viewer(*, viewer_id: str, config: dict[str, Any], device: str) -> dict[str, Any]:
    """Start a managed synthetic rover SLAM worker using a JSON handoff."""
    clean_id = _safe_id(viewer_id)
    if not clean_id:
        return {"ok": False, "error": "viewer_id is required"}
    if config.get("kind") != "blacknode.warp-slam-demo":
        return {"ok": False, "error": "config must be a blacknode.warp-slam-demo"}
    stop_viewer(clean_id)
    script = Path(__file__).resolve().parents[1] / "scripts" / "warp_slam_discovery_viewer.py"
    if not script.is_file():
        return {"ok": False, "error": f"Warp SLAM viewer helper not found: {script}"}

    handoff = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".warp-slam.json",
        prefix=f"blacknode-{clean_id}-",
        delete=False,
    )
    try:
        with handoff:
            json.dump(config, handoff, allow_nan=False, separators=(",", ":"))
    except Exception:
        Path(handoff.name).unlink(missing_ok=True)
        raise
    config_path = Path(handoff.name)
    arguments = [
        sys.executable,
        str(script),
        "--config-file", str(config_path),
        "--device", str(device or "cuda:0"),
    ]
    log_dir = Path(tempfile.gettempdir()) / "blacknode-warp-viewers"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{clean_id}.log"
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    try:
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(arguments, stdout=log, stderr=subprocess.STDOUT, **kwargs)
    except Exception as exc:
        config_path.unlink(missing_ok=True)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    _VIEWERS[clean_id] = {
        "proc": proc,
        "log_path": str(log_path),
        "config_path": str(config_path),
    }
    time.sleep(0.4)
    if proc.poll() is not None:
        detail = log_path.read_text(encoding="utf-8", errors="replace").strip()
        _VIEWERS.pop(clean_id, None)
        config_path.unlink(missing_ok=True)
        return {
            "ok": False,
            "error": "Warp SLAM viewer exited during startup" + (f": {detail}" if detail else ""),
        }
    return {"ok": True, "running": True, "viewer_id": clean_id, "log_path": str(log_path)}
