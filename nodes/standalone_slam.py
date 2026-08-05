"""Jetson-local Warp SLAM application and packaged WebGL viewer."""
from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import shutil
import subprocess
import threading
import webbrowser
from typing import Any
from urllib.parse import urlparse

from . import slam_runtime
from .slam import slam_options


_SLAM_ID = "standalone_slam"


def _stream(topic: str, message_type: str) -> dict[str, Any]:
    return {
        "kind": "blacknode.message-stream",
        "schema_version": 1,
        "stream_id": f"topic-subscriber:{topic}",
        "protocol": "ros2",
        "state": "waiting",
        "managed": True,
        "topic": topic,
        "message_type": message_type,
        "backend": "local",
    }


class StandaloneSlamApplication:
    def __init__(self, options: argparse.Namespace):
        self.options = options
        self.slam_id = _SLAM_ID
        self.lock = threading.RLock()
        self.scan_running = False
        self.odometry_running = False
        self.last_error = ""

    @staticmethod
    def _ros2_runtime():
        try:
            from blacknode.pkg.blacknode_ros2 import ros2_runtime
        except Exception as exc:
            raise RuntimeError(
                "blacknode-ros2 is required for the slam application; install and enable "
                "blacknode-ros2/core and blacknode-ros2/topics"
            ) from exc
        return ros2_runtime

    def start(self) -> dict[str, Any]:
        with self.lock:
            ros2_runtime = self._ros2_runtime()
            scan = ros2_runtime.start_topic_subscriber(
                topic=self.options.scan_topic,
                message_type="sensor_msgs/msg/LaserScan",
                node_name="blacknode_standalone_slam_scan",
                history=20,
                public_node_type="ROS2",
                stale_after_seconds=1.0,
                qos="sensor_data",
            )
            if not scan.get("ok"):
                self.last_error = str(scan.get("error") or "could not start the LaserScan subscription")
                return self.snapshot()
            self.scan_running = True

            odometry_source: dict[str, Any] = {}
            if self.options.odometry_topic:
                odometry = ros2_runtime.start_topic_subscriber(
                    topic=self.options.odometry_topic,
                    message_type="nav_msgs/msg/Odometry",
                    node_name="blacknode_standalone_slam_odometry",
                    history=30,
                    public_node_type="ROS2",
                    stale_after_seconds=1.0,
                    qos="reliable",
                )
                self.odometry_running = bool(odometry.get("ok"))
                if self.odometry_running:
                    odometry_source = _stream(
                        self.options.odometry_topic,
                        "nav_msgs/msg/Odometry",
                    )

            values = {
                "filter_min_m": self.options.filter_min,
                "filter_max_m": self.options.filter_max,
                "downsample_stride": 1,
                "sensor_x_m": self.options.sensor_x,
                "sensor_y_m": self.options.sensor_y,
                "sensor_yaw_rad": self.options.sensor_yaw,
                "map_resolution_m": self.options.map_resolution,
                "occupancy_radius_m": self.options.occupancy_radius,
                "fps": self.options.fps,
            }
            trajectory_count = max(64, min(65_536, int(self.options.trajectories)))
            started = slam_runtime.start_slam(
                slam_id=self.slam_id,
                node_id="standalone-slam",
                source=_stream(self.options.scan_topic, "sensor_msgs/msg/LaserScan"),
                odometry_source=odometry_source,
                mode="device" if self.options.native else "editor",
                device=self.options.device,
                options=slam_options(values),
                trajectory_evaluation={
                    "kind": "blacknode.warp-trajectory-evaluator",
                    "schema_version": 1,
                    "enabled": True,
                    "goal_x_m": self.options.goal_x,
                    "goal_y_m": self.options.goal_y,
                    "trajectory_count": trajectory_count,
                    "requested_trajectories": trajectory_count,
                    "time_steps": 48,
                    "horizon_s": 3.0,
                    "maximum_linear_speed_mps": 0.7,
                    "maximum_angular_speed_rps": 1.5,
                    "robot_radius_m": 0.18,
                    "clearance_margin_m": 0.18,
                    "display_trajectories": min(96, trajectory_count),
                    "compare_cpu": False,
                    "comparison_limited": False,
                    "commands_motion": False,
                },
            )
            status = started.get("status") if isinstance(started.get("status"), dict) else {}
            self.last_error = str(status.get("error") or "") if status.get("state") == "error" else ""
            return started

    def stop(self) -> dict[str, Any]:
        with self.lock:
            stopped = slam_runtime.stop_slam(self.slam_id)
            try:
                ros2_runtime = self._ros2_runtime()
                if self.scan_running:
                    ros2_runtime.stop_topic_subscriber(self.options.scan_topic)
                if self.odometry_running:
                    ros2_runtime.stop_topic_subscriber(self.options.odometry_topic)
            finally:
                self.scan_running = False
                self.odometry_running = False
            return {"ok": True, "running": False, "stopped": stopped.get("stopped", 0)}

    def snapshot(self) -> dict[str, Any]:
        value = slam_runtime.slam_status(self.slam_id)
        if self.last_error and not value.get("running"):
            value = {
                **value,
                "status": {
                    "kind": "blacknode.slam-status",
                    "schema_version": 1,
                    "state": "error",
                    "error": self.last_error,
                },
                "report": self.last_error,
            }
        return value

    def control(self, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with self.lock:
            payload = payload or {}
            if action == "start":
                return self.start()
            if action == "clear":
                return slam_runtime.clear_slam(self.slam_id)
            if action == "pause":
                return slam_runtime.set_mapping(self.slam_id, False)
            if action == "resume":
                return slam_runtime.set_mapping(self.slam_id, True)
            if action == "stop":
                self.stop()
                return self.snapshot()
            if action == "set-goal":
                try:
                    goal_x_m = float(payload.get("goal_x_m"))
                    goal_y_m = float(payload.get("goal_y_m"))
                except (TypeError, ValueError) as exc:
                    raise ValueError("SLAM goal coordinates must be finite numbers") from exc
                if not math.isfinite(goal_x_m) or not math.isfinite(goal_y_m):
                    raise ValueError("SLAM goal coordinates must be finite numbers")
                self.options.goal_x = goal_x_m
                self.options.goal_y = goal_y_m
                return slam_runtime.set_trajectory_goal(self.slam_id, goal_x_m, goal_y_m)
            return {
                "running": False,
                "status": {"state": "error", "error": f"unsupported control action {action!r}"},
            }


class SlamViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], application: StandaloneSlamApplication, assets: Path):
        self.application = application
        self.assets = assets.resolve()
        super().__init__(address, SlamViewerHandler)


class SlamViewerHandler(BaseHTTPRequestHandler):
    server: SlamViewerServer

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def _json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        try:
            encoded = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except (TypeError, ValueError):
            encoded = json.dumps({"running": False, "status": {"state": "error", "error": "viewer state could not be encoded"}}).encode("utf-8")
            status = HTTPStatus.INTERNAL_SERVER_ERROR
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        path = urlparse(self.path).path
        if path == "/api/state":
            self._json(self.server.application.snapshot())
            return
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        candidate = (self.server.assets / relative).resolve()
        try:
            candidate.relative_to(self.server.assets)
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = {
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".png": "image/png",
            ".svg": "image/svg+xml",
        }.get(candidate.suffix.lower(), "application/octet-stream")
        body = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache" if candidate.suffix == ".html" else "public, max-age=31536000, immutable")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        path = urlparse(self.path).path
        prefix = "/api/control/"
        if not path.startswith(prefix):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        action = path[len(prefix):].strip().lower()
        if action not in {"start", "clear", "pause", "resume", "stop", "set-goal"}:
            self._json(
                {"running": False, "status": {"state": "error", "error": "unsupported control action"}},
                HTTPStatus.BAD_REQUEST,
            )
            return
        content_length = int(self.headers.get("Content-Length") or 0)
        if content_length < 0 or content_length > 4096:
            self._json(
                {"running": False, "status": {"state": "error", "error": "control payload is too large"}},
                HTTPStatus.BAD_REQUEST,
            )
            return
        try:
            payload = json.loads(self.rfile.read(content_length)) if content_length else {}
            if not isinstance(payload, dict):
                raise ValueError("control payload must be an object")
            result = self.server.application.control(action, payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self._json(
                {"running": False, "status": {"state": "error", "error": str(exc)}},
                HTTPStatus.BAD_REQUEST,
            )
            return
        self._json(result)


def _open_viewer(url: str, *, fullscreen: bool) -> subprocess.Popen | None:
    if fullscreen:
        browser = next(
            (candidate for name in ("chromium-browser", "chromium", "google-chrome") if (candidate := shutil.which(name))),
            None,
        )
        if browser:
            return subprocess.Popen(
                [browser, "--kiosk", "--no-first-run", "--disable-session-crashed-bubble", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    webbrowser.open(url, new=1)
    return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="blacknode run slam", description="Run Jetson-local Warp SLAM and its packaged viewer")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--odometry-topic", default="/odom", help="set to an empty string to run scan-only")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7780)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--native", action="store_true", help="open the CUDA/OpenGL interop viewer instead of a browser")
    parser.add_argument("--filter-min", type=float, default=0.1)
    parser.add_argument("--filter-max", type=float, default=12.0)
    parser.add_argument("--map-resolution", type=float, default=0.05)
    parser.add_argument("--occupancy-radius", type=float, default=20.0)
    parser.add_argument("--sensor-x", type=float, default=0.0)
    parser.add_argument("--sensor-y", type=float, default=0.0)
    parser.add_argument("--sensor-yaw", type=float, default=0.0)
    parser.add_argument("--goal-x", type=float, default=3.0, help="initial map-frame trajectory goal X in metres")
    parser.add_argument("--goal-y", type=float, default=0.0, help="initial map-frame trajectory goal Y in metres")
    parser.add_argument("--trajectories", type=int, default=2_048, help="parallel candidate trajectories per scan")
    parser.add_argument("--fps", type=int, default=30)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(argv or [])
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    options = _parser().parse_args(arguments)
    assets = Path(__file__).resolve().parents[1] / "viewer"
    if not (assets / "index.html").is_file():
        raise RuntimeError(
            "The packaged SLAM viewer is missing. Reinstall blacknode-cuda or build it with "
            "npm run build:slam-viewer from the Blacknode editor workspace."
        )
    application = StandaloneSlamApplication(options)
    started = application.start()
    status = started.get("status") if isinstance(started.get("status"), dict) else {}
    if status.get("state") in {"error", "unavailable"}:
        application.stop()
        raise RuntimeError(str(status.get("error") or started.get("report") or "SLAM could not start"))

    server = SlamViewerServer((options.host, max(0, min(65535, options.port))), application, assets)
    host, port = server.server_address[:2]
    visible_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{visible_host}:{port}/"
    print(f"Blacknode Warp SLAM is running on {options.device}.")
    print(f"Viewer: {url}")
    print("Press Ctrl+C to stop.")
    browser_process = None
    if not options.no_open and not options.native:
        browser_process = _open_viewer(url, fullscreen=options.fullscreen)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        application.stop()
        if browser_process is not None and browser_process.poll() is None:
            browser_process.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
