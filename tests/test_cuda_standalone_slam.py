from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.request

from blacknode.pkg.blacknode_cuda import standalone_slam


def test_packaged_slam_viewer_contains_built_application_assets():
    viewer = Path(standalone_slam.__file__).resolve().parents[1] / "viewer"
    html = (viewer / "index.html").read_text(encoding="utf-8")
    javascript_assets = list((viewer / "assets").glob("*.js"))
    stylesheet_assets = list((viewer / "assets").glob("*.css"))

    assert "Blacknode Warp SLAM" in html
    assert (viewer / "blacknode-logo-dark.png").is_file()
    assert len(javascript_assets) == 1
    assert len(stylesheet_assets) == 1
    assert javascript_assets[0].name in html
    assert stylesheet_assets[0].name in html
    javascript = javascript_assets[0].read_text(encoding="utf-8")
    # The standalone build imports the current shared role-aware editor viewer.
    # Keep the polished SLAM presentation in the release bundle: the B robot,
    # metric floor, animated ray sweep, and map palette all belong to the same
    # component operators see inside the editor.
    for marker in (
        "viewerRole",
        "B robot",
        "Robot ↑",
        "CURRENT SCAN",
        "active beam",
        "fixed free-floor cells",
        "occupied wall cells",
        "WARP MAP",
        "#72ff9d",
    ):
        assert marker in javascript


def test_standalone_application_starts_ros_streams_and_warp_slam(monkeypatch):
    options = standalone_slam._parser().parse_args([
        "--native",
        "--device", "cuda:0",
        "--scan-topic", "/laser",
        "--odometry-topic", "/wheel_odom",
        "--trajectories", "2048",
    ])
    calls = []

    class FakeRosRuntime:
        @staticmethod
        def start_topic_subscriber(**kwargs):
            calls.append(("ros-start", kwargs))
            return {"ok": True, "backend": "native"}

        @staticmethod
        def stop_topic_subscriber(topic):
            calls.append(("ros-stop", topic))
            return {"ok": True, "stopped": 1}

    monkeypatch.setattr(
        standalone_slam.StandaloneSlamApplication,
        "_ros2_runtime",
        staticmethod(lambda: FakeRosRuntime),
    )

    captured = {}

    def fake_start_slam(**kwargs):
        captured.update(kwargs)
        return {
            "running": True,
            "live": False,
            "scene": {},
            "status": {"state": "waiting", "error": ""},
        }

    monkeypatch.setattr(standalone_slam.slam_runtime, "start_slam", fake_start_slam)
    monkeypatch.setattr(standalone_slam.slam_runtime, "stop_slam", lambda _slam_id: {"ok": True, "stopped": 1})
    application = standalone_slam.StandaloneSlamApplication(options)

    result = application.start()
    stopped = application.stop()

    assert result["running"] is True
    assert captured["mode"] == "device"
    assert captured["device"] == "cuda:0"
    assert captured["source"]["topic"] == "/laser"
    assert captured["odometry_source"]["topic"] == "/wheel_odom"
    assert captured["options"]["stride"] == 1
    assert captured["trajectory_evaluation"]["kind"] == "blacknode.warp-trajectory-evaluator"
    assert captured["trajectory_evaluation"]["trajectory_count"] == 2_048
    assert captured["trajectory_evaluation"]["commands_motion"] is False
    assert [item[1]["topic"] for item in calls if item[0] == "ros-start"] == [
        "/laser", "/wheel_odom",
    ]
    assert stopped["running"] is False


def test_standalone_lidar_view_starts_without_navigation_overlays(monkeypatch):
    options = standalone_slam._parser().parse_args([])
    captured = {}

    class FakeRosRuntime:
        @staticmethod
        def start_topic_subscriber(**_kwargs):
            return {"ok": True, "backend": "native"}

    monkeypatch.setattr(
        standalone_slam.StandaloneSlamApplication,
        "_ros2_runtime",
        staticmethod(lambda: FakeRosRuntime),
    )
    monkeypatch.setattr(
        standalone_slam.slam_runtime,
        "start_slam",
        lambda **kwargs: captured.update(kwargs) or {
            "running": True,
            "status": {"state": "waiting", "error": ""},
        },
    )

    standalone_slam.StandaloneSlamApplication(options).start()

    assert options.trajectories == 0
    assert captured["trajectory_evaluation"] == {}


def test_standalone_viewer_serves_state_static_assets_and_controls(tmp_path):
    (tmp_path / "index.html").write_text("<title>viewer</title>", encoding="utf-8")

    class FakeApplication:
        def __init__(self):
            self.actions = []
            self.payloads = []

        def snapshot(self):
            return {"running": True, "scene": {"sequence": 7}, "status": {"state": "ready"}}

        def control(self, action, payload=None):
            self.actions.append(action)
            self.payloads.append(payload or {})
            return {"running": action != "stop", "status": {"state": action}}

    application = FakeApplication()
    server = standalone_slam.SlamViewerServer(("127.0.0.1", 0), application, tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(f"{url}/api/state", timeout=2) as response:
            state = json.loads(response.read())
        request = urllib.request.Request(f"{url}/api/control/clear", method="POST")
        with urllib.request.urlopen(request, timeout=2) as response:
            controlled = json.loads(response.read())
        goal_request = urllib.request.Request(
            f"{url}/api/control/set-goal",
            data=json.dumps({"goal_x_m": -1.25, "goal_y_m": 0.75}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(goal_request, timeout=2) as response:
            goal = json.loads(response.read())
        with urllib.request.urlopen(f"{url}/", timeout=2) as response:
            html = response.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert state["scene"]["sequence"] == 7
    assert controlled["status"]["state"] == "clear"
    assert goal["status"]["state"] == "set-goal"
    assert application.actions == ["clear", "set-goal"]
    assert application.payloads == [{}, {"goal_x_m": -1.25, "goal_y_m": 0.75}]
    assert "viewer" in html
