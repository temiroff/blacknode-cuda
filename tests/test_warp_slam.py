"""Synthetic Warp SLAM discovery scenario and managed-viewer contracts."""
import json
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

import blacknode  # noqa: F401
from blacknode.node import _NODE_REGISTRY
from blacknode.pkg.blacknode_cuda import warp_slam
from blacknode.pkg.blacknode_cuda import warp_viewer_runtime


def test_slam_viewer_registers_and_scenario_is_deterministic():
    fn = _NODE_REGISTRY["WarpSLAMDiscoveryViewer"]
    scenario = warp_slam.slam_demo_scenario({"rays_per_scan": 1, "map_capacity": 999_999})

    assert fn._bn_package == "blacknode-cuda"
    assert fn._bn_component == "spatial-processing"
    assert scenario["kind"] == "blacknode.warp-slam-demo"
    assert scenario["rays_per_scan"] == 64
    assert warp_slam.slam_demo_scenario()["rays_per_scan"] == 1_000_000
    assert scenario["map_capacity"] == 250_000
    assert scenario["ray_history_capacity"] == 100_000
    assert scenario["show_paths"] is False
    assert scenario["accumulate_rays"] is True
    assert len(scenario["obstacles"]) == 7
    assert scenario["route"][0] == scenario["route"][-1]
    for start, end in zip(scenario["route"], scenario["route"][1:]):
        for fraction in np.linspace(0.0, 1.0, 101):
            x = start[0] + fraction * (end[0] - start[0])
            y = start[1] + fraction * (end[1] - start[1])
            assert not any(
                xmin <= x <= xmax and ymin <= y <= ymax
                for xmin, ymin, xmax, ymax in scenario["obstacles"]
            )


def test_route_pose_interpolates_constant_speed_and_wraps():
    route = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 0.0]]
    x, y, yaw, progress = warp_slam.route_pose(route, 1.0)
    wrapped = warp_slam.route_pose(route, 1.0 + 4.0 + math.sqrt(8.0))

    assert x == pytest.approx(1.0)
    assert y == pytest.approx(0.0)
    assert yaw == pytest.approx(0.0)
    assert 0.0 < progress < 1.0
    assert wrapped == pytest.approx((x, y, yaw, progress))


def test_ghost_obstacle_geometry_builds_translucent_faces_and_edges():
    faces, edges = warp_slam._ghost_obstacle_geometry([[1.0, 2.0, 4.0, 6.0]])

    assert faces.shape == (36, 3)
    assert edges.shape == (24, 3)
    assert faces.dtype == np.float32
    assert edges.dtype == np.float32
    assert faces[:, 0].min() == pytest.approx(1.0)
    assert faces[:, 0].max() == pytest.approx(4.0)
    assert faces[:, 1].min() == pytest.approx(2.0)
    assert faces[:, 1].max() == pytest.approx(6.0)
    assert faces[:, 2].min() == pytest.approx(0.0)
    assert faces[:, 2].max() == pytest.approx(0.85)


def test_warp_raycast_hits_nearest_box_face_on_cpu():
    if warp_slam.wp is None:
        pytest.skip("Warp is not installed")
    wp = warp_slam.wp
    obstacles = wp.array(np.asarray([[1.0, -0.5, 1.2, 0.5]], dtype=np.float32), dtype=wp.vec4, device="cpu")
    hits = wp.zeros(1, dtype=wp.vec3, device="cpu")
    ranges = wp.zeros(1, dtype=wp.float32, device="cpu")

    wp.launch(
        warp_slam._raycast_aabb_kernel,
        dim=1,
        inputs=[
            obstacles, 1, 0.0, 0.0, 0.45, 0.0,
            0.0, 1.0, 0.0, 0.0, 1, 10.0,
            -5.0, -5.0, 5.0, 5.0, 1.45,
        ],
        outputs=[hits, ranges],
        device="cpu",
    )

    assert ranges.numpy()[0] == pytest.approx(1.0, abs=1.0e-5)
    assert hits.numpy()[0] == pytest.approx([1.0, 0.0, 0.45], abs=1.0e-5)


def test_warp_raycast_hides_max_range_misses_on_cpu():
    if warp_slam.wp is None:
        pytest.skip("Warp is not installed")
    wp = warp_slam.wp
    obstacles = wp.array(np.empty((0, 4), dtype=np.float32), dtype=wp.vec4, device="cpu")
    hits = wp.zeros(1, dtype=wp.vec3, device="cpu")
    ranges = wp.zeros(1, dtype=wp.float32, device="cpu")

    wp.launch(
        warp_slam._raycast_aabb_kernel,
        dim=1,
        inputs=[
            obstacles, 0, 0.0, 0.0, 0.45, 0.0,
            0.0, 1.0, 0.0, 0.0, 1, 2.0,
            -5.0, -5.0, 5.0, 5.0, 1.45,
        ],
        outputs=[hits, ranges],
        device="cpu",
    )

    assert ranges.numpy()[0] == pytest.approx(2.0)
    assert hits.numpy()[0, 2] == pytest.approx(-1000.0)


def test_gpu_occupancy_kernel_deduplicates_cells_on_cpu():
    if warp_slam.wp is None:
        pytest.skip("Warp is not installed")
    wp = warp_slam.wp
    hits = wp.array(
        np.asarray([[1.01, 0.1, 0.1], [1.02, 0.1, 0.2], [1.01, 0.1, 1.1]], dtype=np.float32),
        dtype=wp.vec3,
        device="cpu",
    )
    occupancy = wp.zeros(32, dtype=wp.int32, device="cpu")
    count = wp.zeros(1, dtype=wp.int32, device="cpu")
    points = wp.empty(32, dtype=wp.vec3, device="cpu")
    wp.launch(warp_slam._clear_points_kernel, dim=32, inputs=[points], device="cpu")

    wp.launch(
        warp_slam._accumulate_occupancy_kernel,
        dim=3,
        inputs=[hits, 0.0, 0.0, 1.0, 4, 4, 2, 32, occupancy, count, points],
        device="cpu",
    )

    assert count.numpy().tolist() == [2]
    assert occupancy.numpy()[1] == 1
    assert occupancy.numpy()[17] == 1
    assert points.numpy()[0] == pytest.approx([1.5, 0.5, 0.5])
    assert points.numpy()[1] == pytest.approx([1.5, 0.5, 1.5])


def test_gpu_floor_discovery_accumulates_only_visible_free_cells_on_cpu():
    if warp_slam.wp is None:
        pytest.skip("Warp is not installed")
    wp = warp_slam.wp
    obstacles = wp.array(
        np.asarray([[1.8, 0.0, 2.2, 1.0]], dtype=np.float32),
        dtype=wp.vec4,
        device="cpu",
    )
    discovered = wp.zeros(4, dtype=wp.int32, device="cpu")
    observation_count = wp.zeros(4, dtype=wp.int32, device="cpu")
    points = wp.empty(4, dtype=wp.vec3, device="cpu")
    wp.launch(warp_slam._clear_points_kernel, dim=4, inputs=[points], device="cpu")

    wp.launch(
        warp_slam._accumulate_floor_discovery_kernel,
        dim=4,
        inputs=[
            obstacles, 1, 0.5, 0.5, 10.0, 0.0, 0.0, 1.0, 4,
            discovered, observation_count, points,
        ],
        device="cpu",
    )

    assert discovered.numpy().tolist() == [1, 1, 0, 0]
    assert observation_count.numpy().tolist() == [1, 1, 0, 0]
    assert points.numpy()[0] == pytest.approx([0.5, 0.5, 0.012])
    assert points.numpy()[1] == pytest.approx([1.5, 0.5, 0.012])
    assert points.numpy()[2, 2] == pytest.approx(-1000.0)


def test_gpu_line_kernels_append_every_valid_displayed_beam_on_cpu():
    if warp_slam.wp is None:
        pytest.skip("Warp is not installed")
    wp = warp_slam.wp
    hits = wp.array(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1000.0], [0.0, 2.0, 0.0]], dtype=np.float32),
        dtype=wp.vec3,
        device="cpu",
    )
    current = wp.empty((3, 2), dtype=wp.vec3, device="cpu")
    history = wp.empty((4, 2), dtype=wp.vec3, device="cpu")
    count = wp.zeros(1, dtype=wp.int32, device="cpu")
    wp.launch(warp_slam._clear_lines_kernel, dim=4, inputs=[history], device="cpu")

    wp.launch(
        warp_slam._build_current_lines_kernel,
        dim=3,
        inputs=[hits, 3, -1.0, -2.0, current],
        device="cpu",
    )
    wp.launch(
        warp_slam._append_history_lines_kernel,
        dim=3,
        inputs=[current, 4, count, history],
        device="cpu",
    )

    history_np = history.numpy()
    assert count.numpy().tolist() == [2]
    assert current.numpy()[1, 1, 2] == pytest.approx(-1000.0)
    assert history_np[0, 0] == pytest.approx([-1.0, -2.0, 0.03])
    assert history_np[1, 0] == pytest.approx([-1.0, -2.0, 0.03])
    assert {tuple(history_np[0, 1]), tuple(history_np[1, 1])} == {
        (1.0, 0.0, 0.0),
        (0.0, 2.0, 0.0),
    }


def test_gpu_radial_ring_kernel_builds_concentric_scan_pulses_on_cpu():
    if warp_slam.wp is None:
        pytest.skip("Warp is not installed")
    wp = warp_slam.wp
    lines = wp.empty((8, 2), dtype=wp.vec3, device="cpu")

    wp.launch(
        warp_slam._build_radial_rings_kernel,
        dim=8,
        inputs=[1.0, -2.0, 0.0, 10.0, 2, 4, lines],
        device="cpu",
    )

    lines_np = lines.numpy()
    first_radius = np.linalg.norm(lines_np[0, 0, :2] - np.asarray([1.0, -2.0]))
    second_radius = np.linalg.norm(lines_np[4, 0, :2] - np.asarray([1.0, -2.0]))
    assert first_radius == pytest.approx(0.18, abs=1.0e-5)
    assert second_radius == pytest.approx(5.09, abs=1.0e-5)
    assert lines_np[:, :, 2].min() >= 0.1


def test_slam_steady_state_has_no_gpu_to_numpy_geometry_roundtrip():
    source = inspect.getsource(warp_slam.run_slam_discovery_viewer)

    assert "hits_wp.numpy()" not in source
    assert "np.unique" not in source
    assert "map_points_wp.assign" not in source
    assert "RegisteredGLBuffer" in source
    assert "glDrawArrays(gl.GL_LINES" in source
    assert "draw_grid=False" in source
    assert "glDrawArrays(gl.GL_TRIANGLES" in source
    assert "glDepthMask(gl.GL_FALSE)" in source
    assert "enable_mouse_interaction=True" in source
    assert "grid_depth" in source
    assert "vertical_layers = 64" in source
    assert "_accumulate_floor_discovery_kernel" in source
    assert "floor_buffer.map" in source
    assert 'b"white_mode"' in source


def test_slam_viewer_node_uses_managed_runtime(monkeypatch):
    captured = {}

    def fake_start(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "running": True, "viewer_id": "slam"}

    monkeypatch.setattr(warp_slam.viewer_rt, "start_slam_viewer", fake_start)
    result = _NODE_REGISTRY["WarpSLAMDiscoveryViewer"]({
        "action": "start",
        "viewer_id": "slam",
        "device": "cpu",
        "rays_per_scan": 4096,
        "scan_hz": 15.0,
        "show_ground_truth": True,
        "show_paths": False,
        "accumulate_rays": True,
        "ray_history_capacity": 8000,
    })

    assert result["running"] is True
    assert captured["viewer_id"] == "slam"
    assert captured["device"] == "cpu"
    assert captured["config"]["rays_per_scan"] == 4096
    assert captured["config"]["scan_hz"] == 15.0
    assert captured["config"]["show_ground_truth"] is True
    assert captured["config"]["show_paths"] is False
    assert captured["config"]["accumulate_rays"] is True
    assert captured["config"]["ray_history_capacity"] == 8000
    assert result["viewer"]["controls"]["r"] == "clear the map and restart"
    assert result["viewer"]["controls"]["left_drag"] == "rotate camera"
    assert result["viewer"]["controls"]["wasd_or_arrows"] == "move camera"
    assert result["viewer"]["memory_path"] == "cpu-registered-buffer-fallback"


def test_slam_runtime_uses_config_file_handoff(monkeypatch):
    captured = {}

    class FakeProcess:
        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            del timeout
            return 0

    def fake_popen(arguments, **kwargs):
        captured["arguments"] = arguments
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(warp_viewer_runtime.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(warp_viewer_runtime.time, "sleep", lambda _seconds: None)
    config = warp_slam.slam_demo_scenario()
    result = warp_viewer_runtime.start_slam_viewer(
        viewer_id="slam_handoff", config=config, device="cpu",
    )

    try:
        arguments = captured["arguments"]
        assert result["ok"] is True
        assert "--config-file" in arguments
        config_path = Path(arguments[arguments.index("--config-file") + 1])
        assert json.loads(config_path.read_text(encoding="utf-8")) == config
        assert arguments[-2:] == ["--device", "cpu"]
    finally:
        warp_viewer_runtime.stop_viewer("slam_handoff")
