"""Warp hash-grid dynamic occupancy tests."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("warp")
import warp as wp

from blacknode.pkg.blacknode_cuda.warp_dynamic_occupancy import (
    WarpDynamicOccupancyTracker,
    classify_motion_cpu,
)


def _points(moving_x: float = 1.0) -> np.ndarray:
    return np.asarray([
        [0.0, 0.0, 0.0],
        [moving_x, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float32)


def _motion_cluster(moving_x: float = 1.0) -> np.ndarray:
    return np.asarray([
        [0.0, 0.0, 0.0],
        [moving_x, 0.0, 0.0],
        [moving_x, 0.03, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float32)


def test_cpu_reference_keeps_fixed_returns_and_marks_motion():
    velocities, scores, flags = classify_motion_cpu(
        _points(),
        _points(1.2),
        stable_radius_m=0.05,
        tracking_radius_m=0.3,
        dt_s=0.1,
        minimum_speed_mps=0.1,
    )

    assert flags.tolist() == [0, 1, 0]
    assert velocities[1, 0] == pytest.approx(2.0)
    assert scores[1] > 0.0
    assert np.allclose(velocities[[0, 2]], 0.0)


def test_warp_hash_grid_suppresses_isolated_motion_flicker_on_cpu():
    tracker = WarpDynamicOccupancyTracker(device="cpu")
    first = tracker.update(
        _points(),
        1_000_000_000,
        stable_radius_m=0.05,
        tracking_radius_m=0.3,
        minimum_speed_mps=0.1,
        maximum_age_s=0.5,
        display_points=32,
    )
    second = tracker.update(
        _points(1.2),
        1_100_000_000,
        stable_radius_m=0.05,
        tracking_radius_m=0.3,
        minimum_speed_mps=0.1,
        maximum_age_s=0.5,
        display_points=32,
        compare_cpu=True,
    )

    assert first["state"] == "warming"
    assert second["state"] == "ready"
    assert second["backend"] == "warp-hash-grid"
    assert second["input_points"] == 3
    assert second["reference_points"] == 3
    assert second["dynamic_points"] == 0
    assert second["motion_candidates"] == 1
    assert second["rejected_motion_points"] == 1
    assert second["display_points"] == 0
    assert second["_dynamic_mask"].tolist() == [False, True, False]
    assert second["_static_mask"].tolist() == [True, False, True]
    assert second["_motion_mask"].tolist() == [False, False, False]
    assert second["points"] == []
    assert second["pipeline_ms"] >= 0.0
    assert second["max_error"] < 1.0e-5


def test_warp_hash_grid_keeps_spatially_coherent_motion_cluster():
    tracker = WarpDynamicOccupancyTracker(device="cpu")
    options = {
        "stable_radius_m": 0.05,
        "tracking_radius_m": 0.3,
        "minimum_speed_mps": 0.1,
        "maximum_age_s": 0.5,
        "display_points": 32,
    }
    tracker.update(_motion_cluster(), 1_000_000_000, **options)
    result = tracker.update(_motion_cluster(1.2), 1_100_000_000, **options)

    assert result["dynamic_points"] == 2
    assert result["motion_candidates"] == 2
    assert result["rejected_motion_points"] == 0
    assert result["_motion_mask"].tolist() == [False, True, True, False]
    assert [point[0] for point in result["points"]] == pytest.approx([1.2, 1.2])
    assert [velocity[0] for velocity in result["velocities"]] == pytest.approx([2.0, 2.0], abs=1.0e-5)


def test_revealed_known_wall_returns_restore_as_static_instead_of_reverse_motion():
    tracker = WarpDynamicOccupancyTracker(device="cpu")
    options = {
        "stable_radius_m": 0.05,
        "tracking_radius_m": 0.3,
        "minimum_speed_mps": 0.1,
        "maximum_age_s": 0.5,
        "display_points": 32,
    }
    tracker.update(_motion_cluster(1.0), 1_000_000_000, **options)

    resolution = 0.05
    width = 64
    height = 8
    world_min_x = 0.0
    world_min_y = -0.2
    states = np.zeros(width * height, dtype=np.uint8)
    revealed = _motion_cluster(1.2)
    for point in revealed[1:3]:
        column = int(np.floor((point[0] - world_min_x) / resolution))
        row = int(np.floor((point[1] - world_min_y) / resolution))
        states[row * width + column] = 2

    result = tracker.update(
        revealed,
        1_100_000_000,
        **options,
        static_cell_states=wp.array(states, dtype=wp.uint8, device="cpu"),
        static_grid_width=width,
        static_grid_height=height,
        static_world_min_x=world_min_x,
        static_world_min_y=world_min_y,
        static_resolution_m=resolution,
    )

    assert result["known_static_points"] == 2
    assert result["dynamic_points"] == 0
    assert result["motion_candidates"] == 0
    assert result["_known_static_mask"].tolist() == [False, True, True, False]
    assert result["_static_mask"].tolist() == [True, True, True, True]
    assert result["_motion_mask"].tolist() == [False, False, False, False]
    assert result["points"] == []


def test_clear_discards_temporal_reference_without_reclassifying_old_points():
    tracker = WarpDynamicOccupancyTracker(device="cpu")
    options = {
        "stable_radius_m": 0.05,
        "tracking_radius_m": 0.3,
        "minimum_speed_mps": 0.1,
        "maximum_age_s": 0.5,
        "display_points": 32,
    }
    tracker.update(_points(), 1_000_000_000, **options)
    tracker.clear()
    result = tracker.update(_points(1.2), 1_100_000_000, **options)

    assert result["state"] == "warming"
    assert result["dynamic_points"] == 0


def test_new_unmatched_returns_are_transient_until_confirmed():
    tracker = WarpDynamicOccupancyTracker(device="cpu")
    options = {
        "stable_radius_m": 0.04,
        "tracking_radius_m": 0.3,
        "minimum_speed_mps": 0.04,
        "maximum_age_s": 0.6,
        "display_points": 32,
    }
    tracker.update(
        np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        1_000_000_000,
        **options,
    )
    result = tracker.update(_points(), 1_100_000_000, **options)

    assert result["_dynamic_mask"].tolist() == [False, True, False]
    assert result["_static_mask"].tolist() == [True, False, True]
    assert result["transient_points"] == 1
    assert result["dynamic_points"] == 0
    assert result["points"] == []


def test_reference_window_reveals_slow_motion_without_frame_to_frame_spikes():
    tracker = WarpDynamicOccupancyTracker(device="cpu")
    options = {
        "stable_radius_m": 0.04,
        "tracking_radius_m": 0.3,
        "minimum_speed_mps": 0.04,
        "maximum_age_s": 0.6,
        "display_points": 32,
    }
    tracker.update(_motion_cluster(1.0), 1_000_000_000, **options)
    small_step = tracker.update(_motion_cluster(1.005), 1_100_000_000, **options)
    accumulated_step = tracker.update(_motion_cluster(1.025), 1_400_000_000, **options)

    assert small_step["dynamic_points"] == 0
    assert accumulated_step["dynamic_points"] == 2
    assert [velocity[0] for velocity in accumulated_step["velocities"]] == pytest.approx(
        [0.0625, 0.0625], abs=1.0e-5,
    )
