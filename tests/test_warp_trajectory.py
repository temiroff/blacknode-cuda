"""Warp trajectory evaluation tests."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("warp")

from blacknode.pkg.blacknode_cuda.warp_trajectory import (
    WarpTrajectoryEvaluator,
    evaluate_trajectories_cpu,
    integrate_candidate_path,
    trajectory_candidates,
)


def test_candidate_generation_is_deterministic_bounded_and_contains_straight_path():
    first = trajectory_candidates(128, 0.8, 1.2)
    second = trajectory_candidates(128, 0.8, 1.2)

    assert np.array_equal(first, second)
    assert first.shape == (128, 2)
    assert first[0].tolist() == [0.0, 0.0]
    assert first[1].tolist() == pytest.approx([0.8, 0.0])
    assert np.all(first[:, 0] >= 0.0)
    assert np.all(first[:, 0] <= 0.8)
    assert np.all(np.abs(first[:, 1]) <= 1.2)


def test_path_integration_keeps_stationary_candidate_fixed_and_straight_candidate_forward():
    stationary = integrate_candidate_path([1.0, 2.0, 0.0], [0.0, 0.0], horizon_s=2.0, time_steps=8)
    forward = integrate_candidate_path([1.0, 2.0, 0.0], [0.5, 0.0], horizon_s=2.0, time_steps=8)

    assert np.allclose(stationary[:, :2], [1.0, 2.0])
    assert forward[-1, :2].tolist() == pytest.approx([2.0, 2.0])


def test_cpu_reference_marks_a_wall_crossing_unsafe_and_prefers_clear_progress():
    controls = np.asarray([[0.0, 0.0], [0.8, 0.0], [0.6, 0.8]], dtype=np.float32)
    wall = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)
    scores, unsafe, terminal, clearances = evaluate_trajectories_cpu(
        controls,
        wall,
        np.empty((0, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
        [0.0, 0.0, 0.0],
        [2.0, 0.0],
        horizon_s=2.0,
        time_steps=24,
        robot_radius_m=0.2,
        clearance_margin_m=0.2,
        maximum_linear_speed_mps=0.8,
        maximum_angular_speed_rps=1.2,
    )

    assert unsafe.tolist() == [0, 1, 0]
    assert scores[2] > scores[1]
    assert terminal[1] < terminal[0]
    assert clearances[1] <= 0.2


def test_cpu_reference_predicts_a_moving_obstacle_crossing_the_future_path():
    _, unsafe, _, clearances = evaluate_trajectories_cpu(
        np.asarray([[0.5, 0.0]], dtype=np.float32),
        np.empty((0, 3), dtype=np.float32),
        np.asarray([[0.5, 0.5, 0.0]], dtype=np.float32),
        np.asarray([[0.0, -0.5, 0.0]], dtype=np.float32),
        [0.0, 0.0, 0.0],
        [2.0, 0.0],
        horizon_s=2.0,
        time_steps=20,
        robot_radius_m=0.16,
        clearance_margin_m=0.16,
        maximum_linear_speed_mps=0.5,
        maximum_angular_speed_rps=1.0,
    )

    assert unsafe.tolist() == [1]
    assert clearances[0] <= 0.16


def test_warp_cpu_evaluator_matches_reference_and_publishes_compact_paths():
    evaluator = WarpTrajectoryEvaluator(device="cpu")
    fixed = np.asarray([
        [1.0, -0.3, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.3, 0.0],
    ], dtype=np.float32)
    result = evaluator.evaluate(
        fixed,
        np.asarray([[0.6, 0.5, 0.0]], dtype=np.float32),
        np.asarray([[0.0, -0.1, 0.0]], dtype=np.float32),
        [0.0, 0.0, 0.0],
        [2.0, 0.0],
        trajectory_count=96,
        time_steps=16,
        horizon_s=2.0,
        maximum_linear_speed_mps=0.8,
        maximum_angular_speed_rps=1.2,
        robot_radius_m=0.18,
        clearance_margin_m=0.18,
        display_trajectories=24,
        compare_cpu=True,
    )

    assert result["state"] == "ready"
    assert result["backend"] == "warp"
    assert result["trajectory_count"] == 96
    assert result["work_items"] == 96 * 16
    assert result["static_obstacles"] == 3
    assert result["dynamic_obstacles"] == 1
    assert result["safe_trajectories"] + result["unsafe_trajectories"] == 96
    assert result["display_trajectories"] == 24
    assert len(result["paths"]) == 24
    assert all(len(path) == 17 for path in result["paths"])
    assert result["best_candidate"]["commands_motion"] is False
    assert result["max_error"] < 1.0e-4
    assert result["pipeline_ms"] >= result["kernel_ms"]
