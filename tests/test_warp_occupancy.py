"""Warp-backed real-ray occupancy grid tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("warp")

from blacknode.pkg.blacknode_cuda.warp_occupancy import WarpOccupancyGrid


def test_real_rays_discover_fixed_free_cell_centers_on_warp_cpu():
    grid = WarpOccupancyGrid(
        device="cpu",
        resolution_m=0.1,
        radius_m=2.0,
        center_xy=(0.0, 0.0),
        display_capacity=2_000,
    )

    result = grid.update(
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        (0.0, 0.0),
    )

    assert result["backend"] == "warp"
    assert result["device"] == "cpu"
    assert result["rays"] == 2
    assert result["free_cells"] > 0
    assert result["grid_width"] == 40
    assert result["grid_height"] == 40
    assert result["encoding"] == "u2-base64"
    assert result["data"]
    assert result["fixed_origin"] is True
    cell_coordinates = (grid.points[:, :2] - np.asarray([
        grid.world_min_x + 0.05,
        grid.world_min_y + 0.05,
    ])) / 0.1
    assert np.allclose(cell_coordinates, np.rint(cell_coordinates), atol=2.0e-5)


def test_grid_origin_does_not_follow_robot_and_clear_only_zeros_evidence():
    grid = WarpOccupancyGrid(
        device="cpu",
        resolution_m=0.1,
        radius_m=2.0,
        center_xy=(0.0, 0.0),
    )
    fixed_origin = (grid.world_min_x, grid.world_min_y)
    grid.update(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), (0.0, 0.0))
    grid.update(np.asarray([[2.0, 1.0, 0.0]], dtype=np.float32), (1.0, 1.0))

    assert (grid.world_min_x, grid.world_min_y) == fixed_origin
    assert len(grid.points) > 0

    grid.clear()

    assert (grid.world_min_x, grid.world_min_y) == fixed_origin
    assert grid.snapshot()["free_cells"] == 0
    assert grid.snapshot()["data"] == ""
    assert grid.points.shape == (0, 3)
    assert grid.occupied_points.shape == (0, 3)


def test_repeated_real_hits_become_persistent_occupied_wall_cells():
    grid = WarpOccupancyGrid(
        device="cpu",
        resolution_m=0.1,
        radius_m=2.0,
        center_xy=(0.0, 0.0),
        display_capacity=2_000,
    )
    wall_hits = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    for _ in range(4):
        result = grid.update(wall_hits, (0.0, 0.0))

    assert result["occupied_cells"] > 0
    assert result["occupied_display_cells"] == len(grid.occupied_points)
    assert len(grid.occupied_points) > 0
    assert np.any(np.linalg.norm(grid.occupied_points[:, :2] - [1.05, 0.05], axis=1) < 0.08)


def test_angular_beam_footprint_widens_free_space_and_wall_with_distance():
    grid = WarpOccupancyGrid(
        device="cpu",
        resolution_m=0.05,
        radius_m=4.0,
        center_xy=(0.0, 0.0),
    )
    hit = np.asarray([[3.0, 0.0, 0.0]], dtype=np.float32)

    for _ in range(4):
        result = grid.update(
            hit,
            (0.0, 0.0),
            angular_increment_rad=math.radians(4.0),
        )

    near_free = grid.points[np.abs(grid.points[:, 0] - 0.5) < 0.08]
    far_free = grid.points[np.abs(grid.points[:, 0] - 2.5) < 0.08]
    wall = grid.occupied_points[np.abs(grid.occupied_points[:, 0] - 3.0) < 0.08]

    assert np.ptp(far_free[:, 1]) > np.ptp(near_free[:, 1])
    assert np.ptp(wall[:, 1]) >= 0.1
    assert result["beam_model"] == "angular-footprint"
    assert result["angular_increment_rad"] == pytest.approx(math.radians(4.0))
    assert result["free_half_width_limit_cells"] == 4
    assert result["wall_half_width_limit_cells"] == 3
    assert result["discontinuity_gating"] is True


def test_range_discontinuity_narrows_beam_on_object_boundary_side():
    grid = WarpOccupancyGrid(
        device="cpu",
        resolution_m=0.05,
        radius_m=4.0,
        center_xy=(0.0, 0.0),
    )
    angle = math.radians(2.0)
    hits = np.asarray([
        [math.cos(-angle), math.sin(-angle), 0.0],
        [3.0, 0.0, 0.0],
    ], dtype=np.float32)

    grid.update(hits, (0.0, 0.0), angular_increment_rad=angle)
    far_free = grid.points[
        (grid.points[:, 0] > 2.4)
        & (grid.points[:, 0] < 2.6)
    ]

    assert len(far_free) > 0
    assert far_free[:, 1].min() >= -0.01
    assert far_free[:, 1].max() >= 0.06
