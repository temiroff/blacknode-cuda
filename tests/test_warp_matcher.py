"""Warp correlative scan-matching tests."""
from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("warp")

from blacknode.pkg.blacknode_cuda import warp_matcher


def _transform(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    cosine = math.cos(float(pose[2]))
    sine = math.sin(float(pose[2]))
    result = points.copy()
    result[:, 0] = cosine * points[:, 0] - sine * points[:, 1] + float(pose[0])
    result[:, 1] = sine * points[:, 0] + cosine * points[:, 1] + float(pose[1])
    return result


def test_warp_matcher_recovers_pose_on_cuda():
    if not warp_matcher.available("cuda:0"):
        pytest.skip("CUDA Warp device is unavailable")
    horizontal = [[x, 0.0, 0.0] for x in np.linspace(-2.0, 2.0, 60)]
    vertical = [[1.25, y, 0.0] for y in np.linspace(-1.0, 2.0, 45)]
    diagonal = [[-1.5 + value, 1.0 + value * 0.4, 0.0] for value in np.linspace(0.0, 1.3, 35)]
    local = np.asarray(horizontal + vertical + diagonal, dtype=np.float32)
    expected = np.asarray([0.25, -0.15, math.radians(5.0)], dtype=np.float64)
    reference = _transform(local, expected)
    matcher = warp_matcher.WarpCorrelativeMatcher(
        reference,
        resolution=0.05,
        linear_window=0.4,
        device="cuda:0",
    )

    pose, score = matcher.match(
        local,
        np.asarray([0.10, -0.05, math.radians(1.0)]),
        linear_window=0.4,
        angular_window=math.radians(10.0),
        minimum_score_gain=0.02,
    )

    assert np.linalg.norm(pose[:2] - expected[:2]) < 0.10
    assert abs(math.atan2(math.sin(float(pose[2] - expected[2])), math.cos(float(pose[2] - expected[2])))) < math.radians(3.0)
    assert score > 0.8
    assert matcher.last_kernel_ms > 0.0


def test_warp_matcher_reports_unavailable_for_cpu_backend():
    assert warp_matcher.available("cpu") is False
