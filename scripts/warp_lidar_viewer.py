"""Launch a static normalized LaserScan in NVIDIA Warp's OpenGL viewer."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="View one Blacknode LaserScan with NVIDIA Warp")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scan-file")
    source.add_argument("--scan-base64")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--filter-min", type=float, default=0.1)
    parser.add_argument("--filter-max", type=float, default=12.0)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--sensor-x", type=float, default=0.0)
    parser.add_argument("--sensor-y", type=float, default=0.0)
    parser.add_argument("--sensor-yaw", type=float, default=0.0)
    parser.add_argument("--point-radius", type=float, default=0.025)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scan-hz", type=float, default=0.25)
    parser.add_argument("--ray-trail-count", type=int, default=96)
    parser.add_argument("--show-raw", action="store_true")
    parser.add_argument("--show-filtered", action="store_true")
    parser.add_argument("--animate-scan", action="store_true")
    parser.add_argument("--show-rays", action="store_true")
    parser.add_argument("--accumulate-hits", action="store_true")
    parser.add_argument("--persist-scans", action="store_true")
    parser.add_argument("--max-accumulated-points", type=int, default=50_000)
    parser.add_argument("--compare-numpy", action="store_true")
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()

    if args.scan_file:
        scan_path = Path(args.scan_file)
        scan = json.loads(scan_path.read_text(encoding="utf-8"))
        if not args.watch:
            scan_path.unlink(missing_ok=True)
    else:
        scan_path = None
        scan = json.loads(base64.urlsafe_b64decode(args.scan_base64.encode("ascii")).decode("utf-8"))

    latest_scan = scan

    def scan_source():
        nonlocal latest_scan
        if args.watch and scan_path is not None:
            try:
                candidate = json.loads(scan_path.read_text(encoding="utf-8"))
                if isinstance(candidate, dict):
                    latest_scan = candidate
            except (OSError, json.JSONDecodeError):
                pass
        return latest_scan

    import blacknode  # noqa: F401 - installs stable extension-package aliases
    from blacknode.pkg.blacknode_cuda import warp_points
    warp_points.run_viewer_loop(
        scan_source=scan_source,
        device=args.device,
        filter_min_m=args.filter_min,
        filter_max_m=args.filter_max,
        stride=max(1, args.stride),
        sensor_pose=(args.sensor_x, args.sensor_y, args.sensor_yaw),
        show_raw=args.show_raw,
        show_filtered=args.show_filtered,
        point_radius=args.point_radius,
        fps=max(1, args.fps),
        animate_scan=args.animate_scan,
        scan_hz=max(0.01, args.scan_hz),
        show_rays=args.show_rays,
        ray_trail_count=max(1, args.ray_trail_count),
        accumulate_hits=args.accumulate_hits,
        persist_scans=args.persist_scans,
        max_accumulated_points=max(1_000, min(250_000, args.max_accumulated_points)),
        compare_numpy=args.compare_numpy,
        title="Blacknode LiDAR — animated scan",
    )
    if args.watch and scan_path is not None:
        scan_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
