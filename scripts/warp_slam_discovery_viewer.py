"""Launch the managed Warp synthetic rover SLAM discovery window."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="View synthetic rover SLAM discovery with NVIDIA Warp")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config_path = Path(args.config_file)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    finally:
        config_path.unlink(missing_ok=True)

    import blacknode  # noqa: F401 - installs extension-package aliases
    from blacknode.pkg.blacknode_cuda import warp_slam

    warp_slam.run_slam_discovery_viewer(config=config, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
