# Spatial processing

`DepthCloudViewer` projects calibrated metric-depth frames into a live 3D point
cloud with NVIDIA Warp. Its `color_mode` selects the point palette:

- `depth` uses the metric-depth and surface-confidence gradient;
- `rgb` samples the connected `rgb_source` frame stream;
- `ir` samples the connected `ir_source` frame stream as infrared intensity.

Connect `DepthImageProcessor.depth_stream` to `source` and
`WarpDepthProjector.stage` to `depth_projection`. Connect processed camera frame
streams to `rgb_source` and `ir_source`. RGB and IR topics must already be
registered to the depth optical image; matching resolution alone does not
correct camera extrinsics.

The projector uses edge-aware spatial cleanup, small-hole filling, isolated
outlier rejection, and motion-gated temporal smoothing by default. Temporal
blending only applies below `temporal_max_delta_m`, so a moving hand or robot
surface switches to the current measurement instead of leaving a depth trail.
Managed viewers reuse the same Warp buffers across frames and bound projection
candidates by `maximum_points` before copying results to the editor.

When `DepthImageProcessor` starts before its ROS 2 CameraInfo topic has produced
a sample, the depth stream keeps the managed `camera_info_source` handle. The
viewer resolves that handle on later live updates and begins projection as soon
as positive `fx` and `fy` intrinsics arrive.
