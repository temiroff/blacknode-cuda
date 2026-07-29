# Internal CUDA kernels

Shared NVRTC compilation, launch, validation, and timing utilities live here.
Public workflow components consume these helpers through capability,
image-processing, tensor-operations, and optional benchmark nodes.

Kernel implementation details are private package infrastructure and are not a
selectable Blacknode component.
