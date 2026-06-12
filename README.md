# blacknode-cuda

Real GPU compute nodes for [Blacknode](https://github.com/temiroff/Blacknode):

| Node | What it does |
|---|---|
| `CUDAKernelLab` | Curated GPU ops (vector add, matmul, FFT, mandelbrot, ...) with measured GPU vs CPU timings |
| `CUDACustomKernel` | Write and run your own CUDA C kernel, compiled at runtime via NVRTC |
| `CUDAImageFilter` | GPU image filters (grayscale, gaussian blur, sobel, ...) |
| `TensorCoreGEMM` | WMMA Tensor Core matrix multiply via NVRTC |
| `CUTLASS` / `CUTLASSGemm` | CUTLASS GEMM through the sandboxed worker |
| `GPUCapability` / `GPURequirement` | Query and gate on local GPU capabilities |

Example workflows ship in `templates/` and appear in the editor's Templates tab.

## Install

```bash
git clone https://github.com/temiroff/blacknode-cuda packages/blacknode-cuda
pip install -r packages/blacknode-cuda/requirements.txt
```

or, from the Blacknode root:

```bash
blacknode packages install https://github.com/temiroff/blacknode-cuda
```

Restart Blacknode (or press Reload in the editor's Packages tab). The nodes
appear under the **NVIDIA GPU** category.

Requires an NVIDIA GPU with a CUDA 12.x driver for actual compute. Without a
GPU the nodes load fine and return structured errors, so workflows stay
editable anywhere.
