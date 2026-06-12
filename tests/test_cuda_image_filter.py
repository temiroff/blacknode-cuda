"""CUDAImageFilter — GPU image filter node.

GPU-dependent checks skip cleanly when Pillow / an NVIDIA GPU is absent.
The "no image" contract always holds.
"""
import os
import tempfile

import pytest

from blacknode.nodes.image import load_image
from blacknode.pkg.blacknode_cuda.cuda import IMAGE_FILTERS, cuda_image_filter


def _has_pil() -> bool:
    try:
        import PIL  # noqa: F401
        import numpy  # noqa: F401
        return True
    except Exception:
        return False


def _has_gpu() -> bool:
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceProperties(0)
        return True
    except Exception:
        return False


HAS_PIL = _has_pil()
HAS_GPU = _has_gpu()
gpu_only = pytest.mark.skipif(not (HAS_PIL and HAS_GPU), reason="no Pillow / NVIDIA GPU")


def test_cuda_image_filter_no_image_errors():
    r = cuda_image_filter({"op": "grayscale"})
    assert r["image"] == ""
    assert "error" in r["report"]


def _make_test_image() -> str:
    import numpy as np
    from PIL import Image as PILImage
    rng = np.random.default_rng(0)
    arr = (rng.random((24, 32, 3)) * 255).astype("uint8")
    path = os.path.join(tempfile.gettempdir(), "_bn_img_test.png")
    PILImage.fromarray(arr).save(path)
    return path


@gpu_only
@pytest.mark.parametrize("op", IMAGE_FILTERS)
def test_each_filter_returns_image(op):
    loaded = load_image({"source": _make_test_image(), "max_size": 0})
    r = cuda_image_filter({"image": loaded["image"], "op": op, "amount": 0.5})
    assert "error" not in r["report"], r["report"]
    assert r["image"].startswith("data:image/png;base64,")
    assert r["device"]
