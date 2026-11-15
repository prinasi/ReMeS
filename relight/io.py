"""Image I/O and transfer functions.

Conventions
-----------
* All arrays are float32, channel-last RGB (OpenCV's BGR is converted on read).
* "linear" means scene-referred radiance; "display" means a [0, 1] encoded image.
* Synthetic4Relight protocol (InvRender): PNGs are decoded with x**2.2 and
  metrics are computed on clip(x**(1/2.2)). Blender's "Standard" view transform
  uses the piecewise sRGB curve instead; both are provided.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

GAMMA = 2.2


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] >= 3:
        rgb = img[..., 2::-1]
        if img.shape[2] == 4:
            rgb = np.concatenate([rgb, img[..., 3:4]], axis=-1)
        return np.ascontiguousarray(rgb)
    return img


def read_image(path: str) -> np.ndarray:
    """Read EXR/PNG/JPG as float32 RGB(A). Integer images are scaled to [0, 1]."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.float32) / float(np.iinfo(img.dtype).max)
    else:
        img = img.astype(np.float32)
    return _bgr_to_rgb(img)


def read_exr_rgb(path: str, clamp_negative: bool = True) -> np.ndarray:
    img = read_image(path)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    img = img[..., :3]
    if clamp_negative:
        img = np.maximum(img, 0.0)
    return np.ascontiguousarray(img)


def write_exr(path: str, rgb: np.ndarray) -> None:
    rgb = np.asarray(rgb, dtype=np.float32)
    cv2.imwrite(str(path), rgb[..., ::-1] if rgb.ndim == 3 else rgb)


def write_png(path: str, display_rgb: np.ndarray) -> None:
    img = np.clip(np.asarray(display_rgb, dtype=np.float32), 0.0, 1.0)
    img = (img * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), img[..., ::-1] if img.ndim == 3 else img)


# ---------------------------------------------------------------- transfer functions
def _lib(x):
    return torch if isinstance(x, torch.Tensor) else np


def linear_to_gamma(x, gamma: float = GAMMA):
    lib = _lib(x)
    return lib.clip(x, 0.0, None) ** (1.0 / gamma)


def gamma_to_linear(x, gamma: float = GAMMA):
    lib = _lib(x)
    return lib.clip(x, 0.0, None) ** gamma


def linear_to_srgb(x):
    lib = _lib(x)
    x = lib.clip(x, 0.0, None)
    return lib.where(x <= 0.0031308, 12.92 * x, 1.055 * lib.clip(x, 0.0031308, None) ** (1 / 2.4) - 0.055)


def srgb_to_linear(x):
    lib = _lib(x)
    x = lib.clip(x, 0.0, None)
    return lib.where(x <= 0.04045, x / 12.92, ((lib.clip(x, 0.04045, None) + 0.055) / 1.055) ** 2.4)


def tonemap(x, mode: str = "gamma"):
    """Linear -> display in [0, 1]. mode: 'gamma' (dataset protocol) or 'srgb' (Blender Standard)."""
    lib = _lib(x)
    y = linear_to_gamma(x) if mode == "gamma" else linear_to_srgb(x)
    return lib.clip(y, 0.0, 1.0)


def psnr(a, b, mask=None) -> float:
    a = torch.as_tensor(np.asarray(a) if not isinstance(a, torch.Tensor) else a, dtype=torch.float32)
    b = torch.as_tensor(np.asarray(b) if not isinstance(b, torch.Tensor) else b, dtype=torch.float32).to(a.device)
    se = (a - b) ** 2
    if mask is not None:
        m = torch.as_tensor(np.asarray(mask) if not isinstance(mask, torch.Tensor) else mask).to(a.device).bool()
        if se.ndim == m.ndim + 1:
            se = se[m]
        else:
            se = se[m.expand_as(se)]
    mse = se.mean().clamp_min(1e-12)
    return float(-10.0 * torch.log10(mse))
