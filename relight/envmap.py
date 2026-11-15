"""Equirectangular HDR environment maps in Blender's world convention.

World frame: Blender (Z up). A direction d = (x, y, z) maps to equirect
coordinates (Cycles ``direction_to_equirectangular``):

    u = 0.5 - atan2(y, x) / (2*pi)          # u in [0,1), left -> right
    v = 0.5 + atan2(z, hypot(x, y)) / pi    # v in [0,1], bottom -> top

Image row index counts from the top, so t = 1 - v is used for lookups.
The image centre (u = 0.5) looks along +X, the top row is +Z.
This convention is verified against Cycles by scripts/check_envmap_convention.py.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from relight.io import read_exr_rgb


def load_envmap(path: str, device: str | torch.device = "cuda") -> torch.Tensor:
    """Load an equirect HDR map as float32 [H, W, 3] (negative values clamped)."""
    return torch.from_numpy(read_exr_rgb(path)).to(device)


def dir_to_uv(d: torch.Tensor) -> torch.Tensor:
    """Directions [..., 3] (any length) -> (u, t) in [0,1]^2, t measured from the top row."""
    x, y, z = d.unbind(-1)
    u = 0.5 - torch.atan2(y, x) / (2.0 * math.pi)
    v = 0.5 + torch.atan2(z, torch.sqrt(x * x + y * y)) / math.pi
    return torch.stack([u % 1.0, 1.0 - v], dim=-1)


def uv_to_dir(uv: torch.Tensor) -> torch.Tensor:
    """(u, t) [..., 2] -> unit directions [..., 3]; inverse of dir_to_uv."""
    u, t = uv.unbind(-1)
    phi = (0.5 - u) * 2.0 * math.pi              # azimuth = atan2(y, x)
    elev = (0.5 - t) * math.pi                   # elevation = atan2(z, r)
    c = torch.cos(elev)
    return torch.stack([c * torch.cos(phi), c * torch.sin(phi), torch.sin(elev)], dim=-1)


def sample_envmap(env: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Bilinear lookup of env [H, W, C] along directions d [..., 3] (u wraps, t clamps)."""
    H, W, C = env.shape
    uv = dir_to_uv(d.reshape(-1, 3))
    x = uv[:, 0] * W - 0.5
    y = uv[:, 1] * H - 0.5
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x0 = x0.long()
    y0 = y0.long()
    x1 = (x0 + 1) % W
    x0 = x0 % W
    y1 = (y0 + 1).clamp(0, H - 1)
    y0 = y0.clamp(0, H - 1)
    flat = env.reshape(-1, C)
    out = ((1 - fx) * (1 - fy) * flat[y0 * W + x0] + fx * (1 - fy) * flat[y0 * W + x1]
           + (1 - fx) * fy * flat[y1 * W + x0] + fx * fy * flat[y1 * W + x1])
    return out.reshape(*d.shape[:-1], C)


def texel_directions(H: int, W: int, device="cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Texel-centre directions [H, W, 3] and solid angles [H, W] of an equirect grid."""
    t = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    u = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
    tt, uu = torch.meshgrid(t, u, indexing="ij")
    dirs = uv_to_dir(torch.stack([uu, tt], dim=-1))
    elev = (0.5 - tt) * math.pi
    # exact solid angle of a latitude band slice
    band = torch.sin(elev + 0.5 * math.pi / H) - torch.sin(elev - 0.5 * math.pi / H)
    solid = band * (2.0 * math.pi / W)
    return dirs, solid


def downsample_envmap(env: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Energy-preserving (area-average) resize of an equirect map to [H, W, C]."""
    x = env.permute(2, 0, 1)[None]
    y = F.adaptive_avg_pool2d(x, (H, W))
    return y[0].permute(1, 2, 0).contiguous()


def diffuse_irradiance(env: torch.Tensor, normals: torch.Tensor, res: tuple[int, int] = (128, 256),
                       chunk: int = 8192) -> torch.Tensor:
    """Unshadowed irradiance E(n) = ∫ L(w) max(n·w, 0) dw for normals [..., 3] -> [..., 3].

    Brute-force quadrature over an area-averaged copy of the map; exact up to the
    texel discretisation, used as a reference for Lambertian shading.
    """
    env_s = downsample_envmap(env, *res)
    dirs, solid = texel_directions(*res, device=env.device)
    w = (env_s * solid[..., None]).reshape(-1, 3)          # radiance * dω
    dirs = dirs.reshape(-1, 3)
    n = F.normalize(normals.reshape(-1, 3), dim=-1)
    out = torch.empty_like(n)
    for s in range(0, n.shape[0], chunk):
        cos = (n[s:s + chunk] @ dirs.T).clamp_min(0.0)
        out[s:s + chunk] = cos @ w
    return out.reshape(normals.shape)


def envmap_to_numpy(env: torch.Tensor) -> np.ndarray:
    return env.detach().float().cpu().numpy()
