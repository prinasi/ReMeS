"""Differentiable cast shadows for environment lighting via batched orthographic shadow maps.

For K light directions w (unit, pointing from the surface towards the light), the mesh is
rasterised orthographically looking along -w (one batched nvdiffrast call). The z-buffer
keeps the surface nearest to the light; its height h = (p - c)·w is interpolated, so it is
differentiable w.r.t. the occluder's vertices. A receiver point x is visible from w with soft
visibility

    V(x, w) = sigmoid((h_x - h_map(x) + bias) / tau),   h_x = (x - c)·w

which is differentiable w.r.t. both the receiver position and the occluder surface: wrong
geometry produces wrong shadows *and* gradients that move caster and receiver.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import nvdiffrast.torch as dr

from relight.gbuffer import get_context


def _basis(w: torch.Tensor):
    """Orthonormal (u, v) completing each direction w [K, 3]."""
    helper = torch.where(w[:, 2:3].abs() < 0.9, torch.tensor([0.0, 0.0, 1.0], device=w.device),
                         torch.tensor([1.0, 0.0, 0.0], device=w.device))
    u = F.normalize(torch.cross(helper, w, dim=-1), dim=-1)
    v = torch.cross(w, u, dim=-1)
    return u, v


class ShadowMapper:
    def __init__(self, center: torch.Tensor, radius: float, res: int = 512, bias_texels: float = 3.0,
                 tau_texels: float = 0.5):
        self.c = center.float()
        self.R = float(radius)
        self.res = res
        texel = 2 * self.R / res
        self.bias = bias_texels * texel
        self.tau = tau_texels * texel

    def render(self, V: torch.Tensor, faces: torch.Tensor, w: torch.Tensor, chunk: int = 64) -> dict:
        """Height maps [K, S, S] for directions w [K, 3] (batched in chunks of `chunk` directions)."""
        if w.shape[0] > chunk:
            parts = [self.render(V, faces, w[s:s + chunk], chunk) for s in range(0, w.shape[0], chunk)]
            return {k: torch.cat([p[k] for p in parts], 0) for k in ("height", "u", "v", "w")}
        u, v = _basis(w)
        p = V - self.c                                                       # [N, 3]
        x = p @ u.T / self.R                                                 # [N, K]
        y = p @ v.T / self.R
        h = p @ w.T                                                          # height towards the light
        z = -h / (2 * self.R)                                                # nearest to light = smallest z
        clip = torch.stack([x, y, z, torch.ones_like(x)], -1).permute(1, 0, 2).contiguous()   # [K, N, 4]
        rast, _ = dr.rasterize(get_context(V.device), clip, faces, resolution=[self.res, self.res])
        hm, _ = dr.interpolate(h.T.contiguous()[..., None], rast, faces)    # [K, S, S, 1]
        cover = rast[..., 3:4] > 0
        # uncovered texels: far below everything; antialiasing then blends silhouettes against it,
        # which makes shadow *boundaries* differentiable w.r.t. the occluder's silhouette vertices
        hm = torch.where(cover, hm, torch.full_like(hm, -2 * self.R))
        hm = dr.antialias(hm.contiguous(), rast, clip, faces)
        return {"height": hm[..., 0], "u": u, "v": v, "w": w}

    def visibility(self, maps: dict, X: torch.Tensor) -> torch.Tensor:
        """Soft visibility [N, K] of points X [N, 3] for the directions of `maps`."""
        p = X - self.c
        u, v, w = maps["u"], maps["v"], maps["w"]
        gx = (p @ u.T) / self.R                                              # [N, K] in [-1, 1]
        gy = (p @ v.T) / self.R
        hx = p @ w.T
        K = w.shape[0]
        # nvdiffrast's framebuffer row 0 is clip y = -1, i.e. grid_sample y = -1
        grid = torch.stack([gx, gy], -1).permute(1, 0, 2)[:, :, None, :]     # [K, N, 1, 2]
        hmap = maps["height"][:, None]                                       # [K, 1, S, S]
        hs = F.grid_sample(hmap, grid, mode="bilinear", padding_mode="border", align_corners=False)[:, 0, :, 0]
        return torch.sigmoid((hx.T - hs + self.bias) / self.tau).T          # [N, K]


def sample_directions(bins: torch.Tensor, dirs: torch.Tensor, k: int, generator=None):
    """Importance-sample k light bins with probability ∝ bin luminance (with replacement).

    Returns indices [k] and probabilities [k] for the unbiased estimator Σ L V cos / (k p).
    If k >= the number of bins, all bins are used deterministically (p = 1/K, weights = bins).
    """
    if k >= bins.shape[0]:                     # deterministic quadrature over all bins (no sampling noise)
        idx = torch.arange(bins.shape[0], device=bins.device)
        return idx, torch.full((bins.shape[0],), 1.0 / bins.shape[0], device=bins.device)
    lum = bins.detach().mean(-1).clamp_min(0) + 1e-8
    p = lum / lum.sum()
    idx = torch.multinomial(p, k, replacement=True, generator=generator)
    return idx, p[idx]
