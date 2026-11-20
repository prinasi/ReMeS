"""Environment lighting and Cook-Torrance shading with precomputed visibility.

Light model (all directions in the Blender world frame, see relight/envmap.py)
-------------------------------------------------------------------------------
* Diffuse: the envmap is integrated into K direction bins (Fibonacci sphere,
  Voronoi cells): ``bins[k] = ∫_cell L(w) dw``. With per-point binary visibility
  V_k, irradiance is ``E = Σ_k V_k max(n·w_k, 0) bins[k]``.
* Specular: split-sum (Karis 2013). A pyramid of GGX-prefiltered envmaps indexed
  by roughness (level 0 = raw map) times the BRDF LUT ``F0·A + B``, attenuated by
  specular occlusion ``SO = Σ V_k D(w_k) / Σ D(w_k)`` around the reflection
  direction using the same visibility bits.

Material convention (metallic-roughness, glTF / Blender Principled BSDF):
    F0 = lerp(0.04, albedo, metallic),  diffuse = (1 - metallic) · albedo / π,
    GGX alpha = roughness².
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from relight.envmap import downsample_envmap, sample_envmap, texel_directions


def fibonacci_sphere(k: int, device="cuda") -> torch.Tensor:
    """K quasi-uniform unit directions [K, 3] (equal-area cells of 4π/K sr)."""
    i = torch.arange(k, device=device, dtype=torch.float32) + 0.5
    z = 1.0 - 2.0 * i / k
    r = torch.sqrt((1.0 - z * z).clamp_min(0))
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=-1)


def ggx_d(cos_h: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    a2 = alpha * alpha
    c2 = cos_h.clamp(0, 1) ** 2
    return a2 / (math.pi * (c2 * (a2 - 1.0) + 1.0) ** 2).clamp_min(1e-12)


# ----------------------------------------------------------------------- BRDF LUT
_LUT_CACHE: dict = {}


def brdf_lut(res: int = 64, samples: int = 1024, device="cuda") -> torch.Tensor:
    """Split-sum DFG LUT [res(roughness), res(NdotV), 2] = (A, B); GGX + Smith (k = α/2)."""
    key = (res, samples, str(device))
    if key in _LUT_CACHE:
        return _LUT_CACHE[key]
    g = torch.Generator(device="cpu").manual_seed(0)
    xi = torch.rand(samples, 2, generator=g).to(device)
    rough = (torch.arange(res, device=device) + 0.5) / res
    nv = (torch.arange(res, device=device) + 0.5) / res
    R, NV = torch.meshgrid(rough, nv, indexing="ij")                    # [res, res]
    alpha = (R * R)[..., None]
    V = torch.stack([torch.sqrt(1 - NV ** 2), torch.zeros_like(NV), NV], -1)[..., None, :]  # [r,r,1,3]
    phi = 2 * math.pi * xi[:, 0]
    cos_t = torch.sqrt((1 - xi[:, 1]) / (1 + (alpha ** 2 - 1) * xi[:, 1]))          # [r,r,S]
    sin_t = torch.sqrt((1 - cos_t ** 2).clamp_min(0))
    H = torch.stack([sin_t * torch.cos(phi), sin_t * torch.sin(phi), cos_t], -1)    # [r,r,S,3]
    VH = (V * H).sum(-1)
    L = 2 * VH[..., None] * H - V
    NL = L[..., 2].clamp_min(0)
    NH = H[..., 2].clamp_min(1e-6)
    VH = VH.clamp_min(0)
    k = alpha / 2
    nvv = NV[..., None]
    G = (nvv / (nvv * (1 - k) + k)) * (NL / (NL * (1 - k) + k))
    Gv = torch.where(NL > 0, G * VH / (NH * nvv), torch.zeros_like(G))
    Fc = (1 - VH) ** 5
    lut = torch.stack([((1 - Fc) * Gv).mean(-1), (Fc * Gv).mean(-1)], -1)
    _LUT_CACHE[key] = lut
    return lut


def lookup_lut(lut: torch.Tensor, nv: torch.Tensor, rough: torch.Tensor) -> torch.Tensor:
    """Bilinear LUT lookup -> [N, 2]."""
    grid = torch.stack([nv.clamp(0, 1) * 2 - 1, rough.clamp(0, 1) * 2 - 1], -1).reshape(1, -1, 1, 2)
    out = F.grid_sample(lut.permute(2, 0, 1)[None], grid, mode="bilinear", padding_mode="border",
                        align_corners=False)
    return out[0, :, :, 0].T


# ----------------------------------------------------------------------- light
class EnvLight:
    """Fixed (non-learnable) environment light derived from an equirect HDR map."""

    LEVELS = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0)

    def __init__(self, env: torch.Tensor, dirs: torch.Tensor, name: str = ""):
        self.name = name
        self.env = env
        self.dirs = dirs
        self.bins = integrate_bins(env, dirs)
        self.levels = prefilter_pyramid(env, self.LEVELS)

    def specular(self, refl: torch.Tensor, rough: torch.Tensor) -> torch.Tensor:
        return specular_lookup(self.levels, self.LEVELS, refl, rough)


def specular_lookup(levels, roughs, refl: torch.Tensor, rough: torch.Tensor) -> torch.Tensor:
    """Prefiltered radiance along reflection dirs [N,3] for roughness [N] -> [N,3]
    (linear interpolation between the two nearest roughness levels)."""
    lv = torch.tensor(roughs, device=refl.device)
    r = rough.clamp(0, 1)
    idx = torch.searchsorted(lv, r.contiguous(), right=True).clamp(1, len(lv) - 1)
    lo, hi = lv[idx - 1], lv[idx]
    t = (r - lo) / (hi - lo)
    out = torch.zeros_like(refl)
    for j, level in enumerate(levels):
        w = torch.where(idx - 1 == j, 1 - t, torch.zeros_like(r)) + torch.where(idx == j, t, torch.zeros_like(r))
        sel = w > 0
        if sel.any():
            out = out.index_add(0, sel.nonzero()[:, 0], w[sel, None] * sample_envmap(level, refl[sel]))
    return out


class LearnableEnvLight(torch.nn.Module):
    """Learnable equirect HDR light, L = exp(θ) with θ [H, W, 3].

    Bins and the prefiltered pyramid are linear in L, so they are rebuilt with
    gradients by ``refresh()`` (call once per iteration before shading) using
    precomputed operators: a texel->bin assignment and per-level GGX weights.
    """

    LEVELS = EnvLight.LEVELS

    def __init__(self, dirs: torch.Tensor, res=(64, 128), init: float | torch.Tensor = 0.5, name="learned"):
        super().__init__()
        self.name = name
        H, W = res
        dev = dirs.device
        init = torch.as_tensor(init, dtype=torch.float32, device=dev)
        self.log_env = torch.nn.Parameter(torch.log(init.clamp_min(1e-4)).expand(H, W, 3).clone())
        self.register_buffer("dirs", dirs)
        tdir, solid = texel_directions(H, W, device=dev)
        tdir = tdir.reshape(-1, 3)
        self.register_buffer("solid", solid.reshape(-1, 1))
        self.register_buffer("owner", (tdir @ dirs.T).argmax(-1))
        self.ops = []
        for k, r in enumerate(self.LEVELS):
            if r == 0.0:
                self.ops.append(None)
                continue
            dst = (H, W) if r <= 0.2 else (H // 2, W // 2)
            ddir, _ = texel_directions(*dst, device=dev)
            cos = ddir.reshape(-1, 3) @ tdir.T
            cos_h = torch.sqrt(((1 + cos) / 2).clamp_min(0))
            w = ggx_d(cos_h, torch.tensor(r * r, device=dev)) * cos.clamp_min(0) * self.solid[:, 0]
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
            self.register_buffer(f"op{k}", w)
            self.ops.append(dst)
        self.refresh()

    @property
    def env(self) -> torch.Tensor:
        return torch.exp(self.log_env)

    def refresh(self):
        env = self.env
        flat = env.reshape(-1, 3)
        self.bins = torch.zeros(self.dirs.shape[0], 3, device=env.device).index_add(0, self.owner, flat * self.solid)
        self.levels = []
        for k, dst in enumerate(self.ops):
            self.levels.append(env if dst is None else (getattr(self, f"op{k}") @ flat).reshape(*dst, 3))
        return self

    def specular(self, refl, rough):
        return specular_lookup(self.levels, self.LEVELS, refl, rough)

    def tv(self) -> torch.Tensor:
        t = self.log_env
        return (t - t.roll(1, dims=1)).abs().mean() + (t[1:] - t[:-1]).abs().mean()

    def saturation(self) -> torch.Tensor:
        """Solid-angle-weighted variance of the log-chroma (log L minus its channel mean) across directions:
        penalises direction-dependent tint (e.g. object colour leaking into weakly observed light directions)
        while leaving the global light colour and the luminance pattern free."""
        t = self.log_env
        c = t - t.mean(-1, keepdim=True)
        w = self.solid.reshape(t.shape[:2] + (1,))
        w = w / w.sum()
        mean = (c * w).sum((0, 1), keepdim=True)
        return ((c - mean) ** 2 * w).sum()

    def white_balance(self) -> torch.Tensor:
        m = (self.env * self.solid.reshape(self.log_env.shape[:2] + (1,))).sum((0, 1))
        return ((m - m.mean()) / m.mean().clamp_min(1e-6)).pow(2).mean()


def integrate_bins(env: torch.Tensor, dirs: torch.Tensor, res=(128, 256), chunk=8192) -> torch.Tensor:
    """∫ L dw over the Voronoi cell of each direction -> [K, 3]."""
    env_s = downsample_envmap(env, *res)
    tdir, solid = texel_directions(*res, device=env.device)
    tdir, w = tdir.reshape(-1, 3), (env_s * solid[..., None]).reshape(-1, 3)
    owner = torch.empty(tdir.shape[0], dtype=torch.long, device=env.device)
    for s in range(0, tdir.shape[0], chunk):
        owner[s:s + chunk] = (tdir[s:s + chunk] @ dirs.T).argmax(-1)
    bins = torch.zeros(dirs.shape[0], 3, device=env.device)
    bins.index_add_(0, owner, w)
    return bins


def prefilter_pyramid(env: torch.Tensor, levels, chunk: int = 2048) -> list[torch.Tensor]:
    """GGX-prefiltered equirect maps (n = v = R assumption) for each roughness level."""
    out = []
    for r in levels:
        if r == 0.0:
            out.append(env)
            continue
        src_res, dst_res = ((128, 256), (64, 128)) if r <= 0.2 else ((64, 128), (32, 64))
        src = downsample_envmap(env, *src_res)
        sdir, solid = texel_directions(*src_res, device=env.device)
        sdir, srad, solid = sdir.reshape(-1, 3), src.reshape(-1, 3), solid.reshape(-1)
        ddir, _ = texel_directions(*dst_res, device=env.device)
        ddir = ddir.reshape(-1, 3)
        alpha = torch.tensor(r * r, device=env.device)
        res = torch.empty_like(ddir)
        for s in range(0, ddir.shape[0], chunk):
            R = ddir[s:s + chunk]
            cos = R @ sdir.T                                             # R·w
            cos_h = torch.sqrt(((1 + cos) / 2).clamp_min(0))            # R·h, h = (R+w)/|R+w|
            w = ggx_d(cos_h, alpha) * cos.clamp_min(0) * solid
            res[s:s + chunk] = (w @ srad) / w.sum(-1, keepdim=True).clamp_min(1e-12)
        out.append(res.reshape(*dst_res, 3))
    return out


# ----------------------------------------------------------------------- shading
def shade(albedo, rough, metal, n, v, vis, light: EnvLight, lut: torch.Tensor,
          specular_occlusion: bool = True, layering: bool = True, irr_extra: torch.Tensor | None = None,
          occluded=None) -> dict:
    """Outgoing linear radiance for N points.

    albedo [N,3], rough [N], metal [N], n (unit shading normal) [N,3],
    v (unit direction to the camera) [N,3], vis [N,K] in [0,1].
    layering: attenuate diffuse by the specular layer's directional albedo (F0·A + B),
    as Blender's Principled BSDF (v2) does for energy conservation (Phase 5 parity).
    irr_extra [N,3]: additional diffuse irradiance (Phase 4 one-bounce indirect, relight/indirect.py).
    occluded: callable ŵ [N,K] -> [N,3], radiance arriving from the blocked part of the specular lobe
    (normalised lobe weights ŵ; relight/interreflect.py). None = blocked directions are black.
    """
    cos = (n @ light.dirs.T).clamp_min(0)                               # [N, K]
    irradiance = (vis * cos) @ light.bins                               # [N, 3]
    if irr_extra is not None:
        irradiance = irradiance + irr_extra
    diffuse = (1 - metal)[:, None] * albedo / math.pi * irradiance

    nv = (n * v).sum(-1).clamp(1e-4, 1)
    refl = F.normalize(2 * nv[:, None] * n - v, dim=-1)
    pref = light.specular(refl, rough)
    ab = lookup_lut(lut, nv, rough)
    f0 = 0.04 * (1 - metal)[:, None] + albedo * metal[:, None]
    spec_albedo = f0 * ab[:, :1] + ab[:, 1:]
    spec = pref * spec_albedo
    if layering:
        diffuse = diffuse * (1 - spec_albedo).clamp_min(0)
    if specular_occlusion:
        cos_r = refl @ light.dirs.T
        cos_h = torch.sqrt(((1 + cos_r) / 2).clamp_min(1e-8))            # finite gradient at grazing angles
        w = ggx_d(cos_h, (rough * rough).clamp_min(1e-3)[:, None]) * cos_r.clamp_min(0) + 1e-8
        wn = w / w.sum(-1, keepdim=True)
        so = (vis * wn).sum(-1)
        spec = spec * so[:, None]
        if occluded is not None:
            spec = spec + spec_albedo * occluded(wn)
    return {"rgb": diffuse + spec, "diffuse": diffuse, "specular": spec, "irradiance": irradiance}
