"""Deferred reflection-direction specular for the T²-Splat geometry stage.

    mode 1:  color = SH_diffuse(x) + s(x) · L_spec(reflect(-v, n(x)), ρ(x))
    mode 2:  color = SH_diffuse(x) + L_spec(r, ρ) · (F0(x)·A(n·v, ρ) + B(n·v, ρ))     (split-sum DFG;
             F0 = s, init ≈ 0.04: dielectric Fresnel sheen that grows at grazing angles)

* SH is capped at a low degree (``deferred_sh_degree``, default 0 = view-independent),
  so view dependence must come from the specular branch, whose lookup direction
  depends on the rendered normal: glossy highlights then *constrain* the normals
  instead of being faked by geometry (the failure found by the geometry diagnostic).
* s (specular tint) and ρ (roughness) are per-vertex attributes stored in the last
  slot of ``TriangleModel._features_rest`` (unused by SH at the capped degree), so
  densification / pruning / heartbeat / RDT carry them automatically. Zero init
  means s ≈ 0.1, ρ = 0.5.
* They are rendered with the T²-Splat rasterizer itself (second pass with
  ``override_color``, black background), i.e. alpha-blended like colour.
* L_spec is a learnable log-space equirect map with a GGX-prefiltered pyramid
  (``relight.shading.LearnableEnvLight``); it has its own optimizer.
* Composition space: by default additive in display (gamma) space. With ``linear``,
  the SH output of a black-background pass is treated as linear radiance and
  ``display = (diffuse + specular)^(1/2.2) + bg·(1 - α)`` (matches how the data were
  rendered); the light can then be initialised from / frozen to a real HDR map.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from relight.shading import LearnableEnvLight, brdf_lut, fibonacci_sphere, lookup_lut

TINT_OFFSET = {1: -2.2, 2: -3.2}    # zero init -> s ≈ 0.10 (mode 1) / F0 ≈ 0.04 (mode 2)
ROUGH_RANGE = (0.03, 1.0)


class DeferredSpecular:
    def __init__(self, env_res: int = 64, lr: float = 0.01, init: float = 0.5, mode: int = 1,
                 env_tv: float = 0.0, linear: bool = False, env_init: str = "", env_freeze: bool = False,
                 device="cuda"):
        self.mode = mode
        self.env_tv = env_tv
        self.linear = linear
        self.light = LearnableEnvLight(fibonacci_sphere(64, device=device), (env_res, 2 * env_res), init=init).to(device)
        if env_init:
            from relight.envmap import downsample_envmap, load_envmap
            env = downsample_envmap(load_envmap(env_init, device), env_res, 2 * env_res)
            self.light.log_env.data.copy_(torch.log(env.clamp_min(1e-4)))
        self.env_freeze = env_freeze
        self.light.log_env.requires_grad_(not env_freeze)
        self.opt = torch.optim.Adam([self.light.log_env], lr=lr)
        self.lut = brdf_lut(device=device) if mode == 2 else None

    # ------------------------------------------------------------------ attributes
    def vertex_attributes(self, triangles) -> torch.Tensor:
        """Per-vertex [s, ρ, 0] in [0, 1] from the last _features_rest slot."""
        raw = triangles._features_rest[:, -1, :]                         # [V, 3]
        s = torch.sigmoid(raw[:, 0] + TINT_OFFSET[self.mode])
        lo, hi = ROUGH_RANGE
        rho = lo + (hi - lo) * torch.sigmoid(raw[:, 1])
        return torch.stack([s, rho, torch.zeros_like(s)], -1)

    @staticmethod
    def pixel_rays(cam, H: int, W: int, device="cuda") -> torch.Tensor:
        """World-space unit ray directions [H, W, 3] for a T²-Splat (OpenCV-convention) Camera."""
        fx = 0.5 * W / math.tan(0.5 * cam.FoVx)
        fy = 0.5 * H / math.tan(0.5 * cam.FoVy)
        j, i = torch.meshgrid(torch.arange(H, device=device) + 0.5, torch.arange(W, device=device) + 0.5,
                              indexing="ij")
        d_cam = torch.stack([(i - 0.5 * W) / fx, (j - 0.5 * H) / fy, torch.ones_like(i)], -1)
        c2w = torch.inverse(cam.world_view_transform.T)
        return F.normalize(d_cam @ c2w[:3, :3].T, dim=-1)

    # ------------------------------------------------------------------ shading
    def shade(self, cam, triangles, pipe, render_fn, render_pkg) -> dict:
        """Specular image [3, H, W] to add to render_pkg['render'] (+ maps for logging)."""
        feats = render_fn(cam, triangles, pipe, torch.zeros(3, device="cuda"),
                          override_color=self.vertex_attributes(triangles))["render"]     # [3, H, W]
        alpha = render_pkg["rend_alpha"]                                                  # [1, H, W]
        n = F.normalize(render_pkg["rend_normal"], dim=0)                                 # world, [3, H, W]
        _, H, W = n.shape
        d = self.pixel_rays(cam, H, W)                                                    # [H, W, 3]
        n_hw = n.permute(1, 2, 0)
        r = F.normalize(d - 2 * (d * n_hw).sum(-1, keepdim=True) * n_hw, dim=-1)
        rho = (feats[1] / alpha[0].clamp_min(1e-4)).clamp(0, 1)
        self.light.refresh()
        pref = self.light.specular(r.reshape(-1, 3), rho.reshape(-1)).reshape(H, W, 3).permute(2, 0, 1)
        if self.mode == 1:
            spec = feats[0:1] * pref                                                      # s is alpha-weighted
        else:
            nv = (-(d * n_hw).sum(-1)).clamp(1e-4, 1)
            ab = lookup_lut(self.lut, nv.reshape(-1), rho.reshape(-1)).reshape(H, W, 2).permute(2, 0, 1)
            f0 = feats[0:1] / alpha.clamp_min(1e-4)
            spec = alpha * pref * (f0 * ab[0:1] + ab[1:2])
        reg = self.env_tv * self.light.tv() if self.env_tv > 0 else spec.new_zeros(())
        return {"specular": spec, "tint": feats[0:1], "roughness": rho[None], "reg": reg}

    def compose(self, image, spec, render_pkg, bg):
        """Final display image from the SH pass (``image``) and the specular term."""
        if not self.linear:
            return image + spec
        alpha = render_pkg["rend_alpha"]
        lin = (image - bg.view(3, 1, 1) * (1 - alpha)).clamp_min(0) + spec     # remove the rasterizer's bg
        return (lin + 1e-6) ** (1 / 2.2) + bg.view(3, 1, 1) * (1 - alpha)

    def step(self):
        if self.env_freeze:
            return
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    def state_dict(self) -> dict:
        return {"log_env": self.light.log_env.detach().cpu()}
