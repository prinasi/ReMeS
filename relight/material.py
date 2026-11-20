"""Learnable PBR materials on a frozen mesh.

Base: per-vertex logits (albedo [V,3], roughness [V,1], metallic [V,1]) interpolated
with barycentrics. Faces cover ~2.5 px on Synthetic4Relight, so per-face R×R textures
would be heavily over-parameterised; vertices are shared between faces, which also
gives implicit cross-face smoothing on these poorly connected meshes.

Optional: per-face residual logits [F,R,R,C] in T²-Splat's barycentric texel layout
(row = w2, column = w1, texel spacing 1/(R-1)); disabled when ``face_res == 0``.

Optional (exclusive with per-face tiles): a shared UV atlas residual [C, A, A] sampled bilinearly at the
interpolated per-corner UVs (``uv`` [F,3,2], Blender/bottom-left convention; image row 0 = top, i.e. v = 1),
e.g. from relight/blender/uv_unwrap.py. Texels are continuous within a chart, so filtering / mipmaps work across
triangle edges and the export is one standard texture set.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

ALBEDO_RANGE = (0.02, 0.98)
ROUGH_RANGE = (0.05, 1.0)


def _logit(p: float) -> float:
    return float(torch.logit(torch.tensor(p)))


class PBRMaterial(nn.Module):
    def __init__(self, faces: torch.Tensor, num_vertices: int, face_res: int = 0,
                 albedo_init: float | torch.Tensor = 0.5, rough_init: float = 0.5,
                 metal_init: float = 0.0, learn_metallic: bool = False, learn_normals: bool = False,
                 atlas_res: int = 0, uv: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("faces", faces.long())
        dev = faces.device
        a = torch.as_tensor(albedo_init, dtype=torch.float32, device=dev)
        a = a.expand(num_vertices, 3) if a.ndim < 2 else a
        lo, hi = ALBEDO_RANGE
        self.albedo = nn.Parameter(torch.logit(((a - lo) / (hi - lo)).clamp(1e-3, 1 - 1e-3)).clone())
        rlo, rhi = ROUGH_RANGE
        self.rough = nn.Parameter(torch.full((num_vertices, 1), _logit((rough_init - rlo) / (rhi - rlo)), device=dev))
        self.learn_metallic = learn_metallic
        m0 = min(max(metal_init, 1e-3), 1 - 1e-3)
        self.metal = nn.Parameter(torch.full((num_vertices, 1), _logit(m0), device=dev),
                                  requires_grad=learn_metallic)
        self.face_res = face_res
        if face_res > 0:
            self.residual = nn.Parameter(torch.zeros(faces.shape[0], face_res, face_res, 5, device=dev))
        else:
            self.register_parameter("residual", None)
        self.atlas_res = atlas_res
        if atlas_res > 0:
            assert face_res == 0 and uv is not None, "atlas needs UVs and excludes per-face tiles"
            self.register_buffer("uv", uv.float().to(dev))
            self.atlas = nn.Parameter(torch.zeros(1, 5, atlas_res, atlas_res, device=dev))
            self.register_buffer("atlas_cover", torch.zeros(atlas_res, atlas_res, dtype=torch.bool, device=dev))
        else:
            self.register_parameter("atlas", None)
        # learnable shading normals: per-vertex offset added to the geometric vertex normal (exported as NORMAL)
        if learn_normals:
            self.normal_offset = nn.Parameter(torch.zeros(num_vertices, 3, device=dev))
        else:
            self.register_parameter("normal_offset", None)

    def shading_normals(self, vertex_normals: torch.Tensor) -> torch.Tensor:
        """Per-vertex shading normals [V, 3]: geometric normals, plus the learned offset if enabled."""
        if self.normal_offset is None:
            return vertex_normals
        return torch.nn.functional.normalize(vertex_normals + self.normal_offset, dim=-1)

    # ------------------------------------------------------------------ activations
    @staticmethod
    def act_albedo(x):
        lo, hi = ALBEDO_RANGE
        return lo + (hi - lo) * torch.sigmoid(x)

    @staticmethod
    def act_rough(x):
        lo, hi = ROUGH_RANGE
        return lo + (hi - lo) * torch.sigmoid(x)

    def vertex_logits(self) -> torch.Tensor:
        return torch.cat([self.albedo, self.rough, self.metal], dim=-1)  # [V, 5]

    def _activate(self, z: torch.Tensor) -> dict:
        metal = torch.sigmoid(z[:, 4]) if self.learn_metallic else torch.zeros_like(z[:, 4])
        return {"albedo": self.act_albedo(z[:, :3]), "rough": self.act_rough(z[:, 3]), "metal": metal}

    def vertex_values(self) -> dict:
        return self._activate(self.vertex_logits())

    def sample(self, face_id: torch.Tensor, bary: torch.Tensor) -> dict:
        tri = self.faces[face_id]                                        # [N, 3]
        z = (bary[:, :, None] * self.vertex_logits()[tri]).sum(1)        # [N, 5]
        if self.residual is not None:
            z = z + self._sample_residual(face_id, bary)
        if self.atlas is not None:
            z = z + self._sample_atlas(face_id, bary)
        return self._activate(z)

    def _sample_atlas(self, face_id, bary):
        uv = (bary[:, :, None] * self.uv[face_id]).sum(1)                # [N, 2]
        grid = torch.stack([uv[:, 0] * 2 - 1, 1 - 2 * uv[:, 1]], -1).reshape(1, -1, 1, 2)
        out = torch.nn.functional.grid_sample(self.atlas, grid, mode="bilinear", padding_mode="border",
                                              align_corners=False)
        return out[0, :, :, 0].T                                         # [N, 5]

    @torch.no_grad()
    def uv_raster(self, res: int | None = None):
        """Rasterise the UV layout: per texel (face id [-1 = empty], barycentrics) at atlas resolution."""
        import nvdiffrast.torch as dr
        from relight.gbuffer import get_context
        R = res or self.atlas_res
        F_ = self.uv.shape[0]
        uv = self.uv.reshape(-1, 2)
        clip = torch.stack([uv[:, 0] * 2 - 1, 1 - 2 * uv[:, 1], torch.zeros_like(uv[:, 0]),
                            torch.ones_like(uv[:, 0])], -1)[None].contiguous()
        tri = torch.arange(3 * F_, device=uv.device, dtype=torch.int32).reshape(F_, 3)
        rast, _ = dr.rasterize(get_context(uv.device), clip, tri, resolution=[R, R])
        r = rast[0]
        fid = r[..., 3].long() - 1
        bary = torch.stack([r[..., 0], r[..., 1], 1 - r[..., 0] - r[..., 1]], -1)
        return fid, bary

    def atlas_smoothness(self) -> torch.Tensor:
        """Mean |Δ| of the atlas residual between 4-neighbour texels that are both covered (within charts)."""
        t, c = self.atlas[0], self.atlas_cover
        dx = (t[:, :, 1:] - t[:, :, :-1]).abs().mean(0)[c[:, 1:] & c[:, :-1]]
        dy = (t[:, 1:] - t[:, :-1]).abs().mean(0)[c[1:] & c[:-1]]
        return (dx.sum() + dy.sum()) / max(dx.numel() + dy.numel(), 1)

    @torch.no_grad()
    def bake_atlas(self, res: int | None = None, dilate: int = 8, chunk: int = 1 << 20) -> dict:
        """Final albedo / rough / metal textures [R, R, ·] (row 0 = top) for export: base + residual at every
        covered texel, then ``dilate`` rounds of neighbour filling into the empty texels (filtering margins)."""
        fid, bary = self.uv_raster(res)
        R = fid.shape[0]
        out = torch.zeros(R * R, 5, device=fid.device)
        cov = (fid >= 0).reshape(-1)
        idx = cov.nonzero()[:, 0]
        f, b = fid.reshape(-1), bary.reshape(-1, 3)
        for s in range(0, idx.numel(), chunk):
            i = idx[s:s + chunk]
            v = self.sample(f[i], b[i])
            out[i] = torch.cat([v["albedo"], v["rough"][:, None], v["metal"][:, None]], -1)
        img = out.reshape(R, R, 5).permute(2, 0, 1)[None]
        m = cov.reshape(1, 1, R, R).float()
        for _ in range(dilate):
            s_ = torch.nn.functional.avg_pool2d(img * m, 3, 1, 1)
            w = torch.nn.functional.avg_pool2d(m, 3, 1, 1)
            fill = (m == 0) & (w > 0)
            img = torch.where(fill, s_ / w.clamp_min(1e-8), img)
            m = torch.where(fill, torch.ones_like(m), m)
        img = img[0].permute(1, 2, 0)
        return {"albedo": img[..., :3].cpu().numpy(), "rough": img[..., 3].cpu().numpy(),
                "metal": img[..., 4].cpu().numpy(), "coverage": float(cov.float().mean())}

    def _sample_residual(self, face_id, bary):
        R = self.face_res
        tex = self.residual[face_id]                                     # [N, R, R, 5]
        if R == 1:
            return tex[:, 0, 0]
        x = bary[:, 1] * (R - 1)
        y = bary[:, 2] * (R - 1)
        x0 = x.floor().clamp(0, R - 2).long()
        y0 = y.floor().clamp(0, R - 2).long()
        fx = (x - x0)[:, None]
        fy = (y - y0)[:, None]
        ar = torch.arange(len(face_id), device=face_id.device)
        t00, t01 = tex[ar, y0, x0], tex[ar, y0, x0 + 1]
        t10, t11 = tex[ar, y0 + 1, x0], tex[ar, y0 + 1, x0 + 1]
        return (1 - fy) * ((1 - fx) * t00 + fx * t01) + fy * ((1 - fx) * t10 + fx * t11)

    @torch.no_grad()
    def face_texel_values(self, T: int, chunk: int = 65536) -> dict:
        """Final per-face texel values [F, T, T, ·] in the export tile layout (utils/glb_export.bake_tiles):
        texel (row, col) at w1 = col/(T-1), w2 = row/(T-1); texels outside the triangle (row+col > T-1)
        are reflected across the hypotenuse. Includes per-face residual textures if present."""
        v, u = np.mgrid[:T, :T].astype(np.float32) / (T - 1)
        outside = u + v > 1
        b, c = np.where(outside, 1 - v, u), np.where(outside, 1 - u, v)
        bary = torch.as_tensor(np.stack((1 - b - c, b, c), -1).reshape(-1, 3), device=self.faces.device)  # [T*T, 3]
        F_ = self.faces.shape[0]
        out = {"albedo": torch.empty(F_, T, T, 3), "rough": torch.empty(F_, T, T), "metal": torch.empty(F_, T, T)}
        for s in range(0, F_, chunk):
            fid = torch.arange(s, min(s + chunk, F_), device=self.faces.device)
            n = fid.shape[0]
            vals = self.sample(fid.repeat_interleave(T * T), bary.repeat(n, 1))
            out["albedo"][s:s + n] = vals["albedo"].reshape(n, T, T, 3).cpu()
            out["rough"][s:s + n] = vals["rough"].reshape(n, T, T).cpu()
            out["metal"][s:s + n] = vals["metal"].reshape(n, T, T).cpu()
        return {k: v.numpy() for k, v in out.items()}

    # ------------------------------------------------------------------ regularisers
    def edge_smoothness(self, edges: torch.Tensor, weights=(1.0, 1.0), edge_w: torch.Tensor | None = None,
                        normal_weights=(0.0, 0.0)) -> torch.Tensor:
        """Mean |Δ| of activated albedo / roughness across mesh edges [E, 2], optionally weighted per edge
        (edge_w [E], e.g. image-edge-aware exp(-|Δc|/σ)). normal_weights = (smoothness of the learned normal
        offset across edges, |offset|² magnitude penalty)."""
        v = self.vertex_values()
        w = edge_w[:, None] if edge_w is not None else 1.0
        diff = lambda t: (t[edges[:, 0]] - t[edges[:, 1]]).reshape(edges.shape[0], -1).abs()
        da = (w * diff(v["albedo"])).mean()
        dr = (w * diff(v["rough"])).mean()
        out = weights[0] * da + weights[1] * dr
        if self.normal_offset is not None and any(normal_weights):
            o = self.normal_offset
            out = out + normal_weights[0] * (w * diff(o)).mean() \
                + normal_weights[1] * (o ** 2).sum(-1).mean()
        return out


def unique_edges(faces: torch.Tensor) -> torch.Tensor:
    f = faces.long()
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
    e = torch.sort(e, dim=1).values
    return torch.unique(e, dim=0)
