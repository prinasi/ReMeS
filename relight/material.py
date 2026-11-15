"""Learnable PBR materials on a frozen mesh.

Base: per-vertex logits (albedo [V,3], roughness [V,1], metallic [V,1]) interpolated
with barycentrics. Faces cover ~2.5 px on Synthetic4Relight, so per-face R×R textures
would be heavily over-parameterised; vertices are shared between faces, which also
gives implicit cross-face smoothing on these poorly connected meshes.

Optional: per-face residual logits [F,R,R,C] in T²-Splat's barycentric texel layout
(row = w2, column = w1, texel spacing 1/(R-1)); disabled when ``face_res == 0``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

ALBEDO_RANGE = (0.02, 0.98)
ROUGH_RANGE = (0.05, 1.0)


def _logit(p: float) -> float:
    return float(torch.logit(torch.tensor(p)))


class PBRMaterial(nn.Module):
    def __init__(self, faces: torch.Tensor, num_vertices: int, face_res: int = 0,
                 albedo_init: float | torch.Tensor = 0.5, rough_init: float = 0.5,
                 metal_init: float = 0.0, learn_metallic: bool = False):
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
        return self._activate(z)

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

    # ------------------------------------------------------------------ regularisers
    def edge_smoothness(self, edges: torch.Tensor, weights=(1.0, 1.0)) -> torch.Tensor:
        """Mean |Δ| of activated albedo / roughness across mesh edges [E, 2]."""
        v = self.vertex_values()
        da = (v["albedo"][edges[:, 0]] - v["albedo"][edges[:, 1]]).abs().mean()
        dr = (v["rough"][edges[:, 0]] - v["rough"][edges[:, 1]]).abs().mean()
        return weights[0] * da + weights[1] * dr


def unique_edges(faces: torch.Tensor) -> torch.Tensor:
    f = faces.long()
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
    e = torch.sort(e, dim=1).values
    return torch.unique(e, dim=0)
