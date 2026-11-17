"""Phase 4: mesh-native one-bounce indirect diffuse illumination on the frozen mesh.

The per-vertex visibility already tells which of the K Fibonacci directions are blocked. For every vertex v
and every blocked direction w_k above its tangent plane, one closest-hit ray (same origin offsets as the
visibility rays) gives the hit triangle and barycentrics. The light arriving from w_k is then the outgoing
diffuse radiance of the hit point, i.e. the *same* material textures lit by the same environment:

    E_ind(v) = Σ_k  cos(n_v, w_k) · ΔΩ · B(hit_k),     ΔΩ = 4π / K
    B(u)     = (1 - metal_u) · albedo_u / π · E_dir(u),  E_dir(u) = Σ_k V_uk max(n_u·w_k, 0) L_k ΔΩ

(one bounce, Lambertian outgoing radiance at the hit, no recursion). The geometry part is a fixed sparse
matrix M [V, V] with M[v, j] = Σ_k cos·ΔΩ·bary_j, so every iteration costs one sparse mat-mul and E_ind is
differentiable w.r.t. the albedo (``detach_material=False``) and the light (learned light, Phase 3).
Back-face hits (inside a closed surface) carry no light.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from relight.mesh import FrozenMesh
from relight.raytrace import MeshTracer
from relight.visibility import unpack_bits


class IndirectLight:
    def __init__(self, transport: torch.Tensor, direct_op: torch.Tensor):
        self.M = transport                    # sparse CSR [V, V]
        self.A = direct_op                    # [V, K] half: V_vk · max(n_v·w_k, 0)
        self._cache = {}

    # ------------------------------------------------------------------ construction
    @classmethod
    @torch.no_grad()
    def build(cls, mesh: FrozenMesh, vis_packed: torch.Tensor, dirs: torch.Tensor, eps: float, tmin: float,
              chunk_vertices: int = 8192) -> "IndirectLight":
        V, K = mesh.vertices.shape[0], dirs.shape[0]
        tracer = MeshTracer(mesh.vertices, mesh.faces)
        faces = mesh.faces.long()
        fv = mesh.vertices[faces]
        fn = torch.nn.functional.normalize(torch.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0], dim=-1), dim=-1)
        d_omega = 4 * math.pi / K
        rows, cols, vals = [], [], []
        A = torch.empty(V, K, dtype=torch.float16, device=mesh.vertices.device)
        for s in range(0, V, chunk_vertices):
            x = mesh.vertices[s:s + chunk_vertices]
            n = mesh.vertex_normals[s:s + chunk_vertices]
            cos = n @ dirs.T                                                   # [B, K]
            vis = unpack_bits(vis_packed[s:s + chunk_vertices])
            A[s:s + chunk_vertices] = (vis.float() * cos.clamp_min(0)).half()
            bi, ki = ((cos > 0) & ~vis).nonzero(as_tuple=True)
            if bi.numel() == 0:
                continue
            d = dirs[ki]
            o = x[bi] + eps * n[bi] + tmin * d
            hit, _, fid, bary = tracer.closest_hit(o, d)
            front = hit & ((fn[fid.clamp_min(0)] * d).sum(-1) < 0)               # light leaves the hit's front side
            bi, ki, fid, bary = bi[front], ki[front], fid[front], bary[front]
            w = cos[bi, ki] * d_omega
            rows.append((bi + s).repeat_interleave(3))
            cols.append(faces[fid].reshape(-1))
            vals.append((w[:, None] * bary).reshape(-1))
        idx = torch.stack([torch.cat(rows), torch.cat(cols)])
        M = torch.sparse_coo_tensor(idx, torch.cat(vals), (V, V)).coalesce().to_sparse_csr()
        return cls(M, A)

    @classmethod
    def cached(cls, mesh: FrozenMesh, vis_packed: torch.Tensor, dirs: torch.Tensor, path: str | Path,
               eps: float, tmin: float) -> "IndirectLight":
        path = Path(path)
        if path.is_file():
            d = torch.load(path, map_location=mesh.vertices.device)
            if d["V"] == mesh.vertices.shape[0] and d["K"] == dirs.shape[0]:
                M = torch.sparse_csr_tensor(d["crow"], d["col"], d["val"], (d["V"], d["V"]))
                return cls(M, d["A"])
        ind = cls.build(mesh, vis_packed, dirs, eps, tmin)
        torch.save({"V": mesh.vertices.shape[0], "K": dirs.shape[0], "crow": ind.M.crow_indices(),
                    "col": ind.M.col_indices(), "val": ind.M.values(), "A": ind.A}, path)
        return ind

    # ------------------------------------------------------------------ evaluation
    def direct_irradiance(self, light, chunk: int = 1 << 16) -> torch.Tensor:
        """Per-vertex direct irradiance [V, 3] (differentiable w.r.t. learnable light bins)."""
        return torch.cat([self.A[s:s + chunk].float() @ light.bins for s in range(0, self.A.shape[0], chunk)])

    def radiosity(self, mat, light, albedo_scale: torch.Tensor | None = None, detach_material: bool = True):
        v = mat.vertex_values()
        alb, metal = v["albedo"], v["metal"]
        if albedo_scale is not None:
            alb = (alb * albedo_scale).clamp(0, 1)
        if detach_material:
            alb, metal = alb.detach(), metal.detach()
        return (1 - metal)[:, None] * alb / math.pi * self.direct_irradiance(light)

    def irradiance(self, mat, light, albedo_scale=None, detach_material: bool = True) -> torch.Tensor:
        """Per-vertex one-bounce indirect irradiance [V, 3]."""
        return self.M @ self.radiosity(mat, light, albedo_scale, detach_material)

    def stats(self) -> dict:
        return {"nnz": int(self.M.values().numel()),
                "rows_with_indirect": int((self.M.crow_indices()[1:] > self.M.crow_indices()[:-1]).sum())}


def interpolate_vertex(values: torch.Tensor, faces: torch.Tensor, face_id: torch.Tensor,
                       bary: torch.Tensor) -> torch.Tensor:
    """Barycentric interpolation of per-vertex values [V, C] at points (face_id, bary) -> [N, C]."""
    return (bary[:, :, None] * values[faces[face_id].long()]).sum(1)
