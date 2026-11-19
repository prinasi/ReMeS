"""Multi-bounce interreflection (diffuse + specular) on the frozen mesh via a per-vertex radiance cache.

Phase 4 (relight/indirect.py) adds one diffuse bounce to the diffuse irradiance. Two parts of the light transport
are still missing there: (1) the blocked part of every specular lobe is black (specular occlusion only removes
the environment), although in reality it reflects the surface it hits; (2) the light leaving a hit point is only
its direct diffuse radiosity, without its specular reflection toward the receiver or its own indirect light.

Mesh-native fix: for every vertex v and every blocked direction w_k above its tangent plane we keep the closest
hit (face, barycentrics) of the visibility ray (relight/indirect.blocked_hits). A *refresh* shades all hit points
with the full model -- diffuse with the current indirect irradiance plus split-sum specular evaluated for the
view direction -w_k (i.e. toward the receiver), with its own blocked lobe filled from the previous cache -- and
stores the result as incoming radiance

    L_in[v, k] = L_out(hit_vk, -w_k)              (dense [V, K, 3] half; 0 for unblocked directions)

Each refresh adds one bounce (progressive multi-bounce; ~3 refreshes converge for albedos < 0.8). Uses:

* specular:  spec += F0A+B · Σ_k (1-V_k) ŵ_k L_in[k],  ŵ = the normalised GGX lobe weights that specular occlusion
  already uses, interpolated from the three vertices of the shaded point (differentiable w.r.t. roughness and
  normals; L_in itself is detached);
* diffuse:   E_ind[v] = Σ_k cos·ΔΩ · L_in[v, k]. Implemented as the differentiable one-bounce term M·B of
  relight/indirect.py plus a detached correction Σ cos·ΔΩ·(L_in − B(hit)), so the value is the full multi-bounce
  irradiance while gradients still reach the light (and the hit albedo with --indirect-grad) as before.

The cache depends on the light and the material; it is kept per (light name, albedo scale) and refreshed every N
iterations during training, rebuilt with ``bounces`` refreshes for evaluation / relighting.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import torch

from relight.indirect import IndirectLight, blocked_hits, interpolate_vertex
from relight.mesh import FrozenMesh


class Interreflection:
    """Drop-in replacement for IndirectLight (same ``irradiance`` signature) plus ``occluded`` lobe radiance.

    shade_fn(mat, light, albedo_scale, face_id, bary, cam_pos, e_ind_vertex, occluded) -> rgb [N, 3] must shade
    points with the training renderer (scripts/train_pbr.shade_points)."""

    def __init__(self, indirect: IndirectLight, mesh: FrozenMesh, records: dict, dirs: torch.Tensor,
                 shade_fn: Callable, bounces: int = 3, chunk: int = 1 << 16):
        self.ind = indirect
        self.M, self.A = indirect.M, indirect.A
        self.mesh = mesh
        self.r = records
        self.dirs = dirs
        self.V, self.K = mesh.vertices.shape[0], dirs.shape[0]
        self.flat = records["v"].long() * self.K + records["k"].long()          # row in the dense [V*K] cache
        self.shade_fn = shade_fn
        self.bounces = bounces
        self.chunk = chunk
        self.state: dict = {}
        self.aligned_scale: torch.Tensor | None = None   # albedo scale used for every aligned (scaled) render
        self._mat = None

    # ------------------------------------------------------------------ construction
    @classmethod
    def cached(cls, indirect: IndirectLight, mesh: FrozenMesh, vis_packed: torch.Tensor, dirs: torch.Tensor,
               path: str | Path, eps: float, tmin: float, shade_fn: Callable, **kw) -> "Interreflection":
        path = Path(path)
        rec = None
        if path.is_file():
            d = torch.load(path, map_location=mesh.vertices.device)
            if d["V"] == mesh.vertices.shape[0] and d["K"] == dirs.shape[0]:
                rec = d["records"]
        if rec is None:
            rec = blocked_hits(mesh, vis_packed, dirs, eps, tmin)
            torch.save({"V": mesh.vertices.shape[0], "K": dirs.shape[0], "records": rec}, path)
        return cls(indirect, mesh, rec, dirs, shade_fn, **kw)

    # ------------------------------------------------------------------ cache
    def _key(self, light, albedo_scale):
        if albedo_scale is None:
            return (light.name, None), None
        if self.aligned_scale is not None:
            return (light.name, "aligned"), self.aligned_scale
        return (light.name, tuple(round(float(x), 4) for x in albedo_scale)), albedo_scale

    def clear(self):
        self.state = {}

    @torch.no_grad()
    def refresh(self, mat, light, albedo_scale=None, bounces: int = 1):
        """Re-shade all hit points ``bounces`` times (each pass uses the previous pass's cache)."""
        key, scale = self._key(light, albedo_scale)
        faces = self.mesh.faces.long()
        for _ in range(bounces):
            prev = self.state.get(key)
            e_vert = self.ind.irradiance(mat, light, scale) + (prev["E"] if prev is not None else 0)
            B = self.ind.radiosity(mat, light, scale)                           # what M·B already accounts for
            L = torch.zeros(self.V * self.K, 3, dtype=torch.float16, device=self.dirs.device)
            E = torch.zeros(self.V, 3, device=self.dirs.device)
            occl = None if prev is None else self._occluded_fn(prev["L"])
            R = self.flat.shape[0]
            for s in range(0, R, self.chunk):
                fid = self.r["face"][s:s + self.chunk].long()
                bary = self.r["bary"][s:s + self.chunk].float()
                d = self.dirs[self.r["k"][s:s + self.chunk].long()]
                x = (bary[:, :, None] * self.mesh.vertices[faces[fid]]).sum(1)
                rgb = self.shade_fn(mat, light, scale, fid, bary, x - d, e_vert, occl)   # view direction = -d
                L[self.flat[s:s + self.chunk]] = rgb.half()
                corr = rgb - interpolate_vertex(B, faces, fid, bary)
                E.index_add_(0, self.r["v"][s:s + self.chunk].long(), self.r["cosw"][s:s + self.chunk, None] * corr)
            self.state[key] = {"L": L.view(self.V, self.K, 3), "E": E}
        return self.state[key]

    def _get(self, light, albedo_scale):
        key, _ = self._key(light, albedo_scale)
        if key not in self.state:
            assert self._mat is not None, "call bind(mat) first"
            self.refresh(self._mat, light, albedo_scale, self.bounces)
        return self.state[key]

    def bind(self, mat):
        """Material used for lazy cache builds (evaluation / relighting)."""
        self._mat = mat
        return self

    def _occluded_fn(self, L: torch.Tensor):
        faces = self.mesh.faces

        def lobe(face_id: torch.Tensor, bary: torch.Tensor, wn: torch.Tensor) -> torch.Tensor:
            """Σ_j bary_j Σ_k ŵ_k L_in[v_j, k]  -> [N, 3] (ŵ normalised lobe weights [N, K])."""
            tri = faces[face_id].long()
            w16 = wn.to(L.dtype)[:, None, :]
            out = 0
            for j in range(3):
                out = out + bary[:, j:j + 1] * torch.bmm(w16, L[tri[:, j]])[:, 0].float()
            return out
        return lobe

    # ------------------------------------------------------------------ IndirectLight interface
    def irradiance(self, mat, light, albedo_scale=None, detach_material: bool = True) -> torch.Tensor:
        st = self._get(light, albedo_scale)
        return self.ind.irradiance(mat, light, albedo_scale, detach_material) + st["E"]

    def occluded(self, light, albedo_scale=None):
        """Callable (face_id, bary, ŵ) -> blocked-lobe radiance, for relight.shading.shade(occluded=...)."""
        return self._occluded_fn(self._get(light, albedo_scale)["L"])

    def radiosity(self, *a, **kw):
        return self.ind.radiosity(*a, **kw)

    def direct_irradiance(self, *a, **kw):
        return self.ind.direct_irradiance(*a, **kw)

    def stats(self) -> dict:
        return {**self.ind.stats(), "records": int(self.flat.shape[0]),
                "cache_MB": round(self.V * self.K * 3 * 2 / 2 ** 20, 1)}
