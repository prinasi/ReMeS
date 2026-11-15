"""Precomputed per-vertex visibility on the frozen mesh.

For every vertex and every direction of a fixed Fibonacci set, a shadow ray is
traced from ``x + eps·n`` (starting ``tmin`` further along the ray to skip the
vertex's own noisy neighbourhood). Directions below the vertex's tangent plane are
marked occluded without tracing. Results are bit-packed: [V, K/8] uint8.
Per-pixel visibility is the barycentric interpolation of the three vertex masks.
"""
from __future__ import annotations

from pathlib import Path

import torch

from relight.mesh import FrozenMesh
from relight.raytrace import MeshTracer

_BITS = None


def _bits(device):
    global _BITS
    if _BITS is None or _BITS.device != torch.device(device):
        _BITS = (1 << torch.arange(8, device=device, dtype=torch.uint8))
    return _BITS


def pack_bits(mask: torch.Tensor) -> torch.Tensor:
    """bool [..., K] (K % 8 == 0) -> uint8 [..., K/8]."""
    m = mask.reshape(*mask.shape[:-1], -1, 8).to(torch.uint8)
    return (m * _bits(mask.device)).sum(-1).to(torch.uint8)


def unpack_bits(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [..., K/8] -> bool [..., K]."""
    b = (packed[..., None] & _bits(packed.device)) != 0
    return b.reshape(*packed.shape[:-1], -1)


def median_edge_length(mesh: FrozenMesh) -> float:
    v = mesh.vertices[mesh.faces.long()]
    e = (v - v.roll(1, dims=1)).norm(dim=-1).reshape(-1)
    return float(e.median())


@torch.no_grad()
def vertex_visibility(mesh: FrozenMesh, dirs: torch.Tensor, tracer: MeshTracer | None = None,
                      eps: float | None = None, tmin: float | None = None,
                      chunk_vertices: int = 32768) -> torch.Tensor:
    """Packed visibility [V, K/8] (1 = unoccluded and above the tangent plane)."""
    assert mesh.vertex_normals is not None, "needs an oriented mesh with vertex normals"
    K = dirs.shape[0]
    assert K % 8 == 0
    tracer = tracer or MeshTracer(mesh.vertices, mesh.faces)
    h = median_edge_length(mesh)
    eps = 0.5 * h if eps is None else eps
    tmin = 2.0 * h if tmin is None else tmin
    V = mesh.vertices.shape[0]
    out = torch.empty(V, K // 8, dtype=torch.uint8, device=mesh.vertices.device)
    for s in range(0, V, chunk_vertices):
        x = mesh.vertices[s:s + chunk_vertices]
        n = mesh.vertex_normals[s:s + chunk_vertices]
        above = (n @ dirs.T) > 0                                         # [B, K]
        vis = torch.zeros_like(above)
        bi, ki = above.nonzero(as_tuple=True)
        if bi.numel():
            d = dirs[ki]
            o = x[bi] + eps * n[bi] + tmin * d
            vis[bi, ki] = ~tracer.any_hit(o, d)
        out[s:s + chunk_vertices] = pack_bits(vis)
    return out


def interpolate_visibility(packed: torch.Tensor, faces: torch.Tensor, face_id: torch.Tensor,
                           bary: torch.Tensor) -> torch.Tensor:
    """Per-point soft visibility [N, K] in [0, 1] from vertex masks."""
    tri = faces[face_id].long()                                          # [N, 3]
    out = None
    for k in range(3):                                                   # one vertex at a time (memory)
        term = bary[:, k:k + 1] * unpack_bits(packed[tri[:, k]]).to(bary.dtype)
        out = term if out is None else out + term
    return out


def cached_vertex_visibility(mesh: FrozenMesh, dirs: torch.Tensor, path: str | Path, **kw) -> torch.Tensor:
    path = Path(path)
    if path.is_file():
        d = torch.load(path, map_location=mesh.vertices.device)
        if d["vis"].shape[0] == mesh.vertices.shape[0] and d["dirs"].shape == dirs.shape and \
                torch.allclose(d["dirs"].to(dirs.device), dirs):
            return d["vis"]
    vis = vertex_visibility(mesh, dirs, **kw)
    torch.save({"vis": vis, "dirs": dirs}, path)
    return vis
