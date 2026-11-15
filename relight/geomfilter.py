"""Mesh filters used by the geometry diagnostic: Taubin smoothing and Poisson remeshing."""
from __future__ import annotations

import numpy as np
import torch

from relight.mesh import FrozenMesh
from relight.orient import vertex_normals


def _adjacency(faces: torch.Tensor, n: int) -> torch.Tensor:
    f = faces.long()
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
    e = torch.unique(torch.sort(e, 1).values, dim=0)
    idx = torch.cat([e, e.flip(1)], 0).T
    A = torch.sparse_coo_tensor(idx, torch.ones(idx.shape[1], device=f.device), (n, n)).coalesce()
    deg = torch.sparse.sum(A, 1).to_dense().clamp_min(1)
    return A, deg


@torch.no_grad()
def taubin(mesh: FrozenMesh, iters: int, lam: float = 0.5, mu: float = -0.53) -> FrozenMesh:
    """Volume-preserving Taubin smoothing on the vertex adjacency graph (works on non-manifold soups)."""
    V = mesh.vertices.clone()
    A, deg = _adjacency(mesh.faces, V.shape[0])
    for _ in range(iters):
        for w in (lam, mu):
            V = V + w * (torch.sparse.mm(A, V) / deg[:, None] - V)
    return FrozenMesh(V, mesh.faces.clone(), mesh.face_textures, mesh.source + f"+taubin{iters}",
                      vertex_normals(V, mesh.faces))


def sample_surface(mesh: FrozenMesh, n: int, seed: int = 0):
    """Area-weighted surface samples with their face ids and smooth normals."""
    g = torch.Generator(device=mesh.vertices.device).manual_seed(seed)
    area = mesh.face_areas
    fid = torch.multinomial(area / area.sum(), n, replacement=True, generator=g)
    u = torch.rand(n, 2, device=area.device, generator=g)
    flip = u.sum(-1) > 1
    u[flip] = 1 - u[flip]
    bary = torch.stack([1 - u.sum(-1), u[:, 0], u[:, 1]], -1)
    tri = mesh.faces[fid].long()
    pts = (bary[:, :, None] * mesh.vertices[tri]).sum(1)
    nrm = mesh.vertex_normals if mesh.vertex_normals is not None else vertex_normals(mesh.vertices, mesh.faces)
    n_s = torch.nn.functional.normalize((bary[:, :, None] * nrm[tri]).sum(1), dim=-1)
    return pts, n_s, fid


def poisson(mesh: FrozenMesh, depth: int = 9, n_points: int = 2_000_000, density_quantile: float = 0.02) -> FrozenMesh:
    """Screened Poisson remeshing from oriented surface samples (Open3D); trims low-density shells."""
    import open3d as o3d
    pts, nrm, _ = sample_surface(mesh, n_points)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.cpu().numpy().astype(np.float64)))
    pcd.normals = o3d.utility.Vector3dVector(nrm.cpu().numpy().astype(np.float64))
    out, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    out.remove_vertices_by_mask(dens < np.quantile(dens, density_quantile))
    out.remove_unreferenced_vertices()
    V = torch.tensor(np.asarray(out.vertices), dtype=torch.float32, device=mesh.vertices.device)
    Fc = torch.tensor(np.asarray(out.triangles), dtype=torch.int32, device=mesh.vertices.device)
    return FrozenMesh(V, Fc, None, mesh.source + f"+poisson{depth}", vertex_normals(V, Fc))
