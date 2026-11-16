"""Topology adaptation for the Stage B connected mesh: periodic isotropic remeshing.

Every N iterations of Stage B the current mesh is remeshed to (near-)uniform edge length with
pymeshlab's isotropic explicit remeshing (edge split / collapse / swap, tangential smoothing,
reprojection onto the current surface), so triangles stay well shaped while vertices move far
(e.g. pulling hull remnants out of undercuts). Per-vertex attributes (raw logits) are transferred
from the closest point on the previous surface.

Note: pymeshlab is GPL-3 licensed (user-approved dependency).
"""
from __future__ import annotations

import numpy as np
import torch


def isotropic_remesh(V: np.ndarray, F: np.ndarray, target_len: float, iterations: int = 3,
                     max_surf_dist: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V.astype(np.float64), face_matrix=F.astype(np.int32)))
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_isotropic_explicit_remeshing(
        iterations=iterations, targetlen=pymeshlab.PureValue(target_len),
        checksurfdist=True, maxsurfdist=pymeshlab.PureValue(max_surf_dist or 0.5 * target_len),
        splitflag=True, collapseflag=True, swapflag=True, smoothflag=True, reprojectflag=True)
    m = ms.current_mesh()
    return m.vertex_matrix().astype(np.float32), m.face_matrix().astype(np.int64)


@torch.no_grad()
def transfer_attributes(V_old: torch.Tensor, F_old: torch.Tensor, A_old: torch.Tensor,
                        V_new: torch.Tensor) -> torch.Tensor:
    """Per-vertex attributes [N_new, C] from the closest point on the old surface (barycentric)."""
    import cubvh
    _, fid, uvw = cubvh.cuBVH(V_old, F_old).unsigned_distance(V_new, return_uvw=True)
    tri = F_old[fid].long()
    return (uvw[:, :, None] * A_old[tri]).sum(1)
