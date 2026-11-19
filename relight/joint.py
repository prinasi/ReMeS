"""Phase 6: joint geometry + material refinement on the material-stage mesh (scripts/train_pbr.py --joint).

The material stage optimises PBR textures and light on a frozen mesh. Phase 6 additionally moves the vertices,
V = V0 + D, so that the *geometry itself* explains the shading (instead of the learned shading-normal offsets,
which are an engine-side normal hack inconsistent with visibility, indirect light and the silhouette):

* data term: the same image loss; its gradient reaches D through the vertex normals (diffuse cosine, specular
  reflection direction and specular-occlusion lobe) and the surface point (view direction). Pixels keep their
  (face, barycentric) correspondence between re-rasterisations ("re-cache" every N iterations), which is exact
  for the small displacements of a refinement;
* silhouette term: antialiased nvdiffrast coverage vs the soft GT masks of a few training views per iteration
  (as scripts/refine_silhouette.py), so the outline stays put while normals change;
* regularisers: uniform Laplacian of the displacement (smooth correction, detail of V0 kept) and an anchor
  |D|²; optional smoothness-preconditioned "large steps" (relight/largesteps.py).

Visibility, the one-bounce transport and the interreflection hit records are non-differentiable ray queries; they
are rebuilt on every re-cache from the current geometry.
"""
from __future__ import annotations

import nvdiffrast.torch as dr
import torch

from relight.gbuffer import get_context
from relight.mesh import FrozenMesh
from relight.orient import vertex_normals


def coverage(V: torch.Tensor, faces: torch.Tensor, mvp: torch.Tensor, res) -> torch.Tensor:
    """Antialiased coverage [H, W] of the mesh (differentiable w.r.t. V at silhouette edges)."""
    clip = (torch.cat([V, torch.ones_like(V[:, :1])], -1) @ mvp.T)[None].contiguous()
    rast, _ = dr.rasterize(get_context(V.device), clip, faces, resolution=res)
    a = (rast[..., 3:4] > 0).float()
    return dr.antialias(a.contiguous(), rast, clip, faces)[0, ..., 0]


def laplacian(n: int, edges: torch.Tensor):
    """Uniform graph Laplacian operator D -> mean(neighbours) - D."""
    a, b = edges[:, 0].long(), edges[:, 1].long()
    deg = torch.zeros(n, device=edges.device).index_add_(0, a, torch.ones_like(a, dtype=torch.float)) \
        .index_add_(0, b, torch.ones_like(b, dtype=torch.float))

    def apply(D):
        nb = torch.zeros_like(D).index_add_(0, a, D[b]).index_add_(0, b, D[a])
        return nb / deg.clamp_min(1)[:, None] - D
    return apply


class JointGeometry(torch.nn.Module):
    def __init__(self, mesh: FrozenMesh, edges: torch.Tensor):
        super().__init__()
        self.register_buffer("V0", mesh.vertices.detach().clone())
        self.faces = mesh.faces.int().contiguous()
        self.D = torch.nn.Parameter(torch.zeros_like(self.V0))
        self.lap = laplacian(self.V0.shape[0], edges)

    def mesh(self) -> FrozenMesh:
        """Current (differentiable) mesh with area-weighted vertex normals."""
        V = self.V0 + self.D
        return FrozenMesh(V, self.faces, vertex_normals=vertex_normals(V, self.faces))

    @torch.no_grad()
    def snapshot(self) -> FrozenMesh:
        V = (self.V0 + self.D).detach().contiguous()
        return FrozenMesh(V, self.faces, vertex_normals=vertex_normals(V, self.faces))

    def regulariser(self, lam_lap: float, lam_anchor: float) -> torch.Tensor:
        return lam_lap * (self.lap(self.D) ** 2).sum(-1).mean() + lam_anchor * (self.D ** 2).sum(-1).mean()

    def silhouette_loss(self, V: torch.Tensor, views, skip_top_rows: int = 0) -> torch.Tensor:
        """views: [(mvp, soft mask [H, W], [H, W])]."""
        loss = 0.0
        for mvp, m, res in views:
            c = coverage(V, self.faces, mvp, res)
            loss = loss + ((c - m) ** 2)[skip_top_rows:].mean()
        return loss / max(len(views), 1)
