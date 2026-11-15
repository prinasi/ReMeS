"""Consistent outward orientation for T²-Splat meshes.

T²-Splat's appearance model is orientation-agnostic, so face winding comes out
random (~50% of visible pixels see a face's back side). Relighting needs a
view-independent outward normal (per-vertex visibility, shading normals, glTF
export), so faces are re-oriented:

1. Camera votes: every training camera is outside the object, so the side of a
   face that a camera sees is its outside. Each visible pixel votes +1 if the
   winding normal faces the camera, -1 otherwise; faces with negative totals flip.
2. Unseen faces (never visible in any training view) fall back to an escape test:
   rays are cast over the hemispheres of both sides and the side from which more
   rays escape the mesh is taken as outside.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from relight.gbuffer import rasterize
from relight.mesh import FrozenMesh
from relight.raytrace import MeshTracer


@torch.no_grad()
def camera_votes(mesh: FrozenMesh, cameras) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed and absolute per-face pixel votes over the given cameras."""
    nF = mesh.faces.shape[0]
    signed = torch.zeros(nF, device=mesh.vertices.device)
    total = torch.zeros(nF, device=mesh.vertices.device)
    for cam in cameras:
        g = rasterize(mesh, cam)
        m = g["mask"]
        fid = g["face_id"][m]
        sign = 1.0 - 2.0 * g["backfacing"][m].float()
        signed.index_add_(0, fid, sign)
        total.index_add_(0, fid, torch.ones_like(sign))
    return signed, total


def _hemisphere(n: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
    """k uniform directions on the hemisphere around each normal n [N,3] -> [N,k,3]."""
    d = F.normalize(torch.randn(n.shape[0], k, 3, device=n.device, generator=gen), dim=-1)
    flip = (d * n[:, None]).sum(-1, keepdim=True) < 0
    return torch.where(flip, -d, d)


@torch.no_grad()
def escape_side(mesh: FrozenMesh, tracer: MeshTracer, face_ids: torch.Tensor, k: int = 32,
                eps: float = 1e-4, seed: int = 0, chunk: int = 65536) -> torch.Tensor:
    """For the given faces, +1 if the winding side escapes more often than the back side, else -1."""
    gen = torch.Generator(device=mesh.vertices.device).manual_seed(seed)
    out = torch.empty(face_ids.shape[0], device=mesh.vertices.device)
    v = mesh.vertices[mesh.faces.long()]
    for s in range(0, face_ids.shape[0], chunk):
        fid = face_ids[s:s + chunk]
        tri = v[fid]
        c = tri.mean(1)
        n = F.normalize(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1)
        esc = []
        for sgn in (1.0, -1.0):
            ns = sgn * n
            d = _hemisphere(ns, k, gen)
            o = (c + eps * ns)[:, None].expand_as(d)
            hit = tracer.any_hit(o.reshape(-1, 3), d.reshape(-1, 3)).reshape(-1, k)
            esc.append((~hit).float().mean(1))
        out[s:s + chunk] = torch.where(esc[0] >= esc[1], 1.0, -1.0)
    return out


@torch.no_grad()
def max_escape(mesh: FrozenMesh, tracer: MeshTracer, face_ids: torch.Tensor, k: int = 64,
               eps: float = 1e-5, seed: int = 0, chunk: int = 65536) -> torch.Tensor:
    """Largest escape fraction over the two sides of each face (0 = fully enclosed)."""
    gen = torch.Generator(device=mesh.vertices.device).manual_seed(seed)
    out = torch.empty(face_ids.shape[0], device=mesh.vertices.device)
    v = mesh.vertices[mesh.faces.long()]
    for s in range(0, face_ids.shape[0], chunk):
        tri = v[face_ids[s:s + chunk]]
        c = tri.mean(1)
        n = F.normalize(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1), dim=-1)
        best = torch.zeros(c.shape[0], device=c.device)
        for sgn in (1.0, -1.0):
            d = _hemisphere(sgn * n, k, gen)
            o = (c + eps * sgn * n)[:, None].expand_as(d)
            hit = tracer.any_hit(o.reshape(-1, 3), d.reshape(-1, 3)).reshape(-1, k)
            best = torch.maximum(best, (~hit).float().mean(1))
        out[s:s + chunk] = best
    return out


def subset_faces(mesh: FrozenMesh, keep: torch.Tensor) -> FrozenMesh:
    """Keep the selected faces and drop vertices no longer referenced."""
    faces = mesh.faces[keep].long()
    used = torch.zeros(mesh.vertices.shape[0], dtype=torch.bool, device=faces.device)
    used[faces.reshape(-1)] = True
    remap = torch.full((mesh.vertices.shape[0],), -1, dtype=torch.long, device=faces.device)
    remap[used] = torch.arange(int(used.sum()), device=faces.device)
    tex = mesh.face_textures[keep] if mesh.face_textures is not None else None
    vn = mesh.vertex_normals[used] if mesh.vertex_normals is not None else None
    return FrozenMesh(mesh.vertices[used].contiguous(), remap[faces].int().contiguous(), tex, mesh.source, vn)


@torch.no_grad()
def remove_hidden_faces(mesh: FrozenMesh, cameras, k: int = 64, max_escape_frac: float = 1 / 64
                        ) -> tuple[FrozenMesh, dict]:
    """Drop faces that no training camera sees AND from which (almost) no ray escapes on either side.

    Such faces are enclosed interior triangles: they can neither be seen nor lit, so removing
    them leaves exterior visibility unchanged, while they would corrupt vertex normals,
    orientation statistics and exported asset size.
    """
    _, total = camera_votes(mesh, cameras)
    unseen = (total == 0).nonzero()[:, 0]
    ext = (mesh.vertices.max(0).values - mesh.vertices.min(0).values).norm().item()
    keep = torch.ones(mesh.faces.shape[0], dtype=torch.bool, device=mesh.vertices.device)
    if unseen.numel():
        tracer = MeshTracer(mesh.vertices, mesh.faces)
        esc = max_escape(mesh, tracer, unseen, k=k, eps=1e-5 * ext)
        keep[unseen[esc <= max_escape_frac]] = False
    area = mesh.face_areas
    stats = {"faces_before": int(mesh.faces.shape[0]), "unseen_faces": int(unseen.numel()),
             "removed_faces": int((~keep).sum()), "removed_area_share": float(area[~keep].sum() / area.sum())}
    return subset_faces(mesh, keep), stats


def vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals from (consistently oriented) faces."""
    v = vertices[faces.long()]
    fn = torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1)    # length = 2 * area
    vn = torch.zeros_like(vertices)
    for k in range(3):
        vn.index_add_(0, faces[:, k].long(), fn)
    return F.normalize(vn, dim=-1)


@torch.no_grad()
def orient_mesh(mesh: FrozenMesh, cameras, escape_rays: int = 32) -> tuple[FrozenMesh, dict]:
    signed, total = camera_votes(mesh, cameras)
    seen = total > 0
    decision = torch.sign(signed)
    ties = seen & (decision == 0)
    unseen = ~seen | ties
    ext = (mesh.vertices.max(0).values - mesh.vertices.min(0).values).norm().item()
    if unseen.any():
        tracer = MeshTracer(mesh.vertices, mesh.faces)
        decision[unseen] = escape_side(mesh, tracer, unseen.nonzero()[:, 0], k=escape_rays, eps=1e-5 * ext)
    flip = decision < 0
    faces = mesh.faces.clone()
    faces[flip] = faces[flip][:, [0, 2, 1]]
    # T²-Splat face textures index row = w2 (v2), col = w1 (v1); swapping v1<->v2
    # transposes the barycentric texture grid, so transpose those faces' textures too.
    tex = mesh.face_textures
    if tex is not None:
        tex = tex.clone()
        tex[flip] = tex[flip].transpose(1, 2)
    agree = (signed.abs() / total.clamp_min(1))[seen & ~ties]
    stats = {"faces": int(mesh.faces.shape[0]), "flipped": int(flip.sum()),
             "seen_faces": int((seen & ~ties).sum()), "unseen_faces": int(unseen.sum()),
             "vote_agreement_mean": float(agree.mean()) if agree.numel() else float("nan"),
             "vote_agreement_p05": float(agree.quantile(0.05)) if agree.numel() else float("nan")}
    return FrozenMesh(mesh.vertices, faces, tex, mesh.source), stats


def normal_noise_deg(mesh: FrozenMesh, vn: torch.Tensor) -> dict:
    """Angle between each face normal and its averaged vertex normal (area-weighted stats)."""
    fn = mesh.face_normals
    avg = F.normalize(vn[mesh.faces.long()].sum(1), dim=-1)
    ang = torch.rad2deg(torch.acos((fn * avg).sum(-1).clamp(-1, 1)))
    w = mesh.face_areas
    order = torch.argsort(ang)
    cw = torch.cumsum(w[order], 0) / w.sum()
    med = ang[order][torch.searchsorted(cw, torch.tensor(0.5, device=cw.device))]
    p90 = ang[order][torch.searchsorted(cw, torch.tensor(0.9, device=cw.device)).clamp_max(len(order) - 1)]
    return {"area_weighted_median_deg": float(med), "area_weighted_p90_deg": float(p90),
            "mean_deg": float((ang * w).sum() / w.sum())}


__all__ = ["orient_mesh", "remove_hidden_faces", "subset_faces", "vertex_normals", "camera_votes",
           "escape_side", "max_escape", "normal_noise_deg"]
