"""Multi-view photometric consistency for an explicit surface (PGSR-style, adapted to meshes).

PGSR warps reference patches into a neighbouring view with the homography of each pixel's local plane.
With a rasterised mesh every pixel already has its exact (differentiable) surface point X, so the warp is
dense and exact at pixel resolution: X is projected into the neighbour, its *observed* image is sampled
there, and reference / warped patches are compared with normalised cross-correlation (NCC). NCC is
invariant to per-patch gain and offset, so shading, cast shadows and soft highlights do not drive it:
it only rewards the surface point being where both photographs agree — geometry that "explains" a
shadow by moving the surface is penalised whenever the shadow edge / texture is seen from two views.

PGSR's forward-backward geometric (depth reprojection) term is vacuous for a single surface (both
depths come from the same mesh); it is used here only as the occlusion test.
"""
from __future__ import annotations

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F

from relight.gbuffer import get_context, mvp_matrix


def _w2c(cam, device):
    w2c = torch.as_tensor(np.linalg.inv(np.asarray(cam.c2w, np.float64)), dtype=torch.float32, device=device)
    return w2c[:3, :3], w2c[:3, 3]


def project(X: torch.Tensor, cam) -> tuple[torch.Tensor, torch.Tensor]:
    """World points [..., 3] -> continuous pixel coords [..., 2] (x right, y down; centres at i + 0.5)
    and positive depth along the viewing axis [...]."""
    R, t = _w2c(cam, X.device)
    Xc = X @ R.T + t                       # Blender camera: looks along -Z, +Y up
    z = -Xc[..., 2]
    f = cam.focal
    zs = z.clamp_min(1e-6)
    u = f * Xc[..., 0] / zs + 0.5 * cam.width
    v = -f * Xc[..., 1] / zs + 0.5 * cam.height
    return torch.stack([u, v], -1), z


@torch.no_grad()
def depth_map(V: torch.Tensor, faces: torch.Tensor, cam) -> torch.Tensor:
    """Z-buffer depth (along the viewing axis) of the mesh in `cam`; +inf where empty. [H, W]"""
    mvp = mvp_matrix(cam).to(V.device)
    clip = (torch.cat([V, torch.ones_like(V[:, :1])], -1) @ mvp.T)[None].contiguous()
    rast, _ = dr.rasterize(get_context(V.device), clip, faces, resolution=[cam.height, cam.width])
    pos, _ = dr.interpolate(V[None].contiguous(), rast, faces)
    _, z = project(pos[0], cam)
    return torch.where(rast[0, ..., 3] > 0, z, torch.full_like(z, float("inf")))


def _sample(img: torch.Tensor, uv: torch.Tensor, W: int, H: int, mode="bilinear") -> torch.Tensor:
    """img [H, W, C] at continuous pixel coords uv [h, w, 2] -> [h, w, C]."""
    g = torch.stack([uv[..., 0] / W * 2 - 1, uv[..., 1] / H * 2 - 1], -1)[None]
    return F.grid_sample(img.permute(2, 0, 1)[None], g, mode=mode, padding_mode="border",
                         align_corners=False)[0].permute(1, 2, 0)


def gray(img: torch.Tensor) -> torch.Tensor:
    return (img[..., :3] * torch.tensor([0.299, 0.587, 0.114], device=img.device)).sum(-1, keepdim=True)


def ncc_loss(out: dict, ref_img: torch.Tensor, ref_mask: torch.Tensor, nb_cam, nb_img: torch.Tensor,
             nb_mask: torch.Tensor, V: torch.Tensor, faces: torch.Tensor, patch: int = 7,
             tex_thr: float = 0.01, occ_tol: float = 0.01) -> tuple[torch.Tensor, dict]:
    """1 - NCC between reference patches and their warp into one neighbour view.

    out: render of the reference view (needs "position" [H, W, 3] differentiable, "mask");
    ref_img / nb_img: observed images (any fixed encoding, e.g. gamma) [H, W, 3]; *_mask: object masks.
    Pixels count when the whole patch is valid (on the surface, inside both masks, visible in the
    neighbour) and the reference patch is textured (std > tex_thr: NCC is undefined on flat patches)."""
    H, W = ref_img.shape[:2]
    X = out["position"]
    uv, z = project(X, nb_cam)
    with torch.no_grad():
        zb = depth_map(V.detach(), faces, nb_cam)
        z_seen = _sample(zb[..., None].nan_to_num(posinf=1e6), uv, W, H, mode="nearest")[..., 0]
        inside = (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < H)
        nbm = _sample(nb_mask[..., None].float(), uv, W, H, mode="nearest")[..., 0] > 0.5
        valid = out["mask"] & (ref_mask > 0.5) & inside & nbm & ((z - z_seen).abs() < occ_tol * z)
    Ir = gray(ref_img)[..., 0]
    Iw = _sample(gray(nb_img), uv, W, H)[..., 0]
    v = valid.float()
    pool = lambda x: F.avg_pool2d(x[None, None], patch, stride=1, padding=patch // 2)[0, 0]
    n = pool(v)
    ns = n.clamp_min(1e-6)
    mr, mw = pool(v * Ir) / ns, pool(v * Iw) / ns
    var_r = (pool(v * Ir * Ir) / ns - mr ** 2).clamp_min(0)
    var_w = (pool(v * Iw * Iw) / ns - mw ** 2).clamp_min(0)
    cov = pool(v * Ir * Iw) / ns - mr * mw
    ncc = cov / torch.sqrt(var_r * var_w + 1e-8)
    use = valid & (n > 0.99) & (var_r > tex_thr ** 2)
    if not use.any():
        return torch.zeros((), device=X.device), {"mv_pixels": 0, "ncc": float("nan")}
    loss = (1 - ncc[use]).clamp(0, 2).mean()
    return loss, {"mv_pixels": int(use.sum()), "ncc": float(ncc[use].mean())}


def neighbour_table(cameras, k: int = 4, max_angle_deg: float = 45.0) -> list[list[int]]:
    """For each camera, the k nearest others by viewing direction (towards the origin)."""
    C = np.stack([np.asarray(c.center, np.float64) for c in cameras])
    d = C / np.linalg.norm(C, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(d @ d.T, -1, 1)))
    out = []
    for i in range(len(cameras)):
        order = [int(j) for j in np.argsort(ang[i]) if j != i and ang[i, j] <= max_angle_deg][:k]
        out.append(order or [int(np.argsort(ang[i])[1])])
    return out


# ---------------------------------------------------------------------- MVS depth prior
def load_mvs_depths(colmap_dense: str, n_views: int, device="cuda") -> list[torch.Tensor | None]:
    """COLMAP geometric (filtered, consistency-checked) depth maps of the training views, in the
    scripts/mvs_init.py workspace layout (<dense>/stereo/depth_maps/{i:03d}.png.geometric.bin); 0 = no
    estimate. Depth is along the viewing axis, i.e. comparable with `project(...)[1]`."""
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from mvs_init import read_colmap_array
    out = []
    for i in range(n_views):
        p = Path(colmap_dense) / "stereo" / "depth_maps" / f"{i:03d}.png.geometric.bin"
        out.append(torch.from_numpy(read_colmap_array(p).copy()).to(device) if p.is_file() else None)
    return out


def mvs_depth_loss(out: dict, cam, mvs_depth: torch.Tensor, mask: torch.Tensor, trunc: float = 0.02):
    """Truncated relative L1 between the rendered depth and the MVS depth on pixels where both exist.
    Residuals above `trunc` (relative) are treated as MVS outliers and ignored, so wrong matches
    (highlights, occlusion boundaries) cannot pull the surface; it acts where photometric gradients are
    weak (textureless regions) and the surface drifts slowly away from the measurement."""
    _, z = project(out["position"], cam)
    valid = out["mask"] & (mvs_depth > 0) & (mask > 0.5)
    if not valid.any():
        return torch.zeros((), device=z.device), 0
    rel = (z[valid] - mvs_depth[valid]).abs() / mvs_depth[valid]
    use = rel < trunc
    if not use.any():
        return torch.zeros((), device=z.device), 0
    return rel[use].mean(), int(use.sum())
