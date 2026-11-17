"""Phase 2/3: PBR material optimisation on a frozen mesh.

Phase 2 (default): known (oracle) training light. Phase 3 (``--learn-light``): the
training light is a learnable log-space HDR envmap optimised jointly with materials.

Inputs : <model>/relight_mesh.pt (scripts/prepare_relight_mesh.py), Synthetic4Relight
         train views (linear HDR + mask), envmaps (training light + relight lights).
Outputs: <model>/<out>/material.pt, metrics.json, images/*.png

    python scripts/train_pbr.py -m output/synthetic4relight/hotdog --scene hotdog

Evaluation (InvRender protocol, display = clip(x^(1/2.2))):
  NVS      test/*_rgba.png (training light), PSNR inside the GT mask and on the full
           image composited on black;
  albedo   test/*_albedo.png decoded with x^2.2, per-channel least-squares scale
           alignment, PSNR on display values inside the mask (raw = no alignment);
  rough    MSE vs test/*_rough.png inside the mask;
  relight  test_rli/envmap{6,12}_*.png rendered with those envmaps; "aligned" variants
           first rescale albedo by the per-view albedo alignment (needed for an
           unknown light: albedo and light are only defined up to a per-channel scale).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relight.data import RELIGHT_ENVMAPS, TRAIN_ENVMAP, Synthetic4Relight  # noqa: E402
from relight.envmap import load_envmap  # noqa: E402
from relight.gbuffer import rasterize  # noqa: E402
from relight.io import gamma_to_linear, psnr, tonemap, write_png  # noqa: E402
from relight.material import PBRMaterial, unique_edges  # noqa: E402
from relight.mesh import load_relight_mesh  # noqa: E402
from relight.shading import EnvLight, LearnableEnvLight, brdf_lut, fibonacci_sphere, shade  # noqa: E402
from relight.envmap import downsample_envmap  # noqa: E402
from relight.visibility import cached_vertex_visibility, interpolate_visibility  # noqa: E402
from relight.indirect import interpolate_vertex  # noqa: E402

SKIP_TOP_ROWS = 5   # hotdog train EXRs have their top rows zeroed (dataset artifact)


def shading_normal(mesh, face_id, bary, mat=None):
    base = mesh.vertex_normals if mat is None else mat.shading_normals(mesh.vertex_normals)
    vn = base[mesh.faces[face_id].long()]                                # [N, 3, 3]
    return F.normalize((bary[:, :, None] * vn).sum(1), dim=-1)


def surface_point(mesh, face_id, bary):
    return (bary[:, :, None] * mesh.vertices[mesh.faces[face_id].long()]).sum(1)


@torch.no_grad()
def build_pixel_cache(mesh, ds: Synthetic4Relight, alpha_thr: float) -> dict:
    """All interior foreground pixels of all training views (face, bary, camera, GT linear rgb)."""
    fids, barys, cams, rgbs, centers, index = [], [], [], [], [], []
    offset = 0
    for i in range(ds.num_views("train")):
        v = ds.train_view(i)
        g = rasterize(mesh, v.camera)
        sel = g["mask"] & (torch.from_numpy(v.mask).cuda() >= alpha_thr)
        sel[:SKIP_TOP_ROWS] = False
        im = torch.full(sel.shape, -1, dtype=torch.int32, device="cuda")          # pixel -> cache row (patches)
        n_sel = int(sel.sum())
        im[sel] = torch.arange(offset, offset + n_sel, dtype=torch.int32, device="cuda")
        offset += n_sel
        index.append(im)
        fids.append(g["face_id"][sel].int())
        barys.append(g["bary"][sel])
        cams.append(torch.full((int(sel.sum()),), i, dtype=torch.int16, device="cuda"))
        rgbs.append(torch.from_numpy(v.linear).cuda()[sel].half())
        centers.append(torch.as_tensor(v.camera.center, dtype=torch.float32))
    return {"index": torch.stack(index), "face_id": torch.cat(fids), "bary": torch.cat(barys), "cam": torch.cat(cams),
            "rgb": torch.cat(rgbs), "centers": torch.stack(centers).cuda()}


LAYERING = True   # set from --no-layering in main()


def shade_points(mesh, mat, vis_packed, light, lut, face_id, bary, cam_pos, so=True, albedo_scale=None, e_ind=None):
    n = shading_normal(mesh, face_id, bary, mat)
    x = surface_point(mesh, face_id, bary)
    v = F.normalize(cam_pos - x, dim=-1)
    m = mat.sample(face_id, bary)
    if albedo_scale is not None:
        m["albedo"] = (m["albedo"] * albedo_scale).clamp(0, 1)
    vis = interpolate_visibility(vis_packed, mesh.faces, face_id, bary)
    extra = None if e_ind is None else interpolate_vertex(e_ind, mesh.faces, face_id, bary)
    out = shade(m["albedo"], m["rough"], m["metal"], n, v, vis, light, lut, specular_occlusion=so,
                layering=LAYERING, irr_extra=extra)
    out.update(m)
    return out


@torch.no_grad()
def init_albedo(mesh, vis_packed, light, cache, chunk=1 << 17, lo=0.02, hi=0.95, e_ind=None):
    """Per-vertex diffuse albedo estimate: Σ_w L_obs / Σ_w (E_vis/π), w = barycentric weight.

    Ignores specular (small for dielectrics); vertices without observations get the
    median estimate.
    """
    V = mesh.vertices.shape[0]
    num = torch.zeros(V, 3, device="cuda")
    den = torch.zeros(V, 3, device="cuda")
    for s in range(0, cache["face_id"].shape[0], chunk):
        fid = cache["face_id"][s:s + chunk].long()
        bary = cache["bary"][s:s + chunk]
        n = shading_normal(mesh, fid, bary)
        vis = interpolate_visibility(vis_packed, mesh.faces, fid, bary)
        E = ((vis * (n @ light.dirs.T).clamp_min(0)) @ light.bins) / math.pi    # [N, 3]
        if e_ind is not None:
            E = E + interpolate_vertex(e_ind, mesh.faces, fid, bary) / math.pi
        L = cache["rgb"][s:s + chunk].float()
        tri = mesh.faces[fid].long()
        for k in range(3):
            w = bary[:, k:k + 1]
            num.index_add_(0, tri[:, k], w * L)
            den.index_add_(0, tri[:, k], w * E)
    seen = den.sum(-1) > 1e-6
    alb = torch.zeros(V, 3, device="cuda")
    alb[seen] = (num[seen] / den[seen].clamp_min(1e-6)).clamp(lo, hi)
    alb[~seen] = alb[seen].median(0).values
    return alb, float(seen.float().mean())


@torch.no_grad()
def edge_aware_weights(mesh, cache, edges, sigma: float, chunk=1 << 18) -> torch.Tensor:
    """exp(-|c_a - c_b| / sigma) per mesh edge, c = vertex mean of observed gamma-space colour (barycentric
    weights over all cached training pixels): ~1 inside uniform regions, ~0 across texture / shading edges."""
    V = mesh.vertices.shape[0]
    num = torch.zeros(V, 3, device="cuda")
    den = torch.zeros(V, device="cuda")
    for s in range(0, cache["face_id"].shape[0], chunk):
        tri = mesh.faces[cache["face_id"][s:s + chunk].long()].long()
        b = cache["bary"][s:s + chunk]
        c = loss_space(cache["rgb"][s:s + chunk].float())
        for k in range(3):
            num.index_add_(0, tri[:, k], b[:, k:k + 1] * c)
            den.index_add_(0, tri[:, k], b[:, k])
    col = num / den.clamp_min(1e-6)[:, None]
    d = (col[edges[:, 0]] - col[edges[:, 1]]).abs().max(-1).values
    return torch.exp(-d / sigma)


def sample_patches(cache, n: int, size: int):
    """n random size x size patches (random view, centred on a random cached pixel of it). Returns the cache
    rows of the valid patch pixels and their flat positions in the [n, size, size] patch stack."""
    Vn, H, W = cache["index"].shape
    rows = torch.randint(0, cache["face_id"].shape[0], (n,), device="cuda")
    views = cache["cam"][rows].long()
    # recover the anchor pixel coordinates of those rows
    flat = torch.stack([(cache["index"][v] == r).nonzero()[0] for v, r in zip(views.tolist(), rows.tolist())])
    y0 = (flat[:, 0] - size // 2).clamp(0, H - size)
    x0 = (flat[:, 1] - size // 2).clamp(0, W - size)
    ar = torch.arange(size, device="cuda")
    ys = y0[:, None, None] + ar[None, :, None]
    xs = x0[:, None, None] + ar[None, None, :]
    rows_p = cache["index"][views[:, None, None], ys, xs]                          # [n, size, size]
    valid = (rows_p >= 0).reshape(-1)
    return rows_p.reshape(-1)[valid].long(), valid.nonzero()[:, 0]


def patch_dssim(pred: torch.Tensor, gt: torch.Tensor, patch_idx: torch.Tensor) -> torch.Tensor:
    """1 - SSIM over the patch stack; pixels outside the cache are zero in both images."""
    from utils.loss_utils import ssim
    shape = PATCH_SHAPE
    P_ = torch.zeros(shape[0] * shape[1] * shape[2], 3, device=pred.device, dtype=pred.dtype)
    G_ = torch.zeros_like(P_)
    P_[patch_idx] = pred
    G_[patch_idx] = gt
    P_ = P_.reshape(*shape, 3).permute(0, 3, 1, 2)
    G_ = G_.reshape(*shape, 3).permute(0, 3, 1, 2)
    return 1 - ssim(P_, G_)


PATCH_SHAPE = (32, 64, 64)


def loss_space(x):
    return (x.clamp_min(0) + 1e-4) ** (1 / 2.2)


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def render_view(mesh, mat, vis_packed, light, lut, cam, chunk=1 << 16, so=True, albedo_scale=None, indirect=None):
    """indirect: relight.indirect.IndirectLight (Phase 4) or None (direct light only)."""
    e_ind = None if indirect is None else indirect.irradiance(mat, light, albedo_scale)
    g = rasterize(mesh, cam)
    m = g["mask"]
    fid, bary = g["face_id"][m], g["bary"][m]
    cpos = torch.as_tensor(cam.center, dtype=torch.float32, device="cuda")
    H, W = m.shape
    img = {k: torch.zeros(H, W, c, device="cuda") for k, c in
           (("rgb", 3), ("albedo", 3), ("rough", 1), ("diffuse", 3), ("specular", 3))}
    outs = {k: [] for k in img}
    for s in range(0, fid.shape[0], chunk):
        o = shade_points(mesh, mat, vis_packed, light, lut, fid[s:s + chunk], bary[s:s + chunk], cpos, so,
                         albedo_scale, e_ind)
        for k in img:
            outs[k].append(o[k] if o[k].ndim == 2 else o[k][:, None])
    for k in img:
        if outs[k]:
            img[k][m] = torch.cat(outs[k])
    img["mask"] = m
    return img


_LPIPS = None


def ssim_lpips(pred: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    """Full-image SSIM and LPIPS (VGG) of display-space [H, W, 3] images in [0, 1]."""
    global _LPIPS
    from utils.loss_utils import ssim
    if _LPIPS is None:
        import lpips
        _LPIPS = lpips.LPIPS(net="vgg", verbose=False).cuda().eval()
    a, b = pred.clamp(0, 1).permute(2, 0, 1)[None], gt.clamp(0, 1).permute(2, 0, 1)[None]
    return float(ssim(a, b)), float(_LPIPS(a * 2 - 1, b * 2 - 1))


@torch.no_grad()
def render_view_ss(mesh, mat, vis_packed, light, lut, cam, ss: int, **kw) -> dict:
    """Antialiased render: ss x ss supersampling, box-filtered. Returns premultiplied 'rgb', 'albedo', 'rough'
    and the pixel coverage 'alpha' (as the antialiased GT images, which are premultiplied by their alpha)."""
    import dataclasses
    big = dataclasses.replace(cam, width=cam.width * ss, height=cam.height * ss) if ss > 1 else cam
    o = render_view(mesh, mat, vis_packed, light, lut, big, **kw)
    m = o["mask"][..., None].float()
    out = {"rgb": o["rgb"] * m, "albedo": o["albedo"] * m, "rough": o["rough"] * m, "alpha": m}
    if ss > 1:
        out = {k: F.avg_pool2d(v.permute(2, 0, 1)[None], ss)[0].permute(1, 2, 0) for k, v in out.items()}
    return out


def aligned_albedo_psnr(pred_lin, gt_lin, mask):
    p, g = pred_lin[mask], gt_lin[mask]
    scale = (p * g).sum(0) / (p * p).sum(0).clamp_min(1e-8)
    aligned = (pred_lin * scale).clamp(0, 1)
    return (psnr(tonemap(aligned), tonemap(gt_lin), mask), psnr(tonemap(pred_lin), tonemap(gt_lin), mask),
            scale.tolist())


@torch.no_grad()
def evaluate(mesh, mat, vis_packed, lights, train_light, lut, ds, stride, img_dir: Path | None, n_vis=4, so=True,
             indirect=None, ss: int = 2):
    """Masked metrics use the plain render; *_full metrics (InvRender/TensoIR/R3DG protocol) use an antialiased
    ss x ss supersampled, premultiplied render against the (antialiased, premultiplied) GT."""
    rows = {k: [] for k in ("nvs_psnr_mask", "nvs_psnr_full", "albedo_psnr_aligned", "albedo_psnr_raw",
                            "rough_mse", *[f"relight_{e}_psnr_mask" for e in RELIGHT_ENVMAPS],
                            *[f"relight_{e}_psnr_full" for e in RELIGHT_ENVMAPS],
                            *[f"relight_{e}_psnr_mask_aligned" for e in RELIGHT_ENVMAPS],
                            # full-image variants (black background), the InvRender / TensoIR / R3DG protocol
                            "albedo_psnr_aligned_full", "rough_mse_full",
                            *[f"relight_{e}_psnr_full_aligned" for e in RELIGHT_ENVMAPS],
                            "nvs_ssim_full", "nvs_lpips_full", "albedo_ssim_full", "albedo_lpips_full",
                            *[f"relight_{e}_{k}_full_aligned" for e in RELIGHT_ENVMAPS for k in ("ssim", "lpips")])}
    scales = []
    idx = list(range(0, ds.num_views("test"), stride))
    vis_idx = set(idx[:: max(1, len(idx) // n_vis)][:n_vis])
    for i in idx:
        tv = ds.test_view(i)
        gm = torch.from_numpy(tv.mask).cuda()
        mask = gm > 0.5
        out = render_view(mesh, mat, vis_packed, train_light, lut, tv.camera, so=so, indirect=indirect)
        pred = tonemap(out["rgb"])
        gt = torch.from_numpy(tv.rgb).cuda()
        rows["nvs_psnr_mask"].append(psnr(pred, gt, mask))
        full = render_view_ss(mesh, mat, vis_packed, train_light, lut, tv.camera, ss, so=so, indirect=indirect)
        pred_full = tonemap(full["rgb"])
        rows["nvs_psnr_full"].append(psnr(pred_full, gt * gm[..., None]))
        s_, l_ = ssim_lpips(pred_full, gt * gm[..., None])
        rows["nvs_ssim_full"].append(s_)
        rows["nvs_lpips_full"].append(l_)
        gt_alb = gamma_to_linear(torch.from_numpy(tv.albedo).cuda())
        a_al, a_raw, sc = aligned_albedo_psnr(out["albedo"], gt_alb, mask & out["mask"])
        rows["albedo_psnr_aligned"].append(a_al)
        sc_t = torch.tensor(sc, device="cuda")
        om = out["mask"][..., None].float()
        alb_disp = (tonemap((full["albedo"] * sc_t).clamp(0, 1)), tonemap(gt_alb) * gm[..., None])
        rows["albedo_psnr_aligned_full"].append(psnr(*alb_disp))
        s_, l_ = ssim_lpips(*alb_disp)
        rows["albedo_ssim_full"].append(s_)
        rows["albedo_lpips_full"].append(l_)
        rows["albedo_psnr_raw"].append(a_raw)
        scales.append(sc)
        gt_r = torch.from_numpy(tv.rough).cuda()
        rows["rough_mse"].append(float(((out["rough"][..., 0] - gt_r)[mask & out["mask"]] ** 2).mean()))
        rows["rough_mse_full"].append(float(((full["rough"][..., 0] - gt_r * gm) ** 2).mean()))
        relit = {}
        for e in RELIGHT_ENVMAPS:
            o = render_view(mesh, mat, vis_packed, lights[e], lut, tv.camera, so=so, indirect=indirect)
            rgb, alpha = ds.relight_view(i, e)
            g_rel, a_rel = torch.from_numpy(rgb).cuda(), torch.from_numpy(alpha).cuda()
            p_rel = tonemap(o["rgb"])
            rows[f"relight_{e}_psnr_mask"].append(psnr(p_rel, g_rel, a_rel > 0.5))
            o_full = render_view_ss(mesh, mat, vis_packed, lights[e], lut, tv.camera, ss, so=so, indirect=indirect)
            rows[f"relight_{e}_psnr_full"].append(psnr(tonemap(o_full["rgb"]), g_rel * a_rel[..., None]))
            relit[e] = (p_rel * o["mask"][..., None], g_rel * a_rel[..., None])
            o2 = render_view(mesh, mat, vis_packed, lights[e], lut, tv.camera, so=so,
                             albedo_scale=torch.tensor(sc, device="cuda"), indirect=indirect)
            rows[f"relight_{e}_psnr_mask_aligned"].append(psnr(tonemap(o2["rgb"]), g_rel, a_rel > 0.5))
            o2_full = render_view_ss(mesh, mat, vis_packed, lights[e], lut, tv.camera, ss, so=so, indirect=indirect,
                                     albedo_scale=sc_t)
            rel_disp = (tonemap(o2_full["rgb"]), g_rel * a_rel[..., None])
            rows[f"relight_{e}_psnr_full_aligned"].append(psnr(*rel_disp))
            s_, l_ = ssim_lpips(*rel_disp)
            rows[f"relight_{e}_ssim_full_aligned"].append(s_)
            rows[f"relight_{e}_lpips_full_aligned"].append(l_)
        if img_dir is not None and i in vis_idx:
            gray = lambda x: x.expand(*x.shape[:2], 3)
            top = torch.cat([gt * gm[..., None], pred * out["mask"][..., None],
                             torch.from_numpy(tv.albedo).cuda() * gm[..., None],
                             tonemap(out["albedo"]) * out["mask"][..., None]], 1)
            e = RELIGHT_ENVMAPS[-1]
            bot = torch.cat([relit[e][1], relit[e][0], gray(gt_r[..., None]) * gm[..., None],
                             gray(out["rough"]) * out["mask"][..., None]], 1)
            write_png(img_dir / f"test_{i:03d}.png", torch.cat([top, bot], 0).cpu().numpy())
    res = {k: float(np.mean(v)) for k, v in rows.items()}
    res["albedo_scale_mean"] = np.mean(np.array(scales), 0).tolist()
    res["views"] = len(idx)
    return res


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--data", default=str(ROOT / "data/Synthetic4Relight"))
    p.add_argument("--out", default="pbr_oracle")
    p.add_argument("--light", default=TRAIN_ENVMAP, help="training light (oracle)")
    p.add_argument("--learn-light", action="store_true", help="Phase 3: learn the training light")
    p.add_argument("--light-res", type=int, default=64, help="learnable envmap height (width = 2x)")
    p.add_argument("--light-init", default="auto",
                   help="initialise the learned light from a saved log-envmap (e.g. the geometry stage's "
                        "<model>/env.pt, fitted with shadow maps); 'auto' = <model>/env.pt if present")
    p.add_argument("--freeze-light", action="store_true", help="with --light-init: keep the light fixed")
    p.add_argument("--lr-light", type=float, default=2e-2)
    p.add_argument("--light-tv", type=float, default=1e-3, help="TV on the log envmap")
    p.add_argument("--light-wb", type=float, default=0.0, help="white-balance prior on the mean light colour")
    p.add_argument("--dirs", type=int, default=512, help="visibility / light directions")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch", type=int, default=1 << 17)
    p.add_argument("--lr", type=float, default=3e-2)
    p.add_argument("--albedo-init", default="data", help="'data' (radiance / irradiance) or a constant")
    p.add_argument("--lr-final-ratio", type=float, default=0.1)
    p.add_argument("--face-res", type=int, default=0, help="per-face residual texture res (0 = off)")
    p.add_argument("--smooth-albedo", type=float, default=0.005)
    p.add_argument("--smooth-rough", type=float, default=0.02)
    p.add_argument("--learn-metallic", action="store_true")
    p.add_argument("--learn-normals", action="store_true",
                   help="learnable per-vertex shading-normal offset (exported as the glTF NORMAL attribute)")
    p.add_argument("--smooth-normal", type=float, default=0.01, help="edge smoothness of the normal offset")
    p.add_argument("--normal-anchor", type=float, default=0.01, help="|offset|^2 penalty (stay near geometry)")
    p.add_argument("--lr-normal", type=float, default=1e-3)
    p.add_argument("--ssim", type=float, default=0.0,
                   help="weight of D-SSIM in the image loss (patch batches: --patches x --patch-size^2 pixels)")
    p.add_argument("--patches", type=int, default=32)
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--edge-aware", type=float, default=0.0,
                   help="sigma of image-edge-aware smoothness weights exp(-|dc|/sigma) on mesh edges "
                        "(dc = difference of the vertices' mean observed colour); 0 = uniform")
    p.add_argument("--no-visibility", action="store_true", help="ablation: V = 1 above the horizon")
    p.add_argument("--vis-tmin", type=float, default=10.0,
                   help="shadow rays start this many median edge lengths away (skips local mesh noise)")
    p.add_argument("--no-so", action="store_true", help="ablation: no specular occlusion")
    p.add_argument("--indirect", action=argparse.BooleanOptionalAction, default=True,
                   help="Phase 4: one-bounce indirect diffuse light from the mesh itself (relight/indirect.py); "
                        "--no-indirect = direct light only")
    p.add_argument("--indirect-grad", action="store_true",
                   help="let the indirect term's gradient reach the albedo at the hit points (default: detached)")
    p.add_argument("--no-layering", action="store_true",
                   help="no diffuse attenuation by the specular layer (runs before 2026-11-15 Phase 5)")
    p.add_argument("--alpha-thr", type=float, default=0.99)
    p.add_argument("--eval-stride", type=int, default=4)
    p.add_argument("--eval-ss", type=int, default=2, help="supersampling for the full-image (antialiased) metrics")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-only", action="store_true", help="reload <out>/material.pt (+ light.pt) and re-evaluate")
    a = p.parse_args()
    global LAYERING
    LAYERING = not a.no_layering
    torch.manual_seed(a.seed)
    global PATCH_SHAPE
    PATCH_SHAPE = (a.patches, a.patch_size, a.patch_size)
    t0 = time.perf_counter()

    model = Path(a.model)
    out_dir = model / a.out
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    mesh = load_relight_mesh(model)
    ds = Synthetic4Relight(a.data, a.scene)
    dirs = fibonacci_sphere(a.dirs)

    # visibility (cached next to the mesh; geometry is frozen)
    tv = time.perf_counter()
    if a.no_visibility:
        from relight.visibility import pack_bits
        vis_packed = pack_bits((mesh.vertex_normals @ dirs.T) > 0)
    else:
        from relight.visibility import median_edge_length
        h = median_edge_length(mesh)
        vis_packed = cached_vertex_visibility(mesh, dirs, model / f"visibility_K{a.dirs}_t{a.vis_tmin:g}.pt",
                                              eps=0.5 * h, tmin=a.vis_tmin * h)
    indirect = None
    if a.indirect:
        from relight.indirect import IndirectLight
        from relight.visibility import median_edge_length
        h = median_edge_length(mesh)
        ti = time.perf_counter()
        indirect = IndirectLight.cached(mesh, vis_packed, dirs, model / f"indirect_K{a.dirs}_t{a.vis_tmin:g}.pt",
                                        eps=0.5 * h, tmin=a.vis_tmin * h)
        print(f"[indirect] transport {indirect.stats()} ({time.perf_counter() - ti:.1f}s)", flush=True)
    from relight.visibility import unpack_bits
    above = (mesh.vertex_normals @ dirs.T) > 0
    occl = 1 - unpack_bits(vis_packed).float().sum() / above.float().sum()
    print(f"[vis] {mesh.vertices.shape[0]:,} vertices x {a.dirs} dirs, {time.perf_counter() - tv:.1f}s, "
          f"occluded share of upper hemisphere {100 * occl:.1f}%")

    lut = brdf_lut()
    names = sorted({a.light, TRAIN_ENVMAP, *RELIGHT_ENVMAPS})
    lights = {n: EnvLight(load_envmap(ds.envmap_path(n)), dirs, n) for n in names}
    cache = build_pixel_cache(mesh, ds, a.alpha_thr)
    P = cache["face_id"].shape[0]
    print(f"[data] {P:,} interior foreground pixels from {ds.num_views('train')} views")

    if a.learn_light:
        # grey init: albedo 0.5 under uniform light with ~75% visibility reproduces the mean radiance
        level = float(cache["rgb"].float().mean()) / (0.5 * 0.75)
        light = LearnableEnvLight(dirs, (a.light_res, 2 * a.light_res), init=level).cuda()
        src = model / "env.pt" if a.light_init == "auto" else Path(a.light_init) if a.light_init else None
        if src is not None and src.is_file():
            le = torch.load(src, map_location="cuda")["log_env"].float()
            if tuple(le.shape[:2]) != (a.light_res, 2 * a.light_res):
                le = torch.log(downsample_envmap(torch.exp(le), a.light_res, 2 * a.light_res).clamp_min(1e-6))
            light.log_env.data.copy_(le)
            print(f"[light] learnable {a.light_res}x{2 * a.light_res} envmap initialised from {src}"
                  f"{' (frozen)' if a.freeze_light else ''}")
        else:
            print(f"[light] learnable {a.light_res}x{2 * a.light_res} envmap, grey init {level:.3f}")
        if a.freeze_light:
            light.log_env.requires_grad_(False)
    else:
        light = lights[a.light]

    if a.albedo_init == "data":
        alb0, seen = init_albedo(mesh, vis_packed, light, cache)
        if indirect is not None:                     # one fixed-point step: re-estimate with the bounce light
            with torch.no_grad():
                e0 = indirect.M @ (alb0 / math.pi * indirect.direct_irradiance(light))
            alb0, seen = init_albedo(mesh, vis_packed, light, cache, e_ind=e0)
        print(f"[init] data-driven albedo for {100 * seen:.1f}% of vertices, mean {alb0.mean(0).tolist()}")
    else:
        alb0 = float(a.albedo_init)
    mat = PBRMaterial(mesh.faces, mesh.vertices.shape[0], face_res=a.face_res, albedo_init=alb0,
                      learn_metallic=a.learn_metallic, learn_normals=a.learn_normals).cuda()
    params = [mat.albedo, mat.rough] + ([mat.metal] if a.learn_metallic else []) + \
             ([mat.residual] if mat.residual is not None else [])
    groups = [{"params": params, "lr": a.lr}]
    if mat.normal_offset is not None:
        groups.append({"params": [mat.normal_offset], "lr": a.lr_normal})
    if a.learn_light and light.log_env.requires_grad:
        groups.append({"params": [light.log_env], "lr": a.lr_light})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: a.lr_final_ratio ** (it / max(a.iters, 1)))
    edges = unique_edges(mesh.faces)
    edge_w = None
    if a.edge_aware > 0:
        edge_w = edge_aware_weights(mesh, cache, edges, a.edge_aware)
        print(f"[smooth] edge-aware weights sigma {a.edge_aware}: mean {float(edge_w.mean()):.3f}, "
              f"share < 0.5: {float((edge_w < 0.5).float().mean()):.3f}")

    log = []
    if a.eval_only:
        ck = torch.load(out_dir / "material.pt", map_location="cuda", weights_only=False)
        mat.load_state_dict(ck["state_dict"])
        if a.learn_light:
            light.log_env.data.copy_(torch.load(out_dir / "light.pt")["log_env"].cuda())
        prev = out_dir / "metrics.json"
        log = json.loads(prev.read_text()).get("train_log", []) if prev.is_file() else []
    for it in range(1, 0 if a.eval_only else a.iters + 1):
        if a.ssim > 0:                                    # patch batches (SSIM needs image structure)
            idx, patch_idx = sample_patches(cache, a.patches, a.patch_size)
        else:
            idx = torch.randint(0, P, (a.batch,), device="cuda")
        fid, bary = cache["face_id"][idx].long(), cache["bary"][idx]
        cpos = cache["centers"][cache["cam"][idx].long()]
        if a.learn_light:
            light.refresh()
        e_ind = None if indirect is None else indirect.irradiance(mat, light, detach_material=not a.indirect_grad)
        o = shade_points(mesh, mat, vis_packed, light, lut, fid, bary, cpos, so=not a.no_so, e_ind=e_ind)
        l_img = (loss_space(o["rgb"]) - loss_space(cache["rgb"][idx].float())).abs().mean()
        if a.ssim > 0:
            l_img = (1 - a.ssim) * l_img + a.ssim * patch_dssim(loss_space(o["rgb"]),
                                                                 loss_space(cache["rgb"][idx].float()), patch_idx)
        l_smooth = mat.edge_smoothness(edges, (a.smooth_albedo, a.smooth_rough), edge_w,
                                       (a.smooth_normal, a.normal_anchor))
        loss = l_img + l_smooth
        if a.learn_light:
            loss = loss + a.light_tv * light.tv() + a.light_wb * light.white_balance()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if it % 250 == 0 or it == 1:
            with torch.no_grad():
                pp = -10 * math.log10(((tonemap(o["rgb"]) - tonemap(cache["rgb"][idx].float())) ** 2).mean().item())
            log.append({"iter": it, "loss": l_img.item(), "smooth": float(l_smooth), "train_psnr": pp})
            print(f"[train] it {it:5d}  L1 {l_img.item():.5f}  smooth {float(l_smooth):.5f}  "
                  f"batch PSNR {pp:.2f}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    if not a.eval_only:
        torch.save({"state_dict": mat.state_dict(), "args": vars(a)}, out_dir / "material.pt")
    light_metrics = {}
    if a.learn_light:
        with torch.no_grad():
            light.refresh()
            learned = light.env.detach()
            gt = downsample_envmap(lights[TRAIN_ENVMAP].env, *learned.shape[:2])
            scale = (learned * gt).sum((0, 1)) / (learned * learned).sum((0, 1))
            light_metrics = {"light_psnr_aligned": psnr(tonemap(learned * scale), tonemap(gt)),
                             "light_scale_to_gt": scale.tolist()}
            from relight.io import write_exr
            write_exr(out_dir / "learned_envmap.exr", learned.cpu().numpy())
            write_png(out_dir / "envmap_learned_vs_gt.png",
                      torch.cat([tonemap(learned * scale), tonemap(gt)], 0).cpu().numpy())
        if not a.eval_only:
            torch.save({"log_env": light.log_env.detach().cpu()}, out_dir / "light.pt")
    te = time.perf_counter()
    with torch.no_grad():
        metrics = evaluate(mesh, mat, vis_packed, lights, light, lut, ds, a.eval_stride, out_dir / "images",
                           so=not a.no_so, indirect=indirect, ss=a.eval_ss)
    metrics.update(light_metrics)
    metrics.update(seconds_total=time.perf_counter() - t0, seconds_eval=time.perf_counter() - te,
                   args=vars(a), occluded_share=float(occl), train_log=log)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[eval] " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float)))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
