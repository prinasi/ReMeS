"""Stage A: OTO-style opaque triangle soup with structured (diffuse + reflection-specular)
appearance, optimised on nvdiffrast (relight/oto.py). Stage B (``--stage-b-iters`` > 0):
weld the soup, remesh it with T²-Splat's surface RDT, refine the connected mesh, orient it.

    python scripts/train_oto.py --data data/SynthGeo --scene suzanne_glossy --out output/oto/suzanne_glossy

Outputs <out>/soup.pt (parameters), env.pt, soup_mesh.pt (3F corners), relight_mesh.pt
(Stage B connected + oriented mesh, or the soup if Stage B is off; input to Phases 2-5 and
scripts/eval_geometry_gt.py), train_log.json and preview images.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relight.data import Synthetic4Relight  # noqa: E402
from relight.envmap import downsample_envmap, load_envmap  # noqa: E402
from relight.gbuffer import mvp_matrix  # noqa: E402
from relight.io import tonemap, write_png  # noqa: E402
from relight.mesh import FrozenMesh, save_relight_mesh  # noqa: E402
from relight.oto import OpaqueSoup, albedo_smoothness, flatness_loss, texture_weight, visual_hull_soup  # noqa: E402
from relight.shadowmap import ShadowMapper, sample_directions  # noqa: E402
from relight.shading import LearnableEnvLight, brdf_lut, fibonacci_sphere  # noqa: E402

SKIP_TOP_ROWS = 5
MVS_DEPTHS: list = []          # per-view MVS depth maps (--mvs-depth), kept out of the saved args


def gamma(x):
    return (x.clamp_min(0) + 1e-4) ** (1 / 2.2)


def mvs_term(a, o, views, vi):
    """MVS depth prior (relight/multiview.mvs_depth_loss) for view vi; 0 when off."""
    if a.lambda_mvs <= 0 or MVS_DEPTHS[vi] is None:
        return torch.zeros((), device="cuda")
    from relight.multiview import mvs_depth_loss
    cam, m, _, _ = views[vi]
    return mvs_depth_loss(o, cam, MVS_DEPTHS[vi], m, a.mvs_trunc)[0]


def mv_term(a, o, views, vi, V, faces, it):
    """Multi-view NCC (relight/multiview.py) against a random neighbour of view vi; 0 when off."""
    if a.lambda_mv <= 0 or it < a.mv_from:
        return torch.zeros((), device="cuda"), None
    from relight.multiview import ncc_loss
    cam, m, gt, _ = views[vi]
    nc, nm, ngt, _ = views[random.choice(a.mv_nbrs[vi])]
    return ncc_loss(o, gt, m, nc, ngt, nm, V, faces, patch=a.mv_patch, tex_thr=a.mv_tex_thr)


def stage_b(a, soup, views, light, lut, light_opt, log, t0, mapper=None):
    from relight.oto import ConnectedMesh, rdt_remesh, weld_soup
    from relight.orient import orient_components, vertex_normals
    from relight.visibility import median_edge_length
    if light is not None and mapper is not None and a.pbr_from >= 0 and not soup.is_albedo:
        mean_alb = soup.convert_to_albedo(mapper, light)
        soup.is_albedo = True
        print(f"[stage B] switched to physical shading before remeshing; mean albedo {mean_alb:.3f}", flush=True)
    soup_mesh = FrozenMesh(soup.verts.detach().reshape(-1, 3), soup.faces())
    if a.remesh == "poisson":
        Vw, faces, Aw = poisson_remesh(soup, [v[0] for v in views], a.poisson_depth, a.poisson_faces)
    else:
        h = median_edge_length(soup_mesh)
        Vw, Fw, Aw = weld_soup(soup.verts.detach(), soup.corner_attributes().detach(), eps=a.weld_eps * h)
        faces = rdt_remesh(Vw, Fw, mode=a.rdt_mode)
    if a.carve_dilate >= 0:
        from relight.oto import outside_masks
        bad = outside_masks(Vw[faces].mean(1), carve_views(views), a.carve_dilate)
        faces = faces[~bad]
        print(f"[stage B] silhouette carving removed {int(bad.sum()):,} faces", flush=True)
    used = torch.zeros(Vw.shape[0], dtype=torch.bool, device=Vw.device)
    used[faces.reshape(-1)] = True
    remap = torch.full((Vw.shape[0],), -1, dtype=torch.long, device=Vw.device)
    remap[used] = torch.arange(int(used.sum()), device=Vw.device)
    cm = ConnectedMesh(Vw[used], remap[faces], Aw[used], lr_pos=a.lr_pos_b, lr_color=a.lr_color, lr_spec=a.lr_spec)
    print(f"[stage B] {a.remesh}: {soup.verts.shape[0] * 3:,} soup corners -> {cm.faces.shape[0]:,} faces on "
          f"{cm.verts.shape[0]:,} vertices ({time.perf_counter() - t0:.0f}s)", flush=True)
    if a.large_steps > 0:
        cm.enable_large_steps(a.large_steps, a.lr_pos_ls)
        print(f"[stage B] large-steps vertex optimisation, lambda {a.large_steps:g}, lr {a.lr_pos_ls:g}", flush=True)
    lr_pos_b = a.lr_pos_ls if a.large_steps > 0 else a.lr_pos_b
    physical_b = soup.is_albedo and mapper is not None and light is not None
    cm.is_albedo = physical_b
    if a.face_tex_b > 0:
        cm.enable_face_tex(a.face_tex_b, a.lr_face_tex)
        print(f"[stage B] per-face {a.face_tex_b}x{a.face_tex_b} log-albedo textures from iteration {a.face_tex_from}",
              flush=True)
    if a.grad_diag:
        grad_diagnostic(a, cm, views, light, lut, mapper if physical_b else None, a.grad_diag + "_start.pt")
    ind = None
    for it in range(1, a.stage_b_iters + 1):
        if a.indirect_b and physical_b and ((it - 1) % a.indirect_every == 0 or ind is None):
            t_ind = time.perf_counter()
            ind = build_stage_b_indirect(cm, light, a)
            print(f"[B {it - 1:5d}] indirect transport {ind.stats()} ({time.perf_counter() - t_ind:.1f}s)", flush=True)
        if a.remesh_every > 0 and it > 1 and (it - 1) % a.remesh_every == 0 and it <= a.stage_b_iters - a.remesh_every:
            info = cm.remesh(a.remesh_scale)
            print(f"[B {it - 1:5d}] remesh: target edge {info['target_len']:.4f}, V/F {info['before']} -> "
                  f"{info['after']}", flush=True)
        cm.set_lr("verts", lr_pos_b * 0.1 ** (it / a.stage_b_iters))
        vi = random.randrange(len(views))
        cam, m, gt, mvp = views[vi]
        if light is not None:
            light.refresh()
        shadow = shadow_context(mapper, light, cm.verts, cm.faces, a.shadow_dirs) \
            if (soup.is_albedo and mapper is not None and light is not None) else None
        if shadow is not None and ind is not None:
            shadow["e_ind"] = stage_b_e_ind(ind, cm, light)
        o = cm.render(cam, light, lut, mvp=mvp, shadow=shadow, soft_edge_tau=a.soft_edge_tau)
        l_img = (gamma(o["rgb"]) - gt).abs().mean()
        loss = l_img + a.lambda_mask * (o["alpha"] - m).abs().mean() + a.lambda_dn * cm_dn(o) \
            + a.lambda_lap * cm.laplacian_loss() + a.lambda_nc * cm.normal_consistency_loss()
        if a.lambda_flat > 0:
            loss = loss + a.lambda_flat * flatness_loss(o, cam, texture_weight(gt, a.flat_sigma))
        if cm.face_tex is not None:
            loss = loss + a.lambda_face_tex * (cm.face_tex ** 2).mean()
        l_mv, _ = mv_term(a, o, views, vi, cm.verts, cm.faces, 1 << 30)       # always on in Stage B
        loss = loss + a.lambda_mv * l_mv + a.lambda_mvs * mvs_term(a, o, views, vi)
        if shadow is not None and a.lambda_albedo > 0:
            loss = loss + a.lambda_albedo * albedo_smoothness(o, gt)
        if light is not None and a.env_tv > 0 and light.log_env.requires_grad:
            loss = loss + a.env_tv * light.tv()
        loss.backward()
        cm.step()
        if cm.face_tex is not None:
            if it >= a.face_tex_from:
                cm.tex_opt.step()
            cm.tex_opt.zero_grad(set_to_none=True)
        if light_opt is not None and light.log_env.requires_grad:
            light_opt.step()
            light_opt.zero_grad(set_to_none=True)
        if it % 500 == 0:
            with torch.no_grad():
                psnr = -10 * math.log10(((gamma(o["rgb"]) - gt) ** 2).mean().item())
            log.append({"stage": "B", "iter": it, "faces": int(cm.faces.shape[0]), "L1": l_img.item(), "psnr": psnr})
            print(f"[B {it:5d}] faces {cm.faces.shape[0]:,}  L1 {l_img.item():.4f}  PSNR {psnr:.2f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    if a.grad_diag:
        grad_diagnostic(a, cm, views, light, lut, mapper if physical_b else None, a.grad_diag + "_end.pt")
    cm.max_cov.zero_()
    with torch.no_grad():
        for cam, m, gt, mvp in views:
            cm.render(cam, light, lut, mvp=mvp, track_coverage=True)
    # face-level coverage pruning would punch holes into watertight (Poisson) meshes: faces smaller
    # than a pixel never cover a pixel centre
    n_rm = cm.prune_faces(cm.max_cov < 1) if a.remesh == "rdt" else 0
    if a.carve_dilate >= 0 and a.carve_final:
        from relight.oto import outside_masks
        n_rm += cm.prune_faces(outside_masks(cm.verts.detach()[cm.faces.long()].mean(1),
                                             carve_views(views), a.carve_dilate))
    mesh = FrozenMesh(cm.verts.detach().contiguous(), cm.faces)
    mesh, stats = orient_components(mesh, [v[0] for v in views])
    mesh.vertex_normals = vertex_normals(mesh.vertices, mesh.faces)
    print(f"[stage B] pruned {n_rm:,} never-visible faces; oriented {stats['components']:,} components "
          f"(flipped {stats['flipped']:,}/{stats['faces']:,} faces)", flush=True)
    return mesh, cm


@torch.no_grad()
def poisson_remesh(soup, cameras, depth, target_faces=0):
    """Watertight screened-Poisson surface from the opaque soup (outward-oriented by camera votes);
    per-vertex attributes from the nearest soup surface point."""
    import cubvh
    from relight.geomfilter import poisson
    from relight.orient import orient_mesh
    soup_mesh = FrozenMesh(soup.verts.detach().reshape(-1, 3).contiguous(), soup.faces())
    oriented, _ = orient_mesh(soup_mesh, cameras)
    pm = poisson(oriented, depth=depth)
    if target_faces and pm.faces.shape[0] > target_faces:
        import open3d as o3d
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(pm.vertices.cpu().numpy().astype(np.float64)),
                                       o3d.utility.Vector3iVector(pm.faces.cpu().numpy().astype(np.int32)))
        om = om.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        om.remove_degenerate_triangles()
        om.remove_unreferenced_vertices()
        pm = FrozenMesh(torch.tensor(np.asarray(om.vertices), dtype=torch.float32, device="cuda"),
                        torch.tensor(np.asarray(om.triangles), dtype=torch.int32, device="cuda"))
    _, fid, uvw = cubvh.cuBVH(soup_mesh.vertices, soup_mesh.faces).unsigned_distance(pm.vertices, return_uvw=True)
    corner = soup.corner_attributes().detach().reshape(-1, 3, 5)
    A = (uvw[:, :, None] * corner[fid]).sum(1)
    return pm.vertices, pm.faces.long(), A


def shadow_context(mapper, light, V, faces, k):
    """Importance-sample k light bins and render their (differentiable) shadow maps."""
    idx, prob = sample_directions(light.bins, light.dirs, k)
    return {"mapper": mapper, "maps": mapper.render(V, faces, light.dirs[idx]), "idx": idx, "prob": prob}


def carve_views(views):
    """(camera, mask) for silhouette carving; the zeroed top rows are unknown, i.e. treated as inside."""
    out = []
    for cam, m, _, _ in views:
        mm = m.clone()
        mm[:SKIP_TOP_ROWS] = 1
        out.append((cam, mm))
    return out


@torch.no_grad()
def build_stage_b_indirect(cm, light, a):
    """One-bounce transport for the current Stage-B mesh: direct operator on the light's own directions,
    gathering transport on a separate Fibonacci set (relight/indirect.py)."""
    from relight.indirect import IndirectLight
    from relight.orient import vertex_normals
    from relight.visibility import median_edge_length, vertex_visibility
    V = cm.verts.detach().contiguous()
    mesh = FrozenMesh(V, cm.faces, vertex_normals=vertex_normals(V, cm.faces))
    h = median_edge_length(mesh)
    eps, tmin = 0.5 * h, a.vis_tmin_b * h
    gdirs = fibonacci_sphere(a.indirect_dirs)
    gather = IndirectLight.build(mesh, vertex_visibility(mesh, gdirs, eps=eps, tmin=tmin), gdirs, eps, tmin)
    direct = IndirectLight.build_direct(mesh, vertex_visibility(mesh, light.dirs, eps=eps, tmin=tmin), light.dirs)
    return IndirectLight(gather.M, direct)


def stage_b_e_ind(ind, cm, light):
    """Per-vertex indirect irradiance with the current albedo (diffuse attribute) and light; detached."""
    with torch.no_grad():
        alb = cm.attributes()[:, :3]
        return ind.M @ (alb / math.pi * ind.direct_irradiance(light))


class _DetachedReceiver:
    """Shadow mapper proxy whose receiver positions carry no gradient (diagnostic)."""
    def __init__(self, mapper):
        self.m = mapper

    def visibility(self, maps, X):
        return self.m.visibility(maps, X.detach())


def grad_diagnostic(a, cm, views, light, lut, mapper, path):
    """Per-loss-term gradients on the vertex positions, summed over all training views. The image term is
    split by path: shadow occluders (shadow maps), shadow receivers (lookup position), shading normals,
    and the rest (rasterisation / antialias / view direction). Saved to `path` for offline analysis."""
    from relight.oto import render_triangles
    V = cm.verts
    terms = ("img", "img_occ_detached", "img_shadow_detached", "img_shadow_nrm_detached", "mask", "dn", "lap",
             "nc", "albedo")
    acc = {t: torch.zeros_like(V) for t in terms}
    if light is not None:
        light.refresh()
    physical = mapper is not None and light is not None and cm_is_albedo(cm)
    idx = prob = None
    if physical:
        idx, prob = sample_directions(light.bins, light.dirs, a.shadow_dirs)
    for cam, m, gt, mvp in views:
        A = cm.attributes()
        fn = cm.face_normals()
        variants = {}
        if physical:
            maps = mapper.render(V, cm.faces, light.dirs[idx])
            maps_d = mapper.render(V.detach(), cm.faces, light.dirs[idx])
            sh = lambda mp, mpr: {"mapper": mpr, "maps": mp, "idx": idx, "prob": prob}
            variants = {"img": (sh(maps, mapper), fn), "img_occ_detached": (sh(maps_d, mapper), fn),
                        "img_shadow_detached": (sh(maps_d, _DetachedReceiver(mapper)), fn),
                        "img_shadow_nrm_detached": (sh(maps_d, _DetachedReceiver(mapper)), fn.detach())}
        else:
            variants = {"img": (None, fn), "img_shadow_nrm_detached": (None, fn.detach())}
        o = None
        for name, (shadow, fnv) in variants.items():
            o = render_triangles(V, cm.faces, A, fnv, cam, light, lut, mvp, shadow)
            l_img = (gamma(o["rgb"]) - gt).abs().mean()
            g = torch.autograd.grad(l_img, V, retain_graph=True, allow_unused=True)[0]
            if g is not None:
                acc[name] += g
        o = render_triangles(V, cm.faces, A, fn, cam, light, lut, mvp,
                             {"mapper": mapper, "maps": mapper.render(V, cm.faces, light.dirs[idx]), "idx": idx,
                              "prob": prob} if physical else None)
        regs = {"mask": a.lambda_mask * (o["alpha"] - m).abs().mean(), "dn": a.lambda_dn * cm_dn(o),
                "lap": a.lambda_lap * cm.laplacian_loss(), "nc": a.lambda_nc * cm.normal_consistency_loss()}
        if physical and a.lambda_albedo > 0:
            regs["albedo"] = a.lambda_albedo * albedo_smoothness(o, gt)
        for name, l in regs.items():
            g = torch.autograd.grad(l, V, retain_graph=True, allow_unused=True)[0]
            if g is not None:
                acc[name] += g
    torch.save({"verts": V.detach().cpu(), "faces": cm.faces.cpu(), "physical": physical,
                "grads": {k: v.cpu() for k, v in acc.items()}}, path)
    print(f"[grad-diag] -> {path}", flush=True)


def cm_is_albedo(cm):
    return getattr(cm, "is_albedo", True)


def cm_dn(o):
    return OpaqueSoup.depth_normal_loss(o)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=15000)
    p.add_argument("--hull-res", type=int, default=128)
    p.add_argument("--max-tris", type=int, default=600_000)
    p.add_argument("--densify-from", type=int, default=500)
    p.add_argument("--densify-until", type=int, default=10000)
    p.add_argument("--densify-every", type=int, default=500)
    p.add_argument("--cov-split", type=float, default=12.0, help="split if max pixel coverage exceeds this")
    p.add_argument("--cov-prune", type=float, default=1.0, help="prune if max pixel coverage is below this")
    p.add_argument("--split-by", choices=["coverage", "error"], default="coverage",
                   help="'error': split only large triangles whose mean image error exceeds --split-err x the mean "
                        "(coverage-only splitting refines the front-most layer and domes textureless planes)")
    p.add_argument("--split-err", type=float, default=1.0)
    p.add_argument("--lambda-flat", type=float, default=0.0,
                   help="texture-weighted planarity prior (screen-space 2nd differences of inverse depth); 0 = off")
    p.add_argument("--flat-sigma", type=float, default=0.02, help="image-gradient scale of the texture weight")
    p.add_argument("--flat-from", type=int, default=0)
    p.add_argument("--large-steps", type=float, default=0.0,
                   help="Stage B: lambda of the smoothness-preconditioned vertex update (Nicolet et al. 2021; "
                        "relight/largesteps.py); 0 = per-coordinate Adam")
    p.add_argument("--lr-pos-ls", type=float, default=1e-3, help="Stage B vertex lr with --large-steps")
    p.add_argument("--indirect-b", action="store_true",
                   help="Stage B: one-bounce indirect diffuse light in the physical shading (relight/indirect.py)")
    p.add_argument("--indirect-every", type=int, default=500, help="rebuild the transport every N Stage-B its")
    p.add_argument("--indirect-dirs", type=int, default=256, help="gathering directions of the transport")
    p.add_argument("--vis-tmin-b", type=float, default=10.0, help="visibility ray start offset (x median edge)")
    p.add_argument("--face-tex-b", type=int, default=0,
                   help="Stage B: per-face RxR log-albedo residual textures (appearance capacity decoupled from "
                        "the mesh); 0 = per-vertex only")
    p.add_argument("--face-tex-from", type=int, default=1, help="Stage B iteration where the textures start learning")
    p.add_argument("--lr-face-tex", type=float, default=1e-2)
    p.add_argument("--lambda-face-tex", type=float, default=1e-3, help="L2 on the log-albedo residual")
    p.add_argument("--grad-diag", default="", help="Stage B: save per-loss-term vertex gradients (all views) to "
                                                   "<prefix>_start.pt / _end.pt")
    p.add_argument("--lr-pos", type=float, default=2e-4)
    p.add_argument("--lr-pos-final", type=float, default=2e-5)
    p.add_argument("--lr-color", type=float, default=1e-2)
    p.add_argument("--lr-spec", type=float, default=1e-2)
    p.add_argument("--lr-env", type=float, default=5e-3)
    p.add_argument("--env-res", type=int, default=64)
    p.add_argument("--env-tv", type=float, default=1e-3)
    p.add_argument("--oracle-env", default="", help="envmap name to use as a fixed light (diagnostic)")
    p.add_argument("--no-specular", action="store_true", help="ablation: diffuse only")
    p.add_argument("--lambda-mask", type=float, default=0.1)
    p.add_argument("--lambda-dn", type=float, default=0.05, help="normal-depth consistency weight")
    p.add_argument("--lambda-mv", type=float, default=0.0,
                   help="multi-view patch-NCC consistency weight (relight/multiview.py; 0 = off)")
    p.add_argument("--mvs-depth", default="", help="COLMAP dense workspace (scripts/mvs_init.py <out>/colmap/dense)")
    p.add_argument("--lambda-mvs", type=float, default=0.0, help="truncated MVS depth prior weight (0 = off)")
    p.add_argument("--mvs-trunc", type=float, default=0.02, help="relative residual above which MVS depth is ignored")
    p.add_argument("--mv-from", type=int, default=3000, help="Stage A iteration where the NCC term starts")
    p.add_argument("--mv-patch", type=int, default=7)
    p.add_argument("--mv-neighbours", type=int, default=4)
    p.add_argument("--mv-tex-thr", type=float, default=0.01, help="min reference patch std (gamma units)")
    p.add_argument("--pbr-from", type=int, default=4000,
                   help="switch diffuse from free radiance to albedo x shadowed irradiance at this iteration (-1 = never)")
    p.add_argument("--shadow-dirs", type=int, default=32,
                   help="light bins (Fibonacci) for diffuse irradiance; all are shadow-mapped every iteration")
    p.add_argument("--shadow-res", type=int, default=512)
    p.add_argument("--lambda-albedo", type=float, default=0.02, help="edge-aware albedo smoothness (physical mode)")
    p.add_argument("--remesh-every", type=int, default=0,
                   help="Stage B: isotropic remeshing (split/collapse/flip/relax) every N iterations (0 = off)")
    p.add_argument("--remesh-scale", type=float, default=1.0, help="target edge length / current median edge")
    p.add_argument("--soft-edge-tau", type=float, default=0.0,
                   help="OTO two-layer/2D-SDF edge gradient with sigmoid temperature tau (1/px); 0 = antialias only")
    p.add_argument("--max-tris-pbr", type=int, default=450_000,
                   help="triangle cap once physical shading is on (batched shadow maps scale with triangle count)")
    p.add_argument("--dn-from", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage-b-iters", type=int, default=5000, help="0 = stop after the soup")
    p.add_argument("--resume", default="", help="dir with soup.pt (+ env.pt): skip Stage A")
    p.add_argument("--init", default="hull",
                   help="'hull' (visual hull) or 'mesh:<dir or relight_mesh.pt>' (e.g. T²-Splat or MVS surface)")
    p.add_argument("--init-tris", type=int, default=40_000)
    p.add_argument("--weld-eps", type=float, default=0.5, help="weld radius in median soup edge lengths")
    p.add_argument("--rdt-mode", default="manifold", choices=("manifold", "g"))
    p.add_argument("--remesh", default="poisson", choices=("rdt", "poisson"),
                   help="connectivity for Stage B: T²-Splat RDT on the welded soup, or screened Poisson")
    p.add_argument("--poisson-depth", type=int, default=9)
    p.add_argument("--poisson-faces", type=int, default=200_000, help="quadric-decimation target (0 = keep)")
    p.add_argument("--carve-dilate", type=int, default=2,
                   help="remove faces whose centre projects outside a mask dilated by this many px (-1 = off)")
    p.add_argument("--carve-final", action="store_true",
                   help="also carve after Stage B refinement (leaves unrefined gaps; off by default)")
    p.add_argument("--lr-pos-b", type=float, default=5e-5)
    p.add_argument("--lambda-lap", type=float, default=1.0, help="uniform Laplacian (Stage B)")
    p.add_argument("--lambda-nc", type=float, default=0.01, help="adjacent face normal consistency (Stage B)")
    a = p.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    ds = Synthetic4Relight(a.data, a.scene)
    views = []
    for i in range(ds.num_views("train")):
        v = ds.train_view(i)
        m = torch.from_numpy(v.mask).cuda()
        m[:SKIP_TOP_ROWS] = 0
        gt = gamma(torch.from_numpy(v.linear).cuda() * m[..., None])
        views.append((v.camera, m, gt, mvp_matrix(v.camera).cuda()))
    print(f"[data] {len(views)} train views", flush=True)
    if a.lambda_mvs > 0:
        from relight.multiview import load_mvs_depths
        MVS_DEPTHS[:] = load_mvs_depths(a.mvs_depth, len(views))
        cov = [float((d > 0).float().mean()) for d in MVS_DEPTHS if d is not None]
        print(f"[mvs] {len(cov)} depth maps from {a.mvs_depth}, mean coverage {np.mean(cov):.3f}", flush=True)
    if a.lambda_mv > 0:
        from relight.multiview import neighbour_table
        a.mv_nbrs = neighbour_table([v[0] for v in views], k=a.mv_neighbours)

    if a.resume:
        st = torch.load(Path(a.resume) / "soup.pt")
        soup = OpaqueSoup(st["verts"].cuda(), lr_pos=a.lr_pos, lr_color=a.lr_color, lr_spec=a.lr_spec)
        soup.diffuse.data.copy_(st["diffuse"].cuda())
        soup.spec.data.copy_(st["spec"].cuda())
        soup.is_albedo = bool(st.get("is_albedo", False))
        print(f"[resume] {soup.num:,} soup triangles from {a.resume} (diffuse = "
              f"{'albedo' if soup.is_albedo else 'radiance'})", flush=True)
    else:
        if a.init == "hull":
            verts = visual_hull_soup([(c, m.cpu().numpy()) for c, m, _, _ in views], res=a.hull_res, max_tris=a.max_tris)
            print(f"[init] visual hull {a.hull_res}^3 -> {verts.shape[0]:,} triangles", flush=True)
        elif a.init.startswith("mesh:"):
            from relight.mesh import load_relight_mesh
            from relight.oto import surface_soup
            src = load_relight_mesh(a.init[5:])
            verts, info = surface_soup(src, carve_views(views), n_tris=a.init_tris)
            print(f"[init] surface {a.init[5:]} ({src.faces.shape[0]:,} faces) -> Poisson {info['poisson_faces']:,} "
                  f"-> {verts.shape[0]:,} triangles (r={info['radius']:.4f})", flush=True)
        else:
            raise ValueError(a.init)
        soup = OpaqueSoup(verts, lr_pos=a.lr_pos, lr_color=a.lr_color, lr_spec=a.lr_spec)
        soup.is_albedo = False
    with torch.no_grad():
        V0 = soup.verts.reshape(-1, 3)
        lo, hi = V0.min(0).values, V0.max(0).values
    mapper = ShadowMapper(0.5 * (lo + hi), 0.55 * float((hi - lo).norm()), res=a.shadow_res)

    lut = brdf_lut()
    light = None
    if not a.no_specular:
        light = LearnableEnvLight(fibonacci_sphere(a.shadow_dirs), (a.env_res, 2 * a.env_res), init=1.0).cuda()
        if a.oracle_env:
            env = downsample_envmap(load_envmap(ds.envmap_path(a.oracle_env)), a.env_res, 2 * a.env_res)
            light.log_env.data.copy_(torch.log(env.clamp_min(1e-4)))
            light.log_env.requires_grad_(False)
        if a.resume and (Path(a.resume) / "env.pt").is_file() and not a.oracle_env:
            light.log_env.data.copy_(torch.load(Path(a.resume) / "env.pt")["log_env"].cuda())
        light_opt = torch.optim.Adam([light.log_env], lr=a.lr_env)

    log = []
    for it in range(1, 0 if a.resume else a.iters + 1):
        frac = it / a.iters
        soup.set_lr("verts", a.lr_pos * (a.lr_pos_final / a.lr_pos) ** frac)
        vi = random.randrange(len(views))
        cam, m, gt, mvp = views[vi]
        if light is not None:
            light.refresh()
        if light is not None and a.pbr_from >= 0 and it == a.pbr_from and not soup.is_albedo:
            mean_alb = soup.convert_to_albedo(mapper, light)
            soup.is_albedo = True
            print(f"[{it:6d}] switched to physical shading (albedo x shadowed irradiance); mean albedo {mean_alb:.3f}",
                  flush=True)
        shadow = shadow_context(mapper, light, soup.verts.reshape(-1, 3), soup.faces(), a.shadow_dirs) \
            if soup.is_albedo else None
        o = soup.render(cam, light, lut, mvp=mvp, track_coverage=True, shadow=shadow, soft_edge_tau=a.soft_edge_tau)
        res = (gamma(o["rgb"]) - gt).abs()
        l_img = res.mean()
        l_mask = (o["alpha"] - m).abs().mean()
        loss = l_img + a.lambda_mask * l_mask
        if a.split_by == "error":
            soup.track_error(o, res.detach().mean(-1))
        if a.lambda_flat > 0 and it >= a.flat_from:
            loss = loss + a.lambda_flat * flatness_loss(o, cam, texture_weight(gt, a.flat_sigma))
        if shadow is not None and a.lambda_albedo > 0:
            loss = loss + a.lambda_albedo * albedo_smoothness(o, gt)
        l_dn = torch.zeros((), device="cuda")
        if a.lambda_dn > 0 and it >= a.dn_from:
            l_dn = soup.depth_normal_loss(o)
            loss = loss + a.lambda_dn * l_dn
        l_mv, mv_info = mv_term(a, o, views, vi, soup.verts.reshape(-1, 3), soup.faces(), it)
        loss = loss + a.lambda_mv * l_mv + a.lambda_mvs * mvs_term(a, o, views, vi)
        if light is not None and a.env_tv > 0 and light.log_env.requires_grad:
            loss = loss + a.env_tv * light.tv()
        loss.backward()
        soup.step()
        if light is not None and light.log_env.requires_grad:
            light_opt.step()
            light_opt.zero_grad(set_to_none=True)

        # coverage-based density control
        if it % a.densify_every == 0 and it >= a.densify_from:
            n_prune = soup.prune(soup.max_cov < a.cov_prune) if it >= 1000 else 0
            n_split = 0
            if it <= a.densify_until:
                big = soup.max_cov > a.cov_split
                if a.split_by == "error":
                    e = soup.mean_error()
                    seen = soup.err_cnt > 0
                    big &= seen & (e > a.split_err * e[seen].mean())
                cand = big.nonzero()[:, 0]
                cap = min(a.max_tris, a.max_tris_pbr) if soup.is_albedo else a.max_tris
                room = max(0, cap - soup.num)
                if cand.numel() > room:
                    cand = cand[torch.argsort(soup.max_cov[cand], descending=True)[:room]]
                n_split = soup.split(cand)
            soup.reset_coverage()
            with torch.no_grad():
                psnr = -10 * math.log10(((gamma(o["rgb"]) - gt) ** 2).mean().item())
            rec = {"iter": it, "tris": soup.num, "pruned": n_prune, "split": n_split, "L1": l_img.item(),
                   "mask": l_mask.item(), "dn": float(l_dn), "psnr": psnr,
                   "ncc": mv_info["ncc"] if mv_info else None}
            log.append(rec)
            print(f"[{it:6d}] tris {soup.num:>8,} (-{n_prune:,} +{n_split:,})  L1 {l_img.item():.4f}  "
                  f"mask {l_mask.item():.4f}  dn {float(l_dn):.4f}  "
                  f"{'ncc %.3f  ' % mv_info['ncc'] if mv_info else ''}PSNR {psnr:.2f}  ({time.perf_counter() - t0:.0f}s)",
                  flush=True)

    # final cleanup: drop triangles never visible in a full pass over the training views
    soup.reset_coverage()
    with torch.no_grad():
        if light is not None:
            light.refresh()
        for cam, m, gt, mvp in views:
            soup.render(cam, light, lut, mvp=mvp, track_coverage=True)
    n_final = soup.prune(soup.max_cov < 1)
    print(f"[final] removed {n_final:,} never-visible triangles -> {soup.num:,}", flush=True)

    torch.save({**soup.state_dict(), "is_albedo": soup.is_albedo}, out / "soup.pt")
    if light is not None:
        torch.save({"log_env": light.log_env.detach().cpu()}, out / "env.pt")
    V = soup.verts.detach().reshape(-1, 3)
    fn = soup.face_normals().detach()
    mesh = FrozenMesh(V.contiguous(), soup.faces(), vertex_normals=fn.repeat_interleave(3, 0))
    save_relight_mesh(mesh, out / "soup_mesh.pt", {"stage": "oto_soup", "args": vars(a)})
    final_mesh, render_obj = mesh, soup
    if a.stage_b_iters > 0:
        final_mesh, render_obj = stage_b(a, soup, views, light, lut, light_opt if light is not None else None, log, t0,
                                         mapper)
    save_relight_mesh(final_mesh, out / "relight_mesh.pt",
                      {"stage": "oto_connected" if a.stage_b_iters > 0 else "oto_soup", "args": vars(a)})
    with torch.no_grad():
        shadow = None
        if soup.is_albedo and light is not None:
            Vr = render_obj.verts.reshape(-1, 3)
            Fr = render_obj.faces() if callable(render_obj.faces) else render_obj.faces
            shadow = shadow_context(mapper, light, Vr, Fr, a.shadow_dirs)
        for k, (cam, m, gt, mvp) in enumerate(views[::25]):
            o = render_obj.render(cam, light, lut, mvp=mvp, shadow=shadow)
            nimg = (o["normal"] * 0.5 + 0.5) * o["mask"][..., None]
            write_png(out / "images" / f"train_{k}.png",
                      torch.cat([gt, gamma(o["rgb"]).clamp(0, 1), nimg], 1).cpu().numpy())
    (out / "train_log.json").write_text(json.dumps({"args": vars(a), "log": log, "final_tris": int(final_mesh.faces.shape[0]),
                                                    "seconds": time.perf_counter() - t0}, indent=2))
    print(f"-> {out}  ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
