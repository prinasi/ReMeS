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
from relight.oto import OpaqueSoup, albedo_smoothness, visual_hull_soup  # noqa: E402
from relight.shadowmap import ShadowMapper, sample_directions  # noqa: E402
from relight.shading import LearnableEnvLight, brdf_lut, fibonacci_sphere  # noqa: E402

SKIP_TOP_ROWS = 5


def gamma(x):
    return (x.clamp_min(0) + 1e-4) ** (1 / 2.2)


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
    for it in range(1, a.stage_b_iters + 1):
        if a.remesh_every > 0 and it > 1 and (it - 1) % a.remesh_every == 0 and it <= a.stage_b_iters - a.remesh_every:
            info = cm.remesh(a.remesh_scale)
            print(f"[B {it - 1:5d}] remesh: target edge {info['target_len']:.4f}, V/F {info['before']} -> "
                  f"{info['after']}", flush=True)
        cm.set_lr("verts", a.lr_pos_b * 0.1 ** (it / a.stage_b_iters))
        cam, m, gt, mvp = random.choice(views)
        if light is not None:
            light.refresh()
        shadow = shadow_context(mapper, light, cm.verts, cm.faces, a.shadow_dirs) \
            if (soup.is_albedo and mapper is not None and light is not None) else None
        o = cm.render(cam, light, lut, mvp=mvp, shadow=shadow, soft_edge_tau=a.soft_edge_tau)
        l_img = (gamma(o["rgb"]) - gt).abs().mean()
        loss = l_img + a.lambda_mask * (o["alpha"] - m).abs().mean() + a.lambda_dn * cm_dn(o) \
            + a.lambda_lap * cm.laplacian_loss() + a.lambda_nc * cm.normal_consistency_loss()
        if shadow is not None and a.lambda_albedo > 0:
            loss = loss + a.lambda_albedo * albedo_smoothness(o, gt)
        if light is not None and a.env_tv > 0 and light.log_env.requires_grad:
            loss = loss + a.env_tv * light.tv()
        loss.backward()
        cm.step()
        if light_opt is not None and light.log_env.requires_grad:
            light_opt.step()
            light_opt.zero_grad(set_to_none=True)
        if it % 500 == 0:
            with torch.no_grad():
                psnr = -10 * math.log10(((gamma(o["rgb"]) - gt) ** 2).mean().item())
            log.append({"stage": "B", "iter": it, "faces": int(cm.faces.shape[0]), "L1": l_img.item(), "psnr": psnr})
            print(f"[B {it:5d}] faces {cm.faces.shape[0]:,}  L1 {l_img.item():.4f}  PSNR {psnr:.2f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
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

    if a.resume:
        st = torch.load(Path(a.resume) / "soup.pt")
        soup = OpaqueSoup(st["verts"].cuda(), lr_pos=a.lr_pos, lr_color=a.lr_color, lr_spec=a.lr_spec)
        soup.diffuse.data.copy_(st["diffuse"].cuda())
        soup.spec.data.copy_(st["spec"].cuda())
        soup.is_albedo = bool(st.get("is_albedo", False))
        print(f"[resume] {soup.num:,} soup triangles from {a.resume} (diffuse = "
              f"{'albedo' if soup.is_albedo else 'radiance'})", flush=True)
    else:
        verts = visual_hull_soup([(c, m.cpu().numpy()) for c, m, _, _ in views], res=a.hull_res, max_tris=a.max_tris)
        soup = OpaqueSoup(verts, lr_pos=a.lr_pos, lr_color=a.lr_color, lr_spec=a.lr_spec)
        print(f"[init] visual hull {a.hull_res}^3 -> {soup.num:,} triangles", flush=True)
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
        cam, m, gt, mvp = random.choice(views)
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
        l_img = (gamma(o["rgb"]) - gt).abs().mean()
        l_mask = (o["alpha"] - m).abs().mean()
        loss = l_img + a.lambda_mask * l_mask
        if shadow is not None and a.lambda_albedo > 0:
            loss = loss + a.lambda_albedo * albedo_smoothness(o, gt)
        l_dn = torch.zeros((), device="cuda")
        if a.lambda_dn > 0 and it >= a.dn_from:
            l_dn = soup.depth_normal_loss(o)
            loss = loss + a.lambda_dn * l_dn
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
                cand = (soup.max_cov > a.cov_split).nonzero()[:, 0]
                cap = min(a.max_tris, a.max_tris_pbr) if soup.is_albedo else a.max_tris
                room = max(0, cap - soup.num)
                if cand.numel() > room:
                    cand = cand[torch.argsort(soup.max_cov[cand], descending=True)[:room]]
                n_split = soup.split(cand)
            soup.reset_coverage()
            with torch.no_grad():
                psnr = -10 * math.log10(((gamma(o["rgb"]) - gt) ** 2).mean().item())
            rec = {"iter": it, "tris": soup.num, "pruned": n_prune, "split": n_split, "L1": l_img.item(),
                   "mask": l_mask.item(), "dn": float(l_dn), "psnr": psnr}
            log.append(rec)
            print(f"[{it:6d}] tris {soup.num:>8,} (-{n_prune:,} +{n_split:,})  L1 {l_img.item():.4f}  "
                  f"mask {l_mask.item():.4f}  dn {float(l_dn):.4f}  PSNR {psnr:.2f}  ({time.perf_counter() - t0:.0f}s)",
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
