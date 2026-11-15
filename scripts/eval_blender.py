"""Phase 5: relight the exported asset in Blender Cycles and score it.

For each relighting envmap (envmap6, envmap12) the GLB is rendered with Cycles at the
test cameras, (a) with full global illumination and (b) direct light only. Reports
  * round trip : IoU of Cycles alpha vs the ReMeS rasterised mask (geometry/axes);
  * parity     : Cycles direct-only vs the ReMeS renderer (same light, same asset);
  * engine     : Cycles GI vs ground-truth test_rli (the end-to-end metric), next to
                 the ReMeS renderer vs ground truth on the same views.
Albedo is exported with the run's mean per-channel albedo alignment (metrics.json),
as in the "aligned" relighting metric (needed for learned lights; ~1 for oracle).

    python scripts/eval_blender.py -m output/synthetic4relight/hotdog --scene hotdog --run pbr_learned
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from relight.data import RELIGHT_ENVMAPS, Synthetic4Relight  # noqa: E402
from relight.envmap import load_envmap  # noqa: E402
from relight.export import write_pbr_glb  # noqa: E402
from relight.io import psnr, read_image, tonemap, write_png  # noqa: E402
from relight.mesh import load_relight_mesh  # noqa: E402
from relight.shading import EnvLight, brdf_lut, fibonacci_sphere  # noqa: E402
from relight.visibility import cached_vertex_visibility, median_edge_length  # noqa: E402
from export_pbr import load_material  # noqa: E402
from train_pbr import render_view  # noqa: E402


def blender(glb, envmap, transforms, frames, out_dir, samples, direct, width):
    if all((out_dir / f"{f:03d}.exr").is_file() for f in frames):
        return
    cmd = [sys.executable, "-m", "relight.blender.render_relight", "--glb", str(glb), "--envmap", str(envmap),
           "--transforms", str(transforms), "--frames", ",".join(map(str, frames)), "--out-dir", str(out_dir),
           "--samples", str(samples), "--width", str(width), "--height", str(width)]
    if direct:
        cmd.append("--direct-only")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Blender failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--run", default="pbr_learned")
    p.add_argument("--data", default=str(ROOT / "data/Synthetic4Relight"))
    p.add_argument("--frame-stride", type=int, default=20)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--no-align", action="store_true", help="export albedo without alignment")
    a = p.parse_args()

    model, run = Path(a.model), Path(a.model) / a.run
    ds = Synthetic4Relight(a.data, a.scene)
    metrics_run = json.loads((run / "metrics.json").read_text())
    scale = [1.0, 1.0, 1.0] if a.no_align else metrics_run["albedo_scale_mean"]

    mesh = load_relight_mesh(model)
    mat = load_material(mesh, run)
    with torch.no_grad():
        v = mat.vertex_values()
        albedo = (v["albedo"] * torch.tensor(scale, device="cuda")).clamp(0, 1)
    glb = run / ("asset_aligned.glb" if not a.no_align else "asset.glb")
    rep = write_pbr_glb(mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy(), mesh.vertex_normals.cpu().numpy(),
                        albedo.cpu().numpy(), v["rough"].cpu().numpy(), v["metal"].cpu().numpy(), glb)
    print(f"[export] {rep['faces']:,} faces, {rep['pages']} page(s), {rep['bytes'] / 1e6:.1f} MB, albedo scale {scale}")

    args = torch.load(run / "material.pt", map_location="cpu", weights_only=False)["args"]
    dirs = fibonacci_sphere(args["dirs"])
    h = median_edge_length(mesh)
    vis = cached_vertex_visibility(mesh, dirs, model / f"visibility_K{args['dirs']}_t{args['vis_tmin']:g}.pt",
                                   eps=0.5 * h, tmin=args["vis_tmin"] * h)
    lut = brdf_lut()
    frames = list(range(0, ds.num_views("test"), a.frame_stride))
    cams = ds.cameras("test")
    transforms = ds.path / "transforms_test.json"
    res, rows = {}, []
    for e in RELIGHT_ENVMAPS:
        light = EnvLight(load_envmap(ds.envmap_path(e)), dirs, e)
        dirs_out = {m: run / "blender" / f"{e}_{m}" for m in ("gi", "direct")}
        for m, d in dirs_out.items():
            blender(glb, ds.envmap_path(e), transforms, frames, d, a.samples, m == "direct", cams[0].width)
        stats = {k: [] for k in ("mask_iou", "parity_psnr", "cycles_gi_vs_gt_mask", "cycles_gi_vs_gt_full",
                                 "remes_vs_gt_mask", "cycles_direct_vs_gt_mask")}
        for k, f in enumerate(frames):
            gi = torch.from_numpy(read_image(dirs_out["gi"] / f"{f:03d}.exr")).cuda()
            di = torch.from_numpy(read_image(dirs_out["direct"] / f"{f:03d}.exr")).cuda()
            ours = render_view(mesh, mat, vis, light, lut, cams[f], albedo_scale=torch.tensor(scale, device="cuda"))
            om = ours["mask"]
            cm = gi[..., 3] > 0.5
            stats["mask_iou"].append(float((om & cm).sum() / (om | cm).sum()))
            o_disp = tonemap(ours["rgb"]) * om[..., None]
            stats["parity_psnr"].append(psnr(tonemap(di[..., :3]), o_disp, om & cm))
            g_rgb, g_a = ds.relight_view(f, e)
            g = torch.from_numpy(g_rgb).cuda()
            ga = torch.from_numpy(g_a).cuda()
            gmask = ga > 0.5
            gi_disp = tonemap(gi[..., :3])
            stats["cycles_gi_vs_gt_mask"].append(psnr(gi_disp, g, gmask))
            stats["cycles_gi_vs_gt_full"].append(psnr(gi_disp, g * ga[..., None]))
            stats["remes_vs_gt_mask"].append(psnr(o_disp, g, gmask))
            stats["cycles_direct_vs_gt_mask"].append(psnr(tonemap(di[..., :3]), g, gmask))
            if k < 2:
                write_png(run / "blender" / f"cmp_{e}_{f:03d}.png",
                          torch.cat([g * ga[..., None], gi_disp, tonemap(di[..., :3]), o_disp], 1).cpu().numpy())
        res[e] = {k: float(np.mean(v)) for k, v in stats.items()}
        print(f"[{e}] " + "  ".join(f"{k}={v:.3f}" for k, v in res[e].items()), flush=True)
    import_report = json.loads((run / "blender" / f"{RELIGHT_ENVMAPS[0]}_gi" / "import_report.json").read_text())
    out = {"frames": frames, "samples": a.samples, "albedo_scale": scale, "export": rep, "results": res,
           "import": import_report}
    (run / "blender_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"-> {run / 'blender_metrics.json'}")


if __name__ == "__main__":
    main()
