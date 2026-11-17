"""Where does the relighting / albedo error come from? Region split on SynthGeo (GT mesh labels).

For each material run, test views are relit (envmap6/12, run's albedo alignment) and compared with the GT
relit images; pixels are labelled by the GT mesh as object (Suzanne) or slab, and the slab further as
"contact" (within --contact of the object footprint, where shadows / interreflections are) or "open".
Reports PSNR per region and each region's share of the total squared error.

    python scripts/attribute_error.py --scene suzanne_textured gt=output/oracle_geo/suzanne_textured v6=output/oto/init_t2/suzanne_textured
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from relight.data import RELIGHT_ENVMAPS, Synthetic4Relight  # noqa: E402
from relight.envmap import load_envmap  # noqa: E402
from relight.gbuffer import rasterize  # noqa: E402
from relight.io import gamma_to_linear, tonemap  # noqa: E402
from relight.mesh import load_relight_mesh  # noqa: E402
from relight.shading import EnvLight, brdf_lut, fibonacci_sphere  # noqa: E402
from relight.visibility import cached_vertex_visibility, median_edge_length  # noqa: E402
from eval_geometry_gt import load_gt  # noqa: E402
from export_pbr import load_material  # noqa: E402
from train_pbr import render_view  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True)
    p.add_argument("--data", default=str(ROOT / "data/SynthGeo"))
    p.add_argument("--run", default="pbr_oraclegeo")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--contact", type=int, default=40, help="contact band around the object footprint (px)")
    p.add_argument("models", nargs="+", help="name=model_dir")
    a = p.parse_args()
    ds = Synthetic4Relight(a.data, a.scene)
    gt_mesh, n_obj = load_gt(Path(a.data) / a.scene / "gt_mesh.ply")
    frames = list(range(0, ds.num_views("test"), a.stride))
    cams = ds.cameras("test")
    lut = brdf_lut()
    regions = ("object", "slab_contact", "slab_open")
    labels = {}
    for f in frames:
        g = rasterize(gt_mesh, cams[f])
        obj = g["mask"] & (g["face_id"] < n_obj)
        slab = g["mask"] & (g["face_id"] >= n_obj)
        k = 2 * a.contact + 1
        near = F.max_pool2d(obj.float()[None, None], k, stride=1, padding=a.contact)[0, 0] > 0
        labels[f] = {"object": obj, "slab_contact": slab & near, "slab_open": slab & ~near}
    report = {}
    for spec in a.models:
        name, mdir = spec.split("=", 1)
        model, run = Path(mdir), Path(mdir) / a.run
        mesh = load_relight_mesh(model)
        mat = load_material(mesh, run)
        met = json.loads((run / "metrics.json").read_text())
        scale = torch.tensor(met["albedo_scale_mean"], device="cuda")
        args = torch.load(run / "material.pt", map_location="cpu", weights_only=False)["args"]
        dirs = fibonacci_sphere(args["dirs"])
        h = median_edge_length(mesh)
        vis = cached_vertex_visibility(mesh, dirs, model / f"visibility_K{args['dirs']}_t{args['vis_tmin']:g}.pt",
                                       eps=0.5 * h, tmin=args["vis_tmin"] * h)
        sse = {f"{e}/{r}": 0.0 for e in list(RELIGHT_ENVMAPS) + ["albedo"] for r in regions}
        cnt = {k: 0 for k in sse}
        for e in RELIGHT_ENVMAPS:
            light = EnvLight(load_envmap(ds.envmap_path(e)), dirs, e)
            for f in frames:
                o = render_view(mesh, mat, vis, light, lut, cams[f], albedo_scale=scale)
                pred = tonemap(o["rgb"]) * o["mask"][..., None]
                g_rgb, g_a = ds.relight_view(f, e)
                gt = torch.from_numpy(g_rgb).cuda()
                err = ((pred - gt) ** 2).sum(-1)
                for r in regions:
                    m = labels[f][r] & (torch.from_numpy(g_a).cuda() > 0.5)
                    sse[f"{e}/{r}"] += float(err[m].sum())
                    cnt[f"{e}/{r}"] += int(m.sum()) * 3
                if e == RELIGHT_ENVMAPS[0]:
                    tv = ds.test_view(f)
                    ga = gamma_to_linear(torch.from_numpy(tv.albedo).cuda())
                    ea = ((o["albedo"] - ga) ** 2).sum(-1)
                    for r in regions:
                        m = labels[f][r] & o["mask"]
                        sse[f"albedo/{r}"] += float(ea[m].sum())
                        cnt[f"albedo/{r}"] += int(m.sum()) * 3
        res = {}
        for e in list(RELIGHT_ENVMAPS) + ["albedo"]:
            tot = sum(sse[f"{e}/{r}"] for r in regions)
            for r in regions:
                k = f"{e}/{r}"
                res[k] = {"psnr": -10 * np.log10(sse[k] / max(cnt[k], 1) + 1e-12), "share": sse[k] / max(tot, 1e-12)}
        report[name] = res
    print(f"{'region':28s}" + "".join(f"{n:>22s}" for n in report))
    for e in list(RELIGHT_ENVMAPS) + ["albedo"]:
        for r in regions:
            k = f"{e}/{r}"
            print(f"{k:28s}" + "".join(f"{report[n][k]['psnr']:12.2f} ({100 * report[n][k]['share']:4.0f}%)  "
                                        for n in report))
    out = Path(a.models[0].split("=", 1)[1]).parent.parent / "attribution" / f"{a.scene}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
