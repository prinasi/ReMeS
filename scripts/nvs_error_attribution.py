"""Where does the remaining NVS error come from? Region attribution of the full-image (antialiased) error.

The test views are rendered exactly as in the full-image evaluation (ss x ss supersampled, premultiplied) and
the squared display-space error is split over pixel regions:

  edge       2-px band around the GT silhouette (soft alpha / antialiasing / coverage)
  coverage   outside the band, covered by only one of GT / prediction
  specular   interior, our predicted specular >= 30 % of the radiance
  occluded   interior (not specular), >= 40 % of the cosine-weighted hemisphere blocked (concavities,
             where interreflections live)
  diffuse    the remaining interior
  (highlights: interior pixels among the brightest 2 % of the GT object, reported separately, overlapping)

For each region: share of the total squared error, PSNR inside the region and pixel share.

    python scripts/nvs_error_attribution.py -m output/oto_s4r/v6_t2init/jugs --scene jugs --run r4_fr4_nrm_ssim
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from relight.data import TRAIN_ENVMAP, Synthetic4Relight  # noqa: E402
from relight.envmap import load_envmap  # noqa: E402
from relight.gbuffer import rasterize  # noqa: E402
from relight.indirect import IndirectLight  # noqa: E402
from relight.io import tonemap, write_png  # noqa: E402
from relight.mesh import load_relight_mesh  # noqa: E402
from relight.shading import EnvLight, LearnableEnvLight, brdf_lut, fibonacci_sphere  # noqa: E402
from relight.visibility import cached_vertex_visibility, interpolate_visibility, median_edge_length  # noqa: E402
from export_pbr import load_material  # noqa: E402
from train_pbr import render_view, render_view_ss  # noqa: E402

REGIONS = ("edge", "coverage", "specular", "occluded", "diffuse")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--data", default=str(ROOT / "data/Synthetic4Relight"))
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--ss", type=int, default=2)
    a = p.parse_args()
    model, run = Path(a.model), Path(a.model) / a.run
    args = torch.load(run / "material.pt", map_location="cpu", weights_only=False)["args"]
    mesh = load_relight_mesh(model)
    mat = load_material(mesh, run)
    dirs = fibonacci_sphere(args["dirs"])
    h = median_edge_length(mesh)
    vis = cached_vertex_visibility(mesh, dirs, model / f"visibility_K{args['dirs']}_t{args['vis_tmin']:g}.pt",
                                   eps=0.5 * h, tmin=args["vis_tmin"] * h)
    ind = IndirectLight.cached(mesh, vis, dirs, model / f"indirect_K{args['dirs']}_t{args['vis_tmin']:g}.pt",
                               eps=0.5 * h, tmin=args["vis_tmin"] * h) if args.get("indirect") else None
    ds = Synthetic4Relight(a.data, a.scene)
    if args.get("learn_light"):
        light = LearnableEnvLight(dirs, (args["light_res"], 2 * args["light_res"])).cuda()
        light.log_env.data.copy_(torch.load(run / "light.pt")["log_env"].cuda())
        light.refresh()
    else:
        light = EnvLight(load_envmap(ds.envmap_path(args.get("light", TRAIN_ENVMAP))), dirs, "train")
    lut = brdf_lut()
    cosw = None
    acc = {r: [0.0, 0] for r in (*REGIONS, "highlights")}
    edge_gtcov = 0.0
    total_px, total_se = 0, 0.0
    for i in range(0, ds.num_views("test"), a.stride):
        tv = ds.test_view(i)
        cam = tv.camera
        gt = torch.from_numpy(tv.rgb).cuda()
        ga = torch.from_numpy(tv.mask).cuda()
        gt_p = gt * ga[..., None]
        full = render_view_ss(mesh, mat, vis, light, lut, cam, a.ss, indirect=ind)
        se = ((tonemap(full["rgb"]) - gt_p) ** 2).mean(-1)                       # [H, W]
        # same colours with the GT coverage: what remains in the edge band is shading, not silhouette position
        col = tonemap(full["rgb"] / full["alpha"].clamp_min(1e-3))
        se_gtcov = ((col * (full["alpha"] > 1e-3) * ga[..., None] - gt_p) ** 2).mean(-1)
        # labels from the plain render
        o = render_view(mesh, mat, vis, light, lut, cam, indirect=ind)
        g = rasterize(mesh, cam)
        m = o["mask"]
        band = ((ga > 0.01) & (ga < 0.99)).float()
        band = F.max_pool2d(band[None, None], 5, 1, 2)[0, 0] > 0
        inside_gt = ga >= 0.99
        coverage = ~band & (m ^ inside_gt)
        interior = ~band & m & inside_gt
        rad = o["rgb"].sum(-1)
        spec_share = o["specular"].sum(-1) / rad.clamp_min(1e-6)
        occl = torch.zeros_like(rad)
        if interior.any():
            fid, bary = g["face_id"][interior], g["bary"][interior]
            n = F.normalize((bary[:, :, None] * mesh.vertex_normals[mesh.faces[fid].long()]).sum(1), dim=-1)
            cos = (n @ dirs.T).clamp_min(0)
            v = interpolate_visibility(vis, mesh.faces, fid, bary)
            occl[interior] = 1 - (v * cos).sum(-1) / cos.sum(-1).clamp_min(1e-6)
        specular = interior & (spec_share >= 0.3)
        occluded = interior & ~specular & (occl >= 0.4)
        diffuse = interior & ~specular & ~occluded
        lum = gt.mean(-1)
        thr = torch.quantile(lum[inside_gt], 0.98) if inside_gt.any() else 1.0
        highlights = interior & (lum >= thr)
        labels = {"edge": band, "coverage": coverage, "specular": specular, "occluded": occluded,
                  "diffuse": diffuse, "highlights": highlights}
        edge_gtcov += float(se_gtcov[band].sum())
        for r, sel in labels.items():
            acc[r][0] += float(se[sel].sum())
            acc[r][1] += int(sel.sum())
        total_se += float(se.sum())
        total_px += se.numel()
        if i == 0:
            err = (se / se.max().clamp_min(1e-8)).sqrt()
            lab = torch.zeros(*se.shape, 3, device="cuda")
            for r, c in (("edge", (1, 1, 0)), ("coverage", (1, 0, 1)), ("specular", (0, 1, 1)),
                         ("occluded", (1, 0.4, 0)), ("diffuse", (0.3, 0.3, 1))):
                lab[labels[r]] = torch.tensor(c, dtype=torch.float32, device="cuda")
            write_png(run / "nvs_error_attribution.png",
                      torch.cat([gt_p, tonemap(full["rgb"]), err[..., None].expand(-1, -1, 3), lab], 1).cpu().numpy())
    out = {"psnr_full": -10 * math.log10(total_se / total_px), "regions": {},
           "edge_error_with_gt_coverage_share_of_edge": edge_gtcov / max(acc["edge"][0], 1e-12),
           "psnr_full_if_edge_had_gt_coverage": -10 * math.log10((total_se - acc["edge"][0] + edge_gtcov) / total_px)}
    for r, (s, n) in acc.items():
        out["regions"][r] = {"error_share": s / total_se, "pixel_share": n / total_px,
                             "psnr": -10 * math.log10(s / max(n, 1)) if s > 0 else float("inf")}
    (run / "nvs_error_attribution.json").write_text(json.dumps(out, indent=2))
    print(f"[{a.scene}] full-image PSNR {out['psnr_full']:.2f}; edge error left with GT coverage: "
          f"{100 * out['edge_error_with_gt_coverage_share_of_edge']:.1f} % -> PSNR "
          f"{out['psnr_full_if_edge_had_gt_coverage']:.2f}")
    for r, v in out["regions"].items():
        print(f"   {r:<11} error {100 * v['error_share']:5.1f} %   pixels {100 * v['pixel_share']:5.2f} %   "
              f"PSNR {v['psnr']:.2f}")


if __name__ == "__main__":
    main()
