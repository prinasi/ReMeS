"""Geometry diagnostic D1 + D2 for a ReMeS relight mesh (no GT geometry needed).

D1  error profile
    * bump height: |Δx · n| between the mesh and its Taubin-smoothed self (10 / 50 its),
      in median-edge units and in % of the object diagonal;
    * normal deviation: face normal vs the smoothed surface's normal at the same place;
    * layering: gap from the first surface hit to the next one along training-camera
      rays (double sheets / shells show up as gaps of a few edge lengths).
D2  attribution (Synthetic4Relight test views with GT maps)
    per-pixel normal deviation binned by GT roughness (glossy vs rough) and by GT albedo
    gradient strength (textured vs flat) — tests "specular compensated by geometry" and
    "texture leaking into geometry".

    python scripts/diagnose_geometry.py -m output/synthetic4relight/hotdog --scene hotdog
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

from relight.data import Synthetic4Relight  # noqa: E402
from relight.gbuffer import rasterize  # noqa: E402
from relight.geomfilter import taubin  # noqa: E402
from relight.io import write_png  # noqa: E402
from relight.mesh import load_relight_mesh  # noqa: E402
from relight.raytrace import MeshTracer  # noqa: E402
from relight.visibility import median_edge_length  # noqa: E402


def wquant(x, w, qs):
    o = torch.argsort(x)
    cw = torch.cumsum(w[o], 0) / w.sum()
    return [float(x[o][torch.searchsorted(cw, torch.tensor(q, device=x.device)).clamp_max(len(x) - 1)]) for q in qs]


@torch.no_grad()
def profile(mesh, smooth, h, diag):
    d = ((mesh.vertices - smooth.vertices) * smooth.vertex_normals).sum(-1).abs()
    fn = mesh.face_normals
    sn = F.normalize(smooth.vertex_normals[mesh.faces.long()].sum(1), dim=-1)
    ang = torch.rad2deg(torch.acos((fn * sn).sum(-1).abs().clamp(-1, 1)))     # unsigned (orientation-agnostic)
    w = mesh.face_areas
    q = wquant(ang, w, [0.5, 0.9])
    return {"bump_rms_edges": float(d.pow(2).mean().sqrt() / h), "bump_rms_pct_diag": float(100 * d.pow(2).mean().sqrt() / diag),
            "bump_p90_edges": float(torch.quantile(d, 0.9) / h),
            "normal_dev_median_deg": q[0], "normal_dev_p90_deg": q[1]}


@torch.no_grad()
def layering(mesh, ds, h, n_views=10):
    tr = MeshTracer(mesh.vertices, mesh.faces)
    gaps = []
    for i in np.linspace(0, ds.num_views("train") - 1, n_views).astype(int):
        cam = ds.cameras("train")[i].scaled(0.25)
        o, d = cam.rays()
        o, d = o.reshape(-1, 3), d.reshape(-1, 3)
        hit, t, _, _ = tr.closest_hit(o, d)
        o2 = o[hit] + (t[hit] + 0.05 * h)[:, None] * d[hit]
        hit2, t2, _, _ = tr.closest_hit(o2, d[hit])
        g = torch.where(hit2, t2 + 0.05 * h, torch.full_like(t2, float("inf"))) / h
        gaps.append(g)
    g = torch.cat(gaps)
    return {f"second_surface_within_{k}_edges": float((g < k).float().mean()) for k in (1, 3, 10)}


@torch.no_grad()
def attribution(mesh, smooth, ds, stride=8, vis_dir=None):
    rough_bins = [(0.0, 0.3, "glossy (<0.3)"), (0.3, 0.6, "mid"), (0.6, 1.01, "rough (>0.6)")]
    stats = {"by_gt_roughness": {b[2]: [] for b in rough_bins}, "by_albedo_gradient": {}}
    grad_vals, dev_vals = [], []
    fn = mesh.face_normals
    sn = F.normalize(smooth.vertex_normals[mesh.faces.long()].sum(1), dim=-1)
    face_dev = torch.rad2deg(torch.acos((fn * sn).sum(-1).abs().clamp(-1, 1)))
    for k, i in enumerate(range(0, ds.num_views("test"), stride)):
        tv = ds.test_view(i)
        g = rasterize(mesh, tv.camera)
        m = g["mask"] & (torch.from_numpy(tv.mask).cuda() > 0.99)
        dev = torch.zeros_like(g["depth"])
        dev[g["mask"]] = face_dev[g["face_id"][g["mask"]]]
        r = torch.from_numpy(tv.rough).cuda()
        for lo, hi, name in rough_bins:
            sel = m & (r >= lo) & (r < hi)
            if sel.any():
                stats["by_gt_roughness"][name].append(dev[sel])
        lum = torch.from_numpy(tv.albedo).cuda().mean(-1)[None, None]
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device="cuda", dtype=torch.float32)[None, None] / 8
        gm = (F.conv2d(lum, kx, padding=1) ** 2 + F.conv2d(lum, kx.transpose(-1, -2), padding=1) ** 2).sqrt()
        gm = F.avg_pool2d(gm, 5, 1, 2)[0, 0]
        grad_vals.append(gm[m])
        dev_vals.append(dev[m])
        if vis_dir is not None and k < 3:
            write_png(vis_dir / f"normal_dev_{i:03d}.png",
                      torch.cat([(dev / 60).clamp(0, 1)[..., None].expand(-1, -1, 3),
                                 (1 - r).clamp(0, 1)[..., None].expand(-1, -1, 3) * m[..., None],
                                 (gm / gm.max()).clamp(0, 1)[..., None].expand(-1, -1, 3)], 1).cpu().numpy())
    out = {"by_gt_roughness": {}, "by_albedo_gradient": {}}
    for name, v in stats["by_gt_roughness"].items():
        if v:
            v = torch.cat(v)
            out["by_gt_roughness"][name] = {"pixel_share": None, "median_deg": float(v.median()), "mean_deg": float(v.mean()),
                                            "n": int(v.numel())}
    tot = sum(x["n"] for x in out["by_gt_roughness"].values())
    for x in out["by_gt_roughness"].values():
        x["pixel_share"] = x["n"] / tot
    gv, dv = torch.cat(grad_vals), torch.cat(dev_vals)
    qs = torch.quantile(gv[torch.randperm(len(gv), device=gv.device)[:1_000_000]], torch.tensor([0.5, 0.9], device=gv.device))
    for lo, hi, name in ((0, qs[0], "flat (<p50)"), (qs[0], qs[1], "p50-p90"), (qs[1], 1e9, "textured (>p90)")):
        sel = (gv >= lo) & (gv < hi)
        out["by_albedo_gradient"][name] = {"median_deg": float(dv[sel].median()), "mean_deg": float(dv[sel].mean())}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--data", default=str(ROOT / "data/Synthetic4Relight"))
    p.add_argument("--no-attribution", action="store_true", help="skip D2 (needs GT albedo/roughness maps)")
    a = p.parse_args()
    model = Path(a.model)
    mesh = load_relight_mesh(model)
    ds = Synthetic4Relight(a.data, a.scene)
    h = median_edge_length(mesh)
    diag = float((mesh.vertices.max(0).values - mesh.vertices.min(0).values).norm())
    out_dir = model / "geometry_diagnostic"
    out_dir.mkdir(exist_ok=True)
    rep = {"median_edge": h, "diag": diag, "faces": int(mesh.faces.shape[0])}
    for it in (10, 50):
        rep[f"vs_taubin{it}"] = profile(mesh, taubin(mesh, it), h, diag)
    rep["layering"] = layering(mesh, ds, h)
    if not a.no_attribution:
        rep["attribution_vs_taubin50"] = attribution(mesh, taubin(mesh, 50), ds, vis_dir=out_dir)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
