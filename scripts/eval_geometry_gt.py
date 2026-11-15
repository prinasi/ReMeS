"""Geometry diagnostic D4: compare a reconstructed mesh against ground-truth geometry.

Metrics (object = Suzanne only; the slab's 12 GT faces are the last ones in gt_mesh.ply):
  accuracy      mean / median distance from predicted surface samples to the GT surface
  completeness  mean / median distance from GT samples to the predicted surface
  normal error  angle between predicted face normals and the GT normal at the closest
                GT point (orientation-agnostic), and with predicted smooth vertex normals
  screen normal MAE on test views (GT smooth normals vs predicted smooth / face normals)
  layering      share of camera rays with a second predicted surface within 1 / 3 edges
Distances are reported in % of the object bounding-box diagonal.

    python scripts/eval_geometry_gt.py -m output/synthgeo/suzanne_glossy --data data/SynthGeo/suzanne_glossy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cubvh  # noqa: E402
from diagnose_geometry import layering  # noqa: E402
from relight.data import Synthetic4Relight  # noqa: E402
from relight.gbuffer import rasterize  # noqa: E402
from relight.geomfilter import sample_surface  # noqa: E402
from relight.mesh import FrozenMesh, load_relight_mesh  # noqa: E402
from relight.orient import vertex_normals  # noqa: E402
from relight.visibility import median_edge_length  # noqa: E402

SLAB_FACES = 12


def load_gt(path) -> tuple[FrozenMesh, int]:
    m = trimesh.load(path, process=False)
    V = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device="cuda")
    Fc = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device="cuda")
    return FrozenMesh(V, Fc, vertex_normals=vertex_normals(V, Fc)), Fc.shape[0] - SLAB_FACES


@torch.no_grad()
def evaluate(pred: FrozenMesh, gt: FrozenMesh, n_obj: int, ds, n_samples=1_000_000) -> dict:
    obj = FrozenMesh(gt.vertices, gt.faces[:n_obj], vertex_normals=gt.vertex_normals)
    diag = float((obj.vertices.max(0).values - obj.vertices.min(0).values).norm())
    bvh_gt = cubvh.cuBVH(gt.vertices, gt.faces)
    bvh_pred = cubvh.cuBVH(pred.vertices, pred.faces)
    # predicted samples near the object (closest GT face belongs to Suzanne)
    ps, pn, pf = sample_surface(pred, n_samples)
    d, fid, uvw = bvh_gt.unsigned_distance(ps, return_uvw=True)
    on_obj = fid < n_obj
    gfn = gt.face_normals[fid]
    pfn = pred.face_normals[pf]
    ang_face = torch.rad2deg(torch.acos((pfn * gfn).sum(-1).abs().clamp(0, 1)))
    ang_smooth = torch.rad2deg(torch.acos((pn * gfn).sum(-1).abs().clamp(0, 1)))
    gs, _, _ = sample_surface(obj, n_samples)
    dc, _, _ = bvh_pred.unsigned_distance(gs)
    pct = lambda x: 100 * x / diag
    rep = {"diag": diag, "pred_faces": int(pred.faces.shape[0]),
           "accuracy_mean_pct": pct(float(d[on_obj].mean())), "accuracy_median_pct": pct(float(d[on_obj].median())),
           "completeness_mean_pct": pct(float(dc.mean())), "completeness_median_pct": pct(float(dc.median())),
           "normal_err_face_median_deg": float(ang_face[on_obj].median()),
           "normal_err_face_mean_deg": float(ang_face[on_obj].mean()),
           "normal_err_smooth_median_deg": float(ang_smooth[on_obj].median()),
           "normal_err_smooth_mean_deg": float(ang_smooth[on_obj].mean())}
    # screen-space normal MAE on Suzanne pixels
    errs_s, errs_f = [], []
    for i in range(0, ds.num_views("test"), 5):
        cam = ds.cameras("test")[i]
        gg, gp = rasterize(gt, cam), rasterize(pred, cam)
        m = gg["mask"] & gp["mask"] & (gg["face_id"] < n_obj)
        gnorm = F.normalize((gg["bary"][..., :, None] * gt.vertex_normals[gt.faces.long()[gg["face_id"].clamp_min(0)]]).sum(-2), dim=-1)
        pv = pred.vertex_normals[pred.faces.long()[gp["face_id"].clamp_min(0)]]
        psm = F.normalize((gp["bary"][..., :, None] * pv).sum(-2), dim=-1)
        errs_s.append(torch.rad2deg(torch.acos((psm * gnorm).sum(-1).abs().clamp(0, 1)))[m])
        errs_f.append(torch.rad2deg(torch.acos((gp["normal"] * gnorm).sum(-1).abs().clamp(0, 1)))[m])
    es, ef = torch.cat(errs_s), torch.cat(errs_f)
    rep.update(screen_normal_mae_smooth_deg=float(es.mean()), screen_normal_mae_face_deg=float(ef.mean()),
               screen_normal_median_smooth_deg=float(es.median()))
    rep.update(layering(pred, ds, median_edge_length(pred)))
    return rep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", required=True, help="dir with relight_mesh.pt")
    p.add_argument("--data", required=True, help="SynthGeo scene dir (with gt_mesh.ply)")
    a = p.parse_args()
    data = Path(a.data)
    ds = Synthetic4Relight(data.parent, data.name)
    gt, n_obj = load_gt(data / "gt_mesh.ply")
    pred = load_relight_mesh(a.model)
    rep = evaluate(pred, gt, n_obj, ds)
    out = Path(a.model) / "geometry_gt.json"
    out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
