"""Test the RDT hypothesis on SynthGeo (GT geometry): are vertices on the true surface when
RDT runs, does RDT connect them into correct triangles, and does later training repair it?

For every saved checkpoint (point_cloud/iteration_*; train with --save_topology_snapshots
and extra save_iterations), relative to the GT Suzanne surface (% of its diagonal):
  vertices   distance to GT (median / p90 / share > OFF), split by opacity quartile
  triangles  area-weighted centroid distance, normal error vs GT, share of "bad" triangles
             (centroid > OFF from GT or normal error > 30°)
  layering   share of camera rays with a second surface within 1 / 3 median edges

    python scripts/analyze_rdt.py -m output/synthgeo_rdt/suzanne_diffuse --data data/SynthGeo/suzanne_diffuse
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import cubvh  # noqa: E402
from diagnose_geometry import layering  # noqa: E402
from eval_geometry_gt import load_gt  # noqa: E402
from relight.data import Synthetic4Relight  # noqa: E402
from relight.mesh import FrozenMesh  # noqa: E402
from relight.visibility import median_edge_length  # noqa: E402

OFF = 0.25          # % of diagonal considered "off-surface"


def sort_key(name: str):
    m = re.match(r"iteration_(\d+)(?:_(before|after)_rdt)?$", name)
    it, stage = int(m.group(1)), m.group(2)
    return it, {"before": 0, None: 1, "after": 2}[stage]


@torch.no_grad()
def analyze(ckpt: Path, gt: FrozenMesh, n_obj: int, bvh, diag: float, ds) -> dict:
    st = torch.load(ckpt, map_location="cuda", weights_only=False)
    V = st["triangles_points"].float()
    Fc = st["_triangle_indices"].long()
    used = torch.zeros(V.shape[0], dtype=torch.bool, device="cuda")
    used[Fc.reshape(-1)] = True
    op = torch.sigmoid(st["vertex_weight"].float().reshape(-1))
    d, fid, _ = bvh.unsigned_distance(V)
    d = 100 * d / diag
    sel = used & (fid < n_obj)
    dv, ov = d[sel], op[sel]
    q = torch.quantile(ov, torch.tensor([0.25, 0.75], device="cuda"))
    rep = {"vertices": int(used.sum()), "faces": int(Fc.shape[0]),
           "vert_dist_median": float(dv.median()), "vert_dist_p90": float(torch.quantile(dv, 0.9)),
           "vert_off_share": float((dv > OFF).float().mean()),
           "vert_dist_median_low_opacity_q1": float(dv[ov <= q[0]].median()),
           "vert_dist_median_high_opacity_q4": float(dv[ov >= q[1]].median()),
           "opacity_median": float(ov.median())}
    mesh = FrozenMesh(V, Fc.int())
    c = V[Fc].mean(1)
    dc, fc, _ = bvh.unsigned_distance(c)
    dc = 100 * dc / diag
    on = fc < n_obj
    ang = torch.rad2deg(torch.acos((mesh.face_normals * gt.face_normals[fc]).sum(-1).abs().clamp(0, 1)))
    w = mesh.face_areas * on
    bad = ((dc > OFF) | (ang > 30)).float()
    rep.update(tri_centroid_dist_wmean=float((dc * w).sum() / w.sum()),
               tri_normal_err_wmean=float((ang * w).sum() / w.sum()),
               tri_bad_share_area=float((bad * w).sum() / w.sum()))
    rep.update(layering(mesh, ds, median_edge_length(mesh), n_views=6))
    return rep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--data", required=True)
    a = p.parse_args()
    data = Path(a.data)
    ds = Synthetic4Relight(data.parent, data.name)
    gt, n_obj = load_gt(data / "gt_mesh.ply")
    obj = gt.vertices[gt.faces[:n_obj].long()].reshape(-1, 3)
    diag = float((obj.max(0).values - obj.min(0).values).norm())
    bvh = cubvh.cuBVH(gt.vertices, gt.faces)
    ckpts = sorted((p for p in (Path(a.model) / "point_cloud").glob("iteration_*")
                    if (p / "point_cloud_state_dict.pt").is_file()), key=lambda p: sort_key(p.name))
    out = {}
    for c in ckpts:
        out[c.name] = analyze(c / "point_cloud_state_dict.pt", gt, n_obj, bvh, diag, ds)
        r = out[c.name]
        print(f"{c.name:28s} V={r['vertices']:>8,} F={r['faces']:>8,} | vert dist med {r['vert_dist_median']:.3f} "
              f"p90 {r['vert_dist_p90']:.3f} off {100 * r['vert_off_share']:4.1f}% (low-op {r['vert_dist_median_low_opacity_q1']:.3f}"
              f" / high-op {r['vert_dist_median_high_opacity_q4']:.3f}) | tri dist {r['tri_centroid_dist_wmean']:.3f} "
              f"nerr {r['tri_normal_err_wmean']:.1f}° bad {100 * r['tri_bad_share_area']:4.1f}% | layering "
              f"{100 * r['second_surface_within_1_edges']:.0f}%", flush=True)
    (Path(a.model) / "rdt_analysis.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
