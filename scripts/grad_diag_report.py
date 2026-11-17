"""Which loss terms hold the SynthGeo slab top where it is? Region analysis of `train_oto.py --grad-diag`.

Vertices are labelled with the GT mesh: slab-top "open" (up-facing, away from Suzanne's footprint), "contact"
(up-facing, near the footprint), "rim" (up-facing, near the slab edge) and "object". For each loss term the
descent direction -g (summed over all training views) is projected on +z (the slab normal): > 0 means the term
pushes the slab top UP. The image term is split by gradient path:

    shadow occluders  = img - img_occ_detached
    shadow receivers  = img_occ_detached - img_shadow_detached
    shading normals   = img_shadow_detached - img_shadow_nrm_detached
    raster / position = img_shadow_nrm_detached      (silhouette & occlusion antialias, view direction)

    python scripts/grad_diag_report.py --data data/SynthGeo/suzanne_textured <run>/diag_start.pt <run>/diag_end.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_geometry_gt import load_gt  # noqa: E402
from relight.orient import vertex_normals  # noqa: E402

SLAB_TOP = -0.680


def regions(V, N, gt, n_obj, foot=0.12, contact=0.30, rim=0.12):
    obj_v = gt.vertices[gt.faces[:n_obj].long().reshape(-1)].unique(dim=0)
    low = obj_v[obj_v[:, 2] < SLAB_TOP + 0.15]                        # Suzanne's lower part (footprint)
    d_xy = torch.cdist(V[:, :2], low[:, :2]).min(1).values
    slab_v = gt.vertices[gt.faces[n_obj:].long().reshape(-1)]
    xmax, ymax = slab_v[:, 0].abs().max(), slab_v[:, 1].abs().max()
    up = (N[:, 2] > 0.8) & ((V[:, 2] - SLAB_TOP).abs() < 0.15)
    near_edge = (xmax - V[:, 0].abs() < rim) | (ymax - V[:, 1].abs() < rim)
    out = {"rim": up & near_edge, "contact": up & ~near_edge & (d_xy < contact) & (d_xy > foot),
           "open": up & ~near_edge & (d_xy >= contact)}
    out["object"] = (V[:, 2] > SLAB_TOP + 0.05) & ~up
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("diags", nargs="+")
    a = p.parse_args()
    gt, n_obj = load_gt(Path(a.data) / "gt_mesh.ply")
    for path in a.diags:
        d = torch.load(path, map_location="cuda")
        V, Fc, g = d["verts"].cuda(), d["faces"].cuda(), {k: v.cuda() for k, v in d["grads"].items()}
        N = vertex_normals(V, Fc)
        reg = regions(V, N, gt, n_obj)
        if d["physical"]:
            parts = {"shadow occluders": g["img"] - g["img_occ_detached"],
                     "shadow receivers": g["img_occ_detached"] - g["img_shadow_detached"],
                     "shading normals": g["img_shadow_detached"] - g["img_shadow_nrm_detached"],
                     "raster / position": g["img_shadow_nrm_detached"]}
        else:
            parts = {"shading normals": g["img"] - g["img_shadow_nrm_detached"],
                     "raster / position": g["img_shadow_nrm_detached"]}
        for k in ("mask", "dn", "lap", "nc", "albedo"):
            if g[k].abs().sum() > 0:
                parts[k] = g[k]
        total = sum(parts.values())
        parts["TOTAL"] = total
        print(f"\n== {path}  ({'physical' if d['physical'] else 'radiance'} shading)")
        print("   region sizes: " + ", ".join(f"{k} {int(m.sum())}" for k, m in reg.items()))
        for rn in ("open", "contact", "rim"):
            m = reg[rn]
            if not m.any():
                continue
            z = V[m, 2]
            print(f"   [{rn}] slab-top height: median {float(z.median()):+.4f} (GT {SLAB_TOP:+.3f}, "
                  f"offset {float(z.median()) - SLAB_TOP:+.4f})")
            print(f"   {'term':<20} {'Σ push up (-g_z)':>16} {'share of |push|':>16} {'verts pushing up':>17}")
            denom = sum(float((-v[m, 2]).abs().sum()) for k, v in parts.items() if k != "TOTAL") + 1e-20
            for k, v in parts.items():
                up = -v[m, 2]
                print(f"   {k:<20} {float(up.sum()):>+16.3e} {float(up.abs().sum()) / denom:>15.1%} "
                      f"{float((up > 0).float().mean()):>16.1%}")


if __name__ == "__main__":
    main()
