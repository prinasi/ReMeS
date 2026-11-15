"""Geometry diagnostic D3: cleaned variants of a relight mesh for downstream sensitivity tests.

Variants (each saved as <out>/<scene>/<variant>/relight_mesh.pt so train_pbr.py /
eval_blender.py run on them unchanged):
  orig       the Phase 1 relight mesh
  taubin10   Taubin smoothing, 10 iterations (keeps the sheet structure)
  taubin50   Taubin smoothing, 50 iterations
  poisson9   screened Poisson remesh (depth 9) from oriented surface samples -> single sheet
Reports silhouette fidelity (test mask IoU), layering and normal noise per variant.

    python scripts/make_geometry_variants.py --scene hotdog
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_geometry import layering, profile  # noqa: E402
from relight.data import Synthetic4Relight  # noqa: E402
from relight.gbuffer import rasterize  # noqa: E402
from relight.geomfilter import poisson, taubin  # noqa: E402
from relight.mesh import load_relight_mesh, save_relight_mesh  # noqa: E402
from relight.visibility import median_edge_length  # noqa: E402


@torch.no_grad()
def mask_iou(mesh, ds, stride=8):
    ious = []
    for i in range(0, ds.num_views("test"), stride):
        tv = ds.test_view(i, with_materials=False)
        pred = rasterize(mesh, tv.camera)["mask"]
        gt = torch.from_numpy(tv.mask).cuda() > 0.5
        ious.append(float((pred & gt).sum() / (pred | gt).sum()))
    return float(np.mean(ious)), float(np.min(ious))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True)
    p.add_argument("--model-root", default=str(ROOT / "output/synthetic4relight"))
    p.add_argument("--data", default=str(ROOT / "data/Synthetic4Relight"))
    p.add_argument("--out", default=str(ROOT / "output/geomvar"))
    p.add_argument("--variants", nargs="+", default=["orig", "taubin10", "taubin50", "poisson9"])
    a = p.parse_args()
    base = load_relight_mesh(Path(a.model_root) / a.scene)
    ds = Synthetic4Relight(a.data, a.scene)
    diag = float((base.vertices.max(0).values - base.vertices.min(0).values).norm())
    report = {}
    for v in a.variants:
        if v == "orig":
            m = base
        elif v.startswith("taubin"):
            m = taubin(base, int(v[6:]))
        elif v.startswith("poisson"):
            m = poisson(base, depth=int(v[7:]))
        else:
            raise ValueError(v)
        d = Path(a.out) / a.scene / v
        d.mkdir(parents=True, exist_ok=True)
        save_relight_mesh(m, d / "relight_mesh.pt", {"variant": v, "source": base.source})
        h = median_edge_length(m)
        iou = mask_iou(m, ds)
        r = {"faces": int(m.faces.shape[0]), "median_edge": h, "mask_iou_mean": iou[0], "mask_iou_min": iou[1],
             **layering(m, ds, median_edge_length(base)), **profile(m, taubin(m, 50), h, diag)}
        report[v] = r
        print(f"[{a.scene}/{v}] F={r['faces']:,} IoU {iou[0]:.4f} (min {iou[1]:.4f})  2nd surface <1 edge "
              f"{100 * r['second_surface_within_1_edges']:.1f}%  normal dev median {r['normal_dev_median_deg']:.1f}°",
              flush=True)
    (Path(a.out) / a.scene / "variants.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
