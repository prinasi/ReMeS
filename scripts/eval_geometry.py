"""Phase 1: check that a T²-Splat mesh is usable as a frozen relighting mesh.

Reports
  * topology: vertices, faces, connected components (+ area share of the largest),
    boundary / non-manifold edges, winding consistency across manifold edges;
  * silhouette: mask IoU of the rasterized mesh vs GT alpha on held-out test views;
  * orientation: fraction of visible pixels whose winding normal faces away from the
    camera (≈0 for outward, consistently wound meshes; ≈0.5 for random winding);
  * visualisations: normal maps and mask error for a few test views.

    python scripts/eval_geometry.py -m output/synthetic4relight/hotdog --scene hotdog
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

from relight.data import Synthetic4Relight  # noqa: E402
from relight.gbuffer import rasterize  # noqa: E402
from relight.io import write_png  # noqa: E402
from relight.mesh import load_relight_mesh, load_t2splat_mesh  # noqa: E402


def topology(mesh) -> dict:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    F = mesh.faces.long()
    nV = mesh.vertices.shape[0]
    directed = torch.cat([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], dim=0)
    und = torch.sort(directed, dim=1).values
    key = und[:, 0] * nV + und[:, 1]
    uniq, inv, counts = torch.unique(key, return_inverse=True, return_counts=True)
    per_edge = counts[inv]
    boundary = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    # consistent winding: a manifold edge is traversed once in each direction
    dkey = directed[:, 0] * nV + directed[:, 1]
    _, dcounts = torch.unique(dkey[per_edge == 2], return_counts=True)
    inconsistent = int((dcounts > 1).sum())
    manifold_edges = int((counts == 2).sum())

    f_np = F.cpu().numpy()
    rows = np.concatenate([f_np[:, 0], f_np[:, 1], f_np[:, 2]])
    cols = np.concatenate([f_np[:, 1], f_np[:, 2], f_np[:, 0]])
    adj = coo_matrix((np.ones_like(rows), (rows, cols)), shape=(nV, nV))
    ncomp, labels = connected_components(adj, directed=False)
    area = mesh.face_areas.cpu().numpy()
    comp_area = np.bincount(labels[f_np[:, 0]], weights=area, minlength=ncomp)
    return {"vertices": nV, "faces": int(F.shape[0]), "edges": int(uniq.numel()),
            "components": int(ncomp), "largest_component_area_share": float(comp_area.max() / comp_area.sum()),
            "boundary_edges": boundary, "nonmanifold_edges": nonmanifold,
            "winding_inconsistent_edge_share": inconsistent / max(manifold_edges, 1),
            "total_area": float(area.sum())}


@torch.no_grad()
def silhouettes(mesh, ds: Synthetic4Relight, stride: int, vis_dir: Path | None, n_vis: int = 4) -> dict:
    ious, backs, idx_all = [], [], range(0, ds.num_views("test"), stride)
    vis_idx = set(list(idx_all)[:: max(1, len(idx_all) // n_vis)][:n_vis])
    for i in idx_all:
        view = ds.test_view(i, with_materials=False)
        g = rasterize(mesh, view.camera)
        pred = g["mask"]
        gt = torch.from_numpy(view.mask).cuda() > 0.5
        ious.append(((pred & gt).sum() / (pred | gt).sum().clamp_min(1)).item())
        backs.append((g["backfacing"].sum() / pred.sum().clamp_min(1)).item())
        if vis_dir is not None and i in vis_idx:
            n = (g["normal"] * 0.5 + 0.5) * pred[..., None]
            err = torch.stack([(pred & ~gt).float(), (gt & ~pred).float(), torch.zeros_like(gt, dtype=torch.float)], -1)
            bf = torch.stack([g["backfacing"].float()] * 3, -1)
            write_png(vis_dir / f"test_{i:03d}.png", torch.cat([n, err, bf], dim=1).cpu().numpy())
    return {"views": len(ious), "mask_iou_mean": float(np.mean(ious)), "mask_iou_min": float(np.min(ious)),
            "backfacing_pixel_share_mean": float(np.mean(backs))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--data", default=str(ROOT / "data/Synthetic4Relight"))
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--stride", type=int, default=4, help="evaluate every n-th test view")
    p.add_argument("--relight-mesh", action="store_true",
                   help="evaluate <model>/relight_mesh.pt (oriented) instead of the raw checkpoint")
    a = p.parse_args()

    model = Path(a.model)
    mesh = load_relight_mesh(model) if a.relight_mesh else load_t2splat_mesh(model, a.iteration)
    ds = Synthetic4Relight(a.data, a.scene)
    out = model / ("geometry_eval_oriented" if a.relight_mesh else "geometry_eval")
    out.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": mesh.source, "topology": topology(mesh),
              "silhouette": silhouettes(mesh, ds, a.stride, out)}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    t, s = report["topology"], report["silhouette"]
    print(f"[{a.scene}] V={t['vertices']:,} F={t['faces']:,} components={t['components']} "
          f"(largest {100 * t['largest_component_area_share']:.1f}% area) boundary={t['boundary_edges']:,} "
          f"non-manifold={t['nonmanifold_edges']:,} winding-inconsistent={100 * t['winding_inconsistent_edge_share']:.2f}%")
    print(f"[{a.scene}] test mask IoU mean={s['mask_iou_mean']:.4f} min={s['mask_iou_min']:.4f} "
          f"back-facing pixels={100 * s['backfacing_pixel_share_mean']:.1f}%  -> {out}")


if __name__ == "__main__":
    main()
