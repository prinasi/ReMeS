"""Phase 1 post-process: T²-Splat checkpoint -> oriented frozen mesh for relighting.

Steps: load mesh (+ face textures) -> remove hidden interior faces (unseen by every
training camera and fully enclosed) -> orient faces outward (training-camera votes,
ray escape test for unseen faces) -> area-weighted vertex normals -> save
<model>/relight_mesh.pt (Phase 2 input) and <model>/relight_mesh.ply (inspection).

    python scripts/prepare_relight_mesh.py -m output/synthetic4relight/hotdog --scene hotdog
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relight.data import Synthetic4Relight  # noqa: E402
from relight.mesh import RELIGHT_MESH, load_t2splat_mesh, save_relight_mesh  # noqa: E402
from relight.orient import normal_noise_deg, orient_mesh, remove_hidden_faces, vertex_normals  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--data", default=str(ROOT / "data/Synthetic4Relight"))
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--escape-rays", type=int, default=32)
    p.add_argument("--keep-hidden", action="store_true", help="do not remove enclosed, unseen faces")
    a = p.parse_args()

    model = Path(a.model)
    t0 = time.perf_counter()
    mesh = load_t2splat_mesh(model, a.iteration)
    cams = Synthetic4Relight(a.data, a.scene).cameras("train")
    hidden = {}
    if not a.keep_hidden:
        mesh, hidden = remove_hidden_faces(mesh, cams)
    oriented, stats = orient_mesh(mesh, cams, escape_rays=a.escape_rays)
    stats["hidden_removal"] = hidden
    oriented.vertex_normals = vertex_normals(oriented.vertices, oriented.faces)
    stats["normal_noise"] = normal_noise_deg(oriented, oriented.vertex_normals)
    stats["seconds"] = time.perf_counter() - t0
    save_relight_mesh(oriented, model / RELIGHT_MESH, meta={"orientation": stats, "scene": a.scene})

    import trimesh
    trimesh.Trimesh(oriented.vertices.cpu().numpy(), oriented.faces.cpu().numpy(),
                    vertex_normals=oriented.vertex_normals.cpu().numpy(), process=False
                    ).export(model / "relight_mesh.ply")
    (model / "relight_mesh.json").write_text(json.dumps(stats, indent=2))
    nn = stats["normal_noise"]
    if hidden:
        print(f"[{a.scene}] removed {hidden['removed_faces']:,}/{hidden['faces_before']:,} enclosed unseen faces "
              f"({100 * hidden['removed_area_share']:.1f}% of area)")
    print(f"[{a.scene}] flipped {stats['flipped']:,}/{stats['faces']:,} faces; seen {stats['seen_faces']:,}, "
          f"unseen {stats['unseen_faces']:,}; vote agreement mean {stats['vote_agreement_mean']:.3f} "
          f"(p05 {stats['vote_agreement_p05']:.3f}); face-vs-vertex normal deviation median "
          f"{nn['area_weighted_median_deg']:.1f}° p90 {nn['area_weighted_p90_deg']:.1f}° "
          f"({stats['seconds']:.1f}s) -> {model / RELIGHT_MESH}")


if __name__ == "__main__":
    main()
