"""Ground-truth geometry as a relight mesh (oracle-geometry ablation on SynthGeo).

The SynthGeo GT mesh (subdivided Suzanne + a 12-triangle slab cube) is isotropically remeshed to the
median edge length of a reference reconstruction, so per-vertex materials / visibility have the same
resolution as in the reconstructed meshes. Slab vertices are split per cube side so the sides get exact
flat normals; Suzanne gets area-weighted smooth normals (as rendered by Blender's shade_smooth).

    python scripts/gt_relight_mesh.py --data data/SynthGeo/suzanne_glossy --ref output/oto/init_t2/suzanne_glossy \
        --out output/oracle_geo/suzanne_glossy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relight.mesh import FrozenMesh, load_relight_mesh, save_relight_mesh  # noqa: E402
from relight.orient import vertex_normals  # noqa: E402
from relight.remesh import isotropic_remesh  # noqa: E402
from relight.visibility import median_edge_length  # noqa: E402

SLAB_FACES = 12


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="SynthGeo scene dir with gt_mesh.ply")
    p.add_argument("--ref", required=True, help="reconstruction whose median edge length is matched")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    m = trimesh.load(Path(a.data) / "gt_mesh.ply", process=False)
    V, F = np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int64)
    h = float(median_edge_length(load_relight_mesh(a.ref)))
    parts = []
    # object: remesh, smooth normals
    Vo, Fo = isotropic_remesh(V, F[:-SLAB_FACES], h)
    parts.append((Vo, Fo, None))
    # slab: subdivide, then split vertices per cube side (axis of the face normal)
    Vs, Fs = trimesh.remesh.subdivide_to_size(V, F[-SLAB_FACES:], max_edge=1.5 * h)   # planar: exact
    Vs, Fs = Vs.astype(np.float32), Fs.astype(np.int64)
    fn = np.cross(Vs[Fs[:, 1]] - Vs[Fs[:, 0]], Vs[Fs[:, 2]] - Vs[Fs[:, 0]])
    axes = np.concatenate([np.eye(3), -np.eye(3)])
    side = np.argmax(fn @ axes.T, 1)
    for s in range(6):
        fs = Fs[side == s]
        if len(fs) == 0:
            continue
        used, inv = np.unique(fs, return_inverse=True)
        parts.append((Vs[used], inv.reshape(-1, 3), axes[s]))
    Vall, Fall, Nall, off = [], [], [], 0
    for v, f, n in parts:
        vt, ft = torch.tensor(v, device="cuda"), torch.tensor(f, dtype=torch.int32, device="cuda")
        nt = vertex_normals(vt, ft) if n is None else torch.tensor(n, dtype=torch.float32, device="cuda").expand_as(vt)
        Vall.append(vt), Fall.append(ft + off), Nall.append(nt)
        off += len(v)
    mesh = FrozenMesh(torch.cat(Vall), torch.cat(Fall).contiguous(), vertex_normals=torch.cat(Nall).contiguous())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_relight_mesh(mesh, out / "relight_mesh.pt", {"source": "gt_mesh (remeshed)", "edge": h})
    print(f"[gt] edge {h:.4f}: object {len(Fo):,} + slab {len(Fs):,} faces -> {out / 'relight_mesh.pt'}")


if __name__ == "__main__":
    main()
