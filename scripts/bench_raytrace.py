"""Phase 0: correctness + throughput of frozen-mesh ray queries (relight/raytrace.py).

Scene: two tessellated spheres (A at the origin, B beside it). Shadow rays leave
points on A along random directions; the analytic answer is "occluded" if the
ray points into A (d·n < 0) or intersects sphere B. Near-grazing rays are
excluded because tessellation makes them ambiguous.

    python scripts/bench_raytrace.py --faces 1000000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from relight.raytrace import MeshTracer  # noqa: E402

RA, CB, RB = 0.5, torch.tensor([1.0, 0.0, 0.0]), 0.4


def build_scene(target_faces: int):
    sub = max(1, int(round(math.log(max(target_faces / 2, 20) / 20, 4))))
    a = trimesh.creation.icosphere(subdivisions=sub, radius=RA)
    b = trimesh.creation.icosphere(subdivisions=sub, radius=RB)
    b.apply_translation(CB.numpy())
    m = trimesh.util.concatenate([a, b])
    return (torch.tensor(m.vertices, dtype=torch.float32, device="cuda"),
            torch.tensor(m.faces, dtype=torch.long, device="cuda"), len(a.faces))


def timed(fn, reps=3):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) / reps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--faces", type=int, default=1_000_000)
    p.add_argument("--rays", type=int, default=16_000_000)
    p.add_argument("--eps", type=float, default=1e-3)
    p.add_argument("--out", default=str(ROOT / "output/phase0/bench_raytrace.json"))
    a = p.parse_args()
    torch.manual_seed(0)

    V, Fc, fa = build_scene(a.faces)
    n = F.normalize(torch.randn(a.rays, 3, device="cuda"), dim=-1)
    d = F.normalize(torch.randn(a.rays, 3, device="cuda"), dim=-1)
    o = n * RA + a.eps * n                      # rays leave points on the analytic sphere A

    # analytic reference: into A (d·n < 0) or through sphere B
    cos = (d * n).sum(-1)
    oc = o - CB.cuda()
    bq = (oc * d).sum(-1)
    disc = bq * bq - ((oc * oc).sum(-1) - RB ** 2)
    ref = (cos < 0) | ((disc > 0) & (-bq - torch.sqrt(disc.clamp_min(0)) > 0))
    w = cos.clamp_min(0)                        # diffuse weight; lower hemisphere irrelevant

    report = {"faces": Fc.shape[0], "vertices": V.shape[0], "rays": a.rays, "eps": a.eps}
    for verify in (False, True):
        tag = "verified" if verify else "raw"
        t0 = time.perf_counter()
        tracer = MeshTracer(V, Fc, verify=verify)
        torch.cuda.synchronize()
        build_s = time.perf_counter() - t0
        occ, t_any = timed(lambda: tracer.any_hit(o, d))
        (hit, t, fid, bary), _ = timed(lambda: tracer.closest_hit(o, d), reps=1)
        wrong = occ != ref
        up = cos > 0
        werr = (w * wrong.float()).sum().item() / w.sum().item()
        pos = o + t[:, None] * d
        tri = tracer.vertices[tracer.faces[fid[hit]]]
        bary_err = ((bary[hit][:, :, None] * tri).sum(1) - pos[hit]).norm(dim=-1).max().item()
        bands = {}
        for lo, hi in [(0, .02), (.02, .05), (.05, .1), (.1, .2), (.2, 1.01)]:
            b = up & (cos >= lo) & (cos < hi)
            bands[f"{lo}-{hi}"] = wrong[b].float().mean().item()
        print(f"[{tag:8s}] {V.shape[0]:,} verts / {Fc.shape[0]:,} faces, BVH {build_s:.2f}s | "
              f"{a.rays / t_any / 1e6:6.1f} M rays/s | cos-weighted vis. error {100 * werr:.4f}% | "
              f"all-ray error {100 * wrong.float().mean().item():.4f}% | bary err {bary_err:.1e}")
        print("           upper-hemisphere error by cos band: "
              + "  ".join(f"[{k}) {100 * v:.3f}%" for k, v in bands.items()))
        report[tag] = {"bvh_build_s": build_s, "any_hit_mrays_s": a.rays / t_any / 1e6,
                       "cos_weighted_error": werr, "error": wrong.float().mean().item(),
                       "error_by_cos_band": bands, "bary_max_err": bary_err,
                       "precompute_1M_vertices_K512_s": 1e6 * 512 / (a.rays / t_any)}
    print("per-vertex visibility precompute, 1M vertices x 512 dirs: "
          f"{report['verified']['precompute_1M_vertices_K512_s']:.1f}s")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"report -> {a.out}")


if __name__ == "__main__":
    main()
