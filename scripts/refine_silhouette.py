"""Silhouette refinement of a final relight mesh (geometry only, materials untouched).

About half of the remaining full-image NVS error sits in a 2-px band around the silhouette, and 50-85 % of it is
coverage (silhouette position), not shading (scripts/nvs_error_attribution.py). This stage moves the vertices so
that the antialiased coverage of the mesh matches the soft GT masks of all training views:

    D* = argmin Σ_views ‖ AA-coverage(V0 + D) − mask ‖²  +  λ_lap ‖L D‖²  +  λ_anchor ‖D‖²

(nvdiffrast rasterize + antialias gives analytic gradients at silhouette edges; the uniform Laplacian of the
*displacement* keeps the correction smooth without flattening the surface detail). Faces are unchanged, so the
orientation is preserved; vertex normals are recomputed. The result (relight_mesh.pt + env.pt) goes to --out for
the material stage (visibility / indirect caches are rebuilt there).

    python scripts/refine_silhouette.py -m output/oto_s4r/v6_t2init/jugs --scene jugs --out output/oto_s4r/v8_sil/jugs
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import nvdiffrast.torch as dr
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relight.data import Synthetic4Relight  # noqa: E402
from relight.gbuffer import get_context, mvp_matrix  # noqa: E402
from relight.mesh import FrozenMesh, load_relight_mesh, save_relight_mesh  # noqa: E402
from relight.orient import vertex_normals  # noqa: E402

SKIP_TOP_ROWS = 5


def coverage(V, faces, mvp, res):
    clip = (torch.cat([V, torch.ones_like(V[:, :1])], -1) @ mvp.T)[None].contiguous()
    rast, _ = dr.rasterize(get_context(V.device), clip, faces, resolution=res)
    a = (rast[..., 3:4] > 0).float()
    return dr.antialias(a.contiguous(), rast, clip, faces)[0, ..., 0]


def laplacian(n, edges):
    a, b = edges[:, 0], edges[:, 1]
    deg = torch.zeros(n, device=edges.device).index_add_(0, a, torch.ones_like(a, dtype=torch.float)) \
        .index_add_(0, b, torch.ones_like(b, dtype=torch.float))
    def apply(D):
        nb = torch.zeros_like(D).index_add_(0, a, D[b]).index_add_(0, b, D[a])
        return nb / deg.clamp_min(1)[:, None] - D
    return apply


@torch.no_grad()
def coverage_metrics(V, faces, views):
    err, band_err, inter, union = 0.0, 0.0, 0.0, 0.0
    for mvp, m, res in views:
        c = coverage(V, faces, mvp, res)
        c[:SKIP_TOP_ROWS] = m[:SKIP_TOP_ROWS]
        band = (m > 0.01) & (m < 0.99) | ((c > 0.01) & (c < 0.99))
        err += float((c - m).abs().mean())
        band_err += float((c - m).abs()[band].mean())
        inter += float(((c > 0.5) & (m > 0.5)).sum())
        union += float(((c > 0.5) | (m > 0.5)).sum())
    n = len(views)
    return {"abs_err": err / n, "band_abs_err": band_err / n, "iou": inter / union}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--data", default=str(ROOT / "data/Synthetic4Relight"))
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--batch", type=int, default=4, help="views per iteration")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lambda-lap", type=float, default=50.0)
    p.add_argument("--lambda-anchor", type=float, default=1.0)
    p.add_argument("--large-steps", type=float, default=0.0, help="lambda of relight/largesteps.py (0 = Adam)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    t0 = time.perf_counter()
    mesh = load_relight_mesh(a.model)
    faces = mesh.faces.int().contiguous()
    V0 = mesh.vertices.detach().clone()
    ds = Synthetic4Relight(a.data, a.scene)

    def load(split):
        out = []
        for i in range(ds.num_views(split)):
            v = ds.train_view(i) if split == "train" else ds.test_view(i)
            m = torch.from_numpy(v.mask).float().cuda()
            out.append((mvp_matrix(v.camera).cuda(), m, [v.camera.height, v.camera.width]))
        return out
    train = load("train")
    test = load("test")[::10]
    before = coverage_metrics(V0, faces, test)
    print(f"[sil] before: test {before}", flush=True)

    f = faces.long()
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = torch.unique(torch.sort(e, 1).values, dim=0)
    lap = laplacian(V0.shape[0], edges)
    D = torch.zeros_like(V0, requires_grad=True)
    if a.large_steps > 0:
        from relight.largesteps import LargeSteps
        ls = LargeSteps(V0.shape[0], edges, lam=a.large_steps, lr=a.lr)
        opt = None
    else:
        opt = torch.optim.Adam([D], lr=a.lr)
    for it in range(1, a.iters + 1):
        lr = a.lr * 0.1 ** (it / a.iters)
        if opt is not None:
            opt.param_groups[0]["lr"] = lr
        else:
            ls.lr = lr
        V = V0 + D
        l_sil = 0.0
        for mvp, m, res in random.sample(train, a.batch):
            c = coverage(V, faces, mvp, res)
            l_sil = l_sil + ((c - m) ** 2)[SKIP_TOP_ROWS:].mean()
        l_sil = l_sil / a.batch
        loss = l_sil + a.lambda_lap * (lap(D) ** 2).sum(-1).mean() + a.lambda_anchor * (D ** 2).sum(-1).mean()
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        else:
            D.grad = None
            loss.backward()
            ls.step(D)
        if it % 250 == 0 or it == 1:
            with torch.no_grad():
                dmax = float(D.norm(dim=-1).max())
                dmean = float(D.norm(dim=-1).mean())
            print(f"[sil] it {it:5d}  sil {float(l_sil):.6f}  |D| mean {dmean:.5f} max {dmax:.4f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    V = (V0 + D).detach().contiguous()
    after = coverage_metrics(V, faces, test)
    print(f"[sil] after:  test {after}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_relight_mesh(FrozenMesh(V, faces, vertex_normals=vertex_normals(V, faces)), out / "relight_mesh.pt",
                      {"stage": "silhouette_refined", "source": str(a.model), "args": vars(a)})
    if (Path(a.model) / "env.pt").is_file():
        shutil.copy(Path(a.model) / "env.pt", out / "env.pt")
    (out / "silhouette_refine.json").write_text(json.dumps({"before": before, "after": after, "args": vars(a),
                                                            "seconds": time.perf_counter() - t0}, indent=2))


if __name__ == "__main__":
    main()
