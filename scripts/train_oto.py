"""Stage A: OTO-style opaque triangle soup with structured (diffuse + reflection-specular)
appearance, optimised on nvdiffrast (relight/oto.py).

    python scripts/train_oto.py --data data/SynthGeo --scene suzanne_glossy --out output/oto/suzanne_glossy

Outputs <out>/soup.pt (parameters), env.pt, relight_mesh.pt (3F corners, for
scripts/eval_geometry_gt.py / diagnose_geometry.py), train_log.json and preview images.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relight.data import Synthetic4Relight  # noqa: E402
from relight.envmap import downsample_envmap, load_envmap  # noqa: E402
from relight.gbuffer import mvp_matrix  # noqa: E402
from relight.io import tonemap, write_png  # noqa: E402
from relight.mesh import FrozenMesh, save_relight_mesh  # noqa: E402
from relight.oto import OpaqueSoup, visual_hull_soup  # noqa: E402
from relight.shading import LearnableEnvLight, brdf_lut, fibonacci_sphere  # noqa: E402

SKIP_TOP_ROWS = 5


def gamma(x):
    return (x.clamp_min(0) + 1e-4) ** (1 / 2.2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=15000)
    p.add_argument("--hull-res", type=int, default=128)
    p.add_argument("--max-tris", type=int, default=600_000)
    p.add_argument("--densify-from", type=int, default=500)
    p.add_argument("--densify-until", type=int, default=10000)
    p.add_argument("--densify-every", type=int, default=500)
    p.add_argument("--cov-split", type=float, default=12.0, help="split if max pixel coverage exceeds this")
    p.add_argument("--cov-prune", type=float, default=1.0, help="prune if max pixel coverage is below this")
    p.add_argument("--lr-pos", type=float, default=2e-4)
    p.add_argument("--lr-pos-final", type=float, default=2e-5)
    p.add_argument("--lr-color", type=float, default=1e-2)
    p.add_argument("--lr-spec", type=float, default=1e-2)
    p.add_argument("--lr-env", type=float, default=5e-3)
    p.add_argument("--env-res", type=int, default=64)
    p.add_argument("--env-tv", type=float, default=1e-3)
    p.add_argument("--oracle-env", default="", help="envmap name to use as a fixed light (diagnostic)")
    p.add_argument("--no-specular", action="store_true", help="ablation: diffuse only")
    p.add_argument("--lambda-mask", type=float, default=0.1)
    p.add_argument("--lambda-dn", type=float, default=0.05, help="normal-depth consistency weight")
    p.add_argument("--dn-from", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    ds = Synthetic4Relight(a.data, a.scene)
    views = []
    for i in range(ds.num_views("train")):
        v = ds.train_view(i)
        m = torch.from_numpy(v.mask).cuda()
        m[:SKIP_TOP_ROWS] = 0
        gt = gamma(torch.from_numpy(v.linear).cuda() * m[..., None])
        views.append((v.camera, m, gt, mvp_matrix(v.camera).cuda()))
    print(f"[data] {len(views)} train views", flush=True)

    verts = visual_hull_soup([(c, m.cpu().numpy()) for c, m, _, _ in views], res=a.hull_res, max_tris=a.max_tris)
    soup = OpaqueSoup(verts, lr_pos=a.lr_pos, lr_color=a.lr_color, lr_spec=a.lr_spec)
    print(f"[init] visual hull {a.hull_res}^3 -> {soup.num:,} triangles", flush=True)

    lut = brdf_lut()
    light = None
    if not a.no_specular:
        light = LearnableEnvLight(fibonacci_sphere(64), (a.env_res, 2 * a.env_res), init=1.0).cuda()
        if a.oracle_env:
            env = downsample_envmap(load_envmap(ds.envmap_path(a.oracle_env)), a.env_res, 2 * a.env_res)
            light.log_env.data.copy_(torch.log(env.clamp_min(1e-4)))
            light.log_env.requires_grad_(False)
        light_opt = torch.optim.Adam([light.log_env], lr=a.lr_env)

    log = []
    for it in range(1, a.iters + 1):
        frac = it / a.iters
        soup.set_lr("verts", a.lr_pos * (a.lr_pos_final / a.lr_pos) ** frac)
        cam, m, gt, mvp = random.choice(views)
        if light is not None:
            light.refresh()
        o = soup.render(cam, light, lut, mvp=mvp, track_coverage=True)
        l_img = (gamma(o["rgb"]) - gt).abs().mean()
        l_mask = (o["alpha"] - m).abs().mean()
        loss = l_img + a.lambda_mask * l_mask
        l_dn = torch.zeros((), device="cuda")
        if a.lambda_dn > 0 and it >= a.dn_from:
            l_dn = soup.depth_normal_loss(o)
            loss = loss + a.lambda_dn * l_dn
        if light is not None and a.env_tv > 0 and light.log_env.requires_grad:
            loss = loss + a.env_tv * light.tv()
        loss.backward()
        soup.step()
        if light is not None and light.log_env.requires_grad:
            light_opt.step()
            light_opt.zero_grad(set_to_none=True)

        # coverage-based density control
        if it % a.densify_every == 0 and it >= a.densify_from:
            n_prune = soup.prune(soup.max_cov < a.cov_prune) if it >= 1000 else 0
            n_split = 0
            if it <= a.densify_until:
                cand = (soup.max_cov > a.cov_split).nonzero()[:, 0]
                room = max(0, a.max_tris - soup.num)
                if cand.numel() > room:
                    cand = cand[torch.argsort(soup.max_cov[cand], descending=True)[:room]]
                n_split = soup.split(cand)
            soup.reset_coverage()
            with torch.no_grad():
                psnr = -10 * math.log10(((gamma(o["rgb"]) - gt) ** 2).mean().item())
            rec = {"iter": it, "tris": soup.num, "pruned": n_prune, "split": n_split, "L1": l_img.item(),
                   "mask": l_mask.item(), "dn": float(l_dn), "psnr": psnr}
            log.append(rec)
            print(f"[{it:6d}] tris {soup.num:>8,} (-{n_prune:,} +{n_split:,})  L1 {l_img.item():.4f}  "
                  f"mask {l_mask.item():.4f}  dn {float(l_dn):.4f}  PSNR {psnr:.2f}  ({time.perf_counter() - t0:.0f}s)",
                  flush=True)

    # final cleanup: drop triangles never visible in a full pass over the training views
    soup.reset_coverage()
    with torch.no_grad():
        if light is not None:
            light.refresh()
        for cam, m, gt, mvp in views:
            soup.render(cam, light, lut, mvp=mvp, track_coverage=True)
    n_final = soup.prune(soup.max_cov < 1)
    print(f"[final] removed {n_final:,} never-visible triangles -> {soup.num:,}", flush=True)

    torch.save(soup.state_dict(), out / "soup.pt")
    if light is not None:
        torch.save({"log_env": light.log_env.detach().cpu()}, out / "env.pt")
    V = soup.verts.detach().reshape(-1, 3)
    fn = soup.face_normals().detach()
    mesh = FrozenMesh(V.contiguous(), soup.faces(), vertex_normals=fn.repeat_interleave(3, 0))
    save_relight_mesh(mesh, out / "relight_mesh.pt", {"stage": "oto_soup", "args": vars(a)})
    with torch.no_grad():
        for k, (cam, m, gt, mvp) in enumerate(views[::25]):
            o = soup.render(cam, light, lut, mvp=mvp)
            nimg = (o["normal"] * 0.5 + 0.5) * o["mask"][..., None]
            write_png(out / "images" / f"train_{k}.png",
                      torch.cat([gt, gamma(o["rgb"]).clamp(0, 1), nimg], 1).cpu().numpy())
    (out / "train_log.json").write_text(json.dumps({"args": vars(a), "log": log, "final_tris": soup.num,
                                                    "seconds": time.perf_counter() - t0}, indent=2))
    print(f"-> {out}  ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
