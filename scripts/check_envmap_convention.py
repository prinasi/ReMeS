"""Phase 0: validate envmap / camera conventions and shading parity with Blender.

Two independent checks:

1. Dataset backgrounds. Synthetic4Relight train EXRs show the environment
   directly behind the object. Looking up each envmap along the camera rays
   must reproduce those pixels for the true training light. This pins the
   equirect convention, the camera convention and the identity of the training
   envmap without any renderer of ours in the loop.

2. Cycles parity. Render a mirror sphere and a Lambertian sphere in Cycles
   (relight/blender/render_spheres.py) and analytically in PyTorch with the
   same envmap and camera; compare background / sphere regions.

    python scripts/check_envmap_convention.py --out output/phase0/envmap_check
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relight.data import RELIGHT_ENVMAPS, SCENES, TRAIN_ENVMAP, Synthetic4Relight  # noqa: E402
from relight.envmap import diffuse_irradiance, load_envmap, sample_envmap  # noqa: E402
from relight.io import psnr, read_exr_rgb, tonemap, write_png  # noqa: E402

HOTDOG_BLACK_ROWS = 5  # hotdog train EXRs have their top 5 rows zeroed (dataset artifact)


def background_check(root: str, views_per_scene: int) -> dict:
    report = {}
    envnames = [TRAIN_ENVMAP, *RELIGHT_ENVMAPS]
    for scene in SCENES:
        ds = Synthetic4Relight(root, scene)
        envs = {n: load_envmap(ds.envmap_path(n)) for n in envnames}
        n = ds.num_views("train")
        idx = np.linspace(0, n - 1, views_per_scene).round().astype(int)
        scores = {e: [] for e in envnames}
        for i in idx:
            v = ds.train_view(int(i))
            _, d = v.camera.rays()
            gt = torch.from_numpy(v.linear).cuda()
            bg = torch.from_numpy(v.mask).cuda() == 0
            if scene == "hotdog":
                bg[:HOTDOG_BLACK_ROWS] = False
            for e, env in envs.items():
                scores[e].append(psnr(tonemap(sample_envmap(env, d)), tonemap(gt), bg))
        report[scene] = {e: float(np.mean(s)) for e, s in scores.items()}
        best = max(report[scene], key=report[scene].get)
        print(f"[bg] {scene:12s} " + "  ".join(f"{e}={s:6.2f}dB" for e, s in report[scene].items())
              + f"   -> training light: {best}")
    return report


def render_cycles(envmap: Path, transforms: Path, frame: int, material: str, samples: int,
                  width: int, out: Path) -> Path:
    if out.is_file():
        return out
    cmd = [sys.executable, "-m", "relight.blender.render_spheres", "--envmap", str(envmap),
           "--transforms", str(transforms), "--frame", str(frame), "--material", material,
           "--samples", str(samples), "--width", str(width), "--height", str(width), "--out", str(out)]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if res.returncode != 0 or not out.is_file():
        raise RuntimeError(f"Blender render failed:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    return out


@torch.no_grad()
def render_torch(env: torch.Tensor, cam, material: str, radius: float, albedo: float):
    o, d = cam.rays()
    b = (o * d).sum(-1)
    c = (o * o).sum(-1) - radius ** 2
    disc = b * b - c
    hit = disc > 0
    t = -b - torch.sqrt(disc.clamp_min(0))
    hit &= t > 0
    out = sample_envmap(env, d)
    n = F.normalize(o + t[..., None] * d, dim=-1)
    if material == "mirror":
        r = d - 2.0 * (d * n).sum(-1, keepdim=True) * n
        shade = sample_envmap(env, r)
    else:
        shade = torch.zeros_like(out)
        shade[hit] = albedo / math.pi * diffuse_irradiance(env, n[hit])
    out = torch.where(hit[..., None], shade, out)
    return out, hit


def erode(mask: torch.Tensor, px: int) -> torch.Tensor:
    m = (~mask).float()[None, None]
    return F.max_pool2d(m, 2 * px + 1, 1, px)[0, 0] == 0


def parity_check(root: str, out_dir: Path, envnames, frames, samples_mirror, samples_diffuse,
                 width, radius, albedo) -> dict:
    ds = Synthetic4Relight(root, "hotdog")  # any scene: only cameras are used
    transforms = ds.path / "transforms_test.json"
    report = {}
    for e in envnames:
        env = load_envmap(ds.envmap_path(e))
        for frame in frames:
            cam = ds.cameras("test")[frame].scaled(width / 800)
            for material in ("mirror", "diffuse"):
                spp = samples_mirror if material == "mirror" else samples_diffuse
                exr = render_cycles(ds.envmap_path(e), transforms, frame, material, spp, width,
                                    out_dir / f"cycles_{e}_f{frame}_{material}.exr")
                ref = torch.from_numpy(read_exr_rgb(str(exr))).cuda()
                ours, hit = render_torch(env, cam, material, radius, albedo)
                sphere, bg = erode(hit, 2), erode(~hit, 2)
                a, b = tonemap(ours, "srgb"), tonemap(ref, "srgb")
                key = f"{e}/f{frame}/{material}"
                report[key] = {
                    "psnr_background": psnr(a, b, bg),
                    "psnr_sphere": psnr(a, b, sphere),
                    "linear_ratio_sphere": (ref[sphere].mean(0) / ours[sphere].mean(0).clamp_min(1e-8)).tolist(),
                }
                r = report[key]
                print(f"[cycles] {key:28s} bg={r['psnr_background']:6.2f}dB  sphere={r['psnr_sphere']:6.2f}dB  "
                      f"Cycles/ours={np.round(r['linear_ratio_sphere'], 4).tolist()}")
                err = (a - b).abs() * 10
                write_png(out_dir / f"cmp_{e}_f{frame}_{material}.png",
                          torch.cat([a, b, err.clamp(0, 1)], dim=1).cpu().numpy())
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data/Synthetic4Relight"))
    p.add_argument("--out", default=str(ROOT / "output/phase0/envmap_check"))
    p.add_argument("--envmaps", default=",".join([TRAIN_ENVMAP, *RELIGHT_ENVMAPS]))
    p.add_argument("--frames", default="0,100")
    p.add_argument("--width", type=int, default=400)
    p.add_argument("--samples-mirror", type=int, default=64)
    p.add_argument("--samples-diffuse", type=int, default=2048)
    p.add_argument("--radius", type=float, default=0.6)
    p.add_argument("--albedo", type=float, default=0.8)
    p.add_argument("--bg-views", type=int, default=10)
    p.add_argument("--skip-cycles", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    report = {"background": background_check(a.data, a.bg_views)}
    if not a.skip_cycles:
        report["cycles"] = parity_check(a.data, out, a.envmaps.split(","), [int(f) for f in a.frames.split(",")],
                                        a.samples_mirror, a.samples_diffuse, a.width, a.radius, a.albedo)
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"report -> {out / 'report.json'}")


if __name__ == "__main__":
    main()
