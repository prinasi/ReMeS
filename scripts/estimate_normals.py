"""Monocular normal priors for the training views (run with the isolated env: .venv_normals/bin/python).

Each training view (linear EXR -> tonemapped sRGB, object composited on black) is passed to a monocular
normal estimator; the camera-space prediction is converted to world space (Blender camera: x right, y up,
z backwards) and saved as float16 [H, W, 3] next to a confidence map:

    <out>/{i:03d}.npz  {"normal": world-space unit normals (float16), "valid": bool mask}

With --gt-mesh (SynthGeo) the angular error against the ground-truth normals is reported per view, which is
how much the prior can be trusted.

    .venv_normals/bin/python scripts/estimate_normals.py --data data/SynthGeo --scene suzanne_glossy \\
        --out output/priors/suzanne_glossy/marigold --gt-mesh data/SynthGeo/suzanne_glossy/gt_mesh.ply
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relight.data import Synthetic4Relight  # noqa: E402
from relight.io import tonemap, write_png  # noqa: E402

SKIP_TOP_ROWS = 5


def load_model(name: str):
    if name == "marigold":
        from diffusers import MarigoldNormalsPipeline
        pipe = MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1", variant="fp16",
                                                       torch_dtype=torch.float16).to("cuda")
        pipe.set_progress_bar_config(disable=True)

        def run(img: np.ndarray, ensemble: int) -> np.ndarray:
            out = pipe(img, num_inference_steps=4, ensemble_size=ensemble, processing_resolution=768,
                       match_input_resolution=True, output_type="np")
            return np.asarray(out.prediction[0], np.float32)       # [H, W, 3], camera space (x right, y up, z out)
        return run
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="marigold")
    p.add_argument("--ensemble", type=int, default=4)
    p.add_argument("--views", default="", help="comma list of view indices (default all)")
    p.add_argument("--gt-mesh", default="", help="SynthGeo GT mesh: report angular error vs GT normals")
    p.add_argument("--flip", default="", help="camera-space sign flips to apply, e.g. 'y' or 'xz' (convention)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = Synthetic4Relight(a.data, a.scene)
    views = [int(v) for v in a.views.split(",")] if a.views else list(range(ds.num_views("train")))
    run = load_model(a.model)
    gt = None
    if a.gt_mesh:
        sys.path.insert(0, str(ROOT / "scripts"))
        from eval_geometry_gt import load_gt
        from relight.gbuffer import rasterize
        gt, _ = load_gt(a.gt_mesh)
    flip = torch.ones(3, device="cuda")
    for c, ax in enumerate("xyz"):
        if ax in a.flip:
            flip[c] = -1
    errs, t0 = [], time.perf_counter()
    for i in views:
        v = ds.train_view(i)
        m = v.mask > 0.5
        m[:SKIP_TOP_ROWS] = False
        img = (tonemap(torch.from_numpy(v.linear)).numpy() * m[..., None]).clip(0, 1)
        n_cam = torch.from_numpy(run(img, a.ensemble)).cuda()
        n_cam = F.normalize(n_cam * flip, dim=-1)
        R = torch.as_tensor(v.camera.c2w[:3, :3], dtype=torch.float32, device="cuda")   # Blender camera axes
        n_w = F.normalize(n_cam @ R.T, dim=-1)
        valid = torch.from_numpy(m).cuda()
        np.savez_compressed(out / f"{i:03d}.npz", normal=n_w.half().cpu().numpy(), valid=m)
        rec = {"view": i}
        if gt is not None:
            g = rasterize(gt, v.camera)
            both = valid & g["mask"]
            ang = torch.rad2deg(torch.acos((n_w * g["normal"]).sum(-1).clamp(-1, 1)))[both]
            rec.update(mean=float(ang.mean()), median=float(ang.median()), p90=float(ang.quantile(0.9)))
            errs.append(rec)
            if len(errs) <= 2:
                vis = torch.cat([(n_w * 0.5 + 0.5) * valid[..., None], (g["normal"] * 0.5 + 0.5) * both[..., None]], 1)
                write_png(out / f"cmp_{i:03d}.png", vis.cpu().numpy())
        print(f"[{i:03d}] {rec} ({time.perf_counter() - t0:.0f}s)", flush=True)
    if errs:
        summ = {k: float(np.mean([e[k] for e in errs])) for k in ("mean", "median", "p90")}
        (out / "gt_error.json").write_text(json.dumps({"summary": summ, "views": errs}, indent=2))
        print(f"[gt] mean angular error over views: {summ}")


if __name__ == "__main__":
    main()
