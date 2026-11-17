"""Synthetic4Relight comparison in the format of R3DG Tab. 2 (full-image metrics, albedo per-channel aligned;
relighting = mean over the relighting envmaps, aligned albedo). Literature rows are copied from R3DG (Gao et al.,
arXiv 2311.16043, Tab. 2); ours are read from train_pbr.py metrics.json (4-scene means).

    python scripts/compare_s4r.py output/oto_s4r/v6_t2init:pbr_learned_ind_init_lr5e3 [more root:run ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCENES = ("air_baloons", "chair", "hotdog", "jugs")
ENVS = ("envmap6", "envmap12")
# NVS PSNR/SSIM/LPIPS | Relight PSNR/SSIM/LPIPS | Albedo PSNR/SSIM/LPIPS | Rough MSE
LIT = {
    "NeRFactor": (22.80, 0.916, 0.150, 21.54, 0.875, 0.171, 19.49, 0.864, 0.206, None),
    "Nvdiffrec-MC (mesh)": (34.29, 0.967, 0.068, 24.22, 0.943, 0.078, 29.61, 0.945, 0.075, 0.009),
    "InvRender": (30.74, 0.953, 0.086, 28.67, 0.950, 0.091, 28.28, 0.935, 0.072, 0.008),
    "TensoIR": (35.80, 0.978, 0.049, 29.69, 0.951, 0.079, 30.58, 0.946, 0.065, 0.015),
    "R3DG": (36.80, 0.982, 0.028, 31.00, 0.964, 0.050, 28.31, 0.951, 0.058, 0.013),
}


def ours(root: str, run: str):
    rows = []
    for s in SCENES:
        m = json.loads((Path(root) / s / run / "metrics.json").read_text())
        rel = lambda k: float(np.mean([m[f"relight_{e}_{k}_full_aligned"] for e in ENVS]))
        rows.append((m["nvs_psnr_full"], m["nvs_ssim_full"], m["nvs_lpips_full"], rel("psnr"), rel("ssim"),
                     rel("lpips"), m["albedo_psnr_aligned_full"], m["albedo_ssim_full"], m["albedo_lpips_full"],
                     m["rough_mse_full"]))
    return tuple(np.mean(np.array(rows), 0))


def fmt(v, k):
    if v is None:
        return "—"
    return f"{v:.4f}" if k == 9 else f"{v:.3f}" if k % 3 else f"{v:.2f}"


def main():
    print("| Method | NVS PSNR | SSIM | LPIPS | Relight PSNR | SSIM | LPIPS | Albedo PSNR | SSIM | LPIPS | Rough MSE |")
    print("|---|" + "---|" * 10)
    for name, v in LIT.items():
        print(f"| {name} | " + " | ".join(fmt(x, k) for k, x in enumerate(v)) + " |")
    for spec in sys.argv[1:]:
        root, run = spec.rsplit(":", 1)
        v = ours(root, run)
        print(f"| **ours** {Path(root).name}/{run} | " + " | ".join(fmt(x, k) for k, x in enumerate(v)) + " |")


if __name__ == "__main__":
    main()
