"""Collect metrics.json of PBR runs into a Markdown table.

    python scripts/summarize_pbr.py --runs pbr_oracle pbr_oracle_novis
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLS = [("nvs_psnr_mask", "NVS"), ("albedo_psnr_aligned", "Albedo (aligned)"), ("albedo_psnr_raw", "Albedo (raw)"),
        ("rough_mse", "Rough MSE"), ("relight_envmap6_psnr_mask", "Relight env6"),
        ("relight_envmap12_psnr_mask", "Relight env12"), ("nvs_psnr_full", "NVS (full img)"),
        ("relight_envmap12_psnr_full", "Relight env12 (full img)")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT / "output/synthetic4relight"))
    p.add_argument("--runs", nargs="+", default=["pbr_oracle"])
    p.add_argument("--scenes", nargs="+", default=["air_baloons", "chair", "hotdog", "jugs"])
    a = p.parse_args()
    header = "| Run | Scene | " + " | ".join(c[1] for c in COLS) + " |"
    lines = [header, "|" + "---|" * (len(COLS) + 2)]
    for run in a.runs:
        acc = {k: [] for k, _ in COLS}
        for s in a.scenes:
            f = Path(a.root) / s / run / "metrics.json"
            if not f.is_file():
                lines.append(f"| {run} | {s} | " + " | ".join("—" for _ in COLS) + " |")
                continue
            m = json.loads(f.read_text())
            for k, _ in COLS:
                acc[k].append(m[k])
            fmt = lambda k: f"{m[k]:.4f}" if k == "rough_mse" else f"{m[k]:.2f}"
            lines.append(f"| {run} | {s} | " + " | ".join(fmt(k) for k, _ in COLS) + " |")
        if all(len(v) == len(a.scenes) for v in acc.values()):
            mean = {k: sum(v) / len(v) for k, v in acc.items()}
            fmt = lambda k: f"{mean[k]:.4f}" if k == "rough_mse" else f"{mean[k]:.2f}"
            lines.append(f"| **{run}** | **mean** | " + " | ".join(f"**{fmt(k)}**" for k, _ in COLS) + " |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
