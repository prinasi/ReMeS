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
        ("relight_envmap12_psnr_full", "Relight env12 (full img)"),
        ("relight_envmap6_psnr_mask_aligned", "Relight env6 (albedo-aligned)"),
        ("relight_envmap12_psnr_mask_aligned", "Relight env12 (albedo-aligned)"),
        ("light_psnr_aligned", "Light PSNR (aligned)")]


def _fmt(d, k):
    if k not in d:
        return "—"
    return f"{d[k]:.4f}" if k == "rough_mse" else f"{d[k]:.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(ROOT / "output/synthetic4relight"))
    p.add_argument("--runs", nargs="+", default=["pbr_oracle"])
    p.add_argument("--scenes", nargs="+", default=["air_baloons", "chair", "hotdog", "jugs"])
    p.add_argument("--blender", action="store_true", help="also tabulate blender_metrics.json (Phase 5)")
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
                if k in m:
                    acc[k].append(m[k])
            lines.append(f"| {run} | {s} | " + " | ".join(_fmt(m, k) for k, _ in COLS) + " |")
        mean = {k: sum(v) / len(v) for k, v in acc.items() if len(v) == len(a.scenes)}
        if mean:
            lines.append(f"| **{run}** | **mean** | " + " | ".join(f"**{_fmt(mean, k)}**" if k in mean else "—"
                                                                 for k, _ in COLS) + " |")
    print("\n".join(lines))
    if a.blender:
        print()
        print(blender_table(a))


BCOLS = [("mask_iou", "Mask IoU"), ("parity_psnr", "Parity (Cycles direct vs ReMeS)"),
         ("cycles_gi_vs_gt_mask", "Cycles GI vs GT"), ("cycles_direct_vs_gt_mask", "Cycles direct vs GT"),
         ("remes_vs_gt_mask", "ReMeS vs GT"), ("cycles_gi_vs_gt_full", "Cycles GI vs GT (full img)")]


def blender_table(a) -> str:
    lines = ["| Run | Scene | Light | " + " | ".join(c[1] for c in BCOLS) + " |", "|" + "---|" * (len(BCOLS) + 3)]
    for run in a.runs:
        acc = {}
        for s in a.scenes:
            f = Path(a.root) / s / run / "blender_metrics.json"
            if not f.is_file():
                continue
            res = json.loads(f.read_text())["results"]
            for e, r in res.items():
                acc.setdefault(e, []).append(r)
                lines.append(f"| {run} | {s} | {e} | " + " | ".join(
                    f"{r[k]:.3f}" if k == "mask_iou" else f"{r[k]:.2f}" for k, _ in BCOLS) + " |")
        for e, rs in acc.items():
            if len(rs) == len(a.scenes):
                m = {k: sum(r[k] for r in rs) / len(rs) for k, _ in BCOLS}
                lines.append(f"| **{run}** | **mean** | {e} | " + " | ".join(
                    f"**{m[k]:.3f}**" if k == "mask_iou" else f"**{m[k]:.2f}**" for k, _ in BCOLS) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
