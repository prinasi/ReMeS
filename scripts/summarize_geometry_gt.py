"""Tabulate geometry_gt.json (scripts/eval_geometry_gt.py) across SynthGeo methods.

    python scripts/summarize_geometry_gt.py --methods T2-Splat=output/synthgeo deferred=output/synthgeo_deferred
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLS = [("accuracy_mean_pct", "Acc. mean"), ("accuracy_median_pct", "Acc. median"),
        ("completeness_mean_pct", "Compl. mean"), ("normal_err_face_median_deg", "Normal err (face, med)"),
        ("normal_err_smooth_median_deg", "Normal err (smooth, med)"), ("screen_normal_mae_smooth_deg", "Screen normal MAE"),
        ("second_surface_within_1_edges", "Layering"), ("slab_depth_offset_median", "Slab depth offset"),
        ("object_depth_abs_mean_pct", "Object |depth err| mean")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["T2-Splat=output/synthgeo", "deferred=output/synthgeo_deferred"])
    p.add_argument("--scenes", nargs="+", default=["suzanne_diffuse", "suzanne_textured", "suzanne_glossy"])
    a = p.parse_args()
    print("| Scene | Method | " + " | ".join(c[1] for c in COLS) + " |")
    print("|" + "---|" * (len(COLS) + 2))
    for s in a.scenes:
        for m in a.methods:
            name, root = m.split("=", 1)
            f = ROOT / root / s / "geometry_gt.json"
            if not f.is_file():
                print(f"| {s} | {name} | " + " | ".join("—" for _ in COLS) + " |")
                continue
            r = json.loads(f.read_text())
            cells = []
            for k, _ in COLS:
                if k not in r:
                    cells.append("—")
                    continue
                v = r[k]
                cells.append(f"{100 * v:.0f}%" if k == "second_surface_within_1_edges"
                             else f"{v:.1f}°" if k.endswith("_deg") else f"{v:.3f}")
            print(f"| {s} | {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
