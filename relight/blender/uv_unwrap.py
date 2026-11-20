"""Shared UV atlas for a mesh via Blender's Smart UV Project (run with the env's python, which has bpy).

    python -m relight.blender.uv_unwrap --mesh in.npz --out uv.npz [--angle 66] [--margin 0.002]

in.npz: vertices [V,3] float32, faces [F,3] int. out.npz: uv [F,3,2] float32 (per face corner, glTF/Blender
convention: origin bottom-left), plus the chart count and the used UV area. Faces keep their order.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import bpy  # noqa: F401  (must precede bmesh in the bpy module build)
import bmesh
import numpy as np


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--angle", type=float, default=66.0, help="Smart UV Project angle limit (degrees)")
    p.add_argument("--margin", type=float, default=0.002, help="island margin (UV units)")
    p.add_argument("--pack", action=argparse.BooleanOptionalAction, default=True, help="re-pack islands")
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    d = np.load(a.mesh)
    V, F = d["vertices"].astype(np.float64), d["faces"].astype(np.int64)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    me = bpy.data.meshes.new("m")
    me.from_pydata(V.tolist(), [], F.tolist())
    me.update()
    assert len(me.polygons) == len(F), "Blender dropped faces (degenerate / duplicate?)"
    ob = bpy.data.objects.new("m", me)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(a.angle), island_margin=a.margin,
                             area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
    if a.pack:                                     # Blender's concave-shape packer: much denser than Smart UV's
        bpy.ops.uv.select_all(action="SELECT")
        bpy.ops.uv.pack_islands(udim_source="CLOSEST_UDIM", rotate=True, margin_method="FRACTION",
                                margin=a.margin, shape_method="CONCAVE")
    bm = bmesh.from_edit_mesh(me)
    uv_layer = bm.loops.layers.uv.active
    bm.faces.ensure_lookup_table()
    uv = np.zeros((len(F), 3, 2), np.float32)
    for f in bm.faces:
        for k, loop in enumerate(f.loops):
            uv[f.index, k] = loop[uv_layer].uv
    # charts = connected components over UV-continuous edges
    charts = 0
    seen = np.zeros(len(F), bool)
    for f0 in bm.faces:
        if seen[f0.index]:
            continue
        charts += 1
        stack = [f0]
        seen[f0.index] = True
        while stack:
            f = stack.pop()
            for loop in f.loops:
                for other in loop.edge.link_loops:
                    g = other.face
                    if seen[g.index]:
                        continue
                    # same chart if the shared edge's UVs match on both sides
                    a0, a1 = loop[uv_layer].uv, loop.link_loop_next[uv_layer].uv
                    b0, b1 = other.link_loop_next[uv_layer].uv, other[uv_layer].uv
                    if (a0 - b0).length < 1e-6 and (a1 - b1).length < 1e-6:
                        seen[g.index] = True
                        stack.append(g)
    bpy.ops.object.mode_set(mode="OBJECT")
    e1, e2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
    np.savez(a.out, uv=uv, charts=charts, uv_area=area)
    print(json.dumps({"faces": len(F), "charts": charts, "uv_area_used": float(area),
                      "seconds": time.perf_counter() - t0}))


if __name__ == "__main__":
    main()
