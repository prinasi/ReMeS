"""Render a mirror or diffuse sphere under an equirect HDR in Cycles (linear EXR).

    python -m relight.blender.render_spheres --envmap X.exr --transforms T.json \
        --frame 0 --material mirror --out out.exr

Used by scripts/check_envmap_convention.py to validate world/envmap/camera
conventions and Lambert/mirror shading against the PyTorch renderer.
"""
from __future__ import annotations

import argparse
import json

import bpy

from relight.blender.common import add_camera, output_exr, reset_scene, setup_cycles, setup_world_envmap


def add_sphere(radius: float, material: str, albedo: float):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=512, ring_count=256, radius=radius, location=(0, 0, 0))
    obj = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    mat = bpy.data.materials.new(material)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if material == "mirror":
        bsdf = nt.nodes.new("ShaderNodeBsdfGlossy")
        bsdf.distribution = "GGX"
        bsdf.inputs["Roughness"].default_value = 0.0
        bsdf.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    elif material == "diffuse":
        bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse")
        bsdf.inputs["Roughness"].default_value = 0.0
        bsdf.inputs["Color"].default_value = (albedo, albedo, albedo, 1.0)
    else:
        raise ValueError(material)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    obj.data.materials.append(mat)
    return obj


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--envmap", required=True)
    p.add_argument("--transforms", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--material", choices=("mirror", "diffuse", "none"), default="mirror")
    p.add_argument("--radius", type=float, default=0.6)
    p.add_argument("--albedo", type=float, default=0.8)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    with open(a.transforms) as f:
        meta = json.load(f)
    scene = reset_scene()
    setup_cycles(scene, samples=a.samples, width=a.width, height=a.height, device=a.device)
    setup_world_envmap(scene, a.envmap)
    add_camera(scene, meta["frames"][a.frame]["transform_matrix"], meta["camera_angle_x"])
    if a.material != "none":
        add_sphere(a.radius, a.material, a.albedo)
    output_exr(scene, a.out)
    bpy.ops.render.render(write_still=True)
    print(f"[blender] wrote {a.out}")


if __name__ == "__main__":
    main()
