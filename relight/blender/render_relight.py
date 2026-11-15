"""Relight an exported ReMeS GLB in Blender (Cycles) under an equirect HDR.

    python -m relight.blender.render_relight --glb asset.glb --envmap envmap12.exr \
        --transforms transforms_test.json --frames 0,20,40 --out-dir renders/ [--direct-only]

Writes <out-dir>/<frame>.exr (linear RGBA, film transparent) and import_report.json
(material wiring and imported bounding box, to verify the glTF round trip).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy
from mathutils import Matrix

from relight.blender.common import add_camera, output_exr, reset_scene, setup_cycles, setup_world_envmap


def configure_materials(distribution: str) -> list[dict]:
    """Force the Principled BSDF microfacet model and report how the importer wired textures."""
    report = []
    for mat in bpy.data.materials:
        if not mat.node_tree:
            continue
        for node in mat.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            node.distribution = distribution
            wiring = {}
            for name in ("Base Color", "Roughness", "Metallic", "Normal"):
                sock = node.inputs[name]
                if sock.is_linked:
                    link = sock.links[0]
                    wiring[name] = f"{link.from_node.type}:{link.from_socket.name}"
                else:
                    dv = sock.default_value
                    wiring[name] = list(dv) if hasattr(dv, "__len__") else float(dv)
            report.append({"material": mat.name, "distribution": node.distribution, "inputs": wiring})
    return report


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--glb", required=True)
    p.add_argument("--envmap", required=True)
    p.add_argument("--transforms", required=True)
    p.add_argument("--frames", required=True, help="comma-separated frame indices")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--width", type=int, default=800)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--direct-only", action="store_true", help="no indirect bounces (parity with ReMeS shading)")
    p.add_argument("--bounces", type=int, default=8)
    p.add_argument("--distribution", default="GGX", choices=("GGX", "MULTI_GGX"))
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(Path(a.transforms).read_text())
    scene = reset_scene()
    setup_cycles(scene, samples=a.samples, width=a.width, height=a.height,
                 max_bounces=0 if a.direct_only else a.bounces, device=a.device)
    scene.render.film_transparent = True
    setup_world_envmap(scene, a.envmap)

    bpy.ops.import_scene.gltf(filepath=a.glb)
    meshes = [o for o in scene.objects if o.type == "MESH"]
    corners = [o.matrix_world @ v.co for o in meshes for v in o.data.vertices]
    bbox = {"min": [min(c[i] for c in corners) for i in range(3)], "max": [max(c[i] for c in corners) for i in range(3)]}
    materials = configure_materials(a.distribution)
    (out / "import_report.json").write_text(json.dumps(
        {"objects": len(meshes), "faces": sum(len(o.data.polygons) for o in meshes), "bbox_world": bbox,
         "materials": materials[:2], "n_materials": len(materials), "direct_only": a.direct_only,
         "samples": a.samples}, indent=2))

    cam = add_camera(scene, meta["frames"][0]["transform_matrix"], meta["camera_angle_x"])
    s = scene.render.image_settings
    for f in [int(x) for x in a.frames.split(",")]:
        cam.matrix_world = Matrix([list(map(float, r)) for r in meta["frames"][f]["transform_matrix"]])
        output_exr(scene, str(out / f"{f:03d}.exr"))
        s.color_mode = "RGBA"
        bpy.ops.render.render(write_still=True)
        print(f"[blender] frame {f} -> {out / f'{f:03d}.exr'}", flush=True)


if __name__ == "__main__":
    main()
