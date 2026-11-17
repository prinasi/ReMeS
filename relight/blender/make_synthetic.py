"""Controlled geometry-diagnostic scene with known ground-truth geometry (Synthetic4Relight layout).

A subdivided Suzanne on a slab, lit by an equirect HDR, rendered at the cameras of an
existing Synthetic4Relight scene. Material variants isolate appearance effects that a
reconstruction method may "explain" with geometry:
  diffuse   Principled, roughness 1, no specular layer  (no view dependence)
  glossy    Principled, roughness 0.1, dielectric       (view-dependent highlights)
  textured  diffuse with a high-frequency procedural albedo (texture vs geometry)
The slab material is the same in all variants.

    python -m relight.blender.make_synthetic --out data/SynthGeo/suzanne_glossy --material glossy \
        --cameras data/Synthetic4Relight/hotdog --envmap data/Synthetic4Relight/envmaps/envmap3.exr

Writes train/{i}_rgb.exr (linear, black background) + {i}_mask.png, test/{i}_rgba.png
(16-bit, gamma 2.2), transforms_{train,test}.json and gt_mesh.ply (world-space, evaluated).

--extras adds the Synthetic4Relight material / relighting ground truth at the existing test cameras
(also on an already generated scene with --extras-only): test/{i}_albedo.png (base colour, gamma
1/2.2), test/{i}_rough.png (linear) and test_rli/{envmap}_{i}.png (16-bit, gamma 2.2) under the
relighting envmaps found next to --envmap.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import bpy
import cv2
import numpy as np
from mathutils import Matrix

from relight.blender.common import add_camera, output_exr, reset_scene, setup_cycles, setup_world_envmap


def principled(name, color=(0.6, 0.6, 0.6), roughness=0.5, specular=0.5):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Specular IOR Level"].default_value = specular
    bsdf.distribution = "GGX"
    return mat, bsdf


def build_scene(material: str):
    bpy.ops.mesh.primitive_monkey_add(size=1.4, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.rotation_euler = (0, 0, np.pi)            # face +Y (towards most cameras of the source scene)
    mod = obj.modifiers.new("subsurf", "SUBSURF")
    mod.levels = mod.render_levels = 3
    bpy.ops.object.shade_smooth()
    if material == "diffuse":
        mat, _ = principled("obj", (0.6, 0.45, 0.35), roughness=1.0, specular=0.0)
    elif material == "glossy":
        mat, _ = principled("obj", (0.6, 0.45, 0.35), roughness=0.1, specular=0.5)
    elif material == "textured":
        mat, bsdf = principled("obj", roughness=1.0, specular=0.0)
        nt = mat.node_tree
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 25.0
        noise.inputs["Detail"].default_value = 4.0
        ramp = nt.nodes.new("ShaderNodeValToRGB")
        ramp.color_ramp.elements[0].color = (0.15, 0.08, 0.05, 1)
        ramp.color_ramp.elements[1].color = (0.85, 0.7, 0.5, 1)
        nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
        nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        raise ValueError(material)
    obj.data.materials.append(mat)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    zmin = min((obj.matrix_world @ v.co).z for v in obj.evaluated_get(dg).data.vertices)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, zmin - 0.05))
    slab = bpy.context.active_object
    slab.scale = (2.0, 1.4, 0.1)
    smat, _ = principled("slab", (0.35, 0.33, 0.3), roughness=0.8, specular=0.5)
    slab.data.materials.append(smat)
    return [obj, slab]


def _emission_override(objs, what: str):
    """Replace every object's material output by an emission of its base colour / roughness."""
    for o in objs:
        for mat in o.data.materials:
            nt = mat.node_tree
            bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
            out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
            em = nt.nodes.new("ShaderNodeEmission")
            em.inputs["Strength"].default_value = 1.0
            src = bsdf.inputs["Base Color" if what == "albedo" else "Roughness"]
            if src.is_linked:
                nt.links.new(src.links[0].from_socket, em.inputs["Color"])
            elif what == "albedo":
                em.inputs["Color"].default_value = tuple(src.default_value)
            else:
                r = float(src.default_value)
                em.inputs["Color"].default_value = (r, r, r, 1.0)
            nt.links.new(em.outputs["Emission"], out.inputs["Surface"])


def _render_rgba(scene, tmp: Path):
    output_exr(scene, str(tmp))
    scene.render.image_settings.color_mode = "RGBA"
    bpy.ops.render.render(write_still=True)
    img = cv2.imread(str(tmp), cv2.IMREAD_UNCHANGED)          # BGRA, premultiplied, linear
    tmp.unlink()
    alpha = np.clip(img[..., 3], 0, 1)
    return img[..., :3] / np.maximum(alpha[..., None], 1e-6), alpha


def _write_png16(path, rgb, alpha):
    rgba = np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], -1)
    cv2.imwrite(str(path), np.round(rgba * 65535).astype(np.uint16))


def render_extras(out: Path, envmap: str, samples: int, device: str, material: str):
    """Test-view GT albedo / roughness and relit images (Synthetic4Relight layout)."""
    meta = json.loads((out / "transforms_test.json").read_text())
    names = [Path(f["file_path"]).name for f in meta["frames"]]
    env_dir = Path(envmap).parent
    (out / "test_rli").mkdir(exist_ok=True)
    for kind in ("envmap6", "envmap12", "albedo", "rough"):
        scene = reset_scene()
        setup_cycles(scene, samples=samples if kind.startswith("envmap") else 16, device=device)
        scene.render.film_transparent = True
        if kind.startswith("envmap"):
            scene.cycles.use_denoising = True
            setup_world_envmap(scene, str(env_dir / f"{kind}.exr"))
        else:
            setup_world_envmap(scene, envmap, strength=0.0)
            scene.view_settings.view_transform = "Standard"
        objs = build_scene(material)
        if not kind.startswith("envmap"):
            _emission_override(objs, kind)
        cam = add_camera(scene, meta["frames"][0]["transform_matrix"], meta["camera_angle_x"], name="cam_extra")
        for i, (fr, name) in enumerate(zip(meta["frames"], names)):
            cam.matrix_world = Matrix([list(map(float, r)) for r in fr["transform_matrix"]])
            rgb, alpha = _render_rgba(scene, out / "test" / f"_tmp_{kind}_{i:03d}.exr")
            if kind != "rough":                                      # roughness stays linear
                rgb = np.clip(rgb, 0, None) ** (1 / 2.2)
            dst = out / "test_rli" / f"{kind}_{name}.png" if kind.startswith("envmap") else \
                out / "test" / f"{name}_{kind}.png"
            _write_png16(dst, rgb, alpha)
        print(f"[make_synthetic] extras {kind}: {len(names)} views", flush=True)


def export_gt_mesh(objs, path: Path):
    dg = bpy.context.evaluated_depsgraph_get()
    verts, faces, off = [], [], 0
    for o in objs:
        ev = o.evaluated_get(dg)
        me = ev.to_mesh()
        me.calc_loop_triangles()
        mw = o.matrix_world
        verts += [tuple(mw @ v.co) for v in me.vertices]
        faces += [(t.vertices[0] + off, t.vertices[1] + off, t.vertices[2] + off) for t in me.loop_triangles]
        off += len(me.vertices)
        ev.to_mesh_clear()
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(verts)}\nproperty float x\nproperty float y\n"
                f"property float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        f.writelines(f"{x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in verts)
        f.writelines(f"3 {a} {b} {c}\n" for a, b, c in faces)
    return len(verts), len(faces)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--material", choices=("diffuse", "glossy", "textured"), required=True)
    p.add_argument("--cameras", required=True, help="Synthetic4Relight scene whose cameras to reuse")
    p.add_argument("--envmap", required=True)
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--extras", action="store_true", help="also render GT albedo / roughness / relit test views")
    p.add_argument("--extras-only", action="store_true", help="only the extras, on an existing scene dir")
    a = p.parse_args(argv)
    if a.extras_only:
        render_extras(Path(a.out), a.envmap, a.samples, a.device, a.material)
        return
    out, src = Path(a.out), Path(a.cameras)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "test").mkdir(parents=True, exist_ok=True)

    scene = reset_scene()
    setup_cycles(scene, samples=a.samples, device=a.device)
    scene.cycles.use_denoising = True                 # clean training images
    scene.render.film_transparent = True
    setup_world_envmap(scene, a.envmap)
    objs = build_scene(a.material)
    nv, nf = export_gt_mesh(objs, out / "gt_mesh.ply")

    for split, n in (("train", a.n_train), ("test", a.n_test)):
        meta = json.loads((src / f"transforms_{split}.json").read_text())
        frames = meta["frames"][:n] if split == "train" else meta["frames"][:: max(1, len(meta["frames"]) // n)][:n]
        cam = add_camera(scene, frames[0]["transform_matrix"], meta["camera_angle_x"], name=f"cam_{split}")
        out_frames = []
        for i, fr in enumerate(frames):
            cam.matrix_world = Matrix([list(map(float, r)) for r in fr["transform_matrix"]])
            tmp = out / split / f"_tmp_{i:03d}.exr"
            output_exr(scene, str(tmp))
            scene.render.image_settings.color_mode = "RGBA"
            bpy.ops.render.render(write_still=True)
            img = cv2.imread(str(tmp), cv2.IMREAD_UNCHANGED)          # BGRA, premultiplied, linear
            tmp.unlink()
            alpha = np.clip(img[..., 3], 0, 1)
            stem = out / split / f"{i:03d}"
            if split == "train":
                cv2.imwrite(f"{stem}_rgb.exr", img[..., :3].astype(np.float32))
                cv2.imwrite(f"{stem}_mask.png", np.round(alpha * 255).astype(np.uint8))
            else:
                un = img[..., :3] / np.maximum(alpha[..., None], 1e-6)
                disp = np.clip(un, 0, None) ** (1 / 2.2)
                rgba = np.concatenate([np.clip(disp, 0, 1), alpha[..., None]], -1)
                cv2.imwrite(f"{stem}_rgba.png", np.round(rgba * 65535).astype(np.uint16))
            out_frames.append({"file_path": f"{split}/{i:03d}", "transform_matrix": fr["transform_matrix"]})
            print(f"[make_synthetic] {split} {i + 1}/{len(frames)}", flush=True)
        (out / f"transforms_{split}.json").write_text(json.dumps(
            {"camera_angle_x": meta["camera_angle_x"], "frames": out_frames}, indent=1))
    (out / "scene.json").write_text(json.dumps({"material": a.material, "gt_vertices": nv, "gt_faces": nf,
                                                "envmap": a.envmap, "cameras_from": str(src)}, indent=2))
    if a.extras:
        render_extras(out, a.envmap, a.samples, a.device, a.material)


if __name__ == "__main__":
    main()
