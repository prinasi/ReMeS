"""Shared bpy helpers (import only inside a process that has bpy).

Run Blender work in a separate process (``python -m relight.blender.<script>``)
so bpy never shares a process with torch/CUDA training code.
"""
from __future__ import annotations

import bpy
from mathutils import Matrix


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def setup_cycles(scene, samples: int = 256, width: int = 800, height: int = 800,
                 max_bounces: int = 4, filter_width: float = 0.01, device: str = "auto",
                 seed: int = 0):
    """Deterministic, unbiased Cycles settings for parity tests (no denoiser, no clamping)."""
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    cy = scene.cycles
    cy.samples = samples
    cy.use_adaptive_sampling = False
    cy.use_denoising = False
    cy.seed = seed
    cy.filter_width = filter_width          # ~point sampling at pixel centres
    cy.sample_clamp_direct = 0.0
    cy.sample_clamp_indirect = 0.0
    cy.max_bounces = max_bounces
    cy.diffuse_bounces = max_bounces
    cy.glossy_bounces = max_bounces
    cy.transmission_bounces = max_bounces
    cy.caustics_reflective = True
    cy.caustics_refractive = True
    cy.blur_glossy = 0.0
    _select_device(scene, device)

    vs = scene.view_settings
    vs.view_transform = "Standard"
    vs.look = "None"
    vs.exposure = 0.0
    vs.gamma = 1.0
    scene.display_settings.display_device = "sRGB"


def _select_device(scene, device: str):
    if device == "cpu":
        scene.cycles.device = "CPU"
        return
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
            prefs.refresh_devices()
            gpus = [d for d in prefs.devices if d.type == backend]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                print(f"[blender] Cycles device: {backend} x{len(gpus)}")
                return
        except TypeError:
            continue
    scene.cycles.device = "CPU"
    print("[blender] Cycles device: CPU (no GPU backend found)")


def setup_world_envmap(scene, exr_path: str, strength: float = 1.0, rotation_z: float = 0.0):
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    tex = nt.nodes.new("ShaderNodeTexEnvironment")
    tex.image = bpy.data.images.load(exr_path)
    tex.image.colorspace_settings.name = "Linear Rec.709"
    tex.interpolation = "Linear"
    tex.projection = "EQUIRECTANGULAR"
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = strength
    out = nt.nodes.new("ShaderNodeOutputWorld")
    if rotation_z != 0.0:
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value[2] = rotation_z
        nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    return world


def add_camera(scene, c2w, camera_angle_x: float, name: str = "Camera"):
    """Camera from a NeRF-synthetic / Blender transform_matrix (already Blender convention)."""
    cam_data = bpy.data.cameras.new(name)
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.angle_x = camera_angle_x
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0
    cam = bpy.data.objects.new(name, cam_data)
    scene.collection.objects.link(cam)
    cam.matrix_world = Matrix([list(map(float, row)) for row in c2w])
    scene.camera = cam
    return cam


def output_exr(scene, path: str):
    s = scene.render.image_settings
    s.file_format = "OPEN_EXR"
    s.color_depth = "32"
    s.color_mode = "RGB"
    scene.render.filepath = path
