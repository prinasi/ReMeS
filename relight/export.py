"""PBR glTF 2.0 (GLB) export of a frozen mesh with per-vertex materials.

Per-vertex albedo / roughness / metallic are baked into per-face texture tiles
(T²-Splat layout, see utils/glb_export.bake_tiles: texel (row, col) holds the
barycentric blend with w1 = col/(T-1), w2 = row/(T-1); corner v0 -> texel (0,0),
v1 -> (0,T-1), v2 -> (T-1,0)), packed into atlas pages. Each page is one primitive
with a metallic-roughness material:

* baseColorTexture          albedo, sRGB-encoded (glTF convention);
* metallicRoughnessTexture  linear, G = roughness, B = metallic (R unused = 1);
* NORMAL                    smooth vertex normals (what the ReMeS renderer shades with).

Positions are written Y-up (glTF); Blender's importer converts back to Z-up.
Texture filtering is LINEAR without mipmaps (mips would bleed across tiny tiles).
"""
from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from relight.io import linear_to_srgb
from utils.glb_export import _place_tiles, bake_tiles


def _zup_to_yup(x: np.ndarray) -> np.ndarray:
    y = x[:, [0, 2, 1]].copy()
    y[:, 2] *= -1
    return y


def write_pbr_glb(vertices: np.ndarray, faces: np.ndarray, normals: np.ndarray, albedo: np.ndarray,
                  rough: np.ndarray, metal: np.ndarray, out: str | Path, tile_size: int = 4,
                  padding: int = 1, atlas_size: int = 4096, double_sided: bool = True,
                  chunk_faces: int = 65536) -> dict:
    """vertices/normals [V,3] (Blender Z-up world), faces [F,3], albedo [V,3] linear,
    rough/metal [V]. Returns an export report."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    faces = np.asarray(faces, dtype=np.int64)
    base_v = np.clip(linear_to_srgb(np.asarray(albedo, np.float32)), 0, 1)
    orm_v = np.stack([np.ones_like(rough), rough, metal], -1).astype(np.float32)
    nrm = np.asarray(normals, np.float32)
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
    cell = tile_size + 2 * padding
    grid = atlas_size // cell
    capacity = grid * grid
    gltf = {"asset": {"version": "2.0", "generator": "ReMeS relight/export.py"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": out.stem}],
            "meshes": [{"primitives": []}], "buffers": [], "bufferViews": [], "accessors": [],
            "images": [], "textures": [], "materials": [],
            "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}]}
    blob = io.BytesIO()

    def view(data: bytes, target=None) -> int:
        blob.write(b"\0" * (-blob.tell() % 4))
        v = {"buffer": 0, "byteOffset": blob.tell(), "byteLength": len(data)}
        if target is not None:
            v["target"] = target
        blob.write(data)
        gltf["bufferViews"].append(v)
        return len(gltf["bufferViews"]) - 1

    def accessor(values, kind, bounds=False) -> int:
        values = np.ascontiguousarray(values, dtype="<f4")
        e = {"bufferView": view(values.tobytes(), 34962), "componentType": 5126, "count": len(values), "type": kind}
        if bounds:
            e.update(min=values.min(0).tolist(), max=values.max(0).tolist())
        gltf["accessors"].append(e)
        return len(gltf["accessors"]) - 1

    def image(atlas: np.ndarray) -> int:
        png = io.BytesIO()
        Image.fromarray(atlas).save(png, format="PNG", compress_level=3)
        gltf["images"].append({"bufferView": view(png.getvalue()), "mimeType": "image/png"})
        gltf["textures"].append({"source": len(gltf["images"]) - 1, "sampler": 0})
        return len(gltf["textures"]) - 1

    pages = 0
    for first in range(0, len(faces), capacity):
        last = min(first + capacity, len(faces))
        atlases = [np.zeros((atlas_size, atlas_size, 3), np.uint8) for _ in range(2)]
        for s in range(first, last, chunk_faces):
            e = min(s + chunk_faces, last)
            f = faces[s:e]
            for atlas, vals in zip(atlases, (base_v, orm_v)):
                tiles, _ = bake_tiles(vals[f], None, tile_size)
                tiles = np.pad(tiles, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode="edge")
                _place_tiles(atlas, tiles, s - first, grid, cell)
        f = faces[first:last]
        pos = _zup_to_yup(np.asarray(vertices, np.float32)[f].reshape(-1, 3))
        nor = _zup_to_yup(nrm[f].reshape(-1, 3))
        idx = np.arange(last - first)
        origin = np.stack((idx % grid, idx // grid), -1) * cell + padding + 0.5
        uv = (origin[:, None] + np.array([[0, 0], [tile_size - 1, 0], [0, tile_size - 1]])) / atlas_size
        tb, tm = image(atlases[0]), image(atlases[1])
        gltf["materials"].append({"name": f"remes_pbr_{pages}", "doubleSided": double_sided, "alphaMode": "OPAQUE",
                                  "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1],
                                                           "baseColorTexture": {"index": tb},
                                                           "metallicFactor": 1.0, "roughnessFactor": 1.0,
                                                           "metallicRoughnessTexture": {"index": tm}}})
        gltf["meshes"][0]["primitives"].append({"attributes": {
            "POSITION": accessor(pos, "VEC3", bounds=True), "NORMAL": accessor(nor, "VEC3"),
            "TEXCOORD_0": accessor(uv.reshape(-1, 2), "VEC2")}, "material": pages, "mode": 4})
        pages += 1

    data = blob.getvalue()
    gltf["buffers"] = [{"byteLength": len(data)}]
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    data += b"\0" * (-len(data) % 4)
    total = 12 + 8 + len(js) + 8 + len(data)
    with open(out, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A))
        fh.write(js)
        fh.write(struct.pack("<II", len(data), 0x004E4942))
        fh.write(data)
    report = {"faces": int(len(faces)), "vertices": int(len(vertices)), "pages": pages, "tile_size": tile_size,
              "padding": padding, "atlas_size": atlas_size, "bytes": total,
              "material": "pbrMetallicRoughness (baseColor sRGB, MR: G=roughness B=metallic), smooth normals"}
    Path(str(out) + ".json").write_text(json.dumps(report, indent=2))
    return report
