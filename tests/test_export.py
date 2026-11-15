"""PBR GLB export: structure, axes and texel layout round trip.

    python -m unittest tests.test_export -v
"""
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


def read_glb(path):
    data = Path(path).read_bytes()
    _, _, total = struct.unpack("<III", data[:12])
    jlen, _ = struct.unpack("<II", data[12:20])
    gltf = json.loads(data[20:20 + jlen])
    blen, _ = struct.unpack("<II", data[20 + jlen:28 + jlen])
    return gltf, data[28 + jlen:28 + jlen + blen]


def accessor(gltf, blob, i):
    a = gltf["accessors"][i]
    v = gltf["bufferViews"][a["bufferView"]]
    n = {"VEC2": 2, "VEC3": 3}[a["type"]]
    return np.frombuffer(blob[v["byteOffset"]:v["byteOffset"] + v["byteLength"]], "<f4").reshape(-1, n)


def image(gltf, blob, tex):
    v = gltf["bufferViews"][gltf["images"][gltf["textures"][tex]["source"]]["bufferView"]]
    return np.asarray(Image.open(io.BytesIO(blob[v["byteOffset"]:v["byteOffset"] + v["byteLength"]])))


class ExportTest(unittest.TestCase):
    def test_round_trip(self):
        from relight.export import write_pbr_glb
        from relight.io import linear_to_srgb
        rng = np.random.default_rng(0)
        V = rng.normal(size=(40, 3)).astype(np.float32)
        Fc = rng.integers(0, 40, size=(70, 3))
        N = V / np.linalg.norm(V, axis=1, keepdims=True)
        alb, rough, metal = rng.uniform(0, 1, (40, 3)), rng.uniform(0, 1, 40), rng.uniform(0, 1, 40)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.glb"
            # tiny atlas -> several pages
            rep = write_pbr_glb(V, Fc, N, alb, rough, metal, path, tile_size=4, padding=1, atlas_size=32)
            gltf, blob = read_glb(path)
        self.assertEqual(rep["pages"], len(gltf["meshes"][0]["primitives"]))
        self.assertGreater(rep["pages"], 1)
        face0 = 0
        for prim in gltf["meshes"][0]["primitives"]:
            pos = accessor(gltf, blob, prim["attributes"]["POSITION"])
            uv = accessor(gltf, blob, prim["attributes"]["TEXCOORD_0"])
            mat = gltf["materials"][prim["material"]]["pbrMetallicRoughness"]
            base = image(gltf, blob, mat["baseColorTexture"]["index"]) / 255.0
            mr = image(gltf, blob, mat["metallicRoughnessTexture"]["index"]) / 255.0
            nf = len(pos) // 3
            f = Fc[face0:face0 + nf].reshape(-1)
            # Y-up conversion: (x, y, z) -> (x, z, -y)
            np.testing.assert_allclose(pos, np.stack([V[f, 0], V[f, 2], -V[f, 1]], -1), atol=1e-6)
            # corner UVs hit texel centres holding exactly that vertex's values
            px = np.floor(uv * base.shape[0]).astype(int)
            np.testing.assert_allclose(base[px[:, 1], px[:, 0]], np.clip(linear_to_srgb(alb[f]), 0, 1), atol=1 / 255 + 1e-6)
            np.testing.assert_allclose(mr[px[:, 1], px[:, 0], 1], rough[f], atol=1 / 255 + 1e-6)
            np.testing.assert_allclose(mr[px[:, 1], px[:, 0], 2], metal[f], atol=1 / 255 + 1e-6)
            face0 += nf
        self.assertEqual(face0, len(Fc))


if __name__ == "__main__":
    unittest.main()
