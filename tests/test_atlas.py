"""Shared UV atlas material (relight/material.py) and its GLB export (relight/export.py).

1. bake_atlas reproduces PBRMaterial.sample at covered texels (same UV / row conventions in training and bake);
2. the exported GLB's TEXCOORD_0, looked up in its baseColor image with the glTF convention (origin top-left),
   gives the trained albedo.

    python -m unittest tests.test_atlas -v
"""
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
class AtlasTest(unittest.TestCase):
    def setUp(self):
        from relight.material import PBRMaterial
        torch.manual_seed(0)
        # a 2x2 grid of quads (8 triangles) mapped to two separate UV charts (left / right half of the atlas)
        V = torch.tensor([[x, y, 0.0] for y in range(3) for x in range(3)], device="cuda") / 2
        F_ = []
        for y in range(2):
            for x in range(2):
                a = y * 3 + x
                F_ += [[a, a + 1, a + 4], [a, a + 4, a + 3]]
        self.F = torch.tensor(F_, device="cuda", dtype=torch.int32)
        uv = V[self.F.long()][..., :2] * 0.4                                  # chart 1: [0, 0.4]^2
        uv[4:] += torch.tensor([0.55, 0.0], device="cuda")                    # chart 2 for the upper quads
        uv += 0.03
        self.mat = PBRMaterial(self.F, V.shape[0], albedo_init=torch.rand(V.shape[0], 3, device="cuda") * 0.8 + 0.1,
                               atlas_res=64, uv=uv).cuda()
        with torch.no_grad():
            self.mat.atlas.normal_(0, 0.5)
        self.V = V

    def test_bake_matches_sample(self):
        fid, bary = self.mat.uv_raster()
        cov = fid >= 0
        tex = self.mat.bake_atlas(dilate=0)
        s = self.mat.sample(fid[cov], bary[cov])
        got = torch.from_numpy(tex["albedo"]).cuda()[cov]
        self.assertLess(float((got - s["albedo"]).abs().max()), 1e-5)

    def test_glb_roundtrip(self):
        from relight.export import write_pbr_glb
        from relight.io import linear_to_srgb
        tex = self.mat.bake_atlas()
        tex["uv"] = self.mat.uv.cpu().numpy()
        v = self.mat.vertex_values()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "a.glb"
            write_pbr_glb(self.V.cpu().numpy(), self.F.cpu().numpy(), np.tile([0, 0, 1.0], (9, 1)),
                          v["albedo"].detach().cpu().numpy(), v["rough"].detach().cpu().numpy(),
                          v["metal"].detach().cpu().numpy(), out, uv_atlas=tex)
            raw = out.read_bytes()
        jl = struct.unpack("<I", raw[12:16])[0]
        g = json.loads(raw[20:20 + jl])
        bin_ = raw[20 + jl + 8:]
        acc = g["accessors"][g["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"]]
        bv = g["bufferViews"][acc["bufferView"]]
        uv = np.frombuffer(bin_[bv["byteOffset"]:bv["byteOffset"] + bv["byteLength"]], "<f4").reshape(-1, 2)
        ib = g["bufferViews"][g["images"][0]["bufferView"]]
        img = np.asarray(Image.open(io.BytesIO(bin_[ib["byteOffset"]:ib["byteOffset"] + ib["byteLength"]])))
        R = img.shape[0]
        # face centroids: glTF bilinear lookup (row = v * R from the top) vs the trained material
        c = uv.reshape(-1, 3, 2).mean(1)
        x, y = c[:, 0] * R - 0.5, c[:, 1] * R - 0.5                           # bilinear, texel centres at +0.5
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = (x - x0)[:, None], (y - y0)[:, None]
        at = lambda yy, xx: img[np.clip(yy, 0, R - 1), np.clip(xx, 0, R - 1), :3].astype(np.float64)
        px = (1 - fy) * ((1 - fx) * at(y0, x0) + fx * at(y0, x0 + 1)) + fy * ((1 - fx) * at(y0 + 1, x0) + fx * at(y0 + 1, x0 + 1))
        fid = torch.arange(self.F.shape[0], device="cuda")
        want = self.mat.sample(fid, torch.full((len(fid), 3), 1 / 3, device="cuda"))["albedo"].detach().cpu().numpy()
        want = np.clip(linear_to_srgb(want), 0, 1) * 255
        self.assertLess(float(np.abs(px - want).max()), 3.0)                # sRGB 8-bit + filtering in sRGB space


if __name__ == "__main__":
    unittest.main()
