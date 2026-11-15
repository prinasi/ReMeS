"""nvdiffrast G-buffer conventions (projection, image origin, barycentric order).

    python -m unittest tests.test_gbuffer -v
"""
import json
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Synthetic4Relight"


@unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
class GBufferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import trimesh
        from relight.camera import PinholeCamera
        from relight.mesh import FrozenMesh
        # asymmetric scene so flips/rotations cannot cancel out
        a = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
        b = trimesh.creation.box(extents=(0.3, 0.6, 0.2))
        b.apply_translation((0.6, 0.4, 0.3))
        m = trimesh.util.concatenate([a, b])
        cls.mesh = FrozenMesh(torch.tensor(m.vertices, dtype=torch.float32, device="cuda"),
                              torch.tensor(m.faces, dtype=torch.int32, device="cuda"))
        if (DATA / "hotdog" / "transforms_test.json").is_file():
            meta = json.loads((DATA / "hotdog" / "transforms_test.json").read_text())
            c2w, fov = np.array(meta["frames"][7]["transform_matrix"]), meta["camera_angle_x"]
        else:
            c2w = np.eye(4); c2w[:3, 3] = (0, 0, 4); fov = 0.69
        cls.cam = PinholeCamera(c2w, fov, 320, 240)

    def test_matches_ray_casting(self):
        """Per-pixel face ids agree with BVH ray casting through pixel centres."""
        from relight.gbuffer import rasterize
        from relight.raytrace import MeshTracer
        g = rasterize(self.mesh, self.cam)
        o, d = self.cam.rays()
        hit, t, fid, bary = MeshTracer(self.mesh.vertices, self.mesh.faces).closest_hit(o, d)
        hit, fid = hit.reshape(g["mask"].shape), fid.reshape(g["mask"].shape)
        self.assertGreater(int(g["mask"].sum()), 1000)
        cover = (hit == g["mask"]).float().mean().item()
        both = hit & g["mask"]
        same = (fid[both] == g["face_id"][both]).float().mean().item()
        self.assertGreater(cover, 0.995)
        self.assertGreater(same, 0.98)

    def test_barycentric_order(self):
        """bary are weights of (v0, v1, v2): they reconstruct the interpolated position."""
        from relight.gbuffer import rasterize
        g = rasterize(self.mesh, self.cam)
        m = g["mask"]
        tri = self.mesh.vertices[self.mesh.faces.long()[g["face_id"][m]]]
        recon = (g["bary"][m][:, :, None] * tri).sum(1)
        self.assertLess((recon - g["position"][m]).norm(dim=-1).max().item(), 1e-4)

    def test_normals_face_camera(self):
        from relight.gbuffer import rasterize
        g = rasterize(self.mesh, self.cam)
        m = g["mask"]
        view = torch.nn.functional.normalize(
            torch.as_tensor(self.cam.center, dtype=torch.float32, device="cuda") - g["position"][m], dim=-1)
        self.assertTrue(bool(((g["normal"][m] * view).sum(-1) >= -1e-6).all()))
        # trimesh primitives are consistently wound outward -> nothing should be back-facing
        self.assertLess(g["backfacing"][m].float().mean().item(), 0.01)


if __name__ == "__main__":
    unittest.main()
