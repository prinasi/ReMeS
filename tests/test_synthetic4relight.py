"""Phase 0 checks for the Synthetic4Relight loader and conventions.

    python -m unittest tests.test_synthetic4relight -v
"""
import os
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Synthetic4Relight"


@unittest.skipUnless((DATA / "hotdog").is_dir() and torch.cuda.is_available(), "dataset/GPU not available")
class Synthetic4RelightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from relight.data import Synthetic4Relight
        cls.ds = Synthetic4Relight(DATA, "jugs")

    def test_detection(self):
        from relight.data import is_synthetic4relight
        self.assertTrue(is_synthetic4relight(DATA / "jugs"))
        self.assertFalse(is_synthetic4relight(DATA))

    def test_views(self):
        v = self.ds.train_view(0)
        self.assertEqual(v.linear.shape, (800, 800, 3))
        self.assertGreater(v.linear.max(), 0.0)
        self.assertTrue(0.0 <= v.mask.min() and v.mask.max() <= 1.0)
        t = self.ds.test_view(0)
        self.assertEqual(t.albedo.shape, (800, 800, 3))
        self.assertEqual(t.rough.shape, (800, 800))
        rgb, alpha = self.ds.relight_view(0, "envmap12")
        self.assertEqual(rgb.shape, (800, 800, 3))
        self.assertEqual(self.ds.num_views("test"), 200)

    def test_background_matches_training_envmap(self):
        """Envmap lookup along camera rays reproduces the EXR background (conventions + light id)."""
        from relight.envmap import load_envmap, sample_envmap
        from relight.io import psnr, tonemap
        from relight.data import TRAIN_ENVMAP
        env = load_envmap(self.ds.envmap_path(TRAIN_ENVMAP))
        v = self.ds.train_view(5)
        _, d = v.camera.rays()
        bg = torch.from_numpy(v.mask).cuda() == 0
        score = psnr(tonemap(sample_envmap(env, d)), tonemap(torch.from_numpy(v.linear).cuda()), bg)
        self.assertGreater(score, 60.0)

    def test_t2splat_scene_info(self):
        """T²-Splat reader: camera extrinsics agree with the NeRF-synthetic convention."""
        from scene.dataset_readers import readSynthetic4RelightInfo
        info = readSynthetic4RelightInfo(str(DATA / "jugs"), white_background=False, eval=True)
        self.assertEqual(len(info.train_cameras), self.ds.num_views("train"))
        self.assertEqual(len(info.test_cameras), self.ds.num_views("test"))
        cam = info.train_cameras[0]
        c2w = self.ds.cameras("train")[0].c2w.copy()
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        np.testing.assert_allclose(cam.R, w2c[:3, :3].T, atol=1e-6)
        np.testing.assert_allclose(cam.T, w2c[:3, 3], atol=1e-6)
        img = np.asarray(cam.image)
        self.assertEqual(img.shape, (800, 800, 4))
        self.assertTrue(os.path.isfile(cam.mask_path))
        # alpha channel carries the soft mask (compositing happens in train.py)
        mask = self.ds.train_view(0).mask
        np.testing.assert_allclose(img[..., 3] / 255.0, mask, atol=1 / 255 + 1e-6)

    def test_t2splat_camera_premultiplies_alpha(self):
        """loadCam turns RGBA into gt_alpha_mask; Camera zeroes the background."""
        from argparse import Namespace
        from scene.dataset_readers import readSynthetic4RelightInfo
        from utils.camera_utils import loadCam
        info = readSynthetic4RelightInfo(str(DATA / "jugs"), white_background=False, eval=False)
        args = Namespace(resolution=1, data_device="cpu")
        cam = loadCam(args, 0, info.train_cameras[0], 1.0)
        self.assertIsNotNone(cam.gt_alpha_mask)
        self.assertEqual(tuple(cam.original_image.shape), (3, 800, 800))
        bg = cam.gt_alpha_mask[0] == 0
        self.assertEqual(float(cam.original_image[:, bg].abs().max()), 0.0)
        self.assertEqual(len(info.test_cameras), 0)

    def test_tracer_barycentrics(self):
        import trimesh
        from relight.raytrace import MeshTracer
        m = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
        tr = MeshTracer(torch.tensor(m.vertices), torch.tensor(m.faces))
        o = torch.tensor([[0.0, 0.0, 3.0]], device="cuda").repeat(4, 1)
        d = torch.tensor([[0, 0, -1.0], [0.01, 0, -1.0], [0, 0, 1.0], [0.0, 1.0, 0.0]], device="cuda")
        hit, t, fid, bary = tr.closest_hit(o, d)
        self.assertEqual(hit.tolist(), [True, True, False, False])
        self.assertAlmostEqual(t[0].item(), 2.0, delta=0.01)
        p = (bary[:2, :, None] * tr.vertices[tr.faces[fid[:2]]]).sum(1)
        np.testing.assert_allclose(p.cpu(), (o[:2] + t[:2, None] * torch.nn.functional.normalize(d[:2], dim=-1)).cpu(),
                                   atol=1e-5)


if __name__ == "__main__":
    unittest.main()
