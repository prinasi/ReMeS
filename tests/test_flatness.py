"""Planarity prior: zero for a plane under perspective, positive for a dome, gradient flattens it.

    python -m unittest tests.test_flatness -v
"""
import math
import unittest

import numpy as np
import torch


@unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
class FlatnessTest(unittest.TestCase):
    def setUp(self):
        from relight.camera import PinholeCamera
        self.cams = []
        for az in (0.0, 2.1, 4.2):                                      # three views around the dome
            pos = np.array([2.6 * math.sin(az), -2.6 * math.cos(az), 1.6])
            f = pos / np.linalg.norm(pos)                               # Blender camera: +z points backwards
            right = np.cross([0, 0, 1], f)
            right /= np.linalg.norm(right)
            c2w = np.eye(4)
            c2w[:3, :3] = np.stack([right, np.cross(f, right), f], 1)
            c2w[:3, 3] = pos
            self.cams.append(PinholeCamera(c2w=c2w, fovx=math.radians(50), width=256, height=256))
        self.cam = self.cams[0]

    def grid(self, h):
        """[-1.5, 1.5]^2 plane with a dome of height h on [-1, 1]^2; returns V, F and the free-vertex mask
        (the flat apron around the dome is fixed, as the silhouette-pinned surroundings of a real plane)."""
        n = 61
        u = torch.linspace(-1.5, 1.5, n, device="cuda")
        X, Y = torch.meshgrid(u, u, indexing="ij")
        inside = (X.abs() < 1) & (Y.abs() < 1)
        Z = torch.where(inside, h * (1 - X ** 2) * (1 - Y ** 2), torch.zeros_like(X))
        V = torch.stack([X, Y, Z], -1).reshape(-1, 3)
        i = torch.arange(n - 1, device="cuda")
        a = (i[:, None] * n + i[None, :]).reshape(-1)
        F_ = torch.cat([torch.stack([a, a + n, a + 1], -1), torch.stack([a + 1, a + n, a + n + 1], -1)]).int()
        self.free = inside.reshape(-1)
        return V, F_

    def loss(self, V, F_, cams=None):
        from relight.mesh import FrozenMesh
        from relight.gbuffer import rasterize
        from relight.oto import flatness_loss
        total = 0
        for cam in cams or [self.cam]:
            g = rasterize(FrozenMesh(V, F_), cam)
            total = total + flatness_loss({"position": g["position"], "mask": g["mask"]}, cam,
                                          torch.ones(256, 256, device="cuda"))
        return total

    def test_plane_vs_dome(self):
        V, F_ = self.grid(0.0)
        plane = float(self.loss(V, F_))
        self.assertLess(plane, 1e-4)
        V, F_ = self.grid(0.15)
        self.assertGreater(float(self.loss(V, F_)), 20 * plane)

    def test_descent_flattens(self):
        V, F_ = self.grid(0.15)
        h0 = float(V[:, 2].max())
        V = V.clone().requires_grad_(True)
        opt = torch.optim.Adam([V], lr=2e-3)
        for _ in range(300):
            opt.zero_grad()
            self.loss(V, F_, self.cams).backward()
            V.grad[~self.free] = 0
            opt.step()
        self.assertLess(float(V[:, 2].max()), 0.3 * h0)
