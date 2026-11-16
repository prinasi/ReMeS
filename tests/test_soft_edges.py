"""OTO-style two-layer / 2D-SDF edge gradients in relight/oto.py:render_triangles.

    python -m unittest tests.test_soft_edges -v
"""
import unittest

import numpy as np
import torch


@unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
class SoftEdgeTest(unittest.TestCase):
    def setUp(self):
        from relight.camera import PinholeCamera
        c2w = np.eye(4)
        c2w[:3, 3] = (0, 0, 3)                                     # looks along -z at the origin
        self.cam = PinholeCamera(c2w, 0.8, 96, 96)
        # front (red, z = 0.5) partly covering a larger back triangle (blue, z = 0)
        self.V = torch.tensor([[-0.3, -0.3, 0.5], [0.3, -0.3, 0.5], [0.0, 0.3, 0.5],
                               [-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], device="cuda")
        self.F = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device="cuda")
        red, blue = [1.0, 0.0, 0.0, 0.04, 0.5], [0.0, 0.0, 1.0, 0.04, 0.5]
        self.A = torch.tensor([red] * 3 + [blue] * 3, device="cuda")

    def render(self, V, tau):
        from relight.oto import render_triangles
        v = V.reshape(-1, 3, 3)
        fn = torch.nn.functional.normalize(torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1), dim=-1)
        return render_triangles(V, self.F, self.A, fn, self.cam, None, None, soft_edge_tau=tau)

    def test_forward_unchanged(self):
        a = self.render(self.V, 0.0)["rgb"]
        b = self.render(self.V, 2.0)["rgb"]
        self.assertLess((a - b).abs().max().item(), 1e-6)

    def test_more_edge_gradient(self):
        target = self.render(self.V, 0.0)["rgb"].detach().clone()
        target[..., 0] = 0.0                                         # wants the red triangle to shrink
        target[..., 2] = 1.0
        grads = {}
        for tau in (0.0, 2.0):
            V = self.V.clone().requires_grad_(True)
            ((self.render(V, tau)["rgb"] - target) ** 2).sum().backward()
            grads[tau] = V.grad[:3].norm().item()
        self.assertGreater(grads[2.0], grads[0.0])


if __name__ == "__main__":
    unittest.main()
