"""Differentiable shadow maps (relight/shadowmap.py).

    python -m unittest tests.test_shadowmap -v
"""
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
class ShadowMapTest(unittest.TestCase):
    def setUp(self):
        import trimesh
        s = trimesh.creation.icosphere(subdivisions=4, radius=0.3)
        s.apply_translation((0, 0, 0.5))
        p = trimesh.creation.box(extents=(2, 2, 0.02))
        m = trimesh.util.concatenate([s, p])
        self.V = torch.tensor(m.vertices, dtype=torch.float32, device="cuda")
        self.F = torch.tensor(m.faces, dtype=torch.int32, device="cuda")

    def mapper(self):
        from relight.shadowmap import ShadowMapper
        return ShadowMapper(torch.zeros(3, device="cuda"), 1.5, res=512)

    def test_sphere_shadow_on_plane(self):
        sm = self.mapper()
        w = torch.tensor([[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]], device="cuda")
        maps = sm.render(self.V, self.F, w)
        X = torch.tensor([[0.0, 0.0, 0.011],      # under the sphere (plane top)
                          [0.9, 0.9, 0.011],      # far from the sphere
                          [0.0, 0.0, 0.8]],       # top of the sphere
                         device="cuda")
        Vis = sm.visibility(maps, X)
        self.assertLess(Vis[0, 0].item(), 0.05)
        self.assertGreater(Vis[1, 0].item(), 0.95)
        self.assertGreater(Vis[2, 0].item(), 0.95)
        # oblique light: the shadow moves to -x (sphere centre + t*(-w) hits z=0 at x = -0.375)
        Xo = torch.tensor([[-0.375, 0.0, 0.011], [0.375, 0.0, 0.011]], device="cuda")
        Vo = sm.visibility(maps, Xo)
        self.assertLess(Vo[0, 1].item(), 0.05)
        self.assertGreater(Vo[1, 1].item(), 0.95)

    def test_gradients_reach_receiver_and_occluder(self):
        sm = self.mapper()
        V = self.V.clone().requires_grad_(True)
        w = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
        maps = sm.render(V, self.F, w)
        # a lit receiver right at the shadow boundary (sphere silhouette radius 0.3)
        X = torch.tensor([[0.301, 0.0, 0.011]], device="cuda", requires_grad=True)
        vis = sm.visibility(maps, X)
        vis.sum().backward()
        sphere = slice(0, 2562)                                              # icosphere(4) vertices come first
        self.assertGreater(V.grad[sphere].abs().sum().item(), 0)             # occluder silhouette moves the shadow
        self.assertTrue(X.grad is not None and torch.isfinite(X.grad).all())


if __name__ == "__main__":
    unittest.main()
