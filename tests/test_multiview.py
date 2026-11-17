"""Multi-view NCC: projection matches the rasteriser, correct geometry beats displaced geometry,
and the loss gives vertex gradients.

    python -m unittest tests.test_multiview -v
"""
import math
import unittest

import numpy as np
import torch


def look_at(eye):
    eye = np.asarray(eye, np.float64)
    f = -eye / np.linalg.norm(eye)                       # forward (camera -Z)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r, u, -f, eye
    return c2w


@unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
class MultiViewTest(unittest.TestCase):
    def setUp(self):
        from relight.camera import PinholeCamera
        n = 64                                            # textured plane z = 0, [-1,1]^2 grid
        xs = torch.linspace(-1, 1, n, device="cuda")
        yy, xx = torch.meshgrid(xs, xs, indexing="ij")
        self.V = torch.stack([xx, yy, torch.zeros_like(xx)], -1).reshape(-1, 3).contiguous()
        idx = torch.arange(n * n, device="cuda").reshape(n, n)
        a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
        self.F = torch.cat([torch.stack([a, b, d], -1).reshape(-1, 3),
                            torch.stack([a, d, c], -1).reshape(-1, 3)]).int().contiguous()
        self.cams = [PinholeCamera(look_at([0.6 * math.sin(t), -2.5, 2.0]), 0.8, 128, 128) for t in (-0.6, 0.6)]

    def tex(self, X):
        return 0.5 + 0.4 * torch.sin(25 * X[..., 0:1]) * torch.cos(21 * X[..., 1:2]).expand(*X.shape[:-1], 3)

    def render(self, V, cam):
        import nvdiffrast.torch as dr
        from relight.gbuffer import get_context, mvp_matrix
        clip = (torch.cat([V, torch.ones_like(V[:, :1])], -1) @ mvp_matrix(cam).cuda().T)[None].contiguous()
        rast, _ = dr.rasterize(get_context(), clip, self.F, resolution=[cam.height, cam.width])
        pos = dr.interpolate(V[None].contiguous(), rast, self.F)[0][0]
        mask = rast[0, ..., 3] > 0
        return {"position": pos, "mask": mask}, self.tex(pos) * mask[..., None]

    def test_projection_matches_raster(self):
        from relight.multiview import project
        out, _ = self.render(self.V, self.cams[0])
        uv, _ = project(out["position"], self.cams[0])
        H, W = out["mask"].shape
        jj, ii = torch.meshgrid(torch.arange(H, device="cuda") + 0.5, torch.arange(W, device="cuda") + 0.5, indexing="ij")
        err = (uv - torch.stack([ii, jj], -1)).norm(dim=-1)[out["mask"]]
        self.assertLess(float(err.max()), 0.05)

    def test_ncc_prefers_true_geometry(self):
        from relight.multiview import ncc_loss
        r_out, r_img = self.render(self.V, self.cams[0])
        n_out, n_img = self.render(self.V, self.cams[1])
        ok, info = ncc_loss(r_out, r_img, r_out["mask"].float(), self.cams[1], n_img, n_out["mask"].float(),
                            self.V, self.F)
        Vb = (self.V + torch.tensor([0, 0, 0.15], device="cuda")).requires_grad_(True)
        b_out, _ = self.render(Vb, self.cams[0])
        bad, _ = ncc_loss(b_out, r_img, r_out["mask"].float(), self.cams[1], n_img, n_out["mask"].float(),
                          Vb, self.F)
        self.assertGreater(info["mv_pixels"], 1000)
        self.assertLess(float(ok), 0.05)
        self.assertGreater(float(bad), float(ok) + 0.1)
        bad.backward()
        self.assertGreater(float(Vb.grad[:, 2].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
