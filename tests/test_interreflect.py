"""Interreflection radiance cache (relight/interreflect.py) on the wall-foot scene of tests/test_indirect.py.

1. With a shading function that returns exactly the diffuse radiosity B (what the one-bounce M·B models), one
   refresh adds no irradiance correction.
2. With a constant hit radiance c, the blocked-lobe radiance equals c · (blocked lobe weight) = c · (1 - SO).
3. Lambertian multi-bounce converges (geometric series) and only adds light.

    python -m unittest tests.test_interreflect -v
"""
import math
import types
import unittest

import torch

from tests.test_indirect import grid


@unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
class InterreflectTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from relight.indirect import IndirectLight, blocked_hits, interpolate_vertex
        from relight.mesh import FrozenMesh
        from relight.shading import fibonacci_sphere
        from relight.visibility import vertex_visibility, interpolate_visibility
        S = 40.0
        Vf, Ff, Nf = grid(torch.tensor([0.0, -S / 2, 0.0]), torch.tensor([S, 0, 0.]), torch.tensor([0, S, 0.]),
                          80, 80, (0, 0, 1))
        Vw, Fw, Nw = grid(torch.tensor([0.0, -S / 2, 0.0]), torch.tensor([0, 0, S]), torch.tensor([0, S, 0.]),
                          80, 80, (1, 0, 0))
        V = torch.cat([Vf, Vw]).cuda()
        Fc = torch.cat([Ff, Fw + Vf.shape[0]]).int().cuda()
        N = torch.cat([Nf, Nw]).cuda()
        cls.mesh = mesh = FrozenMesh(V, Fc, vertex_normals=N)
        cls.dirs = dirs = fibonacci_sphere(1024)
        cls.vis = vertex_visibility(mesh, dirs, eps=1e-3, tmin=1e-3)
        cls.ind = IndirectLight.build(mesh, cls.vis, dirs, eps=1e-3, tmin=1e-3)
        cls.rec = blocked_hits(mesh, cls.vis, dirs, eps=1e-3, tmin=1e-3)
        cls.a = 0.6
        cls.light = types.SimpleNamespace(name="sky", dirs=dirs,
                                          bins=torch.full((dirs.shape[0], 3), 4 * math.pi / dirs.shape[0], device="cuda"))
        cls.mat = types.SimpleNamespace(vertex_values=lambda: {
            "albedo": torch.full((V.shape[0], 3), cls.a, device="cuda"), "metal": torch.zeros(V.shape[0], device="cuda")})
        cls.interp_vis = staticmethod(interpolate_visibility)
        cls.interp = staticmethod(interpolate_vertex)
        cls.floor_k = int(((V[:Vf.shape[0]] - torch.tensor([1.0, 0.0, 0.0], device="cuda")).norm(dim=-1)).argmin())

    def lambert_fn(self):
        """Lambertian hit radiance a/π · (E_dir + e_ind) at the hit point (no specular)."""
        mesh, dirs, vis, a = self.mesh, self.dirs, self.vis, self.a

        def fn(mat, light, scale, fid, bary, cam, e_vert, occl):
            n = (bary[:, :, None] * mesh.vertex_normals[mesh.faces[fid].long()]).sum(1)
            v = self.interp_vis(vis, mesh.faces, fid, bary)
            E = (v * (n @ dirs.T).clamp_min(0)) @ light.bins + self.interp(e_vert, mesh.faces, fid, bary)
            return a / math.pi * E
        return fn

    def make(self, fn):
        from relight.interreflect import Interreflection
        return Interreflection(self.ind, self.mesh, self.rec, self.dirs, fn).bind(self.mat)

    def test_one_bounce_matches_transport(self):
        mesh, dirs, vis, a = self.mesh, self.dirs, self.vis, self.a

        def b_only(mat, light, scale, fid, bary, cam, e_vert, occl):       # direct diffuse radiosity only
            n = (bary[:, :, None] * mesh.vertex_normals[mesh.faces[fid].long()]).sum(1)
            B = self.ind.radiosity(mat, light)
            return self.interp(B, mesh.faces, fid, bary)
        ir = self.make(b_only)
        st = ir.refresh(self.mat, self.light, bounces=1)
        E1 = self.ind.irradiance(self.mat, self.light)
        self.assertLess(float(st["E"].abs().max()), 1e-3 * float(E1.max()))

    def test_occluded_lobe_constant(self):
        c = 0.37
        ir = self.make(lambda mat, light, scale, fid, bary, cam, e_vert, occl:
                       torch.full((fid.shape[0], 3), c, device="cuda"))
        ir.refresh(self.mat, self.light, bounces=1)
        lobe = ir.occluded(self.light)
        # floor vertex near the wall: a broad lobe around a direction tilted toward the wall
        k = self.floor_k
        f = int((self.mesh.faces.long() == k).any(-1).nonzero()[0])
        bary = (self.mesh.faces[f].long() == k).float()[None]
        fid = torch.tensor([f], device="cuda")
        R = torch.nn.functional.normalize(torch.tensor([[-1.0, 0.0, 1.0]], device="cuda"), dim=-1)
        w = torch.exp(10 * ((R @ self.dirs.T) - 1)) * (self.dirs[:, 2] > 0)
        wn = w / w.sum()
        v = self.interp_vis(self.vis, self.mesh.faces, fid, bary)
        blocked_above = ((1 - v) * (self.dirs[:, 2] > 0) * wn).sum()
        got = lobe(fid, bary, wn)[0, 0]
        self.assertGreater(float(blocked_above), 0.2)
        self.assertAlmostEqual(float(got), c * float(blocked_above), delta=0.02 * c)

    def test_multibounce_converges(self):
        ir = self.make(self.lambert_fn())
        k = self.floor_k
        E1 = float(self.ind.irradiance(self.mat, self.light)[k, 0])
        vals = []
        for _ in range(4):
            vals.append(float(ir.refresh(self.mat, self.light, bounces=1)["E"][k, 0]))
        # the first refresh already shades hits with their one-bounce irradiance -> adds the 2nd bounce
        # (≈ a/2 of the first at the wall foot); later refreshes add ever smaller higher-order terms
        self.assertGreater(vals[0], 0.1 * self.a * E1)
        self.assertLess(vals[0], self.a * E1)
        self.assertGreater(vals[1], vals[0])
        self.assertGreaterEqual(vals[2], vals[1] - 1e-5)
        self.assertLess(vals[2] - vals[1], vals[1] - vals[0])
        self.assertLess(vals[3] - vals[2], vals[2] - vals[1] + 1e-5)


if __name__ == "__main__":
    unittest.main()
