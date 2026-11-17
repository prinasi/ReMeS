"""Phase 4 one-bounce indirect light: analytic wall-foot case.

A floor vertex at the foot of a large wall under a uniform sky L: the wall blocks half of the cosine-weighted
hemisphere, so E_dir(floor) = πL/2; the wall itself also sees half the sky, B(wall) = a/π · πL/2 = aL/2, and
E_ind(floor) = B · π/2 = πaL/4  ->  E_ind / E_dir = a/2.

    python -m unittest tests.test_indirect -v
"""
import math
import types
import unittest

import torch


def grid(origin, u, v, nu, nv, normal):
    i = torch.arange(nu + 1, dtype=torch.float32)
    j = torch.arange(nv + 1, dtype=torch.float32)
    I, J = torch.meshgrid(i / nu, j / nv, indexing="ij")
    V = origin + I.reshape(-1, 1) * u + J.reshape(-1, 1) * v
    a = (torch.arange(nu)[:, None] * (nv + 1) + torch.arange(nv)[None, :]).reshape(-1)
    F_ = torch.cat([torch.stack([a, a + nv + 1, a + 1], -1), torch.stack([a + 1, a + nv + 1, a + nv + 2], -1)])
    fn = torch.cross(V[F_[0, 1]] - V[F_[0, 0]], V[F_[0, 2]] - V[F_[0, 0]], dim=0)
    if (fn * torch.tensor(normal, dtype=torch.float32)).sum() < 0:
        F_ = F_[:, [0, 2, 1]]
    return V, F_, torch.tensor(normal, dtype=torch.float32).expand(V.shape[0], 3)


@unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
class IndirectTest(unittest.TestCase):
    def test_wall_foot(self):
        from relight.indirect import IndirectLight
        from relight.mesh import FrozenMesh
        from relight.shading import fibonacci_sphere
        from relight.visibility import vertex_visibility
        S = 40.0
        # floor x in [0, S], wall at x = 0 facing +x (separate vertices, flat normals)
        Vf, Ff, Nf = grid(torch.tensor([0.0, -S / 2, 0.0]), torch.tensor([S, 0, 0.]), torch.tensor([0, S, 0.]),
                          160, 160, (0, 0, 1))
        Vw, Fw, Nw = grid(torch.tensor([0.0, -S / 2, 0.0]), torch.tensor([0, 0, S]), torch.tensor([0, S, 0.]),
                          160, 160, (1, 0, 0))
        V = torch.cat([Vf, Vw]).cuda()
        Fc = torch.cat([Ff, Fw + Vf.shape[0]]).int().cuda()
        N = torch.cat([Nf, Nw]).cuda()
        mesh = FrozenMesh(V, Fc, vertex_normals=N)
        dirs = fibonacci_sphere(4096)
        h = S / 160
        vis = vertex_visibility(mesh, dirs, eps=1e-3, tmin=1e-3)
        ind = IndirectLight.build(mesh, vis, dirs, eps=1e-3, tmin=1e-3)
        L, a = 1.0, 0.6
        light = types.SimpleNamespace(dirs=dirs, bins=torch.full((dirs.shape[0], 3), L * 4 * math.pi / dirs.shape[0],
                                                                 device="cuda"))
        mat = types.SimpleNamespace(vertex_values=lambda: {"albedo": torch.full((V.shape[0], 3), a, device="cuda"),
                                                           "metal": torch.zeros(V.shape[0], device="cuda")})
        E_dir = ind.direct_irradiance(light)[:, 0]
        E_ind = ind.irradiance(mat, light)[:, 0]
        # floor vertex near the wall foot, in the middle of the wall's width
        target = torch.tensor([2 * h, 0.0, 0.0], device="cuda")
        k = int(((V[:Vf.shape[0]] - target).norm(dim=-1)).argmin())
        self.assertAlmostEqual(float(E_dir[k]) / (math.pi * L / 2), 1.0, delta=0.05)
        self.assertAlmostEqual(float(E_ind[k] / E_dir[k]), a / 2, delta=0.05 * a)
        # far from the wall the bounce light vanishes
        far = int(((V[:Vf.shape[0]] - torch.tensor([0.9 * S, 0.0, 0.0], device="cuda")).norm(dim=-1)).argmin())
        self.assertLess(float(E_ind[far] / E_dir[far]), 0.05)


if __name__ == "__main__":
    unittest.main()
