"""Shading / light / visibility sanity tests (furnace tests and reference checks).

    python -m unittest tests.test_shading -v
"""
import math
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ENV12 = ROOT / "data/Synthetic4Relight/envmaps/envmap12.exr"


@unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
class ShadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from relight.shading import EnvLight, brdf_lut, fibonacci_sphere
        cls.dirs = fibonacci_sphere(512)
        cls.white = EnvLight(torch.ones(64, 128, 3, device="cuda"), cls.dirs, "white")
        cls.lut = brdf_lut()

    def test_bins_integrate_sphere(self):
        self.assertAlmostEqual(self.white.bins.sum(0)[0].item(), 4 * math.pi, delta=1e-3)
        cell = self.white.bins[:, 0] / (4 * math.pi / 512)
        self.assertLess((cell - 1).abs().max().item(), 0.35)   # Voronoi cells ~equal area

    def test_white_furnace_diffuse(self):
        from relight.shading import shade
        N = 1000
        n = torch.nn.functional.normalize(torch.randn(N, 3, device="cuda"), dim=-1)
        out = shade(torch.ones(N, 3, device="cuda"), torch.full((N,), 0.5, device="cuda"),
                    torch.zeros(N, device="cuda"), n, n, torch.ones(N, 512, device="cuda"), self.white, self.lut,
                    layering=False)
        self.assertLess((out["irradiance"] / math.pi - 1).abs().max().item(), 0.03)
        self.assertLess((out["diffuse"] - 1).abs().max().item(), 0.03)

    def test_prefilter_constant(self):
        for lvl in self.white.levels:
            self.assertLess((lvl - 1).abs().max().item(), 1e-4)

    def test_lut_energy(self):
        from relight.shading import lookup_lut
        ab = lookup_lut(self.lut, torch.tensor([0.99, 0.99, 0.5], device="cuda"),
                        torch.tensor([0.05, 0.5, 0.5], device="cuda"))
        self.assertTrue(bool((ab >= 0).all() and (ab.sum(-1) <= 1.0 + 1e-3).all()))
        self.assertGreater(ab[0].sum().item(), 0.95)           # smooth, normal incidence: ~no loss

    @unittest.skipUnless(ENV12.is_file(), "envmaps not downloaded")
    def test_irradiance_matches_reference(self):
        """Binned irradiance vs the dense quadrature validated against Cycles in Phase 0."""
        from relight.envmap import diffuse_irradiance, load_envmap
        from relight.shading import EnvLight
        env = load_envmap(ENV12)
        light = EnvLight(env, self.dirs)
        n = torch.nn.functional.normalize(torch.randn(2000, 3, device="cuda"), dim=-1)
        ours = (n @ self.dirs.T).clamp_min(0) @ light.bins
        ref = diffuse_irradiance(env, n)
        rel = ((ours - ref).abs().sum(-1) / ref.sum(-1)).mean().item()
        self.assertLess(rel, 0.03)

    @unittest.skipUnless(ENV12.is_file(), "envmaps not downloaded")
    def test_mirror_specular(self):
        from relight.envmap import load_envmap, sample_envmap
        from relight.shading import EnvLight, shade
        env = load_envmap(ENV12)
        light = EnvLight(env, self.dirs)
        N = 500
        n = torch.nn.functional.normalize(torch.randn(N, 3, device="cuda"), dim=-1)
        out = shade(torch.ones(N, 3, device="cuda"), torch.zeros(N, device="cuda"), torch.ones(N, device="cuda"),
                    n, n, torch.ones(N, 512, device="cuda"), light, self.lut, specular_occlusion=False)
        ref = sample_envmap(env, n)                             # v = n -> reflection = n
        rel = ((out["specular"] - ref).abs().sum(-1) / ref.sum(-1).clamp_min(1e-3)).median().item()
        self.assertLess(rel, 0.05)
        self.assertLess(out["diffuse"].abs().max().item(), 1e-6)

    def test_learnable_light_furnace_and_gradients(self):
        from relight.shading import LearnableEnvLight, shade
        light = LearnableEnvLight(self.dirs, (32, 64), init=1.0).cuda()
        self.assertAlmostEqual(light.bins.sum(0)[0].item(), 4 * math.pi, delta=1e-3)
        for lvl in light.levels:
            self.assertLess((lvl - 1).abs().max().item(), 1e-4)
        N = 256
        n = torch.nn.functional.normalize(torch.randn(N, 3, device="cuda"), dim=-1)
        out = shade(torch.full((N, 3), 0.5, device="cuda"), torch.full((N,), 0.3, device="cuda"),
                    torch.zeros(N, device="cuda"), n, n, torch.ones(N, 512, device="cuda"), light.refresh(), self.lut)
        out["rgb"].sum().backward()
        g = light.log_env.grad
        self.assertTrue(bool(torch.isfinite(g).all()) and g.abs().sum().item() > 0)

    @unittest.skipUnless(ENV12.is_file(), "envmaps not downloaded")
    def test_learnable_light_matches_fixed(self):
        """Same map through LearnableEnvLight and EnvLight gives the same shading."""
        from relight.envmap import downsample_envmap, load_envmap
        from relight.shading import EnvLight, LearnableEnvLight, shade
        env = downsample_envmap(load_envmap(ENV12), 64, 128)
        fixed = EnvLight(env, self.dirs)
        learn = LearnableEnvLight(self.dirs, (64, 128)).cuda()
        with torch.no_grad():
            learn.log_env.copy_(torch.log(env.clamp_min(1e-6)))
        learn.refresh()
        N = 2000
        n = torch.nn.functional.normalize(torch.randn(N, 3, device="cuda"), dim=-1)
        args = (torch.full((N, 3), 0.5, device="cuda"), torch.full((N,), 0.4, device="cuda"),
                torch.zeros(N, device="cuda"), n, n, torch.ones(N, 512, device="cuda"))
        a, b = shade(*args, fixed, self.lut)["rgb"], shade(*args, learn, self.lut)["rgb"].detach()
        self.assertLess(((a - b).abs().sum(-1) / a.sum(-1)).mean().item(), 0.03)

    def test_visibility_pack_and_convex_mesh(self):
        import trimesh
        from relight.mesh import FrozenMesh
        from relight.orient import vertex_normals
        from relight.visibility import interpolate_visibility, pack_bits, unpack_bits, vertex_visibility
        m = torch.rand(10, 512, device="cuda") > 0.5
        self.assertTrue(bool((unpack_bits(pack_bits(m)) == m).all()))
        s = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
        V = torch.tensor(s.vertices, dtype=torch.float32, device="cuda")
        Fc = torch.tensor(s.faces, dtype=torch.int32, device="cuda")
        mesh = FrozenMesh(V, Fc, vertex_normals=vertex_normals(V, Fc))
        vis = unpack_bits(vertex_visibility(mesh, self.dirs))
        above = (mesh.vertex_normals @ self.dirs.T) > 0
        self.assertTrue(bool((vis == above).float().mean() > 0.999))     # convex: nothing self-shadows
        pv = interpolate_visibility(pack_bits(vis), Fc, torch.zeros(1, dtype=torch.long, device="cuda"),
                                    torch.tensor([[1.0, 0.0, 0.0]], device="cuda"))
        self.assertTrue(torch.equal(pv[0] > 0.5, vis[Fc[0, 0].long()]))


if __name__ == "__main__":
    unittest.main()
