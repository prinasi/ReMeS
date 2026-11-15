"""Orientation / hidden-face cleanup of frozen meshes (relight/orient.py).

    python -m unittest tests.test_orient -v
"""
import math
import unittest

import numpy as np
import torch


def _ring_cameras(n=24, radius=4.0):
    from relight.camera import PinholeCamera
    cams = []
    for i in range(n):
        th = 2 * math.pi * i / n
        el = 0.4 * math.sin(3 * th)
        eye = radius * np.array([math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el)])
        fwd = -eye / np.linalg.norm(eye)
        right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, up, -fwd, eye
        cams.append(PinholeCamera(c2w, 0.7, 128, 128))
    return cams


@unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
class OrientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import trimesh
        from relight.mesh import FrozenMesh
        outer = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        inner = trimesh.creation.icosphere(subdivisions=2, radius=0.5)   # enclosed, never visible
        m = trimesh.util.concatenate([outer, inner])
        faces = torch.tensor(m.faces, dtype=torch.int32, device="cuda")
        g = torch.Generator(device="cuda").manual_seed(0)
        flip = torch.rand(faces.shape[0], device="cuda", generator=g) < 0.5
        faces[flip] = faces[flip][:, [0, 2, 1]]
        R = 4
        tex = torch.arange(faces.shape[0] * R * R * 3, dtype=torch.float32, device="cuda").reshape(-1, R, R, 3)
        cls.n_outer = len(outer.faces)
        cls.mesh = FrozenMesh(torch.tensor(m.vertices, dtype=torch.float32, device="cuda"), faces, tex)
        cls.flip = flip
        cls.cams = _ring_cameras()

    def test_remove_hidden_then_orient(self):
        from relight.orient import orient_mesh, remove_hidden_faces, vertex_normals
        clean, hs = remove_hidden_faces(self.mesh, self.cams)
        self.assertEqual(hs["removed_faces"], self.mesh.faces.shape[0] - self.n_outer)
        self.assertEqual(clean.faces.shape[0], self.n_outer)
        oriented, st = orient_mesh(clean, self.cams)
        centroid = oriented.vertices[oriented.faces.long()].mean(1)
        outward = (oriented.face_normals * centroid).sum(-1) > 0
        self.assertTrue(bool(outward.all()), f"{int((~outward).sum())} faces still inward")
        vn = vertex_normals(oriented.vertices, oriented.faces)
        radial = torch.nn.functional.normalize(oriented.vertices, dim=-1)
        self.assertGreater((vn * radial).sum(-1).min().item(), 0.95)

    def test_flipped_faces_transpose_their_textures(self):
        """Swapping v1<->v2 must transpose the face's barycentric texture grid."""
        from relight.orient import orient_mesh
        oriented, _ = orient_mesh(self.mesh, self.cams)
        changed = (oriented.faces != self.mesh.faces).any(-1)
        self.assertTrue(bool(changed.any()))
        np.testing.assert_array_equal(oriented.face_textures[changed].cpu().numpy(),
                                      self.mesh.face_textures[changed].transpose(1, 2).cpu().numpy())
        np.testing.assert_array_equal(oriented.face_textures[~changed].cpu().numpy(),
                                      self.mesh.face_textures[~changed].cpu().numpy())


if __name__ == "__main__":
    unittest.main()


class OrientComponentsTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
    def test_random_flips_become_consistent_outward(self):
        import trimesh
        from relight.camera import PinholeCamera
        from relight.mesh import FrozenMesh
        from relight.orient import orient_components
        s = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        V = torch.tensor(s.vertices, dtype=torch.float32, device="cuda")
        Fc = torch.tensor(s.faces, dtype=torch.int64, device="cuda")
        g = torch.Generator(device="cuda").manual_seed(0)
        flip = torch.rand(Fc.shape[0], device="cuda", generator=g) > 0.5
        Fc[flip] = Fc[flip][:, [0, 2, 1]]
        cams = []
        for az in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            c = np.array([3 * np.cos(az), 3 * np.sin(az), 1.0])
            z = c / np.linalg.norm(c)                       # camera looks along -z towards the origin
            x = np.cross([0, 0, 1.0], z); x /= np.linalg.norm(x)
            y = np.cross(z, x)
            c2w = np.eye(4); c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, c
            cams.append(PinholeCamera(c2w, 0.8, 128, 128))
        out, stats = orient_components(FrozenMesh(V, Fc.int()), cams)
        v = out.vertices[out.faces.long()]
        n = torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1)
        self.assertEqual(stats["components"], 1)
        self.assertTrue(bool(((n * v.mean(1)).sum(-1) > 0).all()))
