"""Screen-space G-buffers of a frozen mesh with nvdiffrast.

Conventions
-----------
* Cameras are Blender/OpenGL (looks along -Z, +Y up), so ``inv(c2w)`` is directly
  the OpenGL view matrix. The projection flips clip-space y so that image row 0 is
  the *top* row (nvdiffrast's framebuffer origin is bottom-left otherwise).
* ``bary`` holds the weights of the face's (v0, v1, v2). nvdiffrast returns (u, v)
  as the weights of (v0, v1); w2 = 1 - u - v (verified in tests/test_gbuffer.py).
* T²-Splat face textures index row = w2 (vertex 2) and column = w1 (vertex 1).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

import nvdiffrast.torch as dr

from relight.camera import PinholeCamera
from relight.mesh import FrozenMesh

_CTX: dict = {}


def get_context(device="cuda"):
    key = str(device)
    if key not in _CTX:
        _CTX[key] = dr.RasterizeCudaContext(device=device)
    return _CTX[key]


def mvp_matrix(cam: PinholeCamera, near: float = 0.05, far: float = 100.0) -> torch.Tensor:
    """OpenGL model-view-projection (with y flipped for top-left image origin), [4, 4]."""
    W, H, f = cam.width, cam.height, cam.focal
    proj = np.array([[2 * f / W, 0, 0, 0],
                     [0, -2 * f / H, 0, 0],            # y flip -> row 0 is the top row
                     [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)],
                     [0, 0, -1, 0]], dtype=np.float64)
    view = np.linalg.inv(np.asarray(cam.c2w, dtype=np.float64))
    return torch.tensor(proj @ view, dtype=torch.float32)


def rasterize(mesh: FrozenMesh, cam: PinholeCamera, antialias_mask: bool = False,
              near: float = 0.05, far: float = 100.0) -> dict:
    """Rasterize one view. Returns per-pixel tensors with shape [H, W, ...].

    keys: mask (bool), face_id (long, -1 = background), bary [H,W,3], position [H,W,3],
          normal [H,W,3] (geometric, flipped to face the camera), backfacing (bool:
          winding normal pointed away from the camera before flipping), depth (view z),
          alpha (antialiased coverage, differentiable w.r.t. nothing; float).
    """
    device = mesh.vertices.device
    V = mesh.vertices
    Fc = mesh.faces.int().contiguous()
    mvp = mvp_matrix(cam, near, far).to(device)
    v_h = torch.cat([V, torch.ones_like(V[:, :1])], dim=-1)
    clip = (v_h @ mvp.T)[None].contiguous()
    rast, _ = dr.rasterize(get_context(device), clip, Fc, resolution=[cam.height, cam.width])
    r = rast[0]
    face_id = r[..., 3].long() - 1
    mask = face_id >= 0
    u, v = r[..., 0], r[..., 1]
    bary = torch.stack([u, v, 1.0 - u - v], dim=-1) * mask[..., None]

    pos, _ = dr.interpolate(V[None].contiguous(), rast, Fc)
    pos = pos[0]
    fn = mesh.face_normals
    n = torch.zeros_like(pos)
    n[mask] = fn[face_id[mask]]
    o = torch.as_tensor(cam.center, dtype=torch.float32, device=device)
    view_dir = F.normalize(o - pos, dim=-1)
    backfacing = ((n * view_dir).sum(-1) < 0) & mask
    n = torch.where(backfacing[..., None], -n, n)

    w2c = torch.as_tensor(np.linalg.inv(cam.c2w), dtype=torch.float32, device=device)
    depth = -(pos @ w2c[:3, :3].T + w2c[:3, 3])[..., 2] * mask

    alpha = mask.float()
    if antialias_mask:
        alpha = dr.antialias(alpha[None, ..., None].contiguous(), rast, clip, Fc)[0, ..., 0]
    return {"mask": mask, "face_id": face_id, "bary": bary, "position": pos, "normal": n,
            "backfacing": backfacing, "depth": depth, "alpha": alpha, "rast": rast, "clip": clip}
