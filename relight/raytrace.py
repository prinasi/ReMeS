"""Non-differentiable ray queries on a frozen triangle mesh (cubvh CUDA BVH).

Geometry is fixed after T²-Splat, so visibility V(x, w) and secondary hits carry
no geometry gradient; they are plain BVH queries.

Robustness: cubvh uses single-precision Möller–Trumbore. For grazing rays against
small triangles (|det| ~ 1e-8) it reports spurious hits (false self-shadowing).
Every reported hit is therefore re-tested in float64 against the reported face;
rejected rays are re-traced from just past the bogus hit (``max_retries`` times).
Barycentrics come from the float64 test.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import cubvh
except ImportError as exc:  # pragma: no cover
    raise ImportError("cubvh is required: pip install --no-deps --no-build-isolation "
                      "git+https://github.com/ashawkey/cubvh") from exc

MISS_T = float("inf")


class MeshTracer:
    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor, verify: bool = True, max_retries: int = 3):
        self.vertices = vertices.detach().float().cuda().contiguous()
        self.faces = faces.detach().long().cuda().contiguous()
        self.bvh = cubvh.cuBVH(self.vertices, self.faces.int())
        self.verify = verify
        self.max_retries = max_retries
        self._v64 = self.vertices.double()
        ext = (self.vertices.max(0).values - self.vertices.min(0).values).norm().item()
        self.skip = 1e-6 * max(ext, 1e-6)       # step past a rejected hit

    # ------------------------------------------------------------------ core
    def _mt64(self, o, d, fid):
        """float64 Möller–Trumbore of rays vs. their reported faces -> (ok, t, u, v)."""
        tri = self._v64[self.faces[fid]]
        a, b, c = tri.unbind(1)
        o, d = o.double(), d.double()
        e1, e2 = b - a, c - a
        h = torch.cross(d, e2, dim=-1)
        det = (e1 * h).sum(-1)
        inv = 1.0 / torch.where(det.abs() > 1e-300, det, torch.full_like(det, 1e-300))
        s = o - a
        q = torch.cross(s, e1, dim=-1)
        u = (s * h).sum(-1) * inv
        v = (d * q).sum(-1) * inv
        t = (e2 * q).sum(-1) * inv
        ok = (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0) & torch.isfinite(t)
        return ok, t, u, v

    @torch.no_grad()
    def _trace(self, o, d):
        """Closest verified hit for flat rays. Returns (t [N] float32, fid [N] long, u, v)."""
        N = o.shape[0]
        t_out = torch.full((N,), MISS_T, device=o.device)
        f_out = torch.full((N,), -1, dtype=torch.long, device=o.device)
        u_out = torch.zeros(N, device=o.device)
        v_out = torch.zeros(N, device=o.device)
        idx = torch.arange(N, device=o.device)
        offset = torch.zeros(N, device=o.device)
        cur_o = o
        for attempt in range(self.max_retries + 1):
            _, f, dep = self.bvh.ray_trace(cur_o, d[idx])
            hit = f >= 0
            if not self.verify:
                ok = hit
                t_abs = offset + dep
                uu = vv = torch.zeros_like(dep)
            else:
                ok = torch.zeros_like(hit)
                t_abs = torch.full_like(dep, MISS_T)
                uu = torch.zeros_like(dep)
                vv = torch.zeros_like(dep)
                if hit.any():
                    h = hit.nonzero()[:, 0]
                    good, t64, u64, v64 = self._mt64(o[idx[h]], d[idx[h]], f[h])
                    ok[h] = good
                    t_abs[h] = t64.float()
                    uu[h] = u64.float()
                    vv[h] = v64.float()
            sel = idx[ok]
            t_out[sel], f_out[sel], u_out[sel], v_out[sel] = t_abs[ok], f[ok], uu[ok], vv[ok]
            retry = hit & ~ok
            if not retry.any() or attempt == self.max_retries:
                break
            # rejected hits: continue the ray from just past the bogus intersection
            offset = offset[retry] + dep[retry] + self.skip
            idx = idx[retry]
            cur_o = (o[idx] + offset[:, None] * d[idx]).contiguous()
        return t_out, f_out, u_out, v_out

    # ------------------------------------------------------------------ API
    @torch.no_grad()
    def closest_hit(self, origins: torch.Tensor, dirs: torch.Tensor, chunk: int = 1 << 23):
        """Returns (hit [N] bool, t [N], face_id [N] long, bary [N,3] weights of (v0, v1, v2))."""
        o = origins.reshape(-1, 3).float().contiguous()
        d = F.normalize(dirs.reshape(-1, 3).float(), dim=-1).contiguous()
        N = o.shape[0]
        t = torch.empty(N, device=o.device)
        fid = torch.empty(N, dtype=torch.long, device=o.device)
        bary = torch.zeros(N, 3, device=o.device)
        for s in range(0, N, chunk):
            ts, fs, us, vs = self._trace(o[s:s + chunk], d[s:s + chunk])
            t[s:s + chunk], fid[s:s + chunk] = ts, fs
            bary[s:s + chunk] = torch.stack([1 - us - vs, us, vs], dim=-1)
        hit = fid >= 0
        bary[~hit] = 0
        return hit, t, fid, bary

    @torch.no_grad()
    def any_hit(self, origins: torch.Tensor, dirs: torch.Tensor, tmax: float | torch.Tensor = MISS_T,
                chunk: int = 1 << 23) -> torch.Tensor:
        """Occlusion test: True where the ray hits geometry with 0 < t < tmax."""
        hit, t, _, _ = self.closest_hit(origins, dirs, chunk)
        tm = tmax.reshape(-1) if isinstance(tmax, torch.Tensor) else tmax
        return (hit & (t < tm)).reshape(origins.shape[:-1])
