"""OTO-style opaque triangle soup on nvdiffrast (Stage A of the ReMeS geometry stage).

Inspired by "1000FPS+ Novel View Synthesis from End-to-End Opaque Triangle Optimization"
(Yan et al.): disjoint triangles that are opaque from the first iteration.

* Forward: exact z-buffer rasterisation (``dr.rasterize``) — one triangle per pixel.
* Gradients: barycentric interpolation (``dr.interpolate``), face normals computed from
  the vertex tensor (shading), and ``dr.antialias`` at silhouette / occlusion edges
  (plays the role of OTO's 2D-SDF and two-layer occlusion approximations).
* Appearance (ReMeS): per-corner diffuse radiance + reflection-direction specular
      L = c_d + L_env(reflect(-v, n_face), ρ) · (F0·A(n·v, ρ) + B(n·v, ρ))
  with per-corner F0 and roughness and a learnable (or fixed) HDR environment. View
  dependence can only come through the triangle normal, so a triangle cannot tilt or
  hide to fake reflections without breaking other views (OTO's stated limitation).
* Density control (OTO, coverage-based): prune triangles that never win the z-test
  (removes hidden sheets) or are tiny, split large ones at the longest-edge midpoint;
  Adam state is carried over.

All geometry is in the Blender world frame of relight.camera.PinholeCamera.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

import nvdiffrast.torch as dr

from relight.gbuffer import get_context, mvp_matrix
from relight.shading import lookup_lut

DIFFUSE_MAX = 2.0           # diffuse radiance = DIFFUSE_MAX * sigmoid(logit)
ROUGH_RANGE = (0.03, 1.0)


def _logit(p):
    p = torch.as_tensor(p, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p))


class OpaqueSoup:
    PARAMS = ("verts", "diffuse", "spec")

    def __init__(self, verts: torch.Tensor, diffuse: float = 0.3, f0: float = 0.04, rough: float = 0.5,
                 lr_pos: float = 2e-4, lr_color: float = 1e-2, lr_spec: float = 1e-2):
        F_ = verts.shape[0]
        dev = verts.device
        self.verts = verts.detach().clone().float().requires_grad_(True)                  # [F, 3, 3]
        self.diffuse = torch.full((F_, 3, 3), float(_logit(diffuse / DIFFUSE_MAX)), device=dev, requires_grad=True)
        lo, hi = ROUGH_RANGE
        spec = torch.stack([torch.full((F_, 3), float(_logit(f0)), device=dev),
                            torch.full((F_, 3), float(_logit((rough - lo) / (hi - lo))), device=dev)], -1)
        self.spec = spec.requires_grad_(True)                                               # [F, 3, 2]
        self.lrs = {"verts": lr_pos, "diffuse": lr_color, "spec": lr_spec}
        self.opt = torch.optim.Adam([{"params": [getattr(self, n)], "lr": self.lrs[n], "name": n} for n in self.PARAMS])
        self.max_cov = torch.zeros(F_, device=dev)
        self.err_sum = torch.zeros(F_, device=dev)                 # accumulated per-pixel image error
        self.err_cnt = torch.zeros(F_, device=dev)

    @property
    def num(self) -> int:
        return self.verts.shape[0]

    def faces(self) -> torch.Tensor:
        return torch.arange(3 * self.num, device=self.verts.device, dtype=torch.int32).reshape(-1, 3)

    def face_normals(self) -> torch.Tensor:
        v = self.verts
        return F.normalize(torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1), dim=-1)

    # ------------------------------------------------------------------ rendering
    def corner_attributes(self) -> torch.Tensor:
        lo, hi = ROUGH_RANGE
        return torch.cat([DIFFUSE_MAX * torch.sigmoid(self.diffuse), torch.sigmoid(self.spec[..., :1]),
                          lo + (hi - lo) * torch.sigmoid(self.spec[..., 1:])], -1).reshape(-1, 5)

    def render(self, cam, light, lut, mvp: torch.Tensor | None = None, track_coverage: bool = False,
               shadow: dict | None = None, soft_edge_tau: float = 0.0) -> dict:
        out = render_triangles(self.verts.reshape(-1, 3), self.faces(), self.corner_attributes(), self.face_normals(),
                               cam, light, lut, mvp, shadow, soft_edge_tau)
        if track_coverage:
            with torch.no_grad():
                m = out["mask"]
                self.max_cov = torch.maximum(self.max_cov, torch.bincount(out["tri"][m], minlength=self.num).float())
        return out

    @torch.no_grad()
    def track_error(self, out: dict, err: torch.Tensor):
        """Accumulate the per-pixel image error err [H, W] onto the visible triangles."""
        m = out["mask"]
        t = out["tri"][m]
        self.err_sum.index_add_(0, t, err[m].float())
        self.err_cnt.index_add_(0, t, torch.ones_like(t, dtype=torch.float))

    def mean_error(self) -> torch.Tensor:
        return self.err_sum / self.err_cnt.clamp_min(1)

    # ------------------------------------------------------------------ priors
    @staticmethod
    def depth_normal_loss(out: dict) -> torch.Tensor:
        """1 - |n_face · n_depth| where n_depth comes from the rendered position map."""
        p, n, m = out["position"], out["normal"], out["mask"]
        dx = p[:-1, 1:] - p[:-1, :-1]
        dy = p[1:, :-1] - p[:-1, :-1]
        nd = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        valid = m[:-1, :-1] & m[:-1, 1:] & m[1:, :-1]
        if not valid.any():
            return p.new_zeros(())
        return (1 - (nd[valid] * n[:-1, :-1][valid]).sum(-1).abs()).mean()

    # ------------------------------------------------------------------ density control
    def _rebuild(self, keep: torch.Tensor, new: dict[str, torch.Tensor] | None = None):
        """Keep rows `keep` of every per-triangle parameter, append `new`, carry Adam state."""
        for group in self.opt.param_groups:
            name = group["name"]
            p = group["params"][0]
            st = self.opt.state.pop(p, None)
            data = p.detach()[keep]
            add = None if new is None else new[name]
            if add is not None:
                data = torch.cat([data, add], 0)
            q = data.clone().requires_grad_(True)
            group["params"][0] = q
            setattr(self, name, q)
            if st is not None:
                for k in ("exp_avg", "exp_avg_sq"):
                    t = st[k][keep]
                    if add is not None:
                        t = torch.cat([t, torch.zeros_like(add)], 0)
                    st[k] = t
                self.opt.state[q] = st
        for name in ("max_cov", "err_sum", "err_cnt"):
            t = getattr(self, name)[keep]
            if new is not None:
                t = torch.cat([t, torch.zeros(new["verts"].shape[0], device=t.device)], 0)
            setattr(self, name, t)

    @torch.no_grad()
    def prune(self, remove: torch.Tensor) -> int:
        if remove.any():
            self._rebuild((~remove).nonzero()[:, 0])
        return int(remove.sum())

    @torch.no_grad()
    def split(self, idx: torch.Tensor) -> int:
        """Split triangles `idx` at the midpoint of their longest edge into two."""
        if idx.numel() == 0:
            return 0
        v = self.verts[idx]
        lens = torch.stack([(v[:, 1] - v[:, 0]).norm(dim=-1), (v[:, 2] - v[:, 1]).norm(dim=-1),
                            (v[:, 0] - v[:, 2]).norm(dim=-1)], -1)
        k = lens.argmax(-1)                                                      # longest edge (k, k+1)
        order = (torch.arange(3, device=v.device)[None] + k[:, None]) % 3       # rotate so it is (0, 1)
        ar = torch.arange(len(idx), device=v.device)[:, None]

        def halves(t):
            t = t[ar, order]                                                     # [S, 3, C]
            m = 0.5 * (t[:, 0] + t[:, 1])
            return (torch.stack([t[:, 0], m, t[:, 2]], 1), torch.stack([m, t[:, 1], t[:, 2]], 1))

        parts = {n: halves(getattr(self, n)[idx]) for n in self.PARAMS}
        keep = torch.ones(self.num, dtype=torch.bool, device=v.device)
        keep[idx] = False
        new = {n: torch.cat(parts[n], 0) for n in self.PARAMS}
        self._rebuild(keep.nonzero()[:, 0], new)
        return int(idx.numel())

    @torch.no_grad()
    def convert_to_albedo(self, mapper, light, k: int = 4096, chunk: int = 65536):
        """Switch the diffuse attribute from radiance to albedo: a = L_d · π / E_vis per corner
        (irradiance on the lit side, with shadow-map visibility)."""
        from relight.shadowmap import sample_directions
        idx, prob = sample_directions(light.bins, light.dirs, k)
        w = light.dirs[idx]
        V = self.verts.reshape(-1, 3)
        maps = mapper.render(V, self.faces(), w)
        fn = self.face_normals().repeat_interleave(3, 0)
        wl = light.bins[idx] / (len(idx) * prob)[:, None]
        rad = DIFFUSE_MAX * torch.sigmoid(self.diffuse.reshape(-1, 3))
        alb = torch.empty_like(rad)
        for s in range(0, V.shape[0], chunk):
            vis = mapper.visibility(maps, V[s:s + chunk])
            n = fn[s:s + chunk]
            e1 = (vis * (n @ w.T).clamp_min(0)) @ wl
            e2 = (vis * (-n @ w.T).clamp_min(0)) @ wl
            e = torch.maximum(e1, e2)
            alb[s:s + chunk] = rad[s:s + chunk] * math.pi / (0.96 * e.clamp_min(1e-3))
        alb = alb.clamp(0.01, 0.98)
        self.diffuse.data.copy_(_logit(alb / DIFFUSE_MAX).reshape(self.diffuse.shape))
        return float(alb.mean())

    def reset_coverage(self):
        self.max_cov.zero_()
        self.err_sum.zero_()
        self.err_cnt.zero_()

    def step(self):
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    def set_lr(self, name: str, lr: float):
        for g in self.opt.param_groups:
            if g["name"] == name:
                g["lr"] = lr

    def state_dict(self) -> dict:
        return {n: getattr(self, n).detach().cpu() for n in self.PARAMS}


# ---------------------------------------------------------------------- shared renderer
def _shade_layer(rast, V, faces, attrs, face_normals, o, light, lut, shadow):
    """Shade one rasterised layer. Returns (rgb [H,W,3] masked, mask, n, pos, tri, interp)."""
    tri = rast[0, ..., 3].long() - 1
    mask = tri >= 0
    interp, _ = dr.interpolate(torch.cat([attrs, V], -1)[None].contiguous(), rast, faces)
    interp = interp[0]
    pos = interp[..., 5:8]
    n = torch.zeros_like(pos)
    n[mask] = face_normals[tri[mask]]
    d = F.normalize(pos - o, dim=-1)
    n = torch.where(((n * d).sum(-1, keepdim=True) > 0), -n, n)                      # face the camera
    rgb = interp[..., 0:3]
    if light is not None and mask.any():
        dm, nm = d[mask], n[mask]
        refl = F.normalize(dm - 2 * (dm * nm).sum(-1, keepdim=True) * nm, dim=-1)
        rho = interp[..., 4][mask]
        nv = (-(dm * nm).sum(-1)).clamp(1e-4, 1)
        ab = lookup_lut(lut, nv, rho)
        spec_albedo = interp[..., 3][mask][:, None] * ab[:, :1] + ab[:, 1:]
        spec = light.specular(refl, rho) * spec_albedo
        base = rgb[mask]
        if shadow is not None:
            w = light.dirs[shadow["idx"]]                                                 # [K, 3]
            vis = shadow["mapper"].visibility(shadow["maps"], pos[mask])                  # [N, K]
            cos = (nm @ w.T).clamp_min(0)
            wl = light.bins[shadow["idx"]] / (len(shadow["idx"]) * shadow["prob"])[:, None]
            irr = (vis * cos) @ wl                                                        # [N, 3]
            if shadow.get("e_ind") is not None:                       # one-bounce indirect (relight/indirect.py)
                e_pix, _ = dr.interpolate(shadow["e_ind"][None].contiguous(), rast, faces)
                irr = irr + e_pix[0][mask]
            base = base / math.pi * irr * (1 - spec_albedo).clamp_min(0)
        rgb = rgb.index_put(mask.nonzero(as_tuple=True), base + spec)
    return rgb * mask[..., None], mask, n, pos, tri, interp


def edge_alpha(clip: torch.Tensor, faces: torch.Tensor, tri: torch.Tensor, mask: torch.Tensor, tau: float):
    """OTO 2D-SDF coverage α = σ(τ · d_in), d_in = pixel distance (px) to the nearest edge of its own front
    triangle, positive inside. Differentiable w.r.t. the projected vertices."""
    H, W = mask.shape
    c = clip[0]
    xy = torch.stack([(c[:, 0] / c[:, 3] * 0.5 + 0.5) * W, (c[:, 1] / c[:, 3] * 0.5 + 0.5) * H], -1)
    j, i = mask.nonzero(as_tuple=True)
    p = torch.stack([i.float() + 0.5, j.float() + 0.5], -1)
    t = xy[faces[tri[mask]].long()]                                                   # [M, 3, 2]
    a, b, cc = t[:, 0], t[:, 1], t[:, 2]

    def cross(u, v):
        return u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    orient = torch.sign(cross(b - a, cc - a)).detach()
    d = [orient * cross(e1 - e0, p - e0) / (e1 - e0).norm(dim=-1).clamp_min(1e-6)
         for e0, e1 in ((a, b), (b, cc), (cc, a))]
    d_in = torch.stack(d, -1).min(-1).values
    return torch.zeros(H, W, device=clip.device).index_put((j, i), torch.sigmoid(tau * d_in))


def render_triangles(V: torch.Tensor, faces: torch.Tensor, attrs: torch.Tensor, face_normals: torch.Tensor, cam, light,
                     lut, mvp: torch.Tensor | None = None, shadow: dict | None = None,
                     soft_edge_tau: float = 0.0) -> dict:
    """Opaque z-buffer render of triangles with per-vertex attrs [N, 5] = (diffuse, F0, roughness).

    shadow=None: the diffuse attribute is outgoing radiance (free appearance).
    shadow={"mapper", "maps", "idx", "prob"[, "e_ind"]}: the diffuse attribute is albedo, shaded physically as
        albedo/π · E · (1 - F0·A - B),  E = Σ_k L_k V_k max(n·w_k, 0) / (K p_k)
    with differentiable shadow-map visibility V (relight/shadowmap.py) over K importance-sampled
    light bins of ``light`` — cast shadows must then be explained by geometry. Optional "e_ind" [V, 3]:
    per-vertex one-bounce indirect irradiance added to E (Stage B, relight/indirect.py).

    soft_edge_tau > 0: OTO two-layer + 2D-SDF gradient, forward unchanged. The second depth layer is peeled
    and shaded (C2); image = C1_aa + (α1 - sg(α1)) · (C1 - C2), α1 = σ(τ·d_in): pixels within ~1/τ px inside
    an edge move it depending on whether the front or the behind colour fits the image better.
    """
    dev = V.device
    mvp = mvp_matrix(cam).to(dev) if mvp is None else mvp
    clip = (torch.cat([V, torch.ones_like(V[:, :1])], -1) @ mvp.T)[None].contiguous()
    o = torch.as_tensor(np.asarray(cam.center), dtype=torch.float32, device=dev)
    res = [cam.height, cam.width]
    if soft_edge_tau > 0:
        with dr.DepthPeeler(get_context(dev), clip, faces, res) as peeler:
            rast, _ = peeler.rasterize_next_layer()
            rast2, _ = peeler.rasterize_next_layer()
    else:
        rast, _ = dr.rasterize(get_context(dev), clip, faces, resolution=res)
    rgb, mask, n, pos, tri, interp = _shade_layer(rast, V, faces, attrs, face_normals, o, light, lut, shadow)
    rgba = torch.cat([rgb, mask[..., None].float()], -1)
    aa = dr.antialias(rgba[None].contiguous(), rast, clip, faces)[0]
    out_rgb = aa[..., :3]
    if soft_edge_tau > 0 and mask.any():
        rgb2 = _shade_layer(rast2, V, faces, attrs, face_normals, o, light, lut, shadow)[0]
        a1 = edge_alpha(clip, faces, tri, mask, soft_edge_tau)
        out_rgb = out_rgb + ((a1 - a1.detach()) * mask)[..., None] * (rgb - rgb2.detach())
    return {"rgb": out_rgb, "alpha": aa[..., 3], "mask": mask, "normal": n, "position": pos, "tri": tri,
            "albedo": interp[..., 0:3]}


def albedo_smoothness(out: dict, gt: torch.Tensor, alpha: float = 20.0) -> torch.Tensor:
    """Edge-aware screen-space smoothness of the rendered albedo: |∇a| weighted by exp(-alpha |∇I_gt|).
    Keeps smooth shading variation out of the albedo (it must come from normals / shadows)."""
    a, m = out["albedo"], out["mask"]
    loss, n = a.new_zeros(()), 0
    for sl0, sl1 in (((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                     ((slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        valid = m[sl0] & m[sl1]
        if valid.any():
            w = torch.exp(-alpha * (gt[sl0] - gt[sl1]).abs().mean(-1))
            loss = loss + (w * (a[sl0] - a[sl1]).abs().mean(-1))[valid].mean()
            n += 1
    return loss / max(n, 1)


# ---------------------------------------------------------------------- Stage B: weld + RDT + connected mesh
@torch.no_grad()
def weld_soup(verts: torch.Tensor, attrs: torch.Tensor, eps: float):
    """Merge soup corners within ~eps (grid snapping). Returns V [N,3], faces [F',3], attrs [N,C]."""
    P = verts.reshape(-1, 3)
    key = torch.floor(P / eps).long()
    key = key - key.min(0).values
    span = key.max(0).values + 1
    lin = (key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]
    uniq, inv = torch.unique(lin, return_inverse=True)
    cnt = torch.zeros(len(uniq), device=P.device).index_add_(0, inv, torch.ones_like(inv, dtype=torch.float))
    V = torch.zeros(len(uniq), 3, device=P.device).index_add_(0, inv, P) / cnt[:, None]
    A = torch.zeros(len(uniq), attrs.shape[-1], device=P.device).index_add_(0, inv, attrs.reshape(-1, attrs.shape[-1]))
    A = A / cnt[:, None]
    faces = inv.reshape(-1, 3)
    ok = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    faces = faces[ok]
    return V, faces, A


def rdt_remesh(V: torch.Tensor, faces: torch.Tensor, mode: str = "manifold", max_distance_ratio=0.5,
               min_normal_cos=0.15, min_quality=0.02) -> torch.Tensor:
    """T²-Splat's surface-filtered restricted Delaunay remesh of a welded soup.

    mode "manifold": score-ordered greedy selection with <= 2 faces per edge and consistent
    fans/orientation (utils.rdt_surface.select_rdt_surface); "g": up to 4 faces per edge.
    """
    from functools import partial
    import rdel
    from utils.rdt_surface import run_g_surface_rdt, select_rdt_surface
    v_np, f_np = V.detach().cpu().numpy(), faces.detach().cpu().numpy().astype(np.int64)
    kw = dict(max_distance_ratio=max_distance_ratio, min_normal_cos=min_normal_cos, min_quality=min_quality)
    if mode == "g":
        out = run_g_surface_rdt(v_np, f_np, rdt_run=rdel.run, **kw)
    elif mode == "manifold":
        out = rdel.run(v_np, f_np, verbose=False, orient=False, return_debug=False,
                       face_selector=partial(select_rdt_surface, **kw))
    else:
        raise ValueError(mode)
    return torch.as_tensor(np.asarray(out, dtype=np.int64), device=V.device)


class ConnectedMesh:
    """Shared-vertex opaque mesh with per-vertex appearance (Stage B refinement)."""
    PARAMS = ("verts", "diffuse", "spec")

    def __init__(self, V, faces, attrs, lr_pos=5e-5, lr_color=1e-2, lr_spec=1e-2):
        lo, hi = ROUGH_RANGE
        self.faces = faces.int().contiguous()
        self.verts = V.detach().clone().requires_grad_(True)
        self.diffuse = _logit(attrs[:, :3] / DIFFUSE_MAX).to(V.device).requires_grad_(True)
        self.spec = torch.stack([_logit(attrs[:, 3]), _logit((attrs[:, 4] - lo) / (hi - lo))], -1).to(V.device)
        self.spec.requires_grad_(True)
        self.opt = torch.optim.Adam([{"params": [self.verts], "lr": lr_pos, "name": "verts"},
                                     {"params": [self.diffuse], "lr": lr_color, "name": "diffuse"},
                                     {"params": [self.spec], "lr": lr_spec, "name": "spec"}])
        self._edges()
        self.max_cov = torch.zeros(self.faces.shape[0], device=V.device)
        self.large = None

    def enable_large_steps(self, lam: float, lr: float):
        """Optimise vertex positions with relight/largesteps.py (smoothness-preconditioned, uniform Adam)
        instead of per-coordinate Adam."""
        from relight.largesteps import LargeSteps
        self.large = LargeSteps(self.verts.shape[0], self.edges, lam=lam, lr=lr)
        groups = [g for g in self.opt.param_groups if g["name"] != "verts"]
        self.opt = torch.optim.Adam([{"params": g["params"], "lr": g["lr"], "name": g["name"]} for g in groups])

    def _edges(self):
        f = self.faces.long()
        e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
        fid = torch.arange(f.shape[0], device=f.device).repeat(3)
        es = torch.sort(e, 1).values
        key = es[:, 0] * (self.verts.shape[0] + 1) + es[:, 1]
        uniq, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
        self.edges = torch.stack([uniq // (self.verts.shape[0] + 1), uniq % (self.verts.shape[0] + 1)], -1)
        # face pairs across manifold edges
        order = torch.argsort(inv)
        inv_s, fid_s = inv[order], fid[order]
        two = cnt[inv_s] == 2
        first = torch.ones_like(two)
        first[1:] = inv_s[1:] != inv_s[:-1]
        idx = (first & two).nonzero()[:, 0]
        self.face_pairs = torch.stack([fid_s[idx], fid_s[idx + 1]], -1)

    def face_normals(self):
        v = self.verts[self.faces.long()]
        return F.normalize(torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1), dim=-1)

    def attributes(self):
        lo, hi = ROUGH_RANGE
        return torch.cat([DIFFUSE_MAX * torch.sigmoid(self.diffuse), torch.sigmoid(self.spec[:, :1]),
                          lo + (hi - lo) * torch.sigmoid(self.spec[:, 1:])], -1)

    def render(self, cam, light, lut, mvp=None, track_coverage=False, shadow=None, soft_edge_tau=0.0):
        out = render_triangles(self.verts, self.faces, self.attributes(), self.face_normals(), cam, light, lut, mvp,
                               shadow, soft_edge_tau)
        if track_coverage:
            with torch.no_grad():
                m = out["mask"]
                self.max_cov = torch.maximum(self.max_cov, torch.bincount(out["tri"][m], minlength=self.faces.shape[0]).float())
        return out

    def laplacian_loss(self):
        a, b = self.edges[:, 0], self.edges[:, 1]
        n = self.verts.shape[0]
        nb = torch.zeros_like(self.verts).index_add_(0, a, self.verts[b]).index_add_(0, b, self.verts[a])
        deg = torch.zeros(n, device=self.verts.device).index_add_(0, a, torch.ones_like(a, dtype=torch.float))
        deg = deg.index_add_(0, b, torch.ones_like(b, dtype=torch.float)).clamp_min(1)
        return ((nb / deg[:, None] - self.verts) ** 2).sum(-1).mean()

    def normal_consistency_loss(self):
        if self.face_pairs.numel() == 0:
            return self.verts.new_zeros(())
        fn = self.face_normals()
        return (1 - (fn[self.face_pairs[:, 0]] * fn[self.face_pairs[:, 1]]).sum(-1).abs()).mean()

    @torch.no_grad()
    def remesh(self, scale: float = 1.0, iterations: int = 3) -> dict:
        """Isotropic remeshing to `scale` × the current median edge length (relight/remesh.py); attributes
        transferred from the old surface; optimizer state is reset (learning rates kept)."""
        from relight.remesh import isotropic_remesh, transfer_attributes
        V, F_ = self.verts.detach(), self.faces.long()
        v = V[F_]
        target = scale * float((v - v.roll(1, dims=1)).norm(dim=-1).median())
        Vn, Fn = isotropic_remesh(V.cpu().numpy(), F_.cpu().numpy(), target, iterations)
        Vn = torch.as_tensor(Vn, device=V.device)
        Fn = torch.as_tensor(Fn, device=V.device)
        A = transfer_attributes(V, self.faces, torch.cat([self.diffuse, self.spec], -1).detach(), Vn)
        before = (V.shape[0], F_.shape[0])
        lrs = {g["name"]: g["lr"] for g in self.opt.param_groups}
        self.faces = Fn.int().contiguous()
        self.verts = Vn.clone().requires_grad_(True)
        self.diffuse = A[:, :3].clone().requires_grad_(True)
        self.spec = A[:, 3:].clone().requires_grad_(True)
        self.opt = torch.optim.Adam([{"params": [getattr(self, n)], "lr": lrs[n], "name": n} for n in self.PARAMS
                                     if n in lrs])
        self._edges()
        self.max_cov = torch.zeros(self.faces.shape[0], device=V.device)
        if self.large is not None:
            self.enable_large_steps(self.large.lam, self.large.lr)
        return {"target_len": target, "before": before, "after": (Vn.shape[0], Fn.shape[0])}

    @torch.no_grad()
    def prune_faces(self, remove: torch.Tensor) -> int:
        if remove.any():
            self.faces = self.faces[~remove].contiguous()
            self.max_cov = self.max_cov[~remove]
            self._edges()
            if self.large is not None:
                self.enable_large_steps(self.large.lam, self.large.lr)
        return int(remove.sum())

    def step(self):
        if self.large is not None:
            self.large.step(self.verts)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    def set_lr(self, name, lr):
        if name == "verts" and self.large is not None:
            self.large.lr = lr
            return
        for g in self.opt.param_groups:
            if g["name"] == name:
                g["lr"] = lr


@torch.no_grad()
def outside_masks(points: torch.Tensor, views, dilate_px: int = 2) -> torch.Tensor:
    """True for points that project outside the (dilated) object mask in at least one view.

    Visual-hull property: every true surface point lies inside the silhouette in every view.
    views: iterable of (PinholeCamera, mask [H, W] tensor/array).
    """
    out = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    for cam, mask in views:
        m = torch.as_tensor(mask, dtype=torch.float32, device=points.device)[None, None]
        if dilate_px > 0:
            m = F.max_pool2d(m, 2 * dilate_px + 1, 1, dilate_px)
        c2w = torch.as_tensor(cam.c2w, dtype=torch.float32, device=points.device)
        xc = (points - c2w[:3, 3]) @ c2w[:3, :3]
        z = -xc[:, 2]
        u = cam.focal * xc[:, 0] / z.clamp_min(1e-6) + 0.5 * cam.width
        v = -cam.focal * xc[:, 1] / z.clamp_min(1e-6) + 0.5 * cam.height
        inside = (z > 0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        grid = torch.stack([u / cam.width * 2 - 1, v / cam.height * 2 - 1], -1)[None, None]
        mv = F.grid_sample(m, grid, align_corners=False)[0, 0, 0]
        out |= inside & (mv < 0.5)
    return out


# ---------------------------------------------------------------------- initialisation
@torch.no_grad()
def visual_hull_soup(views, res: int = 128, bbox: float = 1.5, size_factor: float = 0.9,
                     max_tris: int = 300_000, seed: int = 0, device="cuda") -> torch.Tensor:
    """Disjoint triangles on the surface of the visual hull carved from soft masks.

    views: iterable of (PinholeCamera, mask [H, W] numpy). Returns verts [F, 3, 3].
    """
    g = torch.Generator(device=device).manual_seed(seed)
    lin = torch.linspace(-bbox, bbox, res, device=device)
    X = torch.stack(torch.meshgrid(lin, lin, lin, indexing="ij"), -1).reshape(-1, 3)
    occ = torch.ones(X.shape[0], dtype=torch.bool, device=device)
    for cam, mask in views:
        c2w = torch.as_tensor(cam.c2w, dtype=torch.float32, device=device)
        xc = (X - c2w[:3, 3]) @ c2w[:3, :3]                                      # camera frame (-z forward)
        z = -xc[:, 2]
        f = cam.focal
        u = f * xc[:, 0] / z.clamp_min(1e-6) + 0.5 * cam.width
        v = -f * xc[:, 1] / z.clamp_min(1e-6) + 0.5 * cam.height
        inside = (z > 0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        m = torch.as_tensor(mask, dtype=torch.float32, device=device)[None, None]
        grid = torch.stack([u / cam.width * 2 - 1, v / cam.height * 2 - 1], -1)[None, None]
        mv = F.grid_sample(m, grid, align_corners=False)[0, 0, 0]
        occ &= ~inside | (mv > 0.5)
    O = occ.reshape(res, res, res).float()
    pad = F.pad(O[None, None], (1, 1, 1, 1, 1, 1), value=0)[0, 0]
    nb_min = torch.stack([pad[2:, 1:-1, 1:-1], pad[:-2, 1:-1, 1:-1], pad[1:-1, 2:, 1:-1],
                          pad[1:-1, :-2, 1:-1], pad[1:-1, 1:-1, 2:], pad[1:-1, 1:-1, :-2]]).min(0).values
    surf = (O > 0) & (nb_min < 1)
    blur = F.avg_pool3d(F.pad(O[None, None], (2,) * 6, mode="replicate"), 5, 1)[0, 0]
    gx, gy, gz = torch.gradient(blur)
    nrm = -F.normalize(torch.stack([gx, gy, gz], -1), dim=-1)                    # outward
    centers = X.reshape(res, res, res, 3)[surf]
    normals = nrm[surf]
    if centers.shape[0] > max_tris:
        keep = torch.randperm(centers.shape[0], generator=g, device=device)[:max_tris]
        centers, normals = centers[keep], normals[keep]
    vox = 2 * bbox / (res - 1)
    a = F.normalize(torch.cross(normals, torch.randn(normals.shape, generator=g, device=device), dim=-1), dim=-1)
    b = torch.cross(normals, a, dim=-1)
    r = size_factor * vox
    angs = torch.tensor([0.0, 2 * math.pi / 3, 4 * math.pi / 3], device=device)
    corners = centers[:, None] + r * (torch.cos(angs)[None, :, None] * a[:, None] + torch.sin(angs)[None, :, None] * b[:, None])
    # winding so that the face normal points outward
    fn = torch.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=-1)
    flip = (fn * normals).sum(-1) < 0
    corners[flip] = corners[flip][:, [0, 2, 1]]
    return corners


def spawn_triangles(centers: torch.Tensor, normals: torch.Tensor, radius, seed: int = 0) -> torch.Tensor:
    """Small equilateral triangles [N, 3, 3] centred at `centers`, in the tangent plane of `normals`
    (random in-plane rotation), wound so that the face normal follows `normals`."""
    g = torch.Generator(device=centers.device).manual_seed(seed)
    n = F.normalize(normals, dim=-1)
    a = F.normalize(torch.cross(n, torch.randn(n.shape, generator=g, device=n.device), dim=-1), dim=-1)
    b = torch.cross(n, a, dim=-1)
    r = torch.as_tensor(radius, dtype=torch.float32, device=centers.device).reshape(-1, 1, 1) \
        if torch.is_tensor(radius) else radius
    angs = torch.tensor([0.0, 2 * math.pi / 3, 4 * math.pi / 3], device=centers.device)
    corners = centers[:, None] + r * (torch.cos(angs)[None, :, None] * a[:, None] + torch.sin(angs)[None, :, None] * b[:, None])
    fn = torch.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=-1)
    flip = (fn * n).sum(-1) < 0
    corners[flip] = corners[flip][:, [0, 2, 1]]
    return corners


@torch.no_grad()
def surface_soup(mesh, views, n_tris: int = 40_000, poisson_depth: int = 9, carve_dilate: int = 2,
                 size_factor: float = 0.7, seed: int = 0) -> tuple[torch.Tensor, dict]:
    """Initial soup on a given surface (instead of the visual hull): single-sheet screened Poisson of the
    oriented input mesh, silhouette-carved, then n_tris small triangles at area-weighted samples."""
    from relight.geomfilter import poisson, sample_surface
    from relight.mesh import FrozenMesh
    pm = poisson(mesh, depth=poisson_depth)
    keep = ~outside_masks(pm.vertices[pm.faces.long()].mean(1), views, carve_dilate)
    pm = FrozenMesh(pm.vertices, pm.faces[keep].contiguous())
    from relight.orient import vertex_normals
    pm.vertex_normals = vertex_normals(pm.vertices, pm.faces)
    pts, nrm, _ = sample_surface(pm, n_tris, seed=seed)
    area = float(pm.face_areas.sum())
    r = size_factor * math.sqrt(area / n_tris)
    return spawn_triangles(pts, nrm, r, seed), {"poisson_faces": int(pm.faces.shape[0]), "area": area, "radius": r}


def texture_weight(gt: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    """exp(-|grad I| / sigma) of a [H, W, 3] image (max over channels, 3x3 max-filtered): ~1 on textureless
    regions, ~0 on texture / shading edges."""
    g = gt.permute(2, 0, 1)[None]
    gx = F.pad((g[..., :, 1:] - g[..., :, :-1]).abs(), (0, 1, 0, 0))
    gy = F.pad((g[..., 1:, :] - g[..., :-1, :]).abs(), (0, 0, 0, 1))
    mag = torch.maximum(gx, gy).amax(1, keepdim=True)
    mag = F.max_pool2d(mag, 3, 1, 1)
    return torch.exp(-mag / sigma)[0, 0]


def flatness_loss(out: dict, cam, weight: torch.Tensor, strides=(8, 32, 64)) -> torch.Tensor:
    """Texture-weighted screen-space second differences of inverse depth at several pixel strides (zero
    exactly for planes under perspective; large strides see the low-frequency bending of a dome, which
    one-pixel differences on a piecewise-planar mesh do not). With the silhouette fixing the boundary, the
    minimiser over a textureless region is the plane spanning it. A triple (x-k, x, x+k) counts only if all
    three pixels are covered and the image is textureless along it (weight min-pooled over the stride).
    Occlusion boundaries are excluded only through the texture weight (they nearly always are image edges):
    an explicit depth-discontinuity cutoff lets the optimiser escape the prior by creating steep kinks.
    Squared (not L1): L1 of second differences is satisfied by a few sharp creases (a pyramid keeps the
    dome's height); L2 spreads the bending and its minimiser with a fixed linear boundary is the plane."""
    c2w = torch.as_tensor(cam.c2w, dtype=torch.float32, device=out["position"].device)
    z = -((out["position"] - c2w[:3, 3]) @ c2w[:3, 2])                  # camera-space depth (OpenGL -z)
    m = out["mask"]
    d = 1.0 / z.clamp_min(1e-3)
    wneg = -weight[None, None]
    terms = []
    for k in strides:
        w = -F.max_pool2d(wneg, 2 * k + 1, 1, k)[0, 0]                   # min weight within the stride
        for dim in (0, 1):
            n = d.shape[dim]
            if n <= 2 * k:
                continue
            sl = lambda t, a, b: t.narrow(dim, a, n - 2 * k) if b else t
            a_, b_, c_ = sl(d, 0, 1), sl(d, k, 1), sl(d, 2 * k, 1)
            valid = sl(m, 0, 1) & sl(m, k, 1) & sl(m, 2 * k, 1)
            r = (a_ - 2 * b_ + c_).abs() / b_.clamp_min(1e-6)
            if valid.any():
                terms.append((sl(w, k, 1)[valid] * r[valid] ** 2).sum() / valid.sum())
    return sum(terms) / max(len(terms), 1) if terms else d.new_zeros(())
