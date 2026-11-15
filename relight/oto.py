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

    def render(self, cam, light, lut, mvp: torch.Tensor | None = None, track_coverage: bool = False) -> dict:
        out = render_triangles(self.verts.reshape(-1, 3), self.faces(), self.corner_attributes(), self.face_normals(),
                               cam, light, lut, mvp)
        if track_coverage:
            with torch.no_grad():
                m = out["mask"]
                self.max_cov = torch.maximum(self.max_cov, torch.bincount(out["tri"][m], minlength=self.num).float())
        return out

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
        mc = self.max_cov[keep]
        if new is not None:
            mc = torch.cat([mc, torch.zeros(new["verts"].shape[0], device=mc.device)], 0)
        self.max_cov = mc

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

    def reset_coverage(self):
        self.max_cov.zero_()

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
def render_triangles(V: torch.Tensor, faces: torch.Tensor, attrs: torch.Tensor, face_normals: torch.Tensor, cam, light,
                     lut, mvp: torch.Tensor | None = None) -> dict:
    """Opaque z-buffer render of triangles with per-vertex attrs [N, 5] = (diffuse rgb, F0, roughness)."""
    dev = V.device
    mvp = mvp_matrix(cam).to(dev) if mvp is None else mvp
    clip = (torch.cat([V, torch.ones_like(V[:, :1])], -1) @ mvp.T)[None].contiguous()
    rast, _ = dr.rasterize(get_context(dev), clip, faces, resolution=[cam.height, cam.width])
    tri = rast[0, ..., 3].long() - 1
    mask = tri >= 0
    interp, _ = dr.interpolate(torch.cat([attrs, V], -1)[None].contiguous(), rast, faces)
    interp = interp[0]
    pos = interp[..., 5:8]
    o = torch.as_tensor(np.asarray(cam.center), dtype=torch.float32, device=dev)
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
        spec = light.specular(refl, rho) * (interp[..., 3][mask][:, None] * ab[:, :1] + ab[:, 1:])
        rgb = rgb.index_put(mask.nonzero(as_tuple=True), rgb[mask] + spec)
    rgba = torch.cat([rgb * mask[..., None], mask[..., None].float()], -1)
    aa = dr.antialias(rgba[None].contiguous(), rast, clip, faces)[0]
    return {"rgb": aa[..., :3], "alpha": aa[..., 3], "mask": mask, "normal": n, "position": pos, "tri": tri}


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

    def render(self, cam, light, lut, mvp=None, track_coverage=False):
        out = render_triangles(self.verts, self.faces, self.attributes(), self.face_normals(), cam, light, lut, mvp)
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
    def prune_faces(self, remove: torch.Tensor) -> int:
        if remove.any():
            self.faces = self.faces[~remove].contiguous()
            self.max_cov = self.max_cov[~remove]
            self._edges()
        return int(remove.sum())

    def step(self):
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    def set_lr(self, name, lr):
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
