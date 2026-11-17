"""Smoothness-preconditioned vertex optimisation (Nicolet, Jacobson & Jakob 2021, "Large Steps in Inverse
Rendering of Geometry"), without the cholespy dependency.

Vertices x are optimised through u = (I + λL) x (L = combinatorial graph Laplacian): the gradient is
g_u = (I + λL)^{-1} g_x and an update Δu maps back as Δx = (I + λL)^{-1} Δu. High-frequency (per-vertex)
gradient noise is damped by (1 + λμ)^{-2} while low-frequency modes pass, and "uniform Adam" (one scalar
second-moment normalisation, the maximum over all entries) keeps the relative scale of the modes, so a
coherent low-frequency error (e.g. a whole plane offset) moves with large steps instead of being drowned in
per-vertex noise as with per-coordinate Adam. Both solves use Jacobi-preconditioned conjugate gradients on GPU.
"""
from __future__ import annotations

import torch


class LargeSteps:
    def __init__(self, n_verts: int, edges: torch.Tensor, lam: float = 19.0, lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8, cg_tol: float = 1e-5, cg_iters: int = 300):
        dev = edges.device
        a, b = edges[:, 0].long(), edges[:, 1].long()
        deg = torch.zeros(n_verts, device=dev).index_add_(0, a, torch.ones_like(a, dtype=torch.float)) \
            .index_add_(0, b, torch.ones_like(b, dtype=torch.float))
        idx = torch.cat([torch.stack([a, b]), torch.stack([b, a]),
                         torch.arange(n_verts, device=dev).repeat(2, 1)], 1)
        val = torch.cat([torch.full((2 * a.numel(),), -lam, device=dev), 1 + lam * deg])
        self.M = torch.sparse_coo_tensor(idx, val, (n_verts, n_verts)).coalesce().to_sparse_csr()
        self.diag = 1 + lam * deg
        self.lam, self.lr, self.betas, self.eps = lam, lr, betas, eps
        self.cg_tol, self.cg_iters = cg_tol, cg_iters
        self.m = torch.zeros(n_verts, 3, device=dev)
        self.v = torch.zeros(n_verts, 3, device=dev)
        self.t = 0
        self.last_cg_iters = 0

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        """(I + λL) X = rhs for [V, 3] (columns independent), Jacobi-preconditioned CG."""
        x = rhs / self.diag[:, None]
        r = rhs - self.M @ x
        z = r / self.diag[:, None]
        p = z.clone()
        rz = (r * z).sum(0)
        b2 = (rhs * rhs).sum(0).clamp_min(1e-30)
        for it in range(self.cg_iters):
            Ap = self.M @ p
            alpha = rz / (p * Ap).sum(0).clamp_min(1e-30)
            x = x + alpha * p
            r = r - alpha * Ap
            if bool(((r * r).sum(0) <= self.cg_tol ** 2 * b2).all()):
                break
            z = r / self.diag[:, None]
            rz_new = (r * z).sum(0)
            p = z + (rz_new / rz.clamp_min(1e-30)) * p
            rz = rz_new
        self.last_cg_iters = it + 1
        return x

    @torch.no_grad()
    def step(self, verts: torch.Tensor):
        if verts.grad is None:
            return
        g = self.solve(verts.grad)
        b1, b2 = self.betas
        self.t += 1
        self.m.mul_(b1).add_(g, alpha=1 - b1)
        self.v.mul_(b2).addcmul_(g, g, value=1 - b2)
        mh = self.m / (1 - b1 ** self.t)
        vh = self.v / (1 - b2 ** self.t)
        du = -self.lr * mh / (vh.max().sqrt() + self.eps)                 # uniform Adam
        verts.add_(self.solve(du))
        verts.grad = None
