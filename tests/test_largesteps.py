"""Large-steps vertex optimiser: CG solve accuracy, low-frequency pass-through, high-frequency damping.

    python -m unittest tests.test_largesteps -v
"""
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "GPU not available")
class LargeStepsTest(unittest.TestCase):
    def setUp(self):
        n = 64
        i = torch.arange(n, device="cuda")
        a = (i[:, None] * n + i[None, :]).reshape(n, n)
        e = torch.cat([torch.stack([a[:, :-1].reshape(-1), a[:, 1:].reshape(-1)], -1),
                       torch.stack([a[:-1].reshape(-1), a[1:].reshape(-1)], -1)])
        self.n, self.edges = n * n, e

    def test_solve(self):
        from relight.largesteps import LargeSteps
        ls = LargeSteps(self.n, self.edges, lam=19.0)
        b = torch.randn(self.n, 3, device="cuda")
        x = ls.solve(b)
        self.assertLess(float((ls.M @ x - b).norm() / b.norm()), 1e-4)

    def test_modes(self):
        from relight.largesteps import LargeSteps
        ls = LargeSteps(self.n, self.edges, lam=19.0, lr=1e-2)
        # constant gradient (a whole-plane offset): every vertex moves by the same amount ~ lr
        x = torch.zeros(self.n, 3, device="cuda", requires_grad=True)
        x.grad = torch.zeros_like(x)
        x.grad[:, 2] = 1.0
        ls.step(x)
        dz = x.detach()[:, 2]
        self.assertAlmostEqual(float(dz.mean()), -1e-2, delta=1e-3)
        self.assertLess(float(dz.std()), 1e-4)
        # per-vertex noise of the same rms: the resulting motion is much smaller
        ls2 = LargeSteps(self.n, self.edges, lam=19.0, lr=1e-2)
        y = torch.zeros(self.n, 3, device="cuda", requires_grad=True)
        y.grad = torch.zeros_like(y)
        y.grad[:, 2] = torch.randn(self.n, device="cuda")
        ls2.step(y)
        self.assertLess(float(y.detach()[:, 2].abs().mean()), 0.2e-2)


if __name__ == "__main__":
    unittest.main()
