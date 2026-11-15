"""Pinhole cameras in the Blender / NeRF-synthetic convention.

``c2w`` is the Blender camera-to-world matrix from ``transforms_*.json``:
the camera looks along -Z with +Y up and +X right. Pixel (i, j) has its centre at
(i + 0.5, j + 0.5), j counted from the top row.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PinholeCamera:
    c2w: np.ndarray          # [4, 4] Blender camera-to-world
    fovx: float              # horizontal field of view (radians)
    width: int
    height: int
    name: str = ""

    @property
    def focal(self) -> float:
        return 0.5 * self.width / math.tan(0.5 * self.fovx)

    @property
    def fovy(self) -> float:
        return 2.0 * math.atan(0.5 * self.height / self.focal)

    @property
    def center(self) -> np.ndarray:
        return self.c2w[:3, 3]

    def scaled(self, factor: float) -> "PinholeCamera":
        return PinholeCamera(self.c2w, self.fovx, int(round(self.width * factor)),
                             int(round(self.height * factor)), self.name)

    def rays(self, device="cuda") -> tuple[torch.Tensor, torch.Tensor]:
        """World-space ray origins and unit directions, both [H, W, 3]."""
        f = self.focal
        i = torch.arange(self.width, device=device, dtype=torch.float32) + 0.5
        j = torch.arange(self.height, device=device, dtype=torch.float32) + 0.5
        jj, ii = torch.meshgrid(j, i, indexing="ij")
        d_cam = torch.stack([(ii - 0.5 * self.width) / f,
                             -(jj - 0.5 * self.height) / f,
                             -torch.ones_like(ii)], dim=-1)
        R = torch.as_tensor(self.c2w[:3, :3], dtype=torch.float32, device=device)
        d = F.normalize(d_cam @ R.T, dim=-1)
        o = torch.as_tensor(self.c2w[:3, 3], dtype=torch.float32, device=device).expand_as(d)
        return o.contiguous(), d.contiguous()

    def opencv_w2c(self) -> np.ndarray:
        """World-to-camera in the OpenCV/COLMAP convention (+Y down, +Z forward)."""
        c2w = np.array(self.c2w, dtype=np.float64)
        c2w[:3, 1:3] *= -1.0
        return np.linalg.inv(c2w)
