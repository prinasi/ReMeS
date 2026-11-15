"""Synthetic4Relight dataset access (InvRender layout).

<root>/<scene>/
    transforms_{train,test}.json       Blender c2w poses, camera_angle_x
    train/{i}_rgb.exr, {i}_mask.png    linear HDR (background visible) + soft mask
    test/{i}_rgba.png                  16-bit, gamma-2.2 encoded, alpha = mask
    test/{i}_albedo.png, {i}_rough.png GT albedo / roughness (gamma-encoded per protocol)
    test_rli/envmap{6,12}_{i}.png      relit GT under envmap6 / envmap12
<root>/envmaps/envmap{3,6,12}.exr      equirect HDR lights (from the InvRender repo)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from relight.camera import PinholeCamera
from relight.io import gamma_to_linear, read_exr_rgb, read_image

SCENES = ("air_baloons", "chair", "hotdog", "jugs")
RELIGHT_ENVMAPS = ("envmap6", "envmap12")
# InvRender's training light. Assumed from the repo contents (envmap3 is the only
# other map shipped); confirmed or refuted by scripts/check_envmap_convention.py.
TRAIN_ENVMAP = "envmap3"


def is_synthetic4relight(path: str | os.PathLike) -> bool:
    path = Path(path)
    tf = path / "transforms_train.json"
    if not tf.is_file():
        return False
    with open(tf) as f:
        frames = json.load(f)["frames"]
    return bool(frames) and (path / (frames[0]["file_path"] + "_rgb.exr")).is_file()


@dataclass
class TrainView:
    camera: PinholeCamera
    linear: np.ndarray       # [H, W, 3] linear HDR radiance (object + visible background)
    mask: np.ndarray         # [H, W] soft foreground mask in [0, 1]


@dataclass
class TestView:
    camera: PinholeCamera
    rgb: np.ndarray          # [H, W, 3] display (gamma-2.2) values in [0, 1]
    mask: np.ndarray         # [H, W] alpha
    albedo: np.ndarray | None = None   # [H, W, 3] display values
    rough: np.ndarray | None = None    # [H, W] roughness in [0, 1]


class Synthetic4Relight:
    def __init__(self, root: str | os.PathLike, scene: str):
        self.root = Path(root)
        self.scene = scene
        self.path = self.root / scene
        if not self.path.is_dir():
            raise FileNotFoundError(self.path)
        self._meta = {}

    # ------------------------------------------------------------------ cameras
    def meta(self, split: str) -> dict:
        if split not in self._meta:
            with open(self.path / f"transforms_{split}.json") as f:
                self._meta[split] = json.load(f)
        return self._meta[split]

    def cameras(self, split: str, width: int = 800, height: int = 800) -> list[PinholeCamera]:
        meta = self.meta(split)
        return [PinholeCamera(np.array(fr["transform_matrix"], dtype=np.float64),
                              float(meta["camera_angle_x"]), width, height,
                              name=Path(fr["file_path"]).name)
                for fr in meta["frames"]]

    def num_views(self, split: str) -> int:
        return len(self.meta(split)["frames"])

    def _stem(self, split: str, i: int) -> Path:
        return self.path / self.meta(split)["frames"][i]["file_path"]

    # ------------------------------------------------------------------ images
    def train_view(self, i: int) -> TrainView:
        stem = self._stem("train", i)
        linear = read_exr_rgb(f"{stem}_rgb.exr")
        mask = read_image(f"{stem}_mask.png")
        if mask.ndim == 3:
            mask = mask[..., 0]
        H, W = mask.shape
        cam = self.cameras("train", W, H)[i]
        return TrainView(cam, linear, mask.astype(np.float32))

    def test_view(self, i: int, with_materials: bool = True) -> TestView:
        stem = self._stem("test", i)
        rgba = read_image(f"{stem}_rgba.png")
        H, W = rgba.shape[:2]
        cam = self.cameras("test", W, H)[i]
        view = TestView(cam, rgba[..., :3], rgba[..., 3])
        if with_materials:
            view.albedo = read_image(f"{stem}_albedo.png")[..., :3]
            view.rough = read_image(f"{stem}_rough.png")[..., 0]
        return view

    def relight_view(self, i: int, envmap: str) -> tuple[np.ndarray, np.ndarray]:
        """Relit GT (display values in [0,1]) and its alpha for test view i."""
        idx = Path(self.meta("test")["frames"][i]["file_path"]).name
        rgba = read_image(self.path / "test_rli" / f"{envmap}_{idx}.png")
        return rgba[..., :3], rgba[..., 3]

    # ------------------------------------------------------------------ lights
    def envmap_path(self, name: str) -> Path:
        p = self.root / "envmaps" / f"{name}.exr"
        if not p.is_file():
            raise FileNotFoundError(f"{p} (download from github.com/zju3dv/InvRender code/envmaps)")
        return p

    @staticmethod
    def display_to_linear(x: np.ndarray) -> np.ndarray:
        return gamma_to_linear(x)
