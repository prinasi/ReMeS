"""Frozen-mesh access for T²-Splat checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class FrozenMesh:
    vertices: torch.Tensor          # [V, 3] float32 (Blender world frame, same as the cameras)
    faces: torch.Tensor             # [F, 3] int32
    face_textures: torch.Tensor | None = None   # [F, R, R, 3] T²-Splat residual/base textures
    source: str = ""
    vertex_normals: torch.Tensor | None = None  # [V, 3] outward shading normals (oriented meshes only)

    @property
    def face_normals(self) -> torch.Tensor:
        """Unnormalised-orientation geometric normals [F, 3] from the winding order."""
        v = self.vertices[self.faces.long()]
        return F.normalize(torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1), dim=-1)

    @property
    def face_areas(self) -> torch.Tensor:
        v = self.vertices[self.faces.long()]
        return 0.5 * torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=-1).norm(dim=-1)

    def to(self, device) -> "FrozenMesh":
        ft = self.face_textures.to(device) if self.face_textures is not None else None
        vn = self.vertex_normals.to(device) if self.vertex_normals is not None else None
        return FrozenMesh(self.vertices.to(device), self.faces.to(device), ft, self.source, vn)


def resolve_checkpoint(model_dir: str | Path, iteration: int = -1) -> Path:
    model_dir = Path(model_dir)
    if model_dir.is_file():
        return model_dir
    root = model_dir / "point_cloud"
    if iteration == -1:
        its = sorted(int(p.name.split("_")[1]) for p in root.glob("iteration_*")
                     if (p / "point_cloud_state_dict.pt").is_file())
        if not its:
            raise FileNotFoundError(f"no checkpoints under {root}")
        iteration = its[-1]
    path = root / f"iteration_{iteration}" / "point_cloud_state_dict.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_t2splat_mesh(model_dir: str | Path, iteration: int = -1, device: str = "cuda",
                      drop_unused: bool = True) -> FrozenMesh:
    """Load vertices/faces (and face textures if present) from a T²-Splat checkpoint."""
    path = resolve_checkpoint(model_dir, iteration)
    state = torch.load(path, map_location="cpu", weights_only=False)
    V = state["triangles_points"].detach().float()
    Fc = state["_triangle_indices"].detach().long()
    tex = state.get("face_textures")
    tex = tex.detach().float() if isinstance(tex, torch.Tensor) else None
    if drop_unused:
        used = torch.zeros(V.shape[0], dtype=torch.bool)
        used[Fc.reshape(-1)] = True
        remap = torch.full((V.shape[0],), -1, dtype=torch.long)
        remap[used] = torch.arange(int(used.sum()))
        V, Fc = V[used], remap[Fc]
    return FrozenMesh(V.contiguous().to(device), Fc.int().contiguous().to(device),
                      tex.to(device) if tex is not None else None, str(path))


RELIGHT_MESH = "relight_mesh.pt"


def save_relight_mesh(mesh: FrozenMesh, path: str | Path, meta: dict | None = None) -> None:
    """Persist an oriented frozen mesh (the Phase 2+ input)."""
    torch.save({"vertices": mesh.vertices.cpu(), "faces": mesh.faces.cpu(),
                "vertex_normals": None if mesh.vertex_normals is None else mesh.vertex_normals.cpu(),
                "face_textures": None if mesh.face_textures is None else mesh.face_textures.cpu(),
                "source": mesh.source, "meta": meta or {}}, path)


def load_relight_mesh(path: str | Path, device: str = "cuda") -> FrozenMesh:
    path = Path(path)
    if path.is_dir():
        path = path / RELIGHT_MESH
    d = torch.load(path, map_location="cpu", weights_only=False)
    opt = lambda k: None if d.get(k) is None else d[k].to(device)
    return FrozenMesh(d["vertices"].to(device), d["faces"].to(device), opt("face_textures"),
                      str(path), opt("vertex_normals"))
