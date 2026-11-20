"""Phase 5: export a trained PBR run as a Blender-ready GLB.

    python scripts/export_pbr.py -m output/synthetic4relight/hotdog --run pbr_learned
    -> <model>/<run>/asset.glb (+ asset.glb.json report)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relight.export import write_pbr_glb  # noqa: E402
from relight.material import PBRMaterial  # noqa: E402
from relight.mesh import load_relight_mesh  # noqa: E402


def load_material(mesh, run_dir: Path) -> PBRMaterial:
    ck = torch.load(run_dir / "material.pt", map_location="cuda", weights_only=False)
    args = ck["args"]
    A = args.get("atlas_res", 0)
    mat = PBRMaterial(mesh.faces, mesh.vertices.shape[0], face_res=args.get("face_res", 0),
                      learn_metallic=args.get("learn_metallic", False),
                      learn_normals=args.get("learn_normals", False), atlas_res=A,
                      uv=torch.zeros(mesh.faces.shape[0], 3, 2) if A else None).cuda()
    mat.load_state_dict(ck["state_dict"])
    return mat


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--run", default="pbr_learned")
    p.add_argument("--out", default=None, help="default: <model>/<run>/asset.glb")
    p.add_argument("--tile-size", type=int, default=4)
    p.add_argument("--atlas-size", type=int, default=4096)
    p.add_argument("--albedo-scale", type=float, nargs=3, default=None,
                   help="optional per-channel albedo multiplier before export")
    a = p.parse_args()
    model = Path(a.model)
    run = model / a.run
    mesh = load_relight_mesh(model)
    mat = load_material(mesh, run)
    with torch.no_grad():
        v = mat.vertex_values()
        albedo = v["albedo"]
        if a.albedo_scale is not None:
            albedo = (albedo * torch.tensor(a.albedo_scale, device=albedo.device)).clamp(0, 1)
    out = Path(a.out) if a.out else run / "asset.glb"
    face_tiles = None
    if mat.residual is not None:                                  # per-face residual textures -> exact tiles
        face_tiles = mat.face_texel_values(a.tile_size)
        if a.albedo_scale is not None:
            face_tiles["albedo"] = np.clip(face_tiles["albedo"] * np.asarray(a.albedo_scale, np.float32), 0, 1)
    with torch.no_grad():
        shade_n = mat.shading_normals(mesh.vertex_normals)                 # learned normals if any (glTF NORMAL)
    uv_atlas = None
    if mat.atlas is not None:
        uv_atlas = mat.bake_atlas()
        if a.albedo_scale is not None:
            uv_atlas["albedo"] = np.clip(uv_atlas["albedo"] * np.asarray(a.albedo_scale, np.float32), 0, 1)
        uv_atlas["uv"] = mat.uv.cpu().numpy()
    rep = write_pbr_glb(mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy(), shade_n.cpu().numpy(),
                        albedo.cpu().numpy(), v["rough"].cpu().numpy(), v["metal"].cpu().numpy(), out,
                        tile_size=a.tile_size, atlas_size=a.atlas_size, face_tiles=face_tiles, uv_atlas=uv_atlas)
    print(f"[export] {rep['faces']:,} faces, {rep['pages']} atlas page(s), {rep['bytes'] / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
