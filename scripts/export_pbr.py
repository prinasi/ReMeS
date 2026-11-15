"""Phase 5: export a trained PBR run as a Blender-ready GLB.

    python scripts/export_pbr.py -m output/synthetic4relight/hotdog --run pbr_learned
    -> <model>/<run>/asset.glb (+ asset.glb.json report)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relight.export import write_pbr_glb  # noqa: E402
from relight.material import PBRMaterial  # noqa: E402
from relight.mesh import load_relight_mesh  # noqa: E402


def load_material(mesh, run_dir: Path) -> PBRMaterial:
    ck = torch.load(run_dir / "material.pt", map_location="cuda", weights_only=False)
    args = ck["args"]
    if args.get("face_res", 0):
        raise NotImplementedError("per-face residual textures are not exported yet (vertex attributes only)")
    mat = PBRMaterial(mesh.faces, mesh.vertices.shape[0], learn_metallic=args.get("learn_metallic", False)).cuda()
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
    rep = write_pbr_glb(mesh.vertices.cpu().numpy(), mesh.faces.cpu().numpy(), mesh.vertex_normals.cpu().numpy(),
                        albedo.cpu().numpy(), v["rough"].cpu().numpy(), v["metal"].cpu().numpy(), out,
                        tile_size=a.tile_size, atlas_size=a.atlas_size)
    print(f"[export] {rep['faces']:,} faces, {rep['pages']} atlas page(s), {rep['bytes'] / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
