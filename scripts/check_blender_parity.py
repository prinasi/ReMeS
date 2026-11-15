"""Phase 5: shading-model parity with Blender on a clean sphere (no geometry noise).

Exports a smooth icosphere with uniform albedo/roughness, renders it with Cycles
(direct light only) and with the ReMeS renderer under the same envmap, and reports
PSNR and the Cycles/ReMeS brightness ratio per roughness.

    python scripts/check_blender_parity.py [--envmap envmap12] [--no-layering]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_pbr  # noqa: E402
from relight.data import Synthetic4Relight  # noqa: E402
from relight.envmap import load_envmap  # noqa: E402
from relight.export import write_pbr_glb  # noqa: E402
from relight.io import psnr, read_image, tonemap  # noqa: E402
from relight.material import PBRMaterial  # noqa: E402
from relight.mesh import FrozenMesh  # noqa: E402
from relight.shading import EnvLight, brdf_lut, fibonacci_sphere  # noqa: E402
from relight.visibility import vertex_visibility  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data/Synthetic4Relight"))
    p.add_argument("--envmap", default="envmap12")
    p.add_argument("--albedo", type=float, default=0.6)
    p.add_argument("--roughness", type=float, nargs="+", default=[0.2, 0.5, 0.9])
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--no-layering", action="store_true")
    p.add_argument("--out", default=str(ROOT / "output/phase5/parity_sphere"))
    a = p.parse_args()
    train_pbr.LAYERING = not a.no_layering
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = Synthetic4Relight(a.data, "hotdog")          # cameras only
    cam = ds.cameras("test")[0].scaled(0.5)
    s = trimesh.creation.icosphere(subdivisions=6, radius=0.6)
    V = torch.tensor(s.vertices, dtype=torch.float32, device="cuda")
    Fc = torch.tensor(s.faces, dtype=torch.int32, device="cuda")
    mesh = FrozenMesh(V, Fc, vertex_normals=F.normalize(V, dim=-1))
    dirs = fibonacci_sphere(512)
    vis = vertex_visibility(mesh, dirs)
    lut = brdf_lut()
    light = EnvLight(load_envmap(ds.envmap_path(a.envmap)), dirs)
    report = {}
    for r in a.roughness:
        mat = PBRMaterial(mesh.faces, V.shape[0], albedo_init=a.albedo, rough_init=r).cuda()
        vv = {k: x.detach().cpu().numpy() for k, x in mat.vertex_values().items()}
        glb = out / f"sphere_r{r:g}.glb"
        write_pbr_glb(V.cpu().numpy(), Fc.cpu().numpy(), mesh.vertex_normals.cpu().numpy(), vv["albedo"], vv["rough"],
                      vv["metal"], glb)
        rdir = out / f"r{r:g}"
        subprocess.run([sys.executable, "-m", "relight.blender.render_relight", "--glb", str(glb), "--envmap",
                        str(ds.envmap_path(a.envmap)), "--transforms", str(ds.path / "transforms_test.json"),
                        "--frames", "0", "--out-dir", str(rdir), "--samples", str(a.samples),
                        "--width", str(cam.width), "--height", str(cam.height), "--direct-only"],
                       check=True, capture_output=True, cwd=ROOT)
        cy = torch.from_numpy(read_image(rdir / "000.exr")).cuda()
        o = train_pbr.render_view(mesh, mat, vis, light, lut, cam)
        m = o["mask"] & (cy[..., 3] > 0.99)
        m = F.max_pool2d((~m).float()[None, None], 5, 1, 2)[0, 0] == 0
        ratio = (cy[..., :3][m].mean(0) / o["rgb"][m].mean(0)).tolist()
        report[f"{r:g}"] = {"psnr": psnr(tonemap(cy[..., :3]), tonemap(o["rgb"]), m), "ratio": ratio}
        print(f"roughness {r:g}: parity {report[f'{r:g}']['psnr']:.2f} dB, Cycles/ReMeS ratio "
              f"{[round(x, 3) for x in ratio]}", flush=True)
    (out / "report.json").write_text(json.dumps({"layering": not a.no_layering, "results": report}, indent=2))


if __name__ == "__main__":
    main()
