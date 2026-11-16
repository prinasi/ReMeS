"""Multi-view-stereo surface for initialising the opaque soup (option 2: carve with per-pixel depth,
not only silhouettes).

COLMAP PatchMatch stereo with the dataset's *known* poses (no SfM): the training views are written as
an undistorted COLMAP workspace, source views are chosen explicitly (nearest cameras by viewing
angle; COLMAP's automatic choice needs sparse points), depth is estimated with geometric consistency.
The filtered geometric depth maps (inside the object masks) are fused with an Open3D TSDF volume —
COLMAP's stereo_fusion cannot be used without SfM points (it picks overlapping views from them) —
and the extracted surface is carved with the silhouettes; the result is saved as
<out>/relight_mesh.pt for ``train_oto.py --init mesh:<out>``.

    python scripts/mvs_init.py --data data/SynthGeo --scene suzanne_glossy --out output/mvs/suzanne_glossy
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relight.data import Synthetic4Relight  # noqa: E402
from relight.io import tonemap, write_png  # noqa: E402
from relight.mesh import FrozenMesh, save_relight_mesh  # noqa: E402
from relight.orient import vertex_normals  # noqa: E402
from relight.oto import outside_masks  # noqa: E402

SKIP_TOP_ROWS = 5


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def read_colmap_array(path) -> np.ndarray:
    """COLMAP dense .bin array ("w&h&c&" header + float32 column-major data) -> [H, W] (first channel)."""
    with open(path, "rb") as fh:
        header, amps = b"", 0
        while amps < 3:
            ch = fh.read(1)
            header += ch
            amps += ch == b"&"
        w, h, c = map(int, header.decode().strip("&").split("&"))
        data = np.fromfile(fh, np.float32).reshape(c, h, w)
    return data[0]


def run(cmd, log):
    print("[colmap] " + " ".join(map(str, cmd)), flush=True)
    with open(log, "a") as fh:
        r = subprocess.run(list(map(str, cmd)), stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"COLMAP failed ({cmd[1]}); see {log}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-src", type=int, default=10, help="source views per reference view")
    p.add_argument("--max-angle", type=float, default=45.0, help="max viewing-angle difference of sources (deg)")
    p.add_argument("--poisson-depth", type=int, default=9)
    p.add_argument("--min-pixels", type=int, default=3, help="StereoFusion.min_num_pixels")
    p.add_argument("--keep-workspace", action="store_true")
    p.add_argument("--skip-patchmatch", action="store_true", help="reuse existing depth maps in <out>/colmap")
    p.add_argument("--voxel", type=float, default=0.006, help="TSDF voxel size (scene units)")
    p.add_argument("--trunc", type=float, default=0.03, help="TSDF truncation distance")
    a = p.parse_args()
    t0 = time.perf_counter()
    out = Path(a.out)
    ws = out / "colmap"
    if ws.exists() and not a.skip_patchmatch:
        shutil.rmtree(ws)
    for d in (ws / "images", ws / "masks", ws / "sparse"):
        d.mkdir(parents=True, exist_ok=True)
    log = out / "colmap.log"
    if not a.skip_patchmatch:
        log.unlink(missing_ok=True)

    ds = Synthetic4Relight(a.data, a.scene)
    views, names, centers = [], [], []
    for i in range(ds.num_views("train")):
        v = ds.train_view(i)
        m = v.mask.copy()
        m[:SKIP_TOP_ROWS] = 0
        name = f"{i:03d}.png"
        write_png(ws / "images" / name, tonemap(v.linear) * (m[..., None] > 0.5))
        write_png(ws / "masks" / (name + ".png"), np.repeat((m > 0.5)[..., None], 3, -1).astype(np.float32))
        views.append((v.camera, torch.from_numpy(np.where(np.arange(m.shape[0])[:, None] < SKIP_TOP_ROWS, 1.0,
                                                          v.mask)).float().cuda()))
        names.append(name)
        centers.append(v.camera.center)
    cam = views[0][0]
    f = cam.focal
    (ws / "sparse" / "cameras.txt").write_text(f"1 PINHOLE {cam.width} {cam.height} {f} {f} {cam.width / 2} {cam.height / 2}\n")
    lines = []
    for k, (c, _) in enumerate(views):
        w2c = c.opencv_w2c()
        q = rot_to_quat(w2c[:3, :3])
        tx, ty, tz = w2c[:3, 3]
        lines.append(f"{k + 1} {q[0]} {q[1]} {q[2]} {q[3]} {tx} {ty} {tz} 1 {names[k]}\n\n")
    (ws / "sparse" / "images.txt").write_text("".join(lines))
    (ws / "sparse" / "points3D.txt").write_text("")
    if not a.skip_patchmatch:
        run(["colmap", "image_undistorter", "--image_path", ws / "images", "--input_path", ws / "sparse",
             "--output_path", ws / "dense", "--output_type", "COLMAP"], log)

    # explicit source views: nearest viewing directions (towards the scene centre)
    C = np.stack(centers)
    dirs = C / np.linalg.norm(C, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(dirs @ dirs.T, -1, 1)))
    cfg = []
    for k in range(len(names)):
        order = [j for j in np.argsort(ang[k]) if j != k and ang[k, j] <= a.max_angle][:a.n_src]
        cfg.append(names[k] + "\n" + ", ".join(names[j] for j in order) + "\n")
    dist = np.linalg.norm(C, axis=1)
    dmin, dmax = max(0.1, float(dist.min()) - 2.5), float(dist.max()) + 2.5
    if not a.skip_patchmatch:
        (ws / "dense" / "stereo" / "patch-match.cfg").write_text("".join(cfg))
        run(["colmap", "patch_match_stereo", "--workspace_path", ws / "dense", "--workspace_format", "COLMAP",
         "--PatchMatchStereo.geom_consistency", "true", "--PatchMatchStereo.depth_min", dmin,
         "--PatchMatchStereo.depth_max", dmax, "--PatchMatchStereo.gpu_index", "0",
         "--PatchMatchStereo.filter", "true"], log)
    # TSDF fusion of the geometric depth maps (inside the masks)
    import open3d as o3d
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=a.voxel, sdf_trunc=a.trunc, color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor)
    intr = o3d.camera.PinholeCameraIntrinsic(cam.width, cam.height, f, f, cam.width / 2, cam.height / 2)
    valid_share = []
    for k, (c, m) in enumerate(views):
        dpath = ws / "dense" / "stereo" / "depth_maps" / f"{names[k]}.geometric.bin"
        depth = read_colmap_array(dpath)
        mk = (m.cpu().numpy() > 0.5)
        mk[:SKIP_TOP_ROWS] = False
        depth = np.where(mk & (depth > 0), depth, 0).astype(np.float32)
        valid_share.append(float((depth > 0).sum() / max(mk.sum(), 1)))
        color = o3d.geometry.Image(np.zeros((cam.height, cam.width, 3), np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, o3d.geometry.Image(depth), depth_scale=1.0, depth_trunc=dmax, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, c.opencv_w2c())
    mesh = vol.extract_triangle_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    pts = np.asarray(mesh.vertices)
    if len(pts) < 1000:
        raise RuntimeError(f"TSDF produced too few vertices ({len(pts)}); see {log}")
    V = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device="cuda")
    Fc = torch.tensor(np.asarray(mesh.triangles), dtype=torch.int32, device="cuda")
    keep = ~outside_masks(V[Fc.long()].mean(1), views, 2)
    Fc = Fc[keep].contiguous()
    fm = FrozenMesh(V, Fc, vertex_normals=vertex_normals(V, Fc))
    save_relight_mesh(fm, out / "relight_mesh.pt", {"source": "colmap_patchmatch+tsdf"})
    stats = {"depth_valid_share_mean": float(np.mean(valid_share)), "faces": int(Fc.shape[0]),
             "carved_faces": int((~keep).sum()),
             "depth_range": [dmin, dmax], "seconds": time.perf_counter() - t0}
    (out / "mvs_stats.json").write_text(json.dumps(stats, indent=2))
    if not a.keep_workspace:
        shutil.rmtree(ws / "dense" / "stereo", ignore_errors=True)
    print(f"[mvs] {stats} -> {out / 'relight_mesh.pt'}")


if __name__ == "__main__":
    main()
