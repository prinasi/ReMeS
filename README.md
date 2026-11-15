<h1 align="center">ReMeS: Relightable Mesh Splatting</h1>

ReMeS turns the triangle meshes reconstructed by
[T<sup>2</sup>-Splat](https://arxiv.org/abs/2611.06974) (this repository's base) into relightable
PBR assets (albedo / roughness / metallic) that relight in our differentiable renderer and in
standard CG engines such as Blender. The design and the phase-by-phase status are in
[docs/RELIGHTING_PLAN.md](docs/RELIGHTING_PLAN.md).

Pipeline: T<sup>2</sup>-Splat geometry (frozen, opaque mesh) → per-face PBR textures under known,
then unknown, environment light with ray-traced visibility on the frozen mesh → glTF PBR export
validated in Blender Cycles.

## Installation

Requirements: Linux, Python 3.11, an NVIDIA GPU, a CUDA toolkit with `nvcc`, and a C++17 compiler.

```bash
git clone https://github.com/t2-splat/T2Splat.git
cd T2Splat

conda create -n t2splat python=3.11 -y
conda activate t2splat
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
bash scripts/setup.sh
# frozen-mesh ray queries (relight/raytrace.py)
pip install --no-deps --no-build-isolation git+https://github.com/ashawkey/cubvh
```

Native extension sources are included in the repository. Set `CUDA_HOME` if the CUDA toolkit is not on `PATH`. The setup script also builds nvdiffrast, which ReMeS uses for G-buffers and GLB baking. Blender comparisons need `bpy` in the same environment.

Before running individual Python programs, load the local extension paths:

```bash
source scripts/env.sh
```

## Data

Synthetic4Relight (InvRender layout) under `data/Synthetic4Relight/`:

```text
data/Synthetic4Relight/
  envmaps/envmap{3,6,12}.exr        # from github.com/zju3dv/InvRender (code/envmaps); envmap3 = training light
  <scene>/                          # air_baloons, chair, hotdog, jugs
    transforms_{train,test}.json    # Blender camera-to-world poses
    train/{i}_rgb.exr, {i}_mask.png # linear HDR + soft mask
    test/{i}_rgba.png, {i}_albedo.png, {i}_rough.png
    test_rli/envmap{6,12}_{i}.png   # relit ground truth
```

The scene type is detected automatically (`transforms_train.json` + `*_rgb.exr`). Training views are
tonemapped with gamma 2.2 (the dataset protocol) and passed to T<sup>2</sup>-Splat as RGBA so that
training composites over a random background and applies a silhouette loss.

## Phase 0 checks

```bash
source scripts/env.sh
python scripts/check_envmap_convention.py   # envmap/camera conventions + Cycles parity
python scripts/bench_raytrace.py            # frozen-mesh ray queries (cubvh) accuracy/throughput
python -m unittest tests.test_synthetic4relight tests.test_gbuffer -v
```

## Geometry (Phase 1)

```bash
bash scripts/run.sh synthetic4relight 0 hotdog           # GPU 0, one scene
bash scripts/run.sh synthetic4relight 1 all              # all four scenes
python scripts/eval_geometry.py -m output/synthetic4relight/hotdog --scene hotdog
```

Parameters live in `configs/common.yaml` and `configs/synthetic4relight.yaml`; override with
`--set section.key=value` or `--config extra.yaml`. Each run trains geometry (30k iterations),
refines the T<sup>2</sup>-Splat appearance (5k), renders and scores the test views, and exports
`mesh.ply`.

## Rendering and mesh export

**PLY:** Bake SH and MipTex into vertex colors while preserving the original vertices and faces.

```bash
python create_ply.py output/synthetic4relight/hotdog --iteration 35000 \
  --out output/synthetic4relight/hotdog/mesh.ply
```

**GLB:** Bake SH and MipTex into embedded textures for Blender and other glTF viewers, keeping the triangle count unchanged.

```bash
python create_glb.py output/synthetic4relight/hotdog --iteration 35000 \
  --out output/synthetic4relight/hotdog/mesh.glb
```

Use the saved iteration corresponding to your model.

## OpenGL rendering benchmark

Measure the off-screen GPU rendering throughput of an exported, vertex-colored PLY mesh. This standalone tool runs on macOS, Windows, or Linux with an OpenGL 4.1 context and GPU timer queries; it does not require CUDA or the training environment.

Install its optional dependencies and benchmark at **1920 × 1080**:

```bash
python3 -m pip install -r requirements-fps.txt
python3 tools/mesh_fps_benchmark.py output/synthetic4relight/hotdog/mesh.ply \
  --resolution 1920x1080 --json output/synthetic4relight/hotdog/fps.json
```

FPS measures GPU rendering time, not end-to-end application or display frame times. The benchmark uses baked vertex colors and does not evaluate the full SH/MipTex appearance model. See [benchmark details](docs/mesh_fps_benchmark.md) for supported PLY formats, rendering settings, the camera path, measurement definitions, and JSON output.

## Related Work

Related research on triangle splatting, differentiable meshes, and surface reconstruction:

- [MeshSplatting: Differentiable Rendering with Opaque Meshes](https://meshsplatting.github.io/)
- [Triangle Splatting for Real-Time Radiance Field Rendering](https://trianglesplatting.github.io/)
- [3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes](https://convexsplatting.github.io/)
- [MILo: Mesh-In-the-Loop Gaussian Splatting for Detailed and Efficient Surface Reconstruction](https://anttwo.github.io/milo/)
- [DMesh++: An Efficient Differentiable Mesh for Complex Shapes](https://sonsang.github.io/dmesh2-project/)
- [DMesh: A Differentiable Mesh Representation](https://sonsang.github.io/dmesh-project/)
- [Gaussian Wrapping — From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented Gaussians](https://diego1401.github.io/BlobsToSpokesWebsite/)

## BibTeX

If you find T<sup>2</sup>-Splat useful for your research, please cite our paper:

```bibtex
@misc{tang2026t2splatadaptivetopologymesh,
      title={T$^2$-Splat: Adaptive Topology Mesh Splatting with Texture Residuals},
      author={Zhihao Tang and Youjia Zhang and Mingbo Zhao and Wei Yang},
      year={2026},
      eprint={2611.06974},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2611.06974},
}
```

The following works helped make our work possible:

```bibtex
@article{Held2025Triangle,
  title   = {Triangle Splatting for Real-Time Radiance Field Rendering},
  author  = {Held, Jan and Vandeghen, Renaud and Deliege, Adrien and Hamdi, Abdullah and Giancola, Silvio and Cioppa, Anthony and Vedaldi, Andrea and Ghanem, Bernard and Tagliasacchi, Andrea and Van Droogenbroeck, Marc},
  journal = {arXiv preprint arXiv:2505.19175},
  year    = {2025},
  url     = {https://arxiv.org/abs/2505.19175}
}

@article{Held2025MeshSplatting,
  title   = {{MeshSplatting}: Differentiable Rendering with Opaque Meshes},
  author  = {Held, Jan and Son, Sanghyun and Vandeghen, Renaud and Rebain, Daniel and Gadelha, Matheus and Zhou, Yi and Cioppa, Anthony and Lin, Ming C. and Van Droogenbroeck, Marc and Tagliasacchi, Andrea},
  journal = {arXiv preprint arXiv:2512.06818},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.06818}
}
```
