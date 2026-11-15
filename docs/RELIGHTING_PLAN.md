# ReMeS: Relightable Mesh Splatting on T²-Splat — Implementation Plan

Goal: turn T²-Splat's reconstructed triangle mesh into a **relightable PBR asset**
(albedo / roughness / metallic + optional normal map) that (a) relights correctly in
our differentiable renderer and (b) relights identically when imported into Blender
(Cycles / EEVEE) as a standard glTF PBR asset.

Core strategy: **geometry first, then materials under known light, then unknown light.**
Geometry is frozen after T²-Splat; ray tracing is a non-differentiable visibility /
hit query on that frozen mesh.

---

## 0. What the codebase and data give us (findings)

### T²-Splat (this repo)
| Fact | Where | Consequence for relighting |
|---|---|---|
| Final triangles are opaque: opacity floor ramps 0.1 → 0.9999, edge sigma → 1e-4 | `train.py:124-131`, `:550-556`, `:892-900` | Frozen mesh can be rasterized exactly with nvdiffrast (one surface point per pixel). |
| Per-face `R×R` textures in barycentric layout (row = wC → v2, col = wB → v1), out-of-triangle texels mirrored across the hypotenuse, MIP pyramid | `scene/triangle_model.py:127-148, 221-381` | **No xatlas needed.** Each face is its own chart; PBR channels live in the same layout. |
| GLB exporter packs each face's tile into padded atlas cells, splits vertices per face | `utils/glb_export.py:237-320` | Reuse for PBR export; currently writes `KHR_materials_unlit` → must switch to `pbrMetallicRoughness`. |
| CUDA rasterizer returns per-pixel triangle id, normal, depth, alpha — but not barycentrics | `triangle_renderer/__init__.py` | Use nvdiffrast (already a submodule) for the PBR G-buffer; keep the CUDA rasterizer for geometry training. |
| GPU LBVH + Taichi segment any-hit query | `submodules/effrdel/src/intersect.py` | Unused and not runnable as written (kernel decorated `@ti.func`); replaced by `cubvh` (`relight/raytrace.py`). |
| Mesh adjacency utilities | `utils/mesh_topology.py` | Cross-face material smoothness on the surface manifold. |
| Blender/NeRF-synthetic loader expects RGBA `.png`; `Camera` clamps image to [0,1] | `scene/dataset_readers.py:325-366`, `scene/cameras.py:49` | Need a Synthetic4Relight loader and an HDR (unclamped) image path. |

### Synthetic4Relight (`data/Synthetic4Relight/{air_baloons,chair,hotdog,jugs}`)
- `train/{i}_rgb.exr` — **linear HDR** float32 800×800 (+ `{i}_mask.png`). 100 views (chair: 200).
- `test/{i}_rgba.png` (16-bit), `{i}_albedo.png`, `{i}_rough.png` — 200 views with **GT albedo and roughness**.
- `test_rli/envmap6_{i}.png`, `envmap12_{i}.png` — relit GT, 200 views × 2 lights.
- HDR environment maps: not in the dataset zip; downloaded from the InvRender repo
  (`code/envmaps/envmap{3,6,12}.exr`) into `data/Synthetic4Relight/envmaps/`.
  **envmap3 is the training light** (verified, see Phase 0 status).
- hotdog train EXRs have their top 5 rows zeroed (dataset artifact) — ignore them when using background pixels.

### Environment
- Python: `/media/mydrive1/miniconda3/envs/t2splat/bin/python` (`conda activate` is broken on this
  machine; call the env's python directly after `source scripts/env.sh`).
- Installed: torch 2.7.1, nvdiffrast, trimesh, open3d, imageio, **bpy 5.0.1**.
- Installed in Phase 0: `cubvh` 0.1.2 (built from GitHub with CUDA 12.5, `--no-deps`).
- Not installed: xatlas (not needed), mitsuba (not needed), OpenEXR (OpenCV with
  `OPENCV_IO_ENABLE_OPENEXR=1` reads the EXRs fine).
- GPUs: 2× RTX A5000 24 GB.

---

## 1. Why meshes/triangles (vs. Gaussians) — the claims to demonstrate

Each claim maps to a component and an experiment, so the paper story is testable.

| # | Mesh advantage | Gaussian-based relighting (R3DG, GS-IR, GaussianShader, …) | Where it shows up |
|---|---|---|---|
| A1 | **Exact surface & normals**: one opaque hit per pixel, geometric normal from the triangle | Depth/normal are alpha-blended averages; normals need pseudo-normal regularizers | Deferred shading on nvdiffrast G-buffer; roughness/normal accuracy |
| A2 | **Exact, cheap hard visibility**: standard triangle BVH, binary V, precomputable on the static mesh | Visibility needs volumetric ray marching / Gaussian BVH with transmittance, or baked approximations | Phase 2/3 shadows; shadow-baking-into-albedo ablation |
| A3 | **Textured secondary hits → physically meaningful indirect light**: hit (face, bary) → look up the *same* material textures | Indirect light is a separate learned SH/cache per Gaussian | Phase 4 one-bounce indirect |
| A4 | **Material resolution decoupled from geometry**: per-face R×R textures; per-face tangent frame is canonical (edge-aligned), so normal maps are native | One material per primitive → detail needs more primitives | Albedo PSNR vs primitive count |
| A5 | **Surface-manifold regularization**: adjacency gives cross-edge smoothness / Laplacian on materials | KNN graphs in 3D, leaks across thin structures | Material regularizers |
| A6 | **Direct CG-engine deployment**: glTF PBR → Principled BSDF in Cycles/EEVEE/Unity/Unreal; editable, physics/collision-ready | Requires custom splat renderers; cannot be relit by a standard engine | **Engine-relighting PSNR** (Cycles render vs GT `test_rli`) — a metric Gaussian methods cannot even report |
| A7 | **Shading cost scales with pixels, not primitives** (deferred, one shade per pixel) | Shade each Gaussian then blend (mixes normals/materials at edges) | Training/inference speed table |

Headline experiment: *same asset, three renderers* — our PyTorch renderer, Blender Cycles,
Blender EEVEE — under unseen `envmap6`/`envmap12`, all compared against GT.

---

## 2. Architecture

```
Stage G (existing T²-Splat, CUDA rasterizer)        Stage M (new, nvdiffrast, frozen mesh)
 HDR exr ─tonemap─▶ SH + face residual training      V,F frozen ─▶ G-buffer (face_id, bary, x, n_geo)
 + mask/silhouette loss                                     │
        │                                                   ├─▶ PBRFaceTextures[F,R,R,C] sample(bary)
        ▼                                                   │      albedo(3) rough(1) metal(1) [normal(2)]
 mesh (V,F), opaque  ───────────────────────────────▶      ├─▶ Visibility (BVH, precomputed per vertex)
                                                            ├─▶ Light: GT envmap │ learnable log-HDR envmap
                                                            └─▶ Shading: Lambert·V + split-sum GGX·SO (+ indirect)
                                                                     │  loss vs linear HDR GT (tonemapped)
                                                                     ▼
                                              Export: PBR GLB (baseColor sRGB, ORM linear, normal) ─▶ Blender
```

New code lives in `relight/` (keeps upstream T²-Splat files mostly untouched):

| File | Content |
|---|---|
| `relight/data.py` | Synthetic4Relight reader: linear HDR train images + masks; test rgba/albedo/rough; relight splits |
| `relight/gbuffer.py` | nvdiffrast rasterize + `dr.antialias`; outputs face id, barycentrics, world pos, geometric normal (flipped to face the camera) |
| `relight/face_material.py` | `PBRFaceTextures` — `[F,R,R,C]` parameters, bilinear sampling from (face, bary), mirror-texel handling (reuse `build_face_tex_mirror`, `redirect_mirror_gradients`), activations, per-channel resolution (albedo R=8, rough/metal R=2) |
| `relight/light.py` | Equirect HDR load (Blender convention), cubemap conversion, split-sum prefilter (port `nvdiffrec/render/light.py` + renderutils, or a PyTorch prefilter), direction set for diffuse sums, learnable log-space envmap |
| `relight/visibility.py` | BVH build (effrdel LBVH or cubvh), per-vertex K-direction visibility bitmask precompute, specular occlusion, closest-hit query for indirect |
| `relight/shading.py` | Lambert + GGX (Smith G, Schlick F, F0 = lerp(0.04, albedo, metal)) — matched to Blender Principled BSDF |
| `relight/train_pbr.py` | Phase 2/3/4 training loop, losses, regularizers, eval |
| `relight/export_pbr_glb.py` | PBR glTF writer reusing atlas logic from `utils/glb_export.py` |
| `relight/blender/*.py` | bpy scripts: convention check, Cycles/EEVEE relight renders, parity metrics |
| `relight/metrics.py` | NVS PSNR/SSIM/LPIPS, scale-aligned albedo PSNR, roughness MSE, relight PSNR |
| `configs/synthetic4relight.yaml` | Geometry preset for object-centric scenes |

---

## 3. Phases

### Phase 0 — Infrastructure & conventions (do first; cheap, catches silent bugs)
1. **Data loader** (`relight/data.py`, plus a `Synthetic4Relight` callback in
   `scene/dataset_readers.py` / `scene/__init__.py`):
   - Geometry stage: feed `sRGB(clamp(linear))` composited on black with the mask (T²-Splat losses
     expect [0,1]); keep `gt_alpha_mask`.
   - PBR stage: keep unclamped linear HDR tensors (bypass `Camera`'s clamp).
2. **Get the envmaps** (blocker, see §0).
3. **Envmap convention check** (`relight/blender/check_envmap.py`): render a mirror sphere and a
   diffuse sphere in Cycles with `envmap12` (Standard view transform, GGX, no denoiser) and in our
   PyTorch shader. Target ≥ 35 dB. This pins Blender's Z-up equirect orientation once.
4. **Ray query benchmark**: effrdel `query_segments_any_hit` vs `cubvh` on a 1M-face mesh;
   target ≥ 100 M rays/s for any-hit. Install `cubvh` if needed (closest-hit is required in Phase 4).

### Phase 1 — Geometry on Synthetic4Relight (T²-Splat, frozen afterwards)
1. `configs/synthetic4relight.yaml`; add `synthetic4relight` to `--dataset` choices in
   `scripts/run_pipeline.py`. Object-centric settings: random init inside bounds (existing Blender
   path), smaller `max_points` (start 0.5–1 M), no depth/normal priors initially.
2. Add a **silhouette loss** (rendered alpha vs mask) to suppress floaters and background triangles.
3. Post-process: drop faces outside the mask hull / unseen faces (`utils/visibility_pruning.py`),
   make winding consistent, report `tools/mesh_stats.py` (non-manifold edges, components).
4. Checks: test-view mask IoU ≥ 0.98; normal maps visually clean; NVS PSNR of T²-Splat's own
   appearance model on `test/*_rgba.png` as the geometry-quality reference.
5. Known risk: shiny objects (`jugs`) — SH-based appearance can push highlights into geometry
   (dents). Track per-scene; mitigation: stronger normal regularization, later normal-map refinement.

### Phase 2 — Material optimization under **oracle** light (validates the whole shader)
1. **Per-face sampling equivalence unit test** (`tests/test_face_sampling.py`): render the trained
   T²-Splat appearance (Gouraud SH + face texture) via our nvdiffrast path vs the CUDA rasterizer;
   target ≥ 45 dB. nvdiffrast barycentrics (u, v) weight (v0, v1) → `wB = v`, `wC = 1-u-v`; the test
   confirms this mapping.
2. **Visibility precompute**: per vertex, K = 512 Fibonacci directions, offset ray origin
   `x + ε·n`, any-hit → bitmask (64 B/vertex, ~64 MB for 1 M vertices). Interpolate with barycentrics.
3. **Shading**
   - Diffuse: `(1-m)·a/π · Σ_k L(ω_k) V(ω_k) max(n·ω_k,0) Δω` (light pre-integrated per direction bin).
   - Specular: split-sum (prefiltered envmap + BRDF LUT) × specular occlusion
     `SO = Σ_k V(ω_k)·D_GGX(ω_k; r, R(ω_o)) / Σ_k D_GGX(...)` from the same bitmask.
   - Output linear radiance; loss on `sRGB(clamp(L))` vs `sRGB(clamp(GT))` inside the mask
     (also try `log(1+L)`), `+ 0.2·D-SSIM`.
4. **Init**: albedo = 0.5 (or `diffuse_color_linear / irradiance` using the oracle light),
   roughness 0.5, metallic 0 (Synthetic4Relight objects are dielectric — freeze metallic at 0 and
   report it; enable later for general data).
5. **Regularizers**: in-face TV (existing `face_tex_tv_loss`), cross-edge smoothness via mesh
   adjacency, edge-aware weighting `exp(-‖Δa‖/τ)` for roughness.
6. **Eval** (oracle upper bound): NVS PSNR on `test`, **albedo PSNR after per-channel scale
   alignment** vs `test/*_albedo.png`, roughness MSE vs `test/*_rough.png`, relight PSNR on
   `test_rli` envmap6/12. Ablations: no visibility, K = 64/128/512.

### Phase 3 — Joint inverse rendering (unknown light)
1. Light = learnable equirect HDR, log-space `L = exp(θ)`, 32×64 for diffuse bins and
   128×256 (or cubemap 6×64²) for specular prefilter; smoothness prior in log space.
2. Priors: light white-balance (penalize channel variance of mean light), albedo range prior,
   optional light-only warm-up epochs.
3. Report the gap to the Phase 2 oracle on every metric; relight eval swaps in the GT
   envmap6/12 (scale-aligned per protocol).

### Phase 4 — Mesh-native one-bounce indirect illumination (A3)
- Per iteration, sample K_ind = 32–64 directions per pixel (or per vertex, cached), closest-hit via
  BVH → (face, bary) at the hit → evaluate the hit's diffuse radiance with the *current* albedo
  texture and direct light (no recursion; detach the hit's material gradients first, then try
  with gradients).
- Expected gains: concave regions (`jugs`, `hotdog` plate rim) stop brightening the albedo.
- Ablation: direct-only vs +indirect on albedo PSNR / relight PSNR.

### Phase 5 — CG-engine export & Blender parity (A6)
1. **PBR GLB writer** (`relight/export_pbr_glb.py`), reusing the per-face tile atlas:
   - `baseColorTexture`: albedo, sRGB-encoded uint8.
   - `metallicRoughnessTexture`: linear, G = roughness, B = metallic (glTF spec packing).
   - Optional `normalTexture` with per-corner tangents = the per-face edge frame (exact, since vertices
     are already split per face).
   - Remove `KHR_materials_unlit`; `pbrMetallicRoughness`, double-sided off once winding is fixed.
   - Atlas capacity: 4096² with 8+2·2 px cells ≈ 116 k faces per atlas → several materials for
     large meshes. Offer `--face-tex-res 4` and/or decimation for engine-friendly exports; optional
     xatlas re-bake path for a compact single-atlas asset.
2. **Blender scripts** (`relight/blender/render_relight.py`, run with the t2splat python + `bpy`):
   import GLB, set World to the HDR envmap (convention from Phase 0), cameras from
   `transforms_test.json` (native Blender c2w; `camera_angle_x` → sensor fit), Principled BSDF with
   distribution = GGX for parity (Blender 5 defaults to multiscatter GGX), view transform Standard,
   fixed sample count. Render Cycles and EEVEE.
3. **Metrics**: (i) PyTorch vs Cycles parity PSNR (direct-only Cycles via light-path bounces = 0 for
   apples-to-apples, plus full GI); (ii) **engine-relight PSNR** Cycles/EEVEE vs GT `test_rli`.

### Phase 6 (optional, novelty) — Unfreeze geometry
- Joint vertex + material refinement: extend T²-Splat's CUDA face-texture path from C = 3 to
  C = 5–7 channels, or use nvdiffrast + `dr.antialias` silhouette gradients; or limit to
  per-face normal-map refinement. Visibility re-precomputed every N iterations.

---

## 4. Evaluation protocol (Synthetic4Relight)
| Metric | Data | Notes |
|---|---|---|
| NVS PSNR / SSIM / LPIPS | `test/*_rgba.png` | training light |
| Albedo PSNR / SSIM | `test/*_albedo.png` | per-channel scale alignment |
| Roughness MSE | `test/*_rough.png` | |
| Relight PSNR / SSIM / LPIPS | `test_rli/envmap{6,12}_*.png` | our renderer |
| Engine-relight PSNR | `test_rli` | Blender Cycles & EEVEE renders of the exported GLB |
| Parity PSNR | ours vs Cycles | engine compatibility |
| Speed / memory | — | training time, render FPS, asset size |

Baselines: InvRender (dataset authors), TensoIR, NVDiffRec/NVDiffRecMC (mesh + split-sum / MC),
and Gaussian relighting (R3DG, GS-IR) — check which report Synthetic4Relight numbers and rerun
otherwise. Ablations: visibility on/off, indirect on/off, texture resolution, geometry backbone
(T²-Splat vs NVDiffRec mesh).

## 5. Risks
| Risk | Mitigation |
|---|---|
| Envmaps not in the local dataset | Obtain originals before Phase 2; Phase 0/1 can proceed without them |
| Inconsistent winding / non-manifold RDT output | Orientation fix + camera-facing normal flip in shading; mesh stats in CI |
| Too many faces for engine export | Lower face-tex res, decimation, or xatlas re-bake export path |
| Geometry errors baked into materials | Per-face normal map; Phase 6 refinement |
| Split-sum ↔ Cycles mismatch | Report parity separately; GGX (not multiscatter) in Blender |
| nvdiffrec code license (NVIDIA non-commercial) | Fine for research; reimplement prefilter in PyTorch if needed |

## Phase 0 status (done 2026-11-14)
| Item | Result |
|---|---|
| Loader | `relight/data.py`; T²-Splat reader `readSynthetic4RelightInfo` (auto-detected in `scene/__init__.py`); `tests/test_synthetic4relight.py` passes |
| Transfer function | Dataset protocol (InvRender): PNG = gamma 2.2, metrics on `clip(x^(1/2.2))`; `relight/io.py` has gamma and piecewise sRGB |
| Envmap / camera convention | Envmap lookup along camera rays reproduces train-EXR backgrounds at 80–84 dB for envmap3 in all 4 scenes (≤ 13 dB for envmap6/12) → convention, camera and training light confirmed |
| Cycles parity (`scripts/check_envmap_convention.py`) | Mirror sphere 46–70 dB, Lambertian sphere 55–63 dB, background 58–92 dB; linear ratio Cycles/ours within 0.3% |
| Ray queries (`scripts/bench_raytrace.py`) | cubvh 33–53 M rays/s; float64 hit verification removes cubvh's grazing-angle false self-hits (cos-weighted visibility error 0.014% at 655k faces); 512-dir per-vertex visibility for 1M vertices ≈ 15 s |

Notes: cubvh has no any-hit early exit, so per-pixel online tracing (800² × 64 rays ≈ 1 s) is too slow
per iteration → precompute/cache visibility per vertex as planned; indirect light must also be cached.

## Phase 1 status (done 2026-11-15)
Preset `configs/synthetic4relight.yaml` (RGBA supervision, random background, `lambda_mask` 0.1,
full 30k opaque schedule, `lambda_normals` 1e-4). Driver: `scripts/phase1_all.sh`. Outputs in
`output/synthetic4relight/<scene>/` — `relight_mesh.pt` is the Phase 2 input.

| Scene | NVS PSNR / SSIM / LPIPS (35k) | Raw faces | Relight faces | Test mask IoU | Back-facing px | Normal dev. median / p90 |
|---|---|---|---|---|---|---|
| air_baloons | 36.50 / 0.972 / 0.043 | 847k | 480k | 0.9934 | 2.4% | 15.2° / 64.7° |
| chair | 41.74 / 0.988 / 0.011 | 1.18M | 578k | 0.9974 | 1.4% | 14.8° / 62.5° |
| hotdog | 38.62 / 0.986 / 0.028 | 1.01M | 527k | 0.9972 | 2.9% | 14.1° / 69.6° |
| jugs | 42.53 / 0.993 / 0.012 | 529k | 281k | 0.9972 | 1.4% | 11.8° / 66.0° |

Findings
- T²-Splat winding is random (50% back-facing pixels) and ~40% of the surface area is enclosed
  interior faces never seen by any camera. `scripts/prepare_relight_mesh.py` removes unseen +
  fully enclosed faces (unchanged test IoU) and orients faces by training-camera votes
  (back-facing pixels → 1.4–2.9%); flipped faces get their face textures transposed.
- Connectivity shares vertices but mostly not edges (boundary edges > faces, no duplicate
  vertices): the mesh is not a manifold surface, so adjacency-based smoothing/regularizers are weak.
- Ablation (`output/ablations/s4r_lambda_normals_1e-5`): `lambda_normals` 1e-5 → 1e-4 reduces normal
  deviation median by ~40% (jugs 19.5°→11.8°, hotdog 21.7°→14.1°) for −0.12 dB (jugs) / −0.21 dB
  (hotdog) NVS PSNR at 35k (42.65→42.53, 38.83→38.62).
- Remaining issues for Phase 2: faceted normals on flat regions (table, plate rim) → spatially
  smoothed shading normals and/or learned per-face normal offsets; air_baloons strings (1–2 px)
  are not reconstructed.

## 6. Immediate next steps
1. Phase 2: per-face PBR textures + nvdiffrast sampling equivalence test vs the T²-Splat CUDA render.
2. Shading normals: spatially smoothed vertex normals (radius-based, orientation-aware) on `relight_mesh.pt`.
3. Per-vertex visibility precompute (cubvh, 512 dirs) and oracle-light (envmap3) material optimization.
