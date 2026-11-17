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

## Phase 2 status (v1 done 2026-11-15)
Code: `relight/shading.py` (binned env diffuse + split-sum specular + specular occlusion),
`relight/visibility.py` (per-vertex packed visibility), `relight/material.py` (per-vertex PBR),
`scripts/train_pbr.py`, driver `scripts/phase2_all.sh`, table `scripts/summarize_pbr.py`.
Tests: `tests/test_shading.py` (white furnace, prefilter, LUT, irradiance vs Phase 0 reference,
mirror specular, visibility).

Design changes vs the plan above (v1, deliberately simple):
- **Per-vertex materials** instead of per-face R×R textures: faces cover ~2.5 px, ~45% of faces are
  not hit by 10 training views; per-face residual textures exist but are off (`--face-res 0`).
  The per-face sampling equivalence test moves to Phase 5 (exporter compatibility).
- Data-driven albedo init (Σ L_obs / Σ E_vis/π per vertex); constant 0.5 stalls at ~14 dB.
- Loss: L1 on `x^(1/2.2)` of linear HDR over random pixel batches (2^17), no D-SSIM.
- Shadow rays start `10 ×` median edge length off the surface (`--vis-tmin`): the bumpy T²-Splat
  surface self-occludes otherwise (occluded share falls 37%→21% on jugs from 0→10 h); albedo
  +2.9 dB, NVS/relight unchanged (jugs sweep 2/5/10 h).
- Metallic frozen at 0 (Synthetic4Relight is dielectric). 3000 iterations, ~5 min/scene.

Oracle light (envmap3), 50 test views, PSNR inside the GT mask unless noted. These runs predate the
diffuse-layering fix (now `pbr_oracle_nolayer`); current numbers are in the Phase 3 + 5 section.

| Run | NVS | Albedo aligned / raw | Rough MSE | Relight env6 | Relight env12 | NVS full img | Relight env12 full img |
|---|---|---|---|---|---|---|---|
| with visibility (mean) | 26.25 | 22.17 / 21.14 | 0.074 | 25.60 | 25.68 | 32.85 | 32.33 |
| no visibility (mean) | 26.70 | 21.10 / 21.09 | 0.100 | 25.16 | 24.43 | 33.23 | 31.12 |

Per scene (with visibility): air_baloons 24.74 / 18.80 / 0.111 / 23.97 / 24.66, chair 26.94 / 23.51 /
0.027 / 25.81 / 26.60, hotdog 23.98 / 20.79 / 0.113 / 22.71 / 22.13, jugs 29.35 / 25.57 / 0.047 / 29.89 /
29.34 (NVS / albedo aligned / rough MSE / env6 / env12).

Findings
- Visibility trades a little training-light fit (NVS −0.45 dB: without it shadows are baked into
  albedo) for better relighting (env12 +1.25 dB, env6 +0.44 dB), albedo (+1.07 dB) and roughness
  (MSE 0.100→0.074) — the expected signature of separating cast shadows from albedo (claim A2).
- Albedo is too bright in occluded regions (aligned scales 0.47–0.94 < 1; jugs worst): the GT has
  interreflections, our model none → Phase 4 one-bounce indirect.
- Glossy flat surfaces (hotdog plate) are dominated by faceted T²-Splat normals: specular streaks and
  roughness driven up (hotdog / air_baloons rough MSE ≈ 0.11). Fix belongs to the geometry stage.
- Gap to T²-Splat's own appearance (full-image NVS 36.5–42.5 dB) is 5–7 dB: physically based shading
  cannot absorb geometry errors the way SH + face textures can.

## Phase 3 + Phase 5 status (v1 done 2026-11-15)
Single entry point for the v1 pipeline after Phase 1: `bash scripts/relight_all.sh <gpu> <scenes...>`
(oracle light → learned light → GLB export + Blender Cycles for both). Tables:
`python scripts/summarize_pbr.py --runs pbr_oracle pbr_learned --blender`.

**Phase 3 (learned light)** — `train_pbr.py --learn-light`: `LearnableEnvLight` (64×128 log-space HDR,
grey init from mean radiance, TV 1e-3 on log L, white-balance prior available but off), diffuse bins
and GGX pyramid rebuilt each iteration with precomputed linear operators (`tests/test_shading.py`
checks furnace, gradients, and agreement with the fixed `EnvLight`). Albedo/light are defined up to a
per-channel scale, so relighting is also reported after per-view albedo alignment ("albedo-aligned").

**Phase 5 (engine export)** — `relight/export.py` writes glTF PBR: per-vertex values baked into 4×4
per-face tiles (T²-Splat layout), baseColor sRGB + metallicRoughness (G/B), smooth normals, Y-up,
no mipmaps (`tests/test_export.py`: multi-page, axes, exact corner texels). `relight/blender/render_relight.py`
imports into Blender 5.0 (importer wires Base Color ← texture, Roughness ← G, Metallic ← B), forces
Principled distribution GGX, renders Cycles (GI and direct-only) at the test cameras.
`scripts/eval_blender.py` exports with the run's mean albedo alignment and scores 10 views × 2 lights.
`scripts/check_blender_parity.py` isolates the shading model on a clean sphere.

Shading-model change found in Phase 5: Blender's Principled v2 attenuates diffuse by the specular layer's
directional albedo; ReMeS now does the same (`shade(..., layering=True)`, `--no-layering` for old runs).
Clean-sphere parity (Cycles direct vs ReMeS, envmap12): 33.6 / 36.5 / 40.0 dB → **47.0 / 46.2 / 45.4 dB**
(roughness 0.2 / 0.5 / 0.9), brightness ratio 0.94–0.96 → 0.98–0.99. Runs before the change are kept as
`pbr_{oracle,learned}_nolayer`.

ReMeS renderer, 50 test views, PSNR inside GT mask (means over 4 scenes):

| Run | NVS | Albedo aligned / raw | Rough MSE | Relight env6 / env12 | Relight albedo-aligned env6 / env12 | Light PSNR (aligned) |
|---|---|---|---|---|---|---|
| oracle light | 25.85 | 22.16 / 20.70 | 0.118 | 25.41 / 25.57 | 23.96 / 23.86 | — |
| learned light | 27.68 | 20.79 / 15.50 | 0.093 | 21.08 / 19.85 | 23.19 / 22.90 | 13.98 |

Blender Cycles, 10 test views, albedo-aligned asset, means over 4 scenes (env6 / env12):

| Run | Mask IoU | Parity (Cycles direct vs ReMeS) | Cycles GI vs GT | ReMeS vs GT | Cycles GI vs GT (full img) |
|---|---|---|---|---|---|
| oracle light | 1.000 | 23.84 / 23.68 | 21.38 / 21.52 | 23.81 / 23.79 | 28.38 / 28.43 |
| learned light | 1.000 | 23.38 / 23.53 | 20.99 / 20.83 | 23.11 / 22.82 | 27.96 / 27.75 |

Findings
- The whole pipeline runs end to end (images → mesh → materials + light → GLB → Cycles); glTF round trip
  is exact (mask IoU 1.000), material wiring correct.
- Learned light costs ~0.8 dB albedo-aligned relighting and 1.4 dB albedo vs the oracle; it fits the
  training views better (NVS +1.8 dB) by absorbing model/geometry errors into the light. Recovered light
  is low-frequency (13–16 dB vs GT); no sharp sun/horizon.
- Scene parity (~23.5 dB) is far below clean-sphere parity (~46 dB): the gap is geometric — Cycles
  traces exact per-pixel shadows on the bumpy T²-Splat surface (visible pitting), ReMeS uses smoothed
  per-vertex visibility with `vis_tmin`. Exact-start visibility in ReMeS closes only ~1 dB (jugs).
- Cycles GI scores ~2.4 dB below the ReMeS renderer against GT: albedo trained without interreflections
  is too bright, and Cycles adds real interreflections on top → Phase 4 (indirect-aware training).
- With layering, oracle roughness MSE rose (0.074 → 0.118; hotdog 0.24, air_baloons 0.16): roughness is
  poorly constrained on these faceted meshes; revisit with geometry / priors.

## Geometry diagnostic (2026-11-15)
Scripts: `scripts/diagnose_geometry.py` (D1/D2), `scripts/make_geometry_variants.py` + `scripts/geomvar_eval.sh`
(D3), `relight/blender/make_synthetic.py` + `scripts/synthgeo_all.sh` + `scripts/eval_geometry_gt.py` (D4),
filters in `relight/geomfilter.py`. Outputs: `output/synthetic4relight/<scene>/geometry_diagnostic/`,
`output/geomvar/`, `output/synthgeo*/`.

**D1 — error profile (4 real scenes).** "Layering" = share of camera rays that hit a second surface
within 1 median edge length behind the first hit (clean mesh: 0.5%).

| Scene | Bump RMS (edges) | Normal dev. vs smoothed, median / p90 | Layering <1 / <3 edges |
|---|---|---|---|
| air_baloons | 0.41 | 33° / 77° | 86% / 90% |
| chair | 0.45 | 41° / 79° | 94% / 97% |
| hotdog | 0.45 | 34° / 77° | 91% / 95% |
| jugs | 0.41 | 31° / 76° | 85% / 92% |

T²-Splat's opaque triangles form a crumpled multi-sheet shell: small displacements (~0.4 edges) but
steep facets, stacked 2+ deep almost everywhere.

**D2 — attribution (GT roughness / albedo maps).** Normal deviation is higher on glossy pixels where
they are common (jugs 39° vs 21°, hotdog 28° vs 21°, air_baloons 26° vs 23°) and on strongly textured
pixels in 3/4 scenes (+6–8°).

**D4 — controlled scenes with GT geometry** (Suzanne on a slab, Synthetic4Relight cameras + envmap3,
only the object material changes). Distances in % of the object diagonal.

| Material | Mesh | Accuracy mean / median | Completeness mean | Normal error face / smooth (median) | Screen normal MAE | Layering |
|---|---|---|---|---|---|---|
| diffuse | T²-Splat | 0.098 / 0.042 | 0.073 | 5.0° / 5.0° | 9.5° | 87% |
| diffuse | + Poisson | 0.109 / 0.053 | 0.138 | 5.5° / 4.5° | 5.0° | 0% |
| textured | T²-Splat | 0.098 / 0.032 | 0.058 | 7.0° / 6.7° | 8.5° | 80% |
| textured | + Poisson | 0.084 / 0.046 | 0.117 | 5.7° / 4.5° | 4.7° | 0% |
| glossy | T²-Splat | 0.443 / 0.157 | 0.158 | 20.3° / 17.3° | 20.4° | 86% |
| glossy | + Poisson | 0.649 / 0.216 | 0.262 | 17.1° / 15.5° | 16.3° | 0% |

**D3 — downstream sensitivity** (jugs + hotdog, oracle light; Poisson depth 9 single-sheet remesh;
Taubin smoothing rejected: silhouette IoU drops to 0.73–0.90 on these meshes).

| Mesh | Test mask IoU | Albedo aligned | Rough MSE | Relight env6 / env12 (ReMeS) | Parity Cycles-direct vs ReMeS | Cycles GI vs GT env6 / env12 |
|---|---|---|---|---|---|---|
| T²-Splat (orig) | 0.997 | 23.13 | 0.144 | 26.17 / 25.62 | 24.6 / 24.2 | 22.0 / 21.4 |
| Poisson, vis_tmin 10 | 0.96–0.97 | 23.63 | 0.100 | 25.73 / 25.43 | 32.6 / 33.0 | 24.9 / 24.5 |
| Poisson, vis_tmin 1 | 0.96–0.97 | 22.59 | 0.103 | 25.60 / 25.34 | 34.8 / 35.5 | 24.5 / 24.1 |

Conclusions
1. **View-dependent appearance is the root cause of geometric error.** With identical geometry, cameras
   and light, a glossy material makes T²-Splat's surface 4.5× less accurate and its normals 4× worse
   than a diffuse one (SH cannot represent the reflections, geometry compensates). Post-hoc remeshing
   does not fix it (real displacement, not noise). Texture leakage is minor (+2° normal error).
2. **Layering is a separate structural defect** of the triangle representation (80–94% everywhere, even
   for diffuse objects whose surface position is accurate). It dominates engine behaviour: a single-sheet
   mesh raises Cycles/ReMeS parity by +8–11 dB and Cycles-GI-vs-GT relighting by ~+3 dB, and roughness
   error drops ~30%. Poisson costs ~3% silhouette IoU (thin structures, edges), which lowers masked NVS.
3. Hence the geometry work splits into (a) an appearance model inside reconstruction that can explain
   specular reflection (and ties it to the normal), and (b) single-surface output (anti-layering
   regularisation during training and/or silhouette-preserving surface extraction).

## Geometry fix: opaque triangle soup → connected mesh (2026-11-15/16)
Experiments on SynthGeo (GT geometry), glossy Suzanne unless noted; distances in % of the object diagonal.

**(a) Deferred specular inside T²-Splat** (`relight/deferred.py`, `--deferred_spec`): SH capped, reflection-
direction specular from rendered normals. Glossy normal error 20.3° → 12–15° (v1 14.1°, split-sum E1 14.7°,
+normal consistency E2 12.1°, linear + learned light E4 13.8°, **oracle GT light E3 12.2°**), layering unchanged
(82–89%). Even a perfect light cannot fix it: in T²-Splat's soup stage normals are blends of stacked sheets.

**RDT analysis** (`scripts/analyze_rdt.py`, snapshots): layering forms in the soft soup stage (90–94% by 6k, before
RDT); RDT then doubles bad triangles (diffuse 36→71%, normal error 21→44°) — wrong connectivity, repaired only
partly by 30k (diffuse 20% bad, glossy 55%). On glossy, 54% of vertices are already off-surface before RDT.
Compressing T²-Splat's soft→opaque schedule so RDT sees an opaque soup does **not** help (diffuse 5.0→11°,
layering 86–91%), with or without the specular branch.

**Stage A — OTO-style opaque soup** (`relight/oto.py`, `scripts/train_oto.py`; idea from Yan et al., "1000FPS+ NVS
from End-to-End Opaque Triangle Optimization"): disjoint triangles opaque from the first iteration, nvdiffrast
z-buffer forward, gradients via interpolation, face normals and `dr.antialias` (in place of OTO's 2D-SDF /
two-layer approximations); appearance = per-corner diffuse + reflection-direction split-sum specular with a
learnable HDR env (view dependence only through the triangle normal — OTO's per-triangle SH lets triangles fake
reflections, its stated limitation); mask loss, normal–depth consistency, coverage-based prune (never wins the
z-test → hidden sheets removed) / split; visual-hull init. 15k iterations, ~25 min.

**Stage B — connectivity + refinement**: screened Poisson (depth 9) on the outward-oriented soup, quadric
decimation to 200k faces, attributes from the nearest soup point, connected refinement (image, mask,
normal–depth, Laplacian, adjacent-normal consistency), per-component orientation (`relight/orient.py:
orient_components`). Alternative `--remesh rdt` (weld + T²-Splat manifold RDT) leaves ~26% boundary edges.

| Glossy Suzanne | Normal err (face, med) | Screen normal MAE | Acc. median | Layering | Boundary edges |
|---|---|---|---|---|---|
| T²-Splat | 20.3° | 20.4° | 0.157 | 86% | 66% |
| Stage A soup | 5.3° | 5.7° | 0.048 | 14% | — |
| Stage B, RDT (manifold) | 5.7° | 7.8° | 0.058 | 3% | 26% |
| Stage B, Poisson | 5.3° | 5.3° | 0.073 | 0.3% | 0.9% |

All materials (Stage B RDT, `output/oto/v2`): diffuse 3.5° / textured 3.6° / glossy 5.7° face-normal error vs
T²-Splat 5.0° / 7.0° / 20.3°; non-manifold edges 0, winding-inconsistent 0 (T²-Splat 5–6% / 33–38%).

Open issues: (1) textureless-plane ambiguity — the uniform slab's top bulges ~0.08 above GT (T²-Splat ~0.044),
inherited from the visual-hull roof; free per-corner colour can bake shading/shadows, so nothing constrains it
(shadow-aware shading or depth priors would) — this dominates mean accuracy (0.43–0.59); (2) concavities the hull
cannot carve need insertion / longer optimisation.

### Real scenes with the new geometry (2026-11-16)
Pipeline: `bash scripts/oto_real.sh <gpu> <tag> <scenes...>` (Stage A soup 15k + Stage B Poisson 5k with silhouette
carving before refinement, then Phase 2 oracle light and Phase 5 Blender). Outputs `output/oto_s4r/v4/<scene>/`.
Silhouette carving (`relight/oto.py: outside_masks`) removes Poisson faces whose centre projects outside a
2-px-dilated mask in any view (visual-hull property); carving again after refinement (v3) left unrefined gaps.

Means over the 4 Synthetic4Relight scenes, oracle light (T²-Splat pipeline → new geometry):

| Metric | T²-Splat | New geometry |
|---|---|---|
| Normal deviation vs smoothed (median) / layering | 31–41° / 85–94% | 1–7° / 0–5% |
| Test mask IoU | 0.993–0.997 | 0.987–0.994 |
| NVS masked / full image | 25.85 / 32.55 | 26.33 / 32.44 |
| Albedo aligned / raw | 22.16 / 20.70 | 23.19 / 21.89 |
| Roughness MSE | 0.118 | 0.063 |
| Relight env6 / env12 (masked) | 25.41 / 25.57 | 27.27 / 26.94 |
| Relight albedo-aligned env6 / env12 | 23.96 / 23.86 | 26.03 / 25.27 |
| Blender parity (Cycles direct vs ReMeS) env6 / env12 | 23.8 / 23.7 | 31.0 / 32.2 |
| **Cycles GI vs GT** masked env6 / env12 | 21.4 / 21.5 | **26.2 / 26.0** |
| Cycles GI vs GT full image env6 / env12 | 28.4 / 28.4 | 32.5 / 32.5 |

Per scene the only regressions are chair roughness MSE (0.025 → 0.049) and full-image NVS on jugs/chair (silhouette
IoU 0.987–0.988 vs 0.997). Remaining geometry issues: visual-hull remnants at contact regions (object bases) and
the textureless-plane bulge.

### Shadow-aware geometry (2026-11-16)
`relight/shadowmap.py`: batched orthographic shadow maps in nvdiffrast with antialiased height maps → soft visibility
differentiable w.r.t. receiver and occluder, including shadow boundaries (`tests/test_shadowmap.py`). Geometry stage
(`train_oto.py --pbr-from 4000`, default): diffuse becomes albedo/π · shadowed irradiance · (1 − F0·A − B) over all
32 Fibonacci light bins every iteration (importance-sampled 32 of 512 was too noisy: 27–31 dB vs 33.8 deterministic),
edge-aware albedo smoothness; Stage B keeps it. ~110 ms/iteration, peak 15–17 GB.

SynthGeo glossy (learned light ≈ GT light): face normal error 5.3° → **2.8°**, median accuracy 0.073 → **0.034**,
object mean |depth error| 1.09% → **0.27%** (T²-Splat 0.44%) — neck/chin hull remnants removed. Slab bulge unchanged
(−0.076): uniform plane + soft sky give no shadow signal. Missing GT surface is mostly down-facing (never observed).

Real scenes (v5, means over 4): materials/relighting ≈ v4 (albedo 23.21 vs 23.19, rough MSE 0.066 vs 0.063, relight
27.22/26.99 vs 27.27/26.94), Blender parity 31.8/33.3 vs 31.0/32.2, Cycles GI vs GT 26.31/26.20 vs 26.15/26.03.
Normal deviation 1.0–4.8°, layering 0–3%. Contact-region remnants under undercuts (jug bellies) persist: never
directly observed, mask-uncarvable, and both remnant and true surface are in shadow.

### Follow-ups: soft edges, remeshing, per-face textures (2026-11-16) — all off by default
- `--soft-edge-tau` (OTO two-layer + 2D-SDF edge gradient via `dr.DepthPeeler`, forward unchanged): glossy Suzanne
  face-normal error 2.8° → 3.9° (τ=1) / 5.8° (τ=2), median accuracy 0.034 → 0.081 / 0.165 — worse; antialias-only kept.
- `--remesh-every 1000` (pymeshlab isotropic remeshing in Stage B, GPL-3 dependency): neutral (2.7° vs 2.8°,
  accuracy 0.033 vs 0.034); does not fix undercuts.
- `train_pbr.py --face-res 4` + exact per-face tile export (`PBRMaterial.face_texel_values`): jugs −0.2–0.4 dB vs
  per-vertex (faces cover ~2.5 px, so per-vertex already resolves the image).

## 6. Immediate next steps
1. Undercut / contact-region geometry: better initialisation than the visual hull (e.g. depth-fused surface from
   the opaque soup itself, or inward pressure on regions only observed at grazing angles).
2. Phase 4 indirect light (also a compensation source at contacts) and Phase 3 learned light on the new geometry;
   make `oto_real.sh` the default geometry stage of `relight_all.sh`.
3. Textureless-plane depth (slab bulge): depth/planarity prior.

### Better starting shape than the visual hull (v6, commit c24314a + this)

`train_oto.py --init mesh:<dir>` seeds the soup on the Poisson surface of a given mesh (carved by the
silhouettes) instead of the visual hull. Sources tried: T²-Splat's relight mesh (option 1) and a COLMAP
PatchMatch (known poses) + TSDF surface (`scripts/mvs_init.py`, option 2). Monocular priors (option 3) not needed.

SynthGeo GT geometry after full training (acc/compl mean %, smooth normal err, slab offset, object depth err %):

| Scene | Init | Acc | Compl | Normal | Slab | Obj depth |
|---|---|---|---|---|---|---|
| textured | hull | 0.338 | 0.190 | 2.1° | −0.074 | 0.163 |
| textured | T²-Splat | **0.027** | **0.086** | **1.8°** | −0.045 | **0.031** |
| textured | MVS | 0.028 | 0.091 | 1.9° | −0.043 | 0.035 |
| glossy | hull | 0.350 | 0.273 | 2.6° | −0.076 | 0.269 |
| glossy | T²-Splat | **0.075** | **0.142** | **2.3°** | −0.047 | **0.128** |
| glossy | MVS | 0.250 | 0.201 | 3.6° | −0.043 | 0.354 |

MVS is as good as T²-Splat on texture but fails on glossy surfaces (raw MVS normal err 41°). The raw MVS slab
is at +0.007 yet ends at −0.043 after training: the remaining slab bulge is created by the optimisation
(textureless plane unconstrained), not by the init → next: planarity / anchoring term.

Four real scenes, `OTO_ARGS="--init mesh:output/synthetic4relight/@SCENE@" scripts/oto_real.sh` (v6) vs hull (v5), means:

| | Albedo | Rough MSE | Relight env6/12 | Parity env6/12 | Cycles GI vs GT env6/12 |
|---|---|---|---|---|---|
| v5 hull | 23.21 | 0.0658 | 27.22 / 26.99 | 31.81 / 33.29 | 26.31 / 26.20 |
| v6 T²-Splat init | **24.44** | **0.0609** | **27.95 / 28.21** | **32.19 / 33.54** | **26.98 / 27.19** |

Largest gain on hotdog (Cycles GI +2.1 / +3.1 dB, albedo 20.7 → 24.5); jugs +0.4 dB; chair, air_baloons ≈ equal.
T²-Splat init is now the recommended geometry stage.

### Slab bulge diagnosis (SynthGeo, Stage A soup only, MVS init on textured unless noted)

The remaining slab offset (≈ −0.045 after full training, up to ~10 cm in the middle of the slab top, edges
correct) is the dominant remaining geometry error for relighting: with GT geometry
(`scripts/gt_relight_mesh.py`, `oracle_geo_study.sh`) relighting is ≈ 2.6 dB better than v6, and
`scripts/attribute_error.py` puts almost all of the gap on the slab (object within ≈ 0.7 dB of the GT-geometry fit;
open slab and contact region 4–8 dB worse).

| Stage A variant | Acc | Normal | Slab | Obj depth |
|---|---|---|---|---|
| default | 0.042 | 3.1° | −0.044 | 0.093 |
| no physical shading / shadows (`--pbr-from -1`) | 0.053 | 3.1° | −0.046 | 0.104 |
| no normal–depth term (`--lambda-dn 0`) | 0.092 | 16.8° | −0.034 | 0.079 |
| split only, no prune (`--cov-prune 0`) | 0.048 | 3.3° | −0.037 | 0.073 |
| no split / prune (`--densify-from 100000`) | 0.058 | 3.6° | **−0.002** | 0.108 |
| no split / prune, 150k init triangles | 0.046 | 3.3° | **−0.005** | 0.089 |

* Coverage-driven **splitting** creates the dome: front-most (largest-coverage) triangles are refined, and on a
  textureless plane nothing pushes back, so the front layer drifts outward. Shadows / the shading model are not
  the cause.
* Without splitting the soup keeps its init: fixes the slab only if the init is right there (MVS on textured).
  From the T²-Splat init the slab stays at ≈ −0.03 (T²-Splat itself is domed there; glossy too).
* MVS has no depth on the open (textureless) slab — only near its edges / contact shadows (≈ 16 % of slab
  pixels) — so neither a merged MVS + T²-Splat TSDF init (slab −0.03; glossy object corrupted by wrong
  specular matches) nor the truncated MVS depth prior (`--lambda-mvs`, parallel session: slab −0.03..−0.04,
  glossy normals 7–16°) recovers it.
* ⇒ the open slab has no photometric or stereo depth evidence; what constrains it is the silhouette boundary
  plus a prior. Candidates: minimal-surface / planarity prior in low-texture regions (a dome has more area than
  the plane spanning the same boundary), or depth-neutral densification (split by image error, not coverage).

#### Fix attempts (negative so far): error-driven splitting, planarity prior, MVS anchor

New flags (off by default): `--split-by error [--split-err]` (split only large triangles whose mean image
error is above average) and `--lambda-flat [--flat-sigma]` (texture-weighted multi-stride squared second
differences of screen-space inverse depth, Stage A and B; `tests/test_flatness.py`: flattens a 15 cm dome on a
fixed apron to < 1 mm). SynthGeo textured, T²-Splat init:

| Variant | Acc | Normal | Slab | Obj depth |
|---|---|---|---|---|
| Stage A default | ≈0.04 | ≈3.1° | ≈−0.045 | ≈0.09 |
| A: split by error | 0.067 | 3.9° | −0.029 | 0.095 |
| A: flat 1 / 10 / 300 / 3000 | 0.041 / 0.046 / 0.118 / 1.313 | 3.0 / 3.0 / 3.6 / 12.3° | −0.047 / −0.044 / −0.033 / −0.232 | 0.088 / 0.086 / 0.122 / 0.90 |
| A: split by error + flat 10 | 0.088 | 3.9° | −0.032 | 0.096 |
| Stage B only (resume), no prior | 0.027 | 1.8° | −0.045 | 0.031 |
| B: flat 300 / 3000 / 30000 | 0.046 / 0.223 / 0.395 | 2.1 / 5.5 / 8.8° | −0.051 / −0.054 / −0.056 | 0.069 / 0.52 / 0.90 |
| B: MVS depth 0.5 | 0.032 | 1.8° | −0.043 | 0.058 |
| B: MVS depth 0.5 + flat 300 | 0.039 | 1.9° | −0.050 | 0.069 |

* The slab is **not a dome but a flat plateau** ≈ 3.7 cm above the GT top (interior z ≈ −0.643 vs −0.680) with a
  rounded rim (≈ −0.67 at the edges) — a planarity prior has nothing to correct; in the soup (no connectivity)
  a strong prior is satisfied by floating patches and lifts the slab further.
* Neither the MVS depth term (rim / contact anchors) nor anchor + planarity moves the plateau in Stage B, so it
  is actively held (not merely unconstrained). Next: per-loss-term gradient attribution on slab vertices.

#### Gradient diagnostic: the slab plateau height is unobservable

`train_oto.py --grad-diag <prefix>` saves per-loss-term vertex gradients summed over all training views at the
start / end of Stage B; `scripts/grad_diag_report.py` splits them by region (open slab, contact, rim, object)
and by path (shadow occluders / receivers, shading normals, raster) — SynthGeo, T²-Splat init:

* Open slab (offset +0.047): the summed gradient points down but is incoherent — mean / rms of −g_z is
  −0.03 (textured) and −0.001 (glossy); ≈ 50 % of vertices are pushed each way. Shading normals carry
  55–76 % of the magnitude with near-random sign; on glossy, shadow receivers (+0.34) and occluders (−0.40)
  cancel. After 5000 Stage-B iterations the offset moved from +0.0476 to +0.0471.
* ⇒ the height of a textureless plane is (nearly) a gauge freedom of the photometric loss: whatever Stage A
  leaves there stays. Smoothness-preconditioned "large steps" (Nicolet et al. 2021, `--large-steps λ`,
  `relight/largesteps.py`, CG solves, uniform Adam) cannot amplify a force that is absent:

| Stage B only | Slab | Acc | Normal | Obj depth |
|---|---|---|---|---|
| Adam (textured / glossy) | −0.045 / −0.047 | 0.027 / 0.075 | 1.8 / 2.3° | 0.031 / 0.128 |
| large steps λ=19 | −0.045 / −0.047 | 0.036 / 0.079 | 2.1 / 2.4° | 0.041 / 0.174 |
| large steps λ=100 | −0.047 / −0.048 | 0.037 / 0.079 | 2.1 / 2.5° | 0.042 / 0.176 |

* What does fix it: an init that is right on the plane plus no coverage-driven splitting (MVS init, no split:
  −0.005). A general fix needs an explicit height cue (slab-rim crease / side-face silhouettes) or a learned
  depth prior (option 3).

### Phase 4 — one-bounce indirect diffuse (done, `train_pbr.py --indirect`)

`relight/indirect.py`: for every vertex and every *blocked* visibility direction above its tangent plane, one
closest-hit ray (same offsets as the visibility rays) → hit triangle + barycentrics; a sparse transport matrix
M[v, j] = Σ cos·ΔΩ·bary_j (built in < 2 s, ≈ 15 M non-zeros at 100 k vertices, cached next to the mesh).
Per iteration E_ind = M · B with radiosity B = (1−metal)·albedo/π·E_dir (one sparse mat-mul; differentiable
w.r.t. the light, and w.r.t. the hit albedo with `--indirect-grad`; default detached). `shade(..., irr_extra)`
adds it to the diffuse irradiance in training, evaluation and relighting; the data-driven albedo init does one
fixed-point step with it. `tests/test_indirect.py`: wall-foot analytic case (E_ind/E_dir = a/2) within 5 %.
Blender parity stays direct-vs-direct; "ReMeS vs GT" uses the bounce term.

Four scenes, v6 geometry, oracle light (means):

| | Albedo raw / aligned | Rough MSE | Relight env6 / env12 full | ReMeS vs GT env6 / env12 | Cycles GI vs GT env6 / env12 |
|---|---|---|---|---|---|
| direct only | 22.85 / 24.44 | 0.0609 | 27.95 / 34.38 | 26.37 / 26.26 | 26.98 / 27.19 |
| **+ indirect** | **23.65 / 24.80** | **0.0582** | **28.36 / 34.95** | **27.20 / 27.32** | **27.29 / 27.57** |

Largest gains on hotdog / air_baloons (raw albedo ≈ +1 dB). The ReMeS renderer is now within 0.1–0.25 dB of
Cycles with full GI. Recommended: `--indirect` for all material fits.

`--indirect` is now the `train_pbr.py` default (`--no-indirect` for direct only); `oto_real.sh` runs it, and
`LEARN_LIGHT=1` adds Phase 3 + Blender. Phase 3 (learned light) on v6 geometry, four-scene means
(raw albedo / relighting are scale-ambiguous with a learned light; aligned columns and Blender use the
albedo alignment):

| | Albedo aligned / raw | Rough MSE | Relight env6 / env12 (aligned) | ReMeS vs GT env6 / env12 | Cycles GI vs GT env6 / env12 | Light PSNR |
|---|---|---|---|---|---|---|
| oracle light, direct | 24.44 / 22.85 | 0.0609 | 26.74 / 26.44 | 26.37 / 26.26 | 26.98 / 27.19 | — |
| oracle light, + indirect | **24.80 / 23.65** | **0.0582** | **27.56 / 27.52** | **27.20 / 27.32** | **27.29 / 27.57** | — |
| learned light, direct | 22.93 / 17.46 | 0.0649 | 25.94 / 25.39 | 25.79 / 25.29 | 26.03 / 25.89 | 13.94 |
| learned light, + indirect | **23.32 / 18.02** | **0.0640** | **26.76 / 26.54** | **26.53 / 26.36** | **26.47 / 26.46** | 13.90 |

The bounce term gains about as much with a learned light (+0.8 / +1.15 dB aligned relighting, +0.4 / +0.6 dB
Cycles) as with the oracle; the learned envmap itself stays at ≈ 13.9 dB (aligned) — the remaining gap to the
oracle (≈ 0.8–1.1 dB in Cycles) is light estimation.

### Light estimation, Stage-B indirect, full-image metrics (option 2 / 3 of the post-Phase-4 plan)

* **Light init from the geometry stage** (`train_pbr.py --light-init auto|<path>`, now the default with
  `--learn-light`; falls back to the grey init when `<model>/env.pt` is absent; `--freeze-light`): the
  geometry stage's env (fitted with shadow maps) is far closer to the GT light than Phase 3's grey-init
  estimate (aligned light PSNR 17.95 vs 13.90). Four-scene means, learned light + indirect:

| | Light PSNR | Albedo aligned | Rough MSE | Relight env6/12 aligned | Cycles GI vs GT env6/12 |
|---|---|---|---|---|---|
| grey init | 13.90 | 23.32 | 0.0640 | 26.76 / 26.54 | 26.47 / 26.46 |
| env.pt init, lr 2e-2 | 16.32 | 23.55 | 0.0570 | 27.10 / 26.81 | 26.70 / 26.71 |
| env.pt init, lr 5e-3 | 17.00 | 23.62 | 0.0570 | 27.17 / 26.78 | 26.85 / 26.76 |
| env.pt frozen | 17.95 | 22.87 | 0.1128 | 25.74 / 25.41 | 25.64 / 25.70 |

  (frozen is worse: that env was fitted under the geometry stage's coarse 32-direction shading.)
* **Indirect light in Stage B** (`train_oto.py --indirect-b`, transport rebuilt every 500 its in ≈ 1–2 s;
  v7 = Stage B re-run from the v6 soups): oracle-light means change by +0.04 dB (Cycles 27.34 / 27.61 vs
  27.29 / 27.57) — no measurable benefit; off by default.
* **Full-image metrics** (black background, per-channel albedo alignment; the InvRender / TensoIR / R3DG
  protocol as far as the papers specify) are now also reported by `train_pbr.py` (`*_full`, `*_full_aligned`,
  `rough_mse_full`). Learned light (env.pt init), v6: NVS 33.70, relighting 33.63 / 33.41 (aligned),
  albedo 28.98, roughness MSE 0.0131 — vs R3DG (Tab. 2, Synthetic4Relight) 36.80 / 31.00 / 28.31 / 0.013.
  Relighting is ahead, albedo / roughness on par; the remaining gap is NVS (−3.1 dB), consistent with R3DG's
  free per-Gaussian indirect SH and geometry-decoupled normals (smooth shading normals). Next: a learnable
  (exportable) normal map supervised by a monocular normal prior.

### Gap to R3DG without priors (R3DG-inspired, mesh-native changes)

`scripts/compare_s4r.py` prints R3DG Tab. 2 format (full-image PSNR / SSIM / LPIPS-VGG; relighting = mean of
envmap6/12 with aligned albedo; albedo per-channel aligned). Full-image metrics now render with ss×ss
supersampling (`--eval-ss 2`, premultiplied, as the antialiased GT): our evaluation renders had aliased
silhouettes (jugs: +0.7 dB NVS, LPIPS −12 % from antialiasing alone). New `train_pbr.py` options:
`--learn-normals` (per-vertex shading-normal offset, exported as glTF NORMAL; `--smooth-normal`,
`--normal-anchor`, `--lr-normal`), `--edge-aware σ` (image-edge-aware mesh smoothness weights),
`--ssim λ` (patch batches `--patches × --patch-size²`, D-SSIM in gamma space). Fixes: specular-occlusion
sqrt had an infinite gradient at grazing angles (NaN once normals are learnable).

Four scenes, learned light (env.pt init, lr 5e-3), v6 geometry:

| | NVS PSNR / SSIM / LPIPS | Relight PSNR / SSIM / LPIPS | Albedo PSNR / SSIM / LPIPS | Rough MSE |
|---|---|---|---|---|
| R3DG | 36.80 / 0.982 / 0.028 | 31.00 / 0.964 / 0.050 | 28.31 / 0.951 / 0.058 | 0.0130 |
| baseline | 34.16 / 0.970 / 0.063 | 33.83 / 0.966 / 0.074 | 29.34 / 0.946 / 0.076 | 0.0132 |
| + learned normals | 34.43 / 0.971 / 0.062 | 33.80 / 0.966 / 0.072 | 29.41 / 0.947 / 0.075 | 0.0119 |
| + edge-aware (σ 0.05) | 33.85 / 0.969 / 0.063 | 33.71 / 0.966 / 0.074 | 29.13 / 0.946 / 0.076 | 0.0140 |
| + normals + edge-aware | 34.10 / 0.971 / 0.062 | 33.66 / 0.966 / 0.072 | 29.17 / 0.946 / 0.075 | 0.0129 |
| + face textures 4×4 | 34.18 / 0.972 / 0.052 | 33.89 / 0.967 / 0.064 | 29.29 / 0.947 / 0.075 | 0.0120 |
| + SSIM 0.2 | 34.10 / 0.971 / 0.062 | 33.87 / 0.967 / 0.073 | 29.44 / 0.946 / 0.076 | 0.0121 |

LPIPS is driven by appearance resolution (per-face textures), NVS PSNR / roughness by shading normals;
edge-aware smoothness does not help.

Combinations (same setting):

| | NVS PSNR / SSIM / LPIPS | Relight PSNR / SSIM / LPIPS | Albedo PSNR / SSIM / LPIPS | Rough MSE |
|---|---|---|---|---|
| face 4 + normals | 34.45 / 0.973 / 0.051 | 33.85 / 0.968 / 0.063 | 29.36 / 0.948 / 0.073 | 0.0108 |
| **face 4 + normals + SSIM 0.2** | 34.37 / 0.975 / 0.049 | 33.91 / 0.969 / 0.063 | 29.44 / 0.949 / 0.071 | 0.0105 |
| face 8 + normals + SSIM 0.2 | 34.30 / 0.974 / 0.050 | 33.83 / 0.968 / 0.062 | 29.36 / 0.948 / 0.076 | 0.0091 |

Recommended Phase 3 setting (now in `oto_real.sh` with `LEARN_LIGHT=1`): `--learn-light --lr-light 5e-3
--face-res 4 --learn-normals --ssim 0.2` (indirect and env.pt light init are defaults). Versus R3DG:
relighting +2.9 dB PSNR (SSIM higher), albedo +1.1 dB, roughness MSE 0.0105 vs 0.013; remaining gaps are
NVS (−2.4 dB PSNR, LPIPS 0.049 vs 0.028) and LPIPS in general (relight 0.063 vs 0.050, albedo 0.071 vs 0.058).

### NVS error attribution → silhouette refinement (v8)

`scripts/nvs_error_attribution.py` splits the full-image (antialiased) NVS error by region. On the best v6 run,
a 2-px band around the GT silhouette (2–3 % of the pixels) holds **40–63 %** of the error, and 53–85 % of that
is coverage: with the GT coverage the NVS PSNR would be 1.5–2.3 dB higher. Specular + concave interiors
(the physics we do not model, e.g. specular interreflection) hold only 9–28 %. A Blackman–Harris 1.5 px
reconstruction filter instead of box supersampling changes NVS by only +0.06 / +0.18 dB, so the error is
geometric. (Chair: bright blotches on the thin backrest are not a visibility-offset issue; `--vis-tmin 2`
changes nothing.)

`scripts/refine_silhouette.py`: vertex displacement of the final mesh against the soft GT masks of all
training views (nvdiffrast antialiased coverage, L2) with a Laplacian on the displacement and a small anchor;
≈ 20 s per scene; test-view mask IoU 0.987–0.991 → 0.995–0.999. Now part of `oto_real.sh`
(`NO_SILHOUETTE=1` to skip). Best Phase 3 setting on the refined meshes (4 scenes, learned light):

| | NVS PSNR / SSIM / LPIPS | Relight PSNR / SSIM / LPIPS | Albedo PSNR / SSIM / LPIPS | Rough MSE |
|---|---|---|---|---|
| R3DG | 36.80 / 0.982 / 0.028 | 31.00 / 0.964 / 0.050 | 28.31 / 0.951 / 0.058 | 0.0130 |
| v6 | 34.37 / 0.975 / 0.049 | 33.91 / 0.969 / 0.063 | 29.44 / 0.949 / 0.071 | 0.0105 |
| **v8 (silhouette-refined)** | **35.94 / 0.979 / 0.045** | **34.47 / 0.972 / 0.059** | **29.98 / 0.952 / 0.068** | **0.0104** |
| v8, 2× iterations / lr | 35.97 / 0.979 / 0.045 | 34.43 / 0.972 / 0.059 | 29.95 / 0.952 / 0.068 | 0.0104 |
