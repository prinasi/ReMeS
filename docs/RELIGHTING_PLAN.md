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

## 6. Immediate next steps
1. Real scenes with the new geometry (Stage A+B) → Phase 2 / 5 metrics vs the T²-Splat-based pipeline.
2. Textureless ambiguity: shadow/visibility-aware appearance in the geometry stage, or depth priors.
3. Then Phase 4 (indirect light) on the improved geometry.
