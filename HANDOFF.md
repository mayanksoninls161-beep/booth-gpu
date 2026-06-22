# booth-gpu — Project Handover

> GPU reimplementation of the production exhibition-floor-plan **booth-detection pipeline**.
> Goal: produce the **same hall + booth output as production**, verified by an apples-to-apples
> comparison, then scale it on GPU hardware.

---

## 1. Mission

Input: an exhibition floor-plan (PDF or image).
Output: structured booth boxes — one per exhibitor booth — `{bbox, label, source, text_status, score}`, plus halls/stages.

We are mirroring a working **production** pipeline so we can:
1. Confirm the GPU port detects the **same booths in the same places** as prod.
2. Run the whole pipeline faster on GPU/multi-core hardware.

| | Location | Notes |
|---|---|---|
| **PROD (canonical reference)** | `/home/nls161-pc/Documents/plotyourmap (Copy)/` | Read-only. Source of truth for every parameter. (The original `plotyourmap/` was reverted to a clean checkout; use the `(Copy)` for reference.) |
| **GPU repo (this repo)** | `/home/nls161-pc/booth-gpu/` | Public GitHub repo, pulled into Google Colab (free T4) for runs. |
| **Runner** | `src/run_real.py` | Full pipeline: render → tile → 3-pass ensemble → NMS → big-region → text → FP policy. |
| **Prod batch output (compare against)** | `/home/nls161-pc/batch_results/AMFloorplan-0504/AMFloorplan-0504_booths.json` | bbox is **xywh**. GPU bbox is **xyxy**. |

---

## 2. How the pipeline works

Prod's `EnsembleDetector` pools **three passes** over the rendered plan, then de-duplicates:

| Pass | Source tag | NMS score | Role |
|---|---|---|---|
| Geometric | `opencv_strict` | **1.0** | Line/contour cell detection. Highest priority — wins overlaps. |
| Color | `color` | ~0.75 (`0.5 + 0.5*fill`) | Fills colored booth regions. |
| Bordered | `bordered` | **0.45** | Thin-bordered micro-cells. Intentionally loses ties. |
| Big-region | `bigregion` | 1.0 | Recovers large halls/stages that tiling fragments. |

- **Dense plans tile** the image (`detect_tiled`), run all passes per tile (tile-relative area fractions), stitch, then **NMS** (`non_max_suppression`: sort by score desc, then area desc → geometric wins).
- **Labels** come from the PDF text layer (no OCR for vector PDFs). Each box gets `text_status`: `empty | text | boothlike | facility`.
- **FP policy** decides what to keep (see §4).

### Dense preset (confirmed from prod output)
`dpi=250, max_edge=0 (no cap), tile=1800, overlap=400, close_ksize=3, neutral_gray=null,`
`max_box_area_frac=0.06, bordered_min_area_frac=0.0005, iou_threshold=0.4, big_pass=on`.
Render = **9528×7987 px** — identical in prod and GPU (dpi was never a divergence).

---

## 3. Changes made in this repo (all on `main`)

| Commit | Title | What it does |
|---|---|---|
| `1b53d23` | adaptive FP policy | Adds `resolve_adaptive()` + `apply_fp_policy()`. Fixes an 87-booth under-recall caused by a hard-coded strict filter on a plan with pure-numeric booth IDs. |
| `63fb8dc` | geometric fidelity + NMS source scores | `geometric.py`: corrects 3 drifted params to match prod. `run_real.py`: restores per-source NMS scores (had been flattened to 1.0). |
| `c9b40e8` | bordered recovery + tile concurrency | Adds `recover_uncovered_bordered()` and `ThreadPoolExecutor` tile concurrency. |
| (latest) | GPU geometric backend | New `src/geometric_gpu.py` — CUDA reimplementation of the `opencv_strict` pass (the dominant CPU cost). Opt-in via `--geo-backend {cpu,gpu,auto}` (default `cpu`). |

### GPU geometric detector (`src/geometric_gpu.py`)
Moves the geometric pass's CPU-bound CV ops onto the GPU as pure torch tensor ops:

| CPU op (slow) | GPU replacement |
|---|---|
| `connectedComponentsWithStats` (CPU-only, serial) | label-propagation CCL → `gpu_components.components_stats_gpu` (boxes + areas + centroids, vectorised) |
| `floodFill` (border flood, serial) | "CCL components touching the image border" + `isin` — parallel, equivalent |
| `fill_holes` | CCL on inverted mask + area cap |
| `medianBlur(31)` | grayscale morphological **close** (dark-line floor) / **open** (bright-cell floor) — **the one approximation, not bit-exact** |
| `Sobel` / morphology OPEN/CLOSE | `conv2d` / rectangular max/avg-pool (`(1,L)`/`(L,1)` line elements) |
| `cvtColor BGR→HSV/GRAY` | exact — pass uses only V(=max) and S, never H |

- `_subdivide`, tilt (`HoughLinesP`), `_dedupe_prefer_fine` stay on CPU (cheap / rare / many tiny ops).
- **Fidelity gate:** `geometric_gpu.verify_gpu_vs_cpu(tile_bgr)` returns `{cpu_boxes, gpu_boxes, gpu_matched_to_cpu, recall_vs_cpu, cpu_ms, gpu_ms, speedup}`. Run it in Colab on a real tile and check `recall_vs_cpu` before switching the default to gpu.
- **Not yet numerically validated** (built without a local CUDA env). Needs a Colab pass.

### `src/geometric.py` — params corrected to match prod's `Params`
- `line_len_frac: 0.03 → 0.020`
- `min_area_frac: 1e-4 → 8e-5`
- `min_side_px: 20 → 12`

Effect: `opencv_strict` 226 → 957. Rest of `detect_array` / `_extract_booths` is verbatim-identical to prod.

### `src/run_real.py`
- **Per-source NMS scores:** `_SOURCE_SCORE = {opencv_strict:1.0, color:0.75, bordered:0.45, bigregion:1.0}`, applied at both pool-build sites (was hard-coded 1.0, which destroyed prod's geometric-wins priority). Effect: color 187 → 95, bordered 684 → 14.
- **FP policy:** `resolve_adaptive()` + `apply_fp_policy()`.
  - `adaptive` → `strict` if `n_boothlike >= max(10, 0.15*n)` else `shape`.
  - On pure-numeric-ID plans the regex can't mark booths `boothlike`, so it resolves to **shape**.
  - `shape` keeps color/bordered/bigregion unconditionally; keeps `opencv_strict` unless text empty.
- **`recover_uncovered_bordered(pool, kept, iou_thresh, ios_thresh)`:** re-adds bordered micro-cells that no kept box overlaps (IoU) or nests (IoS). Spatial grid, CELL=400. Recovered 121 cells.
- **Tile concurrency:** `ThreadPoolExecutor` over tiles. Detection runs in pure worker threads (OpenCV releases the GIL → real CPU parallelism; one shared CUDA context = safe). **Stitching stays in the main thread (race-free).** Sets `cv2.setNumThreads(1)` when parallel.

### New CLI flags
| Flag | Default | Meaning |
|---|---|---|
| `--fp-policy {none,strict,shape,adaptive}` | `adaptive` | Which keep-policy to apply. |
| `--no-recover-bordered` | recovery ON | Disable bordered micro-cell recovery. |
| `--workers N` | `0` (auto = `min(8, cpu_count)`) | Tile worker threads. |

---

## 4. FP policy reference

| Policy | Keeps |
|---|---|
| `none` | Everything (prod's batch run → 1907 boxes incl. 902 empty cells). |
| `strict` | Only `boothlike`. |
| `shape` | color/bordered/bigregion unconditionally; `opencv_strict` unless text empty. |
| `adaptive` | `strict` if `n_boothlike >= max(10, 0.15*n)` else `shape`. |

---

## 5. Current fidelity (GPU vs prod, AMFloorplan-0504)

Prod kept 1907 boxes, but most are not booths:
- **902 empty cells** (no text — e.g. the vertical strips inside "MILLENNIUM PARK LOUNGE" / "FREEMAN SERVICE DESK").
- **21 named-only regions** (lounges, aisles: `MICHIGAN AVENUE`, `STATE STREET`, `VIP LOUNGE`, …).
- **984 real booths** (carry a numeric booth ID).

Against the **984 real booths**:
- **950 matched** at IoU ≥ 0.5 with the same booth ID (943 pixel-aligned at IoU ≥ 0.95).
- **985 / 993 distinct booth IDs covered = 99.2%.**
- **Prod-only (GPU missed): 34** — mostly text-tokenization fragments; only **1 genuine miss (booth 1031)** + 1 showcase 17px micro-strip.
- **GPU-only (extra real booths): 134** — mostly legitimate recovered micro-cells, not false positives.
- Source mix now matches prod's geometric-dominated shape.

Comparison viz: `/home/nls161-pc/Downloads/prod_vs_gpu_compare.png`
(green = both, red = prod-only, blue = GPU-only.)

---

## 6. How to run

```bash
# Clean booth output (default; adaptive → shape on this plan)
python src/run_real.py --input <plan.pdf>

# Exact count parity with prod's 1907
python src/run_real.py --input <plan.pdf> --fp-policy none

# Control tile concurrency
python src/run_real.py --input <plan.pdf> --workers 8
```

**Concurrency caveat:** the bottleneck is **host vCPU count, not the T4**.
- Free Colab = 2 cores → ~1.5–2× (detect 165s → 148s).
- 16-core box → far faster.
- Do **NOT** use multiprocessing + CUDA (fork + CUDA risks OOM/crash on a single T4).

**Timing profile:** detect ≈ 80% (CPU connected-components — where concurrency helps), big-region ≈ 14%, rest ~6%.

---

## 7. Open work

1. Recover **booth 1031** + the showcase 17px micro-strip (needs a finer bordered pass on that strip — below current detection floor).
2. Validate concurrency on a real multi-core host (16-core box / GPU server) to confirm scaling past Colab's 2 cores.
3. GPU server deployment (Docker/compose). Detection is CPU-bound; size VRAM for the downstream Qwen-VL labeler (24 GB floor).
4. Optionally move connected-components / morphology to CUDA to speed up detect itself.
5. Confirm `--fp-policy none` reproduces the exact 1907-count parity.

---

## 8. Hard constraints (must preserve)

- `.env` files hold **live Roboflow + AWS secrets** — never print/copy them. **Keep S3/AWS writeback OFF.** Always work on **isolated copies** of source images.
- Never commit without an explicit request. Never skip git hooks.
- A GitHub token in `~/.git-credentials` was previously exposed — **rotate it; never print it.**
- `ROBOFLOW_HALL_API_KEY` is read from env — never print/store it.
- OCR: EasyOCR for blurry raster; PaddleOCR only for crisp PDF text.
- Render detection visualizations **without name labels** (boxes must stay visible).
- Local dev box lacks numpy/cv2/fitz — functional runs happen in Colab; local validation = `ast.parse` syntax check + JSON analysis scripts.

---

## 9. Key files

| File | Role |
|---|---|
| `src/run_real.py` | Runner: tiling, pooling, NMS, FP policy, recovery, concurrency, `--geo-backend`. |
| `src/geometric.py` | Geometric / `opencv_strict` pass — CPU (param-matched to prod). |
| `src/geometric_gpu.py` | GPU reimplementation of the geometric pass (opt-in via `--geo-backend gpu`). |
| `src/gpu_components.py` | GPU CCL: `components_boxes_gpu` + `components_stats_gpu`. |
| `src/gpu_ops.py` | GPU per-pixel + morphology primitives (color pass). |
| `colab_booth_gpu.ipynb` | Colab driver notebook. |
| Prod (read-only, under `plotyourmap (Copy)/`) | `app/pipeline/detectors/{booth,ensemble,opencv,color,bordered}_detector.py`, `app/adaptive/{pipeline,config}.py`, `app/pipeline/utils/geometry.py`. |
| Prod batch output | `/home/nls161-pc/batch_results/AMFloorplan-0504/AMFloorplan-0504_booths.json` (bbox xywh). |
