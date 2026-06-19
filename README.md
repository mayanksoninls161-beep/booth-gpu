# booth-gpu — GPU-native detection core (prototype / benchmark)

Goal: take the booth-detection hot path that is currently CPU-bound
(`findContours` + dedup/merge) and re-express it as **GPU tensor operations**,
so a dense floor-plan PDF that takes ~10–12 min today can run in ~1–3 min.

This repo is **self-contained** so it can be cloned straight into Google Colab
(free T4 GPU) for fast iterate-test cycles, independent of the main pipeline /
Docker container.

```
booth-gpu/
  README.md
  requirements.txt
  colab_booth_gpu.ipynb     # one-click: clone -> install -> benchmark
  src/
    gpu_ops.py              # GPU color mask + morphology (per-pixel, SIMD)
    gpu_components.py       # GPU connected-components -> boxes (replaces findContours)
    gpu_merge.py            # GPU IoU matrix + transitive-closure merge (replaces dedup loop)
    detect_gpu.py           # glue: image -> boxes, fully on GPU
    cpu_ref.py             # OpenCV/loop reference implementations (for parity + timing)
    bench.py               # CPU vs GPU timing harness -> prints the real speedup table
  tests/
    test_parity.py         # GPU output == CPU output (within tolerance)
```

---

## The plan (what moves to GPU, why, projected payoff)

| Stage | Today (CPU) | GPU-native replacement | Why it works on GPU | Projected speedup |
|---|---|---|---|---|
| color / threshold / morphology | `cv2.inRange`, `cv2.morphologyEx` | torch tensor ops (`gpu_ops.py`) | per-pixel SIMD, no data dependency | 5–15× on big renders |
| **contours → boxes** | `cv2.findContours` (serial boundary trace, CPU-only) | **GPU connected-components by label propagation** (`gpu_components.py`) | components found by parallel neighbor max-pooling, not boundary walking | 3–8× |
| **dedup / merge** | python loop, all-pairs `if overlap: merge, recheck` | **N×N IoU matrix (one matmul) + boolean transitive closure** (`gpu_merge.py`) | all-pairs overlap is a single tensor op; clustering is log(N) matmuls | 10–50× on dense plans |
| OCR labeling | EasyOCR (CPU) | EasyOCR `gpu=True` | model already GPU-native | 5–10× |

Key idea for merge: the slow part isn't the math, it's the **serial loop**. We
don't port the loop — we replace the *algorithm*. All-pairs IoU becomes a single
`N×N` matrix; "which boxes belong to the same merged group" becomes connected
components on that adjacency matrix, computed by repeated boolean matmul
(transitive closure) in O(log N) GPU steps.

Critical rule: **keep tensors resident on the GPU** across all stages. Bouncing
CPU↔GPU between every step pays a transfer tax that can erase the gains.

---

## Projected end-to-end (dense PDF, ~11 min today)

These are projections; `bench.py` replaces them with measured numbers per stage.

| Step | Dense PDF time |
|---|---|
| Today (4-core container, all CPU) | ~11 min |
| Unlock 16 cores (config only) | ~7–8 min |
| + multiprocess tiles (CPU) | ~3–3.5 min |
| **+ GPU merge + GPU components (this repo)** | ~1.5–2 min |
| + EasyOCR on GPU | ~1–1.5 min |

---

## Run it on Colab

1. Push this folder to a new GitHub repo (see below).
2. Open `colab_booth_gpu.ipynb` in Colab, Runtime → Change runtime type → **GPU (T4)**.
3. Run all cells: it clones the repo, installs deps, runs `bench.py`, prints the
   CPU-vs-GPU speedup table and parity check.

## Make it a repo
```bash
cd booth-gpu
git init && git add -A && git commit -m "booth-gpu prototype: GPU merge + components + bench"
git branch -M main
git remote add origin git@github.com:<you>/booth-gpu.git   # or https
git push -u origin main
```
