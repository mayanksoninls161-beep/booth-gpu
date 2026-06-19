"""
Fixture generator for the benchmark.

The honest way to benchmark the merge is on the REAL pre-NMS box pool the
pipeline produces (export it with export_real_fixtures.py — see that file). But
to make the Colab notebook run end-to-end with zero dependencies on the private
pipeline/container, this script synthesises a box pool with the SAME pathological
structure a dense floor plan produces, so the speedup numbers are representative:

  * a grid of "true" booth cells (the answer the merge should recover)
  * each true cell detected 2-4x by different passes (geo/color/bordered) with
    small jitter + different scores      -> near-duplicates NMS must collapse
  * small nested label/icon boxes inside cells -> containment (ios) drops
  * a few skinny text slices                    -> the false-split guard
  * a few big "merged block" boxes over 2+ cells -> the merged-block pre-filter

Box-pool generation uses only the standard library (random, json) so it runs
anywhere. The optional synthetic plan IMAGE (for the color-mask / components
benchmark) needs numpy and is written only if numpy is importable.

Usage:
    python make_fixtures.py [--n-cells 600] [--out ../fixtures]
"""
from __future__ import annotations

import argparse
import json
import os
import random


def make_box_pool(n_cells=600, seed=0):
    """Return (pool, n_true) where pool is a list of {'bbox','score'} dicts."""
    rng = random.Random(seed)
    pool = []

    # Lay true cells on a near-square grid with realistic gutters.
    cols = max(1, int(round(n_cells ** 0.5)))
    rows = (n_cells + cols - 1) // cols
    cell_w, cell_h = 120, 90
    gut = 14
    x0, y0 = 40, 40

    true_cells = []
    for r in range(rows):
        for c in range(cols):
            if len(true_cells) >= n_cells:
                break
            x1 = x0 + c * (cell_w + gut)
            y1 = y0 + r * (cell_h + gut)
            # mild per-cell size variation
            w = cell_w + rng.randint(-8, 8)
            h = cell_h + rng.randint(-6, 6)
            true_cells.append((x1, y1, x1 + w, y1 + h))

    def jitter(b, j):
        return [b[0] + rng.randint(-j, j), b[1] + rng.randint(-j, j),
                b[2] + rng.randint(-j, j), b[3] + rng.randint(-j, j)]

    for (x1, y1, x2, y2) in true_cells:
        # 2-4 near-duplicate detections of the same cell (different passes)
        ndup = rng.randint(2, 4)
        for _ in range(ndup):
            pool.append({"bbox": jitter((x1, y1, x2, y2), 4),
                         "score": round(rng.uniform(0.55, 0.99), 3)})
        # ~35% of cells carry a small nested label/icon box (high ios -> dropped)
        if rng.random() < 0.35:
            lw = (x2 - x1) // 4; lh = (y2 - y1) // 4
            lx = x1 + (x2 - x1) // 2 - lw // 2
            ly = y1 + (y2 - y1) // 2 - lh // 2
            pool.append({"bbox": [lx, ly, lx + lw, ly + lh],
                         "score": round(rng.uniform(0.4, 0.8), 3)})
        # ~12% carry a skinny text slice (false-split guard keeps the big box)
        if rng.random() < 0.12:
            ty = y1 + (y2 - y1) - 10
            pool.append({"bbox": [x1 + 4, ty, x2 - 4, ty + 8],
                         "score": round(rng.uniform(0.4, 0.7), 3)})

    # A few big "merged block" boxes spanning 2x2 true cells (pre-filter drops).
    n_blocks = max(2, n_cells // 80)
    for _ in range(n_blocks):
        if len(true_cells) < 4:
            break
        i = rng.randrange(len(true_cells))
        bx1, by1, _, _ = true_cells[i]
        block = [bx1 - 6, by1 - 6,
                 bx1 + 2 * (cell_w + gut), by1 + 2 * (cell_h + gut)]
        pool.append({"bbox": [float(v) for v in block],
                     "score": round(rng.uniform(0.3, 0.6), 3)})

    # cast all to float for json cleanliness
    for b in pool:
        b["bbox"] = [float(v) for v in b["bbox"]]
    rng.shuffle(pool)
    return pool, len(true_cells)


def make_plan_image(n_cells=600, seed=0):
    """Synthetic BGR plan image with filled blue cells + numpy is required.

    Returns (img_bgr, lower, upper) or None if numpy is unavailable.
    """
    try:
        import numpy as np
    except Exception:
        return None
    rng = random.Random(seed)
    cols = max(1, int(round(n_cells ** 0.5)))
    rows = (n_cells + cols - 1) // cols
    cell_w, cell_h, gut, x0, y0 = 120, 90, 14, 40, 40
    H = y0 * 2 + rows * (cell_h + gut)
    W = x0 * 2 + cols * (cell_w + gut)
    img = np.full((H, W, 3), 245, dtype=np.uint8)   # near-white page
    placed = 0
    for r in range(rows):
        for c in range(cols):
            if placed >= n_cells:
                break
            x1 = x0 + c * (cell_w + gut)
            y1 = y0 + r * (cell_h + gut)
            x2 = x1 + cell_w + rng.randint(-8, 8)
            y2 = y1 + cell_h + rng.randint(-6, 6)
            # filled blue-ish cell (BGR): high B, low R
            img[y1:y2, x1:x2] = (rng.randint(180, 255), rng.randint(40, 110),
                                 rng.randint(20, 80))
            placed += 1
    lower = [120, 0, 0]      # BGR lower
    upper = [255, 130, 110]  # BGR upper
    return img, lower, upper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cells", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "fixtures"))
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    pool, n_true = make_box_pool(args.n_cells, args.seed)
    pool_path = os.path.join(out, "boxes_pool.json")
    with open(pool_path, "w") as f:
        json.dump({"n_true_cells": n_true, "boxes": pool}, f)
    print(f"wrote {pool_path}: {len(pool)} boxes ({n_true} true cells)")

    plan = make_plan_image(args.n_cells, args.seed)
    if plan is not None:
        import numpy as np
        img, lower, upper = plan
        npy_path = os.path.join(out, "plan_synth.npy")
        np.save(npy_path, img)
        meta_path = os.path.join(out, "plan_synth.json")
        with open(meta_path, "w") as f:
            json.dump({"shape": list(img.shape), "lower_bgr": lower, "upper_bgr": upper}, f)
        print(f"wrote {npy_path}: {img.shape} + {meta_path}")
        # also a viewable PNG if cv2 is around
        try:
            import cv2
            png = os.path.join(out, "plan_synth.png")
            cv2.imwrite(png, img)
            print(f"wrote {png}")
        except Exception:
            pass
    else:
        print("numpy not available -> skipped synthetic plan image "
              "(box-pool merge benchmark still works)")


if __name__ == "__main__":
    main()
