"""
CPU-vs-GPU benchmark + parity harness. This is the file that turns the README's
*projections* into *measured numbers* for your hardware.

What it times:
  STAGE 3  merge / NMS  (the big one):
      CPU   cpu_ref.non_max_suppression        (verbatim production greedy)
      GPU   nms_gpu_assisted                    (exact same result, GPU overlaps)
      GPU   nms_gpu_cluster                     (pure-GPU O(log N), fastest)
  STAGE 1+2  color mask + connected components (optional, needs numpy+cv2):
      CPU   cv2.inRange/morphology + connectedComponentsWithStats
      GPU   gpu_ops + gpu_components

It also checks PARITY:
  * assisted vs CPU  -> must be identical (same kept boxes, by identity)
  * cluster  vs CPU  -> reports agreement (Jaccard); cluster is intentionally
                        a touch more aggressive, so this shows how much.

Run:
    python bench.py                       # synthetic pool (auto-generated)
    python bench.py --pool ../fixtures/boxes_pool_real.json   # real pool
    python bench.py --n-cells 900         # bigger synthetic dense plan

On a CPU-only box the "GPU" columns run on CPU (torch CPU) — useful for the
parity check but NOT for speed. Real speedups need CUDA: open the Colab notebook.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

import cpu_ref
from gpu_merge import nms_gpu_assisted, nms_gpu_cluster


# --------------------------------------------------------------------------- #
# timing helpers
# --------------------------------------------------------------------------- #
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, iters=3, warmup=1, cuda=False):
    for _ in range(max(0, warmup)):
        fn()
    if cuda:
        _sync()
    t0 = time.perf_counter()
    out = None
    for _ in range(max(1, iters)):
        out = fn()
    if cuda:
        _sync()
    dt = (time.perf_counter() - t0) / max(1, iters)
    return dt, out


def _fmt_ms(s):
    return f"{s * 1e3:9.2f} ms"


def _ids(boxes):
    return {id(b) for b in boxes}


def _jaccard(a, b):
    a, b = set(a), set(b)
    u = len(a | b)
    return (len(a & b) / u) if u else 1.0


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_pool(args):
    if args.pool and os.path.exists(args.pool):
        with open(args.pool) as f:
            data = json.load(f)
        pool = data["boxes"]
        for b in pool:
            b["bbox"] = [float(v) for v in b["bbox"]]
            b.setdefault("score", 1.0)
        print(f"pool: {len(pool)} boxes from {args.pool}")
        return pool
    # synthesise
    from make_fixtures import make_box_pool
    pool, n_true = make_box_pool(args.n_cells, seed=args.seed)
    print(f"pool: {len(pool)} boxes (synthetic, {n_true} true cells)")
    return pool


# --------------------------------------------------------------------------- #
# stage 3: merge
# --------------------------------------------------------------------------- #
def _quad(bbox):
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def bench_merge(pool, device, args):
    print("\n=== STAGE 3: dedup / merge (NMS) ===")
    iou_t, cont_t = 0.3, 0.7

    # Production reality: the real pre-NMS boxes carry polygons, so the CPU NMS
    # takes the cv2.intersectConvexConvex path (the actual slow tax today). Build
    # a poly-carrying copy to time that honestly. Falls back to bbox path if cv2
    # is unavailable (then both CPU numbers are the same cheap path).
    have_cv2 = getattr(cpu_ref, "_HAVE_CV2", False)
    cpu_poly_dt = None
    if have_cv2:
        pool_poly = [dict(b, poly=_quad(b["bbox"])) for b in pool]
        cpu_poly_dt, _ = timeit(
            lambda: cpu_ref.non_max_suppression(pool_poly, iou_t, cont_t),
            iters=args.iters_cpu, warmup=0, cuda=False)

    # Clean bbox path: parity reference for the GPU paths (same objects -> id()).
    cpu_dt, cpu_keep = timeit(
        lambda: cpu_ref.non_max_suppression(pool, iou_t, cont_t),
        iters=args.iters_cpu, warmup=0, cuda=False)

    ass_dt, ass_keep = timeit(
        lambda: nms_gpu_assisted(pool, iou_t, cont_t, device=device),
        iters=args.iters_gpu, warmup=1, cuda=True)

    clu_dt, clu_keep = timeit(
        lambda: nms_gpu_cluster(pool, iou_t, cont_t, device=device),
        iters=args.iters_gpu, warmup=1, cuda=True)

    same_assisted = _ids(cpu_keep) == _ids(ass_keep)
    jac_cluster = _jaccard(_ids(cpu_keep), _ids(clu_keep))

    # Headline baseline = production-representative poly path when available.
    base_dt = cpu_poly_dt if cpu_poly_dt is not None else cpu_dt
    base_lbl = "CPU poly (prod)" if cpu_poly_dt is not None else "CPU bbox"

    if cpu_poly_dt is not None:
        print(f"  CPU  NMS poly (prod) : {_fmt_ms(cpu_poly_dt)}  "
              f"<- today's path (cv2.intersectConvexConvex per pair)")
    print(f"  CPU  NMS bbox (ref)  : {_fmt_ms(cpu_dt)}  -> kept {len(cpu_keep)}")
    print(f"  GPU  assisted (exact): {_fmt_ms(ass_dt)}  -> kept {len(ass_keep)}  "
          f"[{'PARITY OK' if same_assisted else 'MISMATCH'}]  "
          f"x{base_dt / max(ass_dt, 1e-9):.1f} vs {base_lbl}")
    print(f"  GPU  cluster (fast)  : {_fmt_ms(clu_dt)}  -> kept {len(clu_keep)}  "
          f"[agreement {jac_cluster * 100:.1f}%]  "
          f"x{base_dt / max(clu_dt, 1e-9):.1f} vs {base_lbl}")

    if not same_assisted:
        only_cpu = _ids(cpu_keep) - _ids(ass_keep)
        only_gpu = _ids(ass_keep) - _ids(cpu_keep)
        print(f"    ! assisted diff: {len(only_cpu)} only-CPU, {len(only_gpu)} only-GPU "
              f"(float threshold edge cases)")

    return {
        "cpu_poly_ms": (cpu_poly_dt * 1e3) if cpu_poly_dt is not None else None,
        "cpu_ms": cpu_dt * 1e3, "assisted_ms": ass_dt * 1e3, "cluster_ms": clu_dt * 1e3,
        "base_ms": base_dt * 1e3, "base_lbl": base_lbl,
        "cpu_kept": len(cpu_keep), "assisted_kept": len(ass_keep), "cluster_kept": len(clu_keep),
        "assisted_parity": same_assisted, "cluster_agreement": jac_cluster,
    }


# --------------------------------------------------------------------------- #
# stage 1+2: color mask + components (optional)
# --------------------------------------------------------------------------- #
def _match_rate(cpu_boxes, gpu_boxes, iou_min=0.9):
    """Fraction of CPU boxes that have a GPU box at IoU >= iou_min."""
    if len(cpu_boxes) == 0:
        return 1.0 if len(gpu_boxes) == 0 else 0.0
    hit = 0
    gpu = [tuple(map(float, b)) for b in gpu_boxes]
    for cb in cpu_boxes:
        cb = tuple(map(float, cb))
        best = 0.0
        for gb in gpu:
            best = max(best, cpu_ref.calculate_iou(cb, gb))
            if best >= iou_min:
                break
        if best >= iou_min:
            hit += 1
    return hit / len(cpu_boxes)


def bench_image(device, args):
    print("\n=== STAGE 1+2: color mask + connected components ===")
    try:
        import numpy as np
    except Exception:
        print("  skipped (numpy not installed)")
        return None
    try:
        import cv2  # noqa: F401
    except Exception:
        print("  skipped (OpenCV/cv2 not installed — CPU reference unavailable)")
        return None

    # load or synthesise the plan image
    img = None; lower = [120, 0, 0]; upper = [255, 130, 110]
    npy = os.path.join(os.path.dirname(__file__), "..", "fixtures", "plan_synth.npy")
    meta = os.path.join(os.path.dirname(__file__), "..", "fixtures", "plan_synth.json")
    if args.image and os.path.exists(args.image):
        img = cv2.imread(args.image)
    elif os.path.exists(npy):
        img = np.load(npy)
        if os.path.exists(meta):
            m = json.load(open(meta)); lower, upper = m["lower_bgr"], m["upper_bgr"]
    else:
        from make_fixtures import make_plan_image
        out = make_plan_image(args.n_cells, seed=args.seed)
        if out is None:
            print("  skipped (could not build synthetic image)")
            return None
        img, lower, upper = out
    print(f"  image: {img.shape}")

    from gpu_ops import image_to_tensor, color_mask_pipeline_gpu
    from gpu_components import components_boxes_gpu

    # --- CPU ---
    def cpu_fn():
        m = cpu_ref.color_mask_cpu(img, lower, upper)
        return cpu_ref.components_boxes_cpu(m, min_area=80)
    cpu_dt, cpu_boxes = timeit(cpu_fn, iters=args.iters_cpu, warmup=0, cuda=False)

    # --- GPU ---
    img_chw = image_to_tensor(img, device=device)

    def gpu_fn():
        mask = color_mask_pipeline_gpu(img_chw, lower, upper)
        return components_boxes_gpu(mask[0, 0], min_area=80)
    gpu_dt, gpu_boxes_t = timeit(gpu_fn, iters=args.iters_gpu, warmup=1, cuda=True)
    gpu_boxes = gpu_boxes_t.cpu().tolist()

    rate = _match_rate(cpu_boxes.tolist(), gpu_boxes, iou_min=0.9)
    print(f"  CPU  inRange+morph+CC: {_fmt_ms(cpu_dt)}  -> {len(cpu_boxes)} boxes")
    print(f"  GPU  mask+labelprop  : {_fmt_ms(gpu_dt)}  -> {len(gpu_boxes)} boxes  "
          f"[{rate * 100:.1f}% match @IoU0.9]  speedup x{cpu_dt / max(gpu_dt, 1e-9):.1f}")
    return {"cpu_ms": cpu_dt * 1e3, "gpu_ms": gpu_dt * 1e3,
            "cpu_boxes": len(cpu_boxes), "gpu_boxes": len(gpu_boxes), "match": rate}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=None, help="path to a boxes_pool*.json (real fixture)")
    ap.add_argument("--n-cells", type=int, default=600, help="synthetic dense-plan size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters-cpu", type=int, default=1)
    ap.add_argument("--iters-gpu", type=int, default=5)
    ap.add_argument("--image", default=None, help="optional plan image for stage 1+2")
    ap.add_argument("--no-image", action="store_true", help="skip the image stage")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 64)
    print(f" booth-gpu benchmark   torch {torch.__version__}   device={device}")
    if device == "cuda":
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(" NOTE: no CUDA -> 'GPU' columns run on CPU. Parity is real,")
        print("       speed is NOT. Run on Colab (T4) for real speedups.")
    print("=" * 64)

    pool = load_pool(args)
    merge = bench_merge(pool, device, args)
    image = None if args.no_image else bench_image(device, args)

    print("\n=== SUMMARY ===")
    print(f" device: {device}")
    if merge:
        base = merge["base_ms"]
        print(f" merge  baseline {base:.1f}ms ({merge['base_lbl']}) | "
              f"assisted {merge['assisted_ms']:.1f}ms "
              f"(x{base / max(merge['assisted_ms'], 1e-9):.1f}, "
              f"{'parity' if merge['assisted_parity'] else 'DIFF'}) | "
              f"cluster {merge['cluster_ms']:.1f}ms "
              f"(x{base / max(merge['cluster_ms'], 1e-9):.1f}, "
              f"{merge['cluster_agreement'] * 100:.0f}% agree)")
    if image:
        print(f" image  CPU {image['cpu_ms']:.1f}ms | GPU {image['gpu_ms']:.1f}ms "
              f"(x{image['cpu_ms'] / max(image['gpu_ms'], 1e-9):.1f}, "
              f"{image['match'] * 100:.0f}% match)")
    print("=" * 64)


if __name__ == "__main__":
    main()
