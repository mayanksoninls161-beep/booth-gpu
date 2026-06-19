"""
Run the GPU booth hot-path on a REAL floor-plan IMAGE or PDF, end to end, and
save an annotated preview you can view inline in Colab.

This is the "try it on my own plan" entry point (the bench/parity scripts use
synthetic or pool fixtures). Pipeline, all on GPU:

    load image (or render a PDF page)        load_image()
    -> pick booth colour band(s)             auto_hsv_bands()  (or pass --hsv-*)
    -> color mask + morphology (per band)    gpu_ops.color_mask_pipeline_gpu
    -> connected components -> boxes         gpu_components.components_boxes_gpu
    -> dedup / merge                         gpu_merge.nms_gpu_cluster (or assisted)
    -> draw boxes (no labels) + save PNGs

Masking is done in HSV space (the GPU ops are channel-order agnostic, so we feed
HSV planes + HSV bounds) because exhibition booths are far easier to isolate by
hue than by raw BGR. By default we auto-detect the dominant saturated hues; pass
--hsv-lower/--hsv-upper to force one exact band.

Examples (inside Colab, after cloning the repo):
    python src/run_real.py --input my_plan.png
    python src/run_real.py --input plan.pdf --page 0 --dpi 200
    python src/run_real.py --input plan.png --colors 2 --min-area 300
    python src/run_real.py --input plan.png --hsv-lower 90,60,60 --hsv-upper 130,255,255

Outputs go to out/:  *_boxes.png (annotated), *_mask.png (the binary mask).
Nothing is uploaded anywhere; everything stays local to the Colab VM.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from gpu_ops import image_to_tensor, color_mask_pipeline_gpu
from gpu_components import components_boxes_gpu
from gpu_merge import nms_gpu_cluster, nms_gpu_assisted


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# input: image file or PDF page -> BGR uint8 image
# --------------------------------------------------------------------------- #
def load_image(path, dpi=200, page=0):
    """Return a BGR uint8 HxWx3 image from an image file OR a PDF page.

    PDFs are rasterised with PyMuPDF (fitz) at `dpi` (no poppler needed). For a
    multi-page PDF choose the page with --page.
    """
    import cv2
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            import fitz  # PyMuPDF
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "PDF support needs PyMuPDF: pip install PyMuPDF") from e
        doc = fitz.open(path)
        if page < 0 or page >= doc.page_count:
            raise ValueError(f"page {page} out of range (0..{doc.page_count - 1})")
        pg = doc[page]
        zoom = dpi / 72.0
        pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # fitz is RGB; cv2 is BGR
        return np.ascontiguousarray(img)
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


# --------------------------------------------------------------------------- #
# colour band(s): auto-detect dominant saturated hues, or use a forced band
# --------------------------------------------------------------------------- #
def auto_hsv_bands(img_bgr, k=3, s_min=60, v_min=60, hue_halfwidth=8):
    """Find up to k dominant booth hues and return HSV (lower, upper) bands.

    We look at sufficiently saturated, not-too-dark pixels (booth fills, not the
    white paper or black text/lines), histogram their hue, and take the k tallest
    peaks that are separated by more than hue_halfwidth. Each becomes a band
    [h-hw, s_min, v_min] .. [h+hw, 255, 255]. OpenCV hue is 0..179.
    """
    import cv2
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0]; s = hsv[..., 1]; v = hsv[..., 2]
    sat = (s >= s_min) & (v >= v_min)
    hues = h[sat]
    if hues.size == 0:
        return []
    hist = np.bincount(hues.ravel(), minlength=180).astype(np.int64)
    peaks = []
    for hue in np.argsort(hist)[::-1]:
        if hist[hue] == 0:
            break
        hue = int(hue)
        if all(min(abs(hue - p), 180 - abs(hue - p)) > hue_halfwidth for p in peaks):
            peaks.append(hue)
        if len(peaks) >= k:
            break
    bands = []
    for hue in peaks:
        lo = [max(0, hue - hue_halfwidth), s_min, v_min]
        hi = [min(179, hue + hue_halfwidth), 255, 255]
        bands.append((lo, hi))
    return bands


# --------------------------------------------------------------------------- #
def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = load_image(args.input, dpi=args.dpi, page=args.page)
    H, W = img.shape[:2]
    if args.downscale and args.downscale != 1.0:
        import cv2
        img = cv2.resize(img, (int(W * args.downscale), int(H * args.downscale)),
                         interpolation=cv2.INTER_AREA)
        H, W = img.shape[:2]
    print(f"image: {W}x{H}  device={device}")

    # --- choose colour band(s) ---
    if args.hsv_lower and args.hsv_upper:
        lo = [int(x) for x in args.hsv_lower.split(",")]
        hi = [int(x) for x in args.hsv_upper.split(",")]
        bands = [(lo, hi)]
        print(f"colour: forced HSV band {lo} .. {hi}")
    else:
        bands = auto_hsv_bands(img, k=args.colors)
        if not bands:
            print("!! no saturated colour found — is this a white/greyscale plan?")
            print("   try a forced band, e.g. --hsv-lower 0,0,0 --hsv-upper 179,255,120")
            return
        print(f"colour: auto-detected {len(bands)} hue band(s):")
        for lo, hi in bands:
            print(f"          HSV {lo} .. {hi}")

    # --- GPU pipeline (timed) ---
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_t = image_to_tensor(hsv, device=device)

    _sync(); t0 = time.perf_counter()
    mask = None
    for lo, hi in bands:                      # OR the per-band masks together
        m = color_mask_pipeline_gpu(hsv_t, lo, hi,
                                    open_ksize=args.open_ksize,
                                    close_ksize=args.close_ksize)
        mask = m if mask is None else (mask | m)
    boxes_t = components_boxes_gpu(mask[0, 0], min_area=args.min_area,
                                   max_iters=args.max_iters)
    _sync(); t_cc = time.perf_counter()

    boxes = boxes_t.cpu().tolist()
    pool = [{"bbox": [float(x) for x in b], "score": 1.0} for b in boxes]
    merge_fn = nms_gpu_assisted if args.merge == "assisted" else nms_gpu_cluster
    _sync(); t1 = time.perf_counter()
    kept = merge_fn(pool, args.iou, args.containment, device=device)
    _sync(); t2 = time.perf_counter()

    print(f"\ncomponents: {len(boxes)} raw boxes   "
          f"(mask+CC {1e3*(t_cc-t0):.1f} ms)")
    print(f"merge ({args.merge}): {len(kept)} kept   "
          f"({1e3*(t2-t1):.1f} ms)")
    print(f"TOTAL GPU: {1e3*((t_cc-t0)+(t2-t1)):.1f} ms")

    # --- annotate (boxes only, NO labels) + save ---
    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    vis = img.copy()
    for b in kept:
        x1, y1, x2, y2 = (int(round(v)) for v in b["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    boxes_png = os.path.join(args.outdir, f"{stem}_boxes.png")
    mask_png = os.path.join(args.outdir, f"{stem}_mask.png")
    cv2.imwrite(boxes_png, vis)
    cv2.imwrite(mask_png, (mask[0, 0].cpu().numpy() * 255).astype(np.uint8))
    print(f"\nsaved: {boxes_png}")
    print(f"saved: {mask_png}")
    return boxes_png, mask_png


def main():
    ap = argparse.ArgumentParser(description="Run the GPU booth pipeline on a real image/PDF")
    ap.add_argument("--input", required=True, help="path to an image (png/jpg) or .pdf")
    ap.add_argument("--page", type=int, default=0, help="PDF page index (0-based)")
    ap.add_argument("--dpi", type=int, default=200, help="PDF rasterisation DPI")
    ap.add_argument("--downscale", type=float, default=1.0, help="resize factor (<1 speeds up label-prop CC)")
    ap.add_argument("--colors", type=int, default=3, help="how many dominant hue bands to auto-detect")
    ap.add_argument("--hsv-lower", default=None, help="force one HSV lower bound 'H,S,V' (0..179,0..255,0..255)")
    ap.add_argument("--hsv-upper", default=None, help="force one HSV upper bound 'H,S,V'")
    ap.add_argument("--min-area", type=int, default=300, help="drop components smaller than this (px^2)")
    ap.add_argument("--max-iters", type=int, default=256, help="label-prop rounds cap (raise for big cells)")
    ap.add_argument("--open-ksize", type=int, default=3)
    ap.add_argument("--close-ksize", type=int, default=3)
    ap.add_argument("--merge", choices=["cluster", "assisted"], default="cluster")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--containment", type=float, default=0.7)
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
