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

For DENSE plans the detect stage runs TILED by default (--tile 1800 --overlap
400, ported from prod's app/adaptive/tiling.py): the page is cut into overlapping
crops, each crop is colour-masked + connected-component'd on its own, boxes that
touch an inner seam are dropped (the neighbour tile holds them whole), the rest
are offset back to global coordinates and fused by ONE global GPU merge. Tiling
both recovers the tiny cells a single full-page pass fuses / min-area-drops AND
speeds the GPU connected-components (smaller crops converge in far fewer
label-prop rounds). Pass --tile 0 to force the old single full-image pass.

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
import cpu_ref


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
# tiling: overlapping crops so dense little cells survive (ported from prod's
# app/adaptive/tiling.py). A whole-page render shrinks each booth below the
# min-area filter and fuses abutting cells; cropping makes each cell a healthy
# fraction of *its tile* so CC traces it on its own, AND each crop is small so
# the GPU label-prop converges in far fewer rounds.
# --------------------------------------------------------------------------- #
_SEAM_EDGE = 4  # px: a box within this of an inner tile margin is "clipped"


def _tile_origins(extent, tile, step):
    """Tile start offsets so the last tile always reaches `extent`."""
    if extent <= tile:
        return [0]
    xs, x = [], 0
    while True:
        xs.append(x)
        if x + tile >= extent:
            break
        x += step
    return xs


def _clipped_xyxy(x1, y1, x2, y2, cw, ch, gx0, gy0, W, H, edge=_SEAM_EDGE):
    """xyxy variant of prod's _clipped_at_seam: True if a CROP-LOCAL box touches
    an INNER tile edge (a partial booth the neighbouring tile holds whole). Boxes
    flush against the real image border are kept."""
    if x1 <= edge and gx0 > 0:
        return True
    if x2 >= cw - edge and gx0 + cw < W:
        return True
    if y1 <= edge and gy0 > 0:
        return True
    if y2 >= ch - edge and gy0 + ch < H:
        return True
    return False


def _detect_on_crop(crop_bgr, bands, device, args, want_mask=True):
    """Colour-mask + connected-components on ONE BGR crop (or the whole image).

    Returns (boxes_xyxy, mask_np_u8, mask_ms, cc_ms):
      boxes_xyxy : list of [x1,y1,x2,y2] in CROP-LOCAL pixel coords
      mask_np_u8 : binary mask (uint8 0/255) for preview stitching, or None
      mask_ms    : colour-mask+morphology wall time (CUDA-synced), ms
      cc_ms      : connected-components wall time (CUDA-synced), ms
    """
    import cv2
    # colour mask + morphology (GPU), per band, OR'd
    _sync(); t0 = time.perf_counter()
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hsv_t = image_to_tensor(hsv, device=device)
    mask = None
    for lo, hi in bands:
        m = color_mask_pipeline_gpu(hsv_t, lo, hi,
                                    open_ksize=args.open_ksize,
                                    close_ksize=args.close_ksize)
        mask = m if mask is None else (mask | m)
    _sync(); mask_ms = (time.perf_counter() - t0) * 1e3

    # connected components -> tile-local xyxy boxes
    _sync(); t1 = time.perf_counter()
    if args.cc == "cpu":
        mask_cpu = (mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        boxes = cpu_ref.components_boxes_cpu(mask_cpu, min_area=args.min_area).tolist()
    else:
        boxes_t = components_boxes_gpu(mask[0, 0], min_area=args.min_area,
                                       max_iters=args.max_iters)
        boxes = boxes_t.cpu().tolist()
        mask_cpu = None
    _sync(); cc_ms = (time.perf_counter() - t1) * 1e3

    mask_np = None
    if want_mask:
        mask_np = mask_cpu if mask_cpu is not None else \
            (mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
    return boxes, mask_np, mask_ms, cc_ms


# --------------------------------------------------------------------------- #
class Profiler:
    """Tiny step timer. Each .step() closes the previous step (with a CUDA sync
    so GPU work is actually finished before we read the clock) and prints it."""

    def __init__(self):
        self.rows = []
        self._t = None
        self._label = None

    def step(self, label):
        now = time.perf_counter()
        if self._label is not None:
            _sync()
            now = time.perf_counter()
            dt = now - self._t
            self.rows.append((self._label, dt))
            print(f"  [{dt * 1e3:8.1f} ms]  {self._label}")
        self._label = label
        self._t = time.perf_counter()

    def done(self):
        self.step(None)

    def total(self):
        return sum(dt for _, dt in self.rows)

    def summary(self):
        tot = self.total()
        print("  " + "-" * 46)
        for label, dt in self.rows:
            pct = (dt / tot * 100) if tot else 0.0
            print(f"  [{dt * 1e3:8.1f} ms] {pct:5.1f}%  {label}")
        print(f"  [{tot * 1e3:8.1f} ms] 100.0%  TOTAL (end to end)")


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import cv2
    print(f"device={device}  input={args.input}")
    print("per-stage timing (each line is that stage only):")
    prof = Profiler()

    # 1. load image / rasterise PDF
    prof.step("load input (PDF render / imread)")
    img = load_image(args.input, dpi=args.dpi, page=args.page)
    H0, W0 = img.shape[:2]

    # 2. optional downscale
    if args.downscale and args.downscale != 1.0:
        prof.step("downscale")
        img = cv2.resize(img, (int(W0 * args.downscale), int(H0 * args.downscale)),
                         interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]

    # 3. choose colour band(s) ONCE on the full image (reused for every tile)
    prof.step("colour band selection")
    if args.hsv_lower and args.hsv_upper:
        lo = [int(x) for x in args.hsv_lower.split(",")]
        hi = [int(x) for x in args.hsv_upper.split(",")]
        bands = [(lo, hi)]
    else:
        bands = auto_hsv_bands(img, k=args.colors)
        if not bands:
            prof.done()
            print("!! no saturated colour found — is this a white/greyscale plan?")
            print("   try a forced band, e.g. --hsv-lower 0,0,0 --hsv-upper 179,255,120")
            return

    # 4. detect — tiled dense pass (default), or single full-image pass (--tile 0)
    use_tiling = bool(args.tile) and args.tile > 0 and max(H, W) > args.tile
    mask_ms_tot = cc_ms_tot = 0.0
    mask_full = None
    total = 0

    if use_tiling:
        step = max(1, args.tile - args.overlap)
        xs = _tile_origins(W, args.tile, step)
        ys = _tile_origins(H, args.tile, step)
        total = len(xs) * len(ys)
        print(f"  tiling: {len(xs)}x{len(ys)} = {total} tiles "
              f"(tile={args.tile}, overlap={args.overlap}, step={step})")
        prof.step(f"tiled detect: mask+CC over {total} tiles")
        mask_full = np.zeros((H, W), dtype=np.uint8)
        pool = []
        n_raw = 0
        for gy0 in ys:
            for gx0 in xs:
                y2c, x2c = min(gy0 + args.tile, H), min(gx0 + args.tile, W)
                crop = img[gy0:y2c, gx0:x2c]
                ch, cw = crop.shape[:2]
                boxes, mnp, m_ms, c_ms = _detect_on_crop(crop, bands, device, args)
                mask_ms_tot += m_ms; cc_ms_tot += c_ms
                if mnp is not None:                      # stitch (overlap -> max)
                    sub = mask_full[gy0:y2c, gx0:x2c]
                    np.maximum(sub, mnp, out=sub)
                n_raw += len(boxes)
                for bx1, by1, bx2, by2 in boxes:
                    if _clipped_xyxy(bx1, by1, bx2, by2, cw, ch, gx0, gy0, W, H,
                                     edge=args.seam_edge):
                        continue                         # neighbour tile holds it whole
                    pool.append({"bbox": [float(bx1 + gx0), float(by1 + gy0),
                                          float(bx2 + gx0), float(by2 + gy0)],
                                 "score": 1.0})
        print(f"  tiles: {n_raw} raw boxes across tiles -> {len(pool)} after seam-clip")
    else:
        prof.step("color mask + CC (single full-image pass)")
        boxes, mask_full, m_ms, c_ms = _detect_on_crop(img, bands, device, args)
        mask_ms_tot += m_ms; cc_ms_tot += c_ms
        pool = [{"bbox": [float(v) for v in b], "score": 1.0} for b in boxes]

    raw_count = len(pool)

    # 5. merge / dedup (GPU) — fuses the overlap-band duplicates from neighbouring
    #    tiles plus any nested fragments, exactly as the prod global NMS does.
    prof.step(f"merge / dedup ({args.merge}, GPU)")
    merge_fn = nms_gpu_assisted if args.merge == "assisted" else nms_gpu_cluster
    kept = merge_fn(pool, args.iou, args.containment, device=device)

    # 6. annotate + save
    prof.step("annotate + save PNGs")
    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    vis = img.copy()
    for b in kept:
        x1, y1, x2, y2 = (int(round(v)) for v in b["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    boxes_png = os.path.join(args.outdir, f"{stem}_boxes.png")
    mask_png = os.path.join(args.outdir, f"{stem}_mask.png")
    mask_img = mask_full if mask_full is not None else np.zeros((H, W), np.uint8)
    # Saving a full 38MP PNG is slow; downscale the previews so the save stage
    # doesn't dominate the timing (boxes are still placed at full-res accuracy).
    if args.preview_max and max(H, W) > args.preview_max:
        sc = args.preview_max / max(H, W)
        vis = cv2.resize(vis, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
        mask_img = cv2.resize(mask_img, (int(W * sc), int(H * sc)),
                              interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(boxes_png, vis)
    cv2.imwrite(mask_png, mask_img)
    prof.done()

    # --- report ---
    print(f"\nimage: {W0}x{H0}" + (f" -> {W}x{H} (downscaled)" if (W, H) != (W0, H0) else ""))
    if args.hsv_lower and args.hsv_upper:
        print(f"colour: forced HSV band {bands[0][0]} .. {bands[0][1]}")
    else:
        print(f"colour: auto-detected {len(bands)} hue band(s):")
        for lo, hi in bands:
            print(f"          HSV {lo} .. {hi}")
    mode = f"tiled ({total} tiles)" if use_tiling else "single full-image pass"
    print(f"detect mode: {mode}")
    print(f"components: {raw_count} pooled boxes   ->   merge ({args.merge}): {len(kept)} kept")
    print("\n=== where the time went ===")
    prof.summary()
    print("\n=== detect-stage breakdown (summed over tiles) ===")
    print(f"  [{mask_ms_tot:8.1f} ms]  colour mask + morphology (GPU)")
    print(f"  [{cc_ms_tot:8.1f} ms]  connected components (backend={args.cc})")
    print(f"\nsaved: {boxes_png}")
    print(f"saved: {mask_png}")
    return boxes_png, mask_png


def main():
    ap = argparse.ArgumentParser(description="Run the GPU booth pipeline on a real image/PDF")
    ap.add_argument("--input", required=True, help="path to an image (png/jpg) or .pdf")
    ap.add_argument("--page", type=int, default=0, help="PDF page index (0-based)")
    ap.add_argument("--dpi", type=int, default=200, help="PDF rasterisation DPI")
    ap.add_argument("--downscale", type=float, default=1.0, help="resize factor (<1 speeds up label-prop CC)")
    ap.add_argument("--tile", type=int, default=1800,
                    help="tile size in px for the dense pass (0 = single full-image pass)")
    ap.add_argument("--overlap", type=int, default=400,
                    help="overlap between adjacent tiles in px (a booth in the seam is held by the neighbour)")
    ap.add_argument("--seam-edge", type=int, default=4,
                    help="drop boxes within this many px of an INNER tile seam (neighbour holds them whole)")
    ap.add_argument("--colors", type=int, default=3, help="how many dominant hue bands to auto-detect")
    ap.add_argument("--hsv-lower", default=None, help="force one HSV lower bound 'H,S,V' (0..179,0..255,0..255)")
    ap.add_argument("--hsv-upper", default=None, help="force one HSV upper bound 'H,S,V'")
    ap.add_argument("--min-area", type=int, default=300, help="drop components smaller than this (px^2)")
    ap.add_argument("--cc", choices=["gpu", "cpu"], default="gpu",
                    help="connected-components backend: gpu label-prop, or cpu cv2 (faster on big images)")
    ap.add_argument("--max-iters", type=int, default=256, help="label-prop rounds cap (raise for big cells)")
    ap.add_argument("--preview-max", type=int, default=2400,
                    help="longest side of saved preview PNGs (0 = full res); keeps save fast")
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
