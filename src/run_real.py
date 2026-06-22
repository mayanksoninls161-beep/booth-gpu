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
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from gpu_ops import image_to_tensor, color_mask_pipeline_gpu
from gpu_components import components_boxes_gpu
from gpu_merge import nms_gpu_cluster, nms_gpu_assisted
import cpu_ref
import geometric
import text_recover
import roboflow_hall


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


def _area_cap(boxes, cw, ch, max_area_frac):
    """Drop boxes bigger than max_area_frac of the tile. The bordered pass's
    inter-booth background (aisles) is one giant 'cell' that must be discarded;
    real dense booths are only a tiny fraction of a tile."""
    if not max_area_frac or max_area_frac <= 0:
        return boxes
    cap = float(max_area_frac) * float(cw) * float(ch)
    return [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) <= cap]


def _eff_min_area(cw, ch, args):
    """Effective component floor for a crop. Prod scales the bordered floor with
    the rendered size (bordered_min_area_frac of the crop area) so a small booth
    survives at any DPI; --min-area is a hard pixel floor underneath that."""
    eff = int(args.min_area)
    if args.min_area_frac and args.min_area_frac > 0:
        eff = max(eff, int(round(args.min_area_frac * cw * ch)))
    return eff


def _color_boxes(crop_bgr, bands, device, args, min_area):
    """HSV colour-band mask -> morphology -> components (GPU, or cv2 via --cc cpu).

    Keys on the fill COLOUR, so it owns the big colour-filled halls -- but it fuses
    same-colour neighbours whenever --close-ksize bridges the thin separator line
    between them, and it misses pale / low-saturation / white cells entirely.
    Returns (boxes_xyxy, mask_u8, mask_ms, cc_ms)."""
    import cv2
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

    _sync(); t1 = time.perf_counter()
    if args.cc == "cpu":
        boxes = cpu_ref.components_boxes_cpu(
            (mask[0, 0].cpu().numpy() * 255).astype(np.uint8),
            min_area=min_area).tolist()
    else:
        boxes = components_boxes_gpu(mask[0, 0], min_area=min_area,
                                     max_iters=args.max_iters).cpu().tolist()
    _sync(); cc_ms = (time.perf_counter() - t1) * 1e3
    mnp = (mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
    return boxes, mnp, mask_ms, cc_ms


def _bordered_boxes(crop_bgr, args, min_area):
    """Border-keyed cell detection: the cells are the regions ENCLOSED by the dark
    grid lines, so EVERY booth -- pale, white, or coloured -- is found, and two
    touching booths stay separate because the dark line between them is background.

    cv2 components are used here (NOT the GPU label-prop): the inverted mask's
    inter-booth background is one giant component whose pixel-diameter is far past
    `max_iters`, so label-prop would truncate and mislabel; a single cv2 scan is
    both correct and size-independent.  Returns (boxes_xyxy, line_mask_u8, ...)."""
    import cv2
    _sync(); t0 = time.perf_counter()
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lines = (gray < args.line_thresh).astype(np.uint8)        # borders + text
    if args.seal_ksize and args.seal_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT,
                                      (args.seal_ksize, args.seal_ksize))
        lines = cv2.dilate(lines, k)                          # seal 1px gaps
    cells = ((lines == 0).astype(np.uint8)) * 255             # between the lines
    mask_ms = (time.perf_counter() - t0) * 1e3

    t1 = time.perf_counter()
    boxes = cpu_ref.components_boxes_cpu(cells, min_area=min_area).tolist()
    cc_ms = (time.perf_counter() - t1) * 1e3
    return boxes, lines * 255, mask_ms, cc_ms


def _mode_list(mode):
    """Which detector passes a mode runs."""
    if mode == "all":
        return ("color", "bordered", "geometric")
    if mode == "both":
        return ("color", "bordered")
    return (mode,)


def _geometric_boxes(crop_bgr, args, geo_params):
    """Geometric line-based cell extraction (prod source 'opencv_strict').
    Returns (boxes_as_source_dicts, mask_ms, cc_ms); boxes are CROP-LOCAL with a
    'source' tag and optional oriented 'coords'."""
    backend = getattr(args, "geo_backend", "cpu")
    if backend == "auto":
        backend = "gpu" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    if backend == "gpu":
        import geometric_gpu
        dets = geometric_gpu.detect_array_gpu(crop_bgr, geo_params, source="opencv_strict")
    else:
        dets = geometric.detect_array(crop_bgr, geo_params, source="opencv_strict")
    cc_ms = (time.perf_counter() - t0) * 1e3
    out = []
    for d in dets:
        x, y, w, h = d["bbox"]
        out.append({"xyxy": [float(x), float(y), float(x + w), float(y + h)],
                    "source": "opencv_strict", "coords": d.get("coords")})
    return out, 0.0, cc_ms


def _detect_on_crop(crop_bgr, bands, device, args, geo_params=None, want_mask=True):
    """Run the chosen detector(s) on ONE BGR crop (or the whole image).

      --mode color     : HSV colour fills only       (owns big colour-filled halls)
      --mode bordered  : grid-line cells only         (owns dense / pale / white booths)
      --mode geometric : line-based cell extraction    (subdivides fused blocks)
      --mode both       : colour + bordered pooled
      --mode all        : colour + bordered + geometric (full prod ensemble)

    Returns (src_boxes, preview_mask_u8, mask_ms, cc_ms); each src_box is
    {"xyxy":[x1,y1,x2,y2], "source":..., "coords":opt}, CROP-LOCAL."""
    ch, cw = crop_bgr.shape[:2]
    min_area = _eff_min_area(cw, ch, args)
    modes = _mode_list(args.mode)
    boxes = []
    mask_ms = cc_ms = 0.0
    color_prev = line_prev = None
    for md in modes:
        if md == "color":
            bxs, color_prev, m_ms, c_ms = _color_boxes(crop_bgr, bands, device, args, min_area)
            for b in bxs:
                boxes.append({"xyxy": [float(v) for v in b], "source": "color", "coords": None})
        elif md == "bordered":
            bxs, line_prev, m_ms, c_ms = _bordered_boxes(crop_bgr, args, min_area)
            bxs = _area_cap(bxs, cw, ch, args.max_area_frac)
            for b in bxs:
                boxes.append({"xyxy": [float(v) for v in b], "source": "bordered", "coords": None})
        else:  # geometric
            gbxs, m_ms, c_ms = _geometric_boxes(crop_bgr, args, geo_params)
            boxes.extend(gbxs)
        mask_ms += m_ms; cc_ms += c_ms
    preview = line_prev if line_prev is not None else color_prev
    return boxes, preview, mask_ms, cc_ms


# --------------------------------------------------------------------------- #
# Fusion guards ported from prod's EnsembleDetector / tiling.py. Each pool entry
# is {"bbox":[x1,y1,x2,y2], "score":float, "source":str, ...}.
# --------------------------------------------------------------------------- #
def _xyxy_area(b):
    x1, y1, x2, y2 = b["bbox"]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def demerge_with_bordered(pool, img_area, bordered_min_area_frac):
    """Drop a COLOUR/GEOMETRIC box that encloses >=2 BORDERED tiles (each
    0.10-0.70x its area and >0.70 intersection-over-tile-area) -- the colour
    closing fused abutting same-fill booths; the bordered pass tiled them
    correctly, so let the tiles win NMS. Verbatim logic from EnsembleDetector."""
    min_tile = bordered_min_area_frac * img_area if bordered_min_area_frac > 0 else 0.0
    bordered = [b for b in pool if str(b.get("source", "")).startswith("bordered")]
    out, demerged = [], 0
    for b in pool:
        if str(b.get("source", "")).startswith("bordered"):
            out.append(b)
            continue
        bx1, by1, bx2, by2 = b["bbox"]
        barea = (bx2 - bx1) * (by2 - by1)
        tiles = 0
        for t in bordered:
            tx1, ty1, tx2, ty2 = t["bbox"]
            tarea = (tx2 - tx1) * (ty2 - ty1)
            if tarea < min_tile:
                continue
            if not (0.10 * barea <= tarea <= 0.70 * barea):
                continue
            ix = max(0.0, min(tx2, bx2) - max(tx1, bx1))
            iy = max(0.0, min(ty2, by2) - max(ty1, by1))
            if tarea > 0 and (ix * iy) / tarea > 0.70:
                tiles += 1
        if tiles >= 2:
            demerged += 1
            continue
        out.append(b)
    return out, demerged


def drop_contained(booths, ios_thresh=0.6):
    """Drop a box when >= ios_thresh of ITS OWN area sits inside a larger kept
    box (prod tiling._drop_contained). NMS uses IoU, which misses a small box
    nested in a big one."""
    def ios(a, b):
        ax1, ay1, ax2, ay2 = a["bbox"]; bx1, by1, bx2, by2 = b["bbox"]
        ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        s = min(_xyxy_area(a), _xyxy_area(b))
        return inter / s if s > 0 else 0.0
    order = sorted(booths, key=_xyxy_area, reverse=True)
    out = []
    for b in order:
        if any(ios(b, o) >= ios_thresh and _xyxy_area(o) > _xyxy_area(b) for o in out):
            continue
        out.append(b)
    return out


def recover_uncovered_bordered(pool, kept, iou_thresh, ios_thresh):
    """Re-instate bordered boxes that NMS suppressed but which cover a region NO
    kept box occupies. Geometric now outscores bordered (1.0 vs 0.45), so a
    geometric box overlapping a bordered cell wins -- correct for normal grids,
    but it also wipes out micro-cell clusters (e.g. the SHOWCASE strip) and a few
    dense cells that ONLY the bordered pass traces. Prod keeps these (its 372
    bordered survivors sit where geometric didn't reach). This adds back any
    bordered pool box that does not overlap (IoU) or nest inside (IoS) a kept box,
    so it can only raise recall in genuinely empty regions -- never disturb the
    already-matched booths. Returns (kept, n_recovered)."""
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a["bbox"]; bx1, by1, bx2, by2 = b["bbox"]
        ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        u = _xyxy_area(a) + _xyxy_area(b) - inter
        return inter / u if u > 0 else 0.0

    def ios_self(a, b):  # fraction of a's OWN area inside b
        ax1, ay1, ax2, ay2 = a["bbox"]; bx1, by1, bx2, by2 = b["bbox"]
        ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0.0, min(ay2, by2) - max(ay1, by1))
        sa = _xyxy_area(a)
        return (ix * iy) / sa if sa > 0 else 0.0

    import collections
    CELL = 400
    grid = collections.defaultdict(list)
    for k in kept:
        x1, y1, x2, y2 = k["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        grid[(int(cx // CELL), int(cy // CELL))].append(k)
    n_rec = 0
    for b in pool:
        if b.get("source") != "bordered":
            continue
        x1, y1, x2, y2 = b["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        gx, gy = int(cx // CELL), int(cy // CELL)
        neigh = [o for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                 for o in grid.get((gx + dx, gy + dy), [])]
        if any(iou(b, o) >= iou_thresh or ios_self(b, o) >= ios_thresh for o in neigh):
            continue                       # region already covered by a kept box
        kept.append(b)
        grid[(gx, gy)].append(b)           # so two uncovered bordered don't double-add
        n_rec += 1
    return kept, n_rec


def _distinct_label_cells(text_items, B, cell):
    x1, y1, x2, y2 = B["bbox"]
    cells = set()
    for ti in text_items:
        cx, cy = ti["center_px"]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            cells.add((int(cx // cell), int(cy // cell)))
    return len(cells)


def big_region_pass(img, bands, device, args, geo_params, text_items, prof):
    """Full-image pass for halls/stages/standalone-big booths that tiling
    structurally seam-clips (any box wider/taller than the overlap is dropped in
    every tile). Runs colour+bordered(+geometric) SEQUENTIALLY on a downscaled
    copy, keeps only genuinely BIG boxes, NMS, then returns the big candidates.
    Ported from tiling.detect_big_regions."""
    import cv2
    H, W = img.shape[:2]
    page_area = float(W) * float(H)
    long_edge = max(H, W)
    me = args.big_max_edge
    sf = (me / long_edge) if (me and long_edge > me) else 1.0
    small = (cv2.resize(img, (int(round(W * sf)), int(round(H * sf))),
                        interpolation=cv2.INTER_AREA) if sf < 1.0 else img)
    inv = 1.0 / sf
    # big-region uses its own (large) area fractions, not the dense floor
    class _A:  # lightweight args clone with big area params
        pass
    ba = _A()
    ba.__dict__.update(args.__dict__)
    ba.min_area = 1
    ba.min_area_frac = args.big_min_area_frac
    ba.max_area_frac = args.big_max_area_frac
    ba.mode = "all" if args.mode == "all" else "both"
    sboxes, _, _, _ = _detect_on_crop(small, bands, device, ba, geo_params=geo_params)
    raw = []
    for b in sboxes:
        x1, y1, x2, y2 = b["bbox"] if "bbox" in b else b["xyxy"]
        x1, y1, x2, y2 = x1 * inv, y1 * inv, x2 * inv, y2 * inv
        w, h = x2 - x1, y2 - y1
        if min(w, h) < args.big_min_side_px:
            continue
        if w * h > args.big_max_area_frac * page_area:
            continue
        raw.append({"bbox": [x1, y1, x2, y2], "score": float(b.get("score", 1.0)),
                    "source": "bigregion", "coords": None})
    merge_fn = nms_gpu_assisted if args.merge == "assisted" else nms_gpu_cluster
    kept = merge_fn(raw, args.iou, args.containment, device=device)
    for k in kept:
        k.setdefault("source", "bigregion")
    return kept


def merge_big_regions(small, big, text_items, coverage_thresh, max_inner_labels,
                      label_cell_px=250):
    """Arbitrate each big box against the crops: drop it as a HALL CONTAINER if
    crops already cover >= coverage_thresh of it (or it holds > max_inner_labels
    distinct label cells); otherwise keep it as a STANDALONE feature and drop the
    stray crops inside it. Ported from tiling.merge_big_regions."""
    def ov(a_bbox, b_bbox):
        ax1, ay1, ax2, ay2 = a_bbox; bx1, by1, bx2, by2 = b_bbox
        ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0.0, min(ay2, by2) - max(ay1, by1))
        return ix * iy
    drop_inner = set()
    keep_big = []
    for B in big:
        Ba = _xyxy_area(B)
        if Ba <= 0:
            continue
        inner_area = 0.0
        overlapping = []
        for s in small:
            ia = ov(s["bbox"], B["bbox"])
            if ia > 0:
                overlapping.append(s)
                inner_area += ia
        coverage = inner_area / Ba
        n_labels = _distinct_label_cells(text_items, B, label_cell_px)
        if coverage >= coverage_thresh or n_labels > max_inner_labels:
            continue  # container -> drop big, keep crops
        for s in overlapping:
            drop_inner.add(id(s))
        keep_big.append(B)
    out = [s for s in small if id(s) not in drop_inner] + keep_big
    return out, len(keep_big), len(drop_inner)


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


# --- per-source NMS scores (port of prod detector scores) --------------------
# Prod's NMS sorts by (score desc, area desc); each detector stamps a fixed
# score so geometric wins overlaps, then color, then bordered "loses ties"
# (bordered_detector.py: score 0.45 "< color (0.5..1.0) & geometric (1.0)").
# The GPU previously flattened every box to 1.0, which let color/bordered
# survive overlaps prod would discard -- skewing the source mix vs prod.
_SOURCE_SCORE = {
    "opencv_strict": 1.0,   # geometric: highest, wins overlaps
    "color": 0.75,          # prod 0.5+0.5*fill; GPU has no fill -> mid value < 1.0
    "bordered": 0.45,       # prod bordered: loses ties to geometric/color
    "bigregion": 1.0,       # recovered halls/stages: keep
}


def _source_score(source):
    return _SOURCE_SCORE.get(source, 1.0)


# --- false-positive policy (port of prod app/adaptive/labeling.py) -----------
# shape sources are trusted on their own geometry (no text required); the
# geometric pass ("opencv_strict") is kept unless its cell is provably empty.
POLICY_SHAPE_SOURCES = {"color", "bordered", "bigregion"}


def resolve_adaptive(booths):
    """Prod resolve_adaptive: go strict only when the text layer actually carries
    booth numbers (>=15% boothlike, min 10), else fall back to shape so plans
    whose booth IDs the RE_BOOTH regex can't match keep their geometry boxes."""
    n = len(booths)
    n_boothlike = sum(1 for b in booths if b.get("text_status") == "boothlike")
    return "strict" if n_boothlike >= max(10, 0.15 * n) else "shape"


def apply_fp_policy(booths, policy):
    """Filter booths by the resolved policy (port of prod apply_policy)."""
    if policy == "none":
        return booths
    if policy == "strict":
        return [b for b in booths if b.get("text_status") == "boothlike"]
    # shape: trust shape-source geometry; keep opencv_strict unless empty; any
    # other source needs a boothlike label.
    out = []
    for b in booths:
        src = b.get("source", "")
        if src in POLICY_SHAPE_SOURCES:
            out.append(b)
        elif src == "opencv_strict":
            if b.get("text_status", "empty") != "empty":
                out.append(b)
        elif b.get("text_status") == "boothlike":
            out.append(b)
    return out


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import cv2
    print(f"device={device}  input={args.input}")

    # Resolve the GPU connected-components backend once. cuCIM (RAPIDS) is a real
    # union-find CCL that converges in a few passes; label propagation is
    # O(component diameter) and loses to cv2 on big background blobs. 'auto' uses
    # cuCIM when importable, else falls back to propagation (identical results).
    import gpu_components
    want_ccl = "cucim" if args.ccl in ("cucim", "auto") else "prop"
    got_ccl = gpu_components.set_ccl_backend(want_ccl)
    if args.ccl in ("cucim", "auto"):
        if got_ccl == "cucim":
            print("  CCL backend: cuCIM (RAPIDS union-find, GPU)")
        else:
            print("  CCL backend: label-propagation (cuCIM not installed; "
                  "`pip install cupy-cuda12x cucim-cu12` for the fast path)")
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

    # geometric pass params (prod OpenCVDetector tuning; tilt optional per-tile)
    geo_params = geometric.GeoParams(enable_tilt=not args.no_geo_tilt)

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
        # Tile concurrency: the per-tile detect is mostly CPU OpenCV work
        # (geometric/bordered connected-components), so threads give REAL
        # parallelism (OpenCV releases the GIL); the GPU colour mask is shared
        # safely across threads on one CUDA context. The limit is CPU cores, not
        # the GPU -- free Colab T4 boxes typically expose ~2 vCPUs, so the speedup
        # there is ~1.5-2x; a many-core host scales much further. --workers caps it.
        # When the geometric pass runs on the GPU the per-tile detect is
        # GPU-bound, and CPU worker threads only serialize on the single CUDA
        # context (no parallelism, extra contention) -- so default to 1 worker.
        _geo_gpu = args.geo_backend == "gpu" or (
            args.geo_backend == "auto" and torch.cuda.is_available())
        if _geo_gpu and args.workers == 0:
            n_workers = 1
        else:
            n_workers = max(1, args.workers if args.workers > 0
                            else min(8, os.cpu_count() or 4))
        if n_workers > 1:
            try:
                import cv2 as _cv2
                _cv2.setNumThreads(1)   # avoid oversubscribing cores per tile
            except Exception:
                pass
        prof.step(f"tiled detect: mask+CC over {total} tiles ({n_workers} workers)")
        print(f"  tile concurrency: {n_workers} worker thread(s) "
              f"(cpu_count={os.cpu_count()})")
        mask_full = np.zeros((H, W), dtype=np.uint8)
        coords = [(gy0, gx0) for gy0 in ys for gx0 in xs]

        def _work(origin):
            gy0, gx0 = origin
            y2c, x2c = min(gy0 + args.tile, H), min(gx0 + args.tile, W)
            crop = img[gy0:y2c, gx0:x2c]
            ch, cw = crop.shape[:2]
            boxes, mnp, m_ms, c_ms = _detect_on_crop(crop, bands, device, args,
                                                     geo_params=geo_params)
            return gy0, gx0, y2c, x2c, ch, cw, boxes, mnp, m_ms, c_ms

        pool = []
        n_raw = 0
        if n_workers > 1:
            ex = ThreadPoolExecutor(max_workers=n_workers)
            results = ex.map(_work, coords)
        else:
            results = (_work(o) for o in coords)
        # Stitch + offset in the MAIN thread (sequential) so mask_full writes and
        # pool appends are race-free, regardless of worker order.
        for (gy0, gx0, y2c, x2c, ch, cw, boxes, mnp, m_ms, c_ms) in results:
            mask_ms_tot += m_ms; cc_ms_tot += c_ms
            if mnp is not None:                      # stitch (overlap -> max)
                sub = mask_full[gy0:y2c, gx0:x2c]
                np.maximum(sub, mnp, out=sub)
            n_raw += len(boxes)
            for sb in boxes:
                bx1, by1, bx2, by2 = sb["xyxy"]
                if _clipped_xyxy(bx1, by1, bx2, by2, cw, ch, gx0, gy0, W, H,
                                 edge=args.seam_edge):
                    continue                         # neighbour tile holds it whole
                rec = {"bbox": [bx1 + gx0, by1 + gy0, bx2 + gx0, by2 + gy0],
                       "score": _source_score(sb["source"]), "source": sb["source"]}
                if sb.get("coords"):
                    rec["coords"] = [[p[0] + gx0, p[1] + gy0] for p in sb["coords"]]
                pool.append(rec)
        if n_workers > 1:
            ex.shutdown()
        print(f"  tiles: {n_raw} raw boxes across tiles -> {len(pool)} after seam-clip")
    else:
        prof.step("color mask + CC (single full-image pass)")
        boxes, mask_full, m_ms, c_ms = _detect_on_crop(img, bands, device, args,
                                                       geo_params=geo_params)
        mask_ms_tot += m_ms; cc_ms_tot += c_ms
        pool = [{"bbox": [float(v) for v in b["xyxy"]], "score": _source_score(b["source"]),
                 "source": b["source"], **({"coords": b["coords"]} if b.get("coords") else {})}
                for b in boxes]

    raw_count = len(pool)

    img_area = float(W) * float(H)

    # 5a. de-merge: drop a colour/geometric box that the bordered pass already
    #     tiled into >=2 sub-cells (prod EnsembleDetector.demerge_with_bordered).
    demerged = 0
    if args.demerge:
        prof.step("demerge (drop bordered-tiled fused boxes)")
        pool, demerged = demerge_with_bordered(pool, img_area, args.min_area_frac)

    # 5b. merge / dedup (GPU) — fuses overlap-band duplicates + nested fragments,
    #     exactly as the prod global NMS does.
    prof.step(f"merge / dedup ({args.merge}, GPU)")
    merge_fn = nms_gpu_assisted if args.merge == "assisted" else nms_gpu_cluster
    kept = merge_fn(pool, args.iou, args.containment, device=device)
    # IoS containment drop (prod tiling._drop_contained): small box nested in a
    # bigger one survives IoU-NMS, so remove it here.
    kept = drop_contained(kept, args.containment)
    # Recover bordered micro-cells (showcase strips, dense cells) that geometric
    # outscored in NMS but which sit where NO kept box landed -- prod keeps these.
    n_bordered_rec = 0
    if args.recover_bordered:
        kept, n_bordered_rec = recover_uncovered_bordered(
            pool, kept, args.iou, args.containment)
        if n_bordered_rec:
            print(f"  recovered {n_bordered_rec} uncovered bordered micro-cells")
    for k in kept:
        k.setdefault("source", "color")

    # 5c. text-layer extraction (PDF vector text) — feeds big-region arbitration,
    #     labeling, and recovery. Empty for raster inputs.
    text_items = []
    is_pdf = os.path.splitext(args.input)[1].lower() == ".pdf"
    if is_pdf and (args.big_pass or args.text_recover or args.label):
        prof.step("extract PDF text layer (fitz)")
        try:
            text_items = text_recover.extract_text_items_pdf_fitz(args.input, args.dpi, args.page)
        except Exception as e:  # noqa: BLE001
            print(f"  !! text-layer extraction failed ({e}); continuing without text")

    # 5d. big-region pass — recover halls/stages tiling structurally seam-clips,
    #     then arbitrate each against the crops by coverage.
    n_big_kept = n_inner_absorbed = 0
    if args.big_pass:
        prof.step("big-region pass (full-image, downscaled)")
        big = big_region_pass(img, bands, device, args, geo_params, text_items, prof)
        kept, n_big_kept, n_inner_absorbed = merge_big_regions(
            kept, big, text_items, args.big_coverage_thresh, args.big_max_inner_labels)
        print(f"  big-region: {len(big)} candidates -> {n_big_kept} standalone kept, "
              f"{n_inner_absorbed} inner crops absorbed")

    # 5e. label booths from the PDF text, then recover any booth-number token that
    #     no detected box covers (the "are we using the texts?" recall net).
    n_recovered = 0
    if text_items:
        prof.step("label + text recovery")
        text_recover.label_booths(kept, text_items)
        if args.text_recover:
            n_recovered = text_recover.recover_missing(kept, text_items)
        # FP policy (port of prod app/adaptive/labeling.py). "adaptive" is prod's
        # real default: it inspects the boothlike ratio and only goes strict when
        # the text layer actually carries booth numbers, otherwise it falls back
        # to "shape" so non-text plans (e.g. pure-numeric booth IDs the RE_BOOTH
        # regex can't match) keep their geometry boxes.
        before = len(kept)
        policy = args.fp_policy
        if policy == "adaptive":
            policy = resolve_adaptive(kept)
            print(f"  fp-policy adaptive -> resolved '{policy}'")
        kept = apply_fp_policy(kept, policy)
        if policy != "none":
            print(f"  fp-policy '{policy}': {before} -> {len(kept)} booths kept")

    # 5f. Roboflow HALL model (the same model prod runs alongside booths), so the
    #     GPU repo reproduces prod's hall + booth output for a full-pipeline
    #     comparison. Off unless --roboflow-hall (needs ROBOFLOW_HALL_API_KEY +
    #     the `inference` package + network).
    halls = []
    hall_booth_map = None
    if args.roboflow_hall:
        prof.step("roboflow hall model (infer + scale-back)")
        try:
            halls = roboflow_hall.detect_halls(img, conf=args.hall_conf,
                                                max_edge=args.hall_max_edge)
            hall_booth_map = roboflow_hall.build_hall_booth_map(halls, kept)
        except Exception as e:  # noqa: BLE001
            print(f"  !! roboflow hall pass failed ({e}); continuing without halls")

    # 6. annotate + save
    prof.step("annotate + save PNGs")
    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    vis = img.copy()
    for b in kept:
        x1, y1, x2, y2 = (int(round(v)) for v in b["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for hbox in halls:  # halls in magenta, thicker
        cv2.rectangle(vis, (int(hbox["x1"]), int(hbox["y1"])),
                      (int(hbox["x2"]), int(hbox["y2"])), (255, 0, 255), 6)
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

    # full-resolution kept boxes as JSON, for offline verification of the result
    # (the previews are downscaled; this is the exact, full-res truth).
    recs = []
    for b in kept:
        x1, y1, x2, y2 = (float(v) for v in b["bbox"])
        recs.append({"bbox": [x1, y1, x2, y2], "score": float(b.get("score", 1.0)),
                     "w": x2 - x1, "h": y2 - y1, "area": (x2 - x1) * (y2 - y1),
                     "source": b.get("source", ""), "label": b.get("label", ""),
                     "text_status": b.get("text_status", "")})
    json_path = os.path.join(args.outdir, f"{stem}_boxes.json")
    with open(json_path, "w") as f:
        json.dump({"input": os.path.basename(args.input),
                   "image": {"w": W, "h": H},
                   "detector": args.mode,
                   "tiling": (total if use_tiling else 0),
                   "merge": args.merge,
                   "stages": {"demerged": demerged, "big_kept": n_big_kept,
                              "inner_absorbed": n_inner_absorbed,
                              "text_recovered": n_recovered,
                              "n_text_items": len(text_items)},
                   "params": {"tile": args.tile, "overlap": args.overlap,
                              "min_area": args.min_area, "line_thresh": args.line_thresh,
                              "seal_ksize": args.seal_ksize, "max_area_frac": args.max_area_frac,
                              "iou": args.iou, "containment": args.containment},
                   "count": len(recs), "boxes": recs}, f)

    # Roboflow hall output (separate file; the hall->booth map mirrors prod's
    # hall_with_booth_predict response). Only written when the hall pass ran.
    hall_json_path = None
    if args.roboflow_hall:
        hall_json_path = os.path.join(args.outdir, f"{stem}_halls.json")
        with open(hall_json_path, "w") as f:
            json.dump({"input": os.path.basename(args.input),
                       "model": "hall_detection/6",
                       "n_halls": len(halls),
                       "halls": [{k: h[k] for k in ("x1", "y1", "x2", "y2", "area",
                                                     "confidence", "class")} for h in halls],
                       "hall_booth_map": {
                           hk: {"coordinates": hv.get("coordinates"),
                                "n_booths": len(hv.get("booths", []))}
                           for hk, hv in (hall_booth_map or {}).items()}}, f)
    prof.done()

    # --- report ---
    print(f"\nimage: {W0}x{H0}" + (f" -> {W}x{H} (downscaled)" if (W, H) != (W0, H0) else ""))
    if args.hsv_lower and args.hsv_upper:
        print(f"colour: forced HSV band {bands[0][0]} .. {bands[0][1]}")
    else:
        print(f"colour: auto-detected {len(bands)} hue band(s):")
        for lo, hi in bands:
            print(f"          HSV {lo} .. {hi}")
    tiling_desc = f"tiled ({total} tiles)" if use_tiling else "single full-image pass"
    print(f"detector: {args.mode}   |   {tiling_desc}")
    print(f"components: {raw_count} pooled boxes   ->   merge ({args.merge}): {len(kept)} kept")
    by_src = {}
    for r in recs:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print(f"  by source: {by_src}")
    if args.demerge:
        print(f"  demerged (fused boxes dropped): {demerged}")
    if args.big_pass:
        print(f"  big-region: {n_big_kept} standalone kept, {n_inner_absorbed} inner absorbed")
    if text_items:
        n_boothlike = sum(1 for r in recs if r["text_status"] == "boothlike")
        print(f"  text layer: {len(text_items)} spans   |   boothlike-labelled boxes: {n_boothlike}"
              f"   |   text-recovered: {n_recovered}")
    if args.roboflow_hall:
        if hall_booth_map is not None:
            per_hall = {k: len(v.get("booths", [])) for k, v in hall_booth_map.items()}
            print(f"  roboflow halls: {len(halls)} detected   |   booths/hall: {per_hall}")
        else:
            print("  roboflow halls: pass failed (see warning above)")
    if recs:
        a = np.array([r["area"] for r in recs])
        wv = np.array([r["w"] for r in recs])
        hv = np.array([r["h"] for r in recs])
        qs = [0, 5, 25, 50, 75, 95, 100]
        pct = lambda arr: {q: int(np.percentile(arr, q)) for q in qs}
        print("\n=== kept-box size distribution (verify small booths survived) ===")
        print(f"  area  pctl(px^2): {pct(a)}")
        print(f"  width pctl(px)  : {pct(wv)}")
        print(f"  height pctl(px) : {pct(hv)}")
        edges = [0, 500, 1000, 2000, 4000, 8000, 1e18]
        hist, _ = np.histogram(a, bins=edges)
        labels = ["<500", "500-1k", "1k-2k", "2k-4k", "4k-8k", ">8k"]
        print("  area histogram : " + "  ".join(f"{labels[i]}={int(hist[i])}" for i in range(len(hist))))
    print(f"saved JSON: {json_path}")
    print("\n=== where the time went ===")
    prof.summary()
    print("\n=== detect-stage breakdown (summed over tiles) ===")
    print(f"  [{mask_ms_tot:8.1f} ms]  colour mask + morphology (GPU)")
    _geo_b = args.geo_backend if args.geo_backend != "auto" else (
        "gpu" if torch.cuda.is_available() else "cpu")
    print(f"  [{cc_ms_tot:8.1f} ms]  connected components "
          f"(geo={_geo_b}, ccl={gpu_components.get_ccl_backend()})")
    print(f"\nsaved: {boxes_png}")
    print(f"saved: {mask_png}")
    if hall_json_path:
        print(f"saved: {hall_json_path}")
    return boxes_png, mask_png


def main():
    ap = argparse.ArgumentParser(description="Run the GPU booth pipeline on a real image/PDF")
    ap.add_argument("--input", required=True, help="path to an image (png/jpg) or .pdf")
    ap.add_argument("--page", type=int, default=0, help="PDF page index (0-based)")
    ap.add_argument("--dpi", type=int, default=250, help="PDF rasterisation DPI (prod dense = 250)")
    ap.add_argument("--downscale", type=float, default=1.0, help="resize factor (<1 speeds up label-prop CC)")
    ap.add_argument("--tile", type=int, default=1800,
                    help="tile size in px for the dense pass (0 = single full-image pass)")
    ap.add_argument("--overlap", type=int, default=400,
                    help="overlap between adjacent tiles in px (a booth in the seam is held by the neighbour)")
    ap.add_argument("--seam-edge", type=int, default=4,
                    help="drop boxes within this many px of an INNER tile seam (neighbour holds them whole)")
    ap.add_argument("--no-recover-bordered", dest="recover_bordered",
                    action="store_false", default=True,
                    help="disable re-instating bordered micro-cells (showcase/dense "
                         "cells) that geometric outscored in NMS but which cover an "
                         "otherwise-empty region (prod keeps these)")
    ap.add_argument("--workers", type=int, default=0,
                    help="tile detect concurrency (thread pool). 0 = auto "
                         "min(8, cpu_count). The bottleneck is CPU OpenCV CC, not "
                         "the GPU, so the practical ceiling is the host's vCPU count "
                         "(free Colab T4 ~2 cores -> ~1.5-2x; a many-core host scales further)")
    ap.add_argument("--mode", choices=["color", "bordered", "geometric", "both", "all"],
                    default="all",
                    help="detector: colour fills, grid-line cells, geometric line-extraction, "
                         "both (colour+bordered), or all (full prod ensemble; default)")
    ap.add_argument("--no-geo-tilt", action="store_true",
                    help="[geometric] disable the oriented (tilted-hall) pass for speed")
    ap.add_argument("--geo-backend", choices=["cpu", "gpu", "auto"], default="cpu",
                    help="[geometric] CV backend: 'cpu' = stock OpenCV (exact prod), "
                         "'gpu' = CUDA torch reimplementation (geometric_gpu), "
                         "'auto' = gpu when CUDA is available else cpu")
    ap.add_argument("--ccl", choices=["prop", "cucim", "auto"], default="auto",
                    help="GPU connected-components algorithm: 'prop' = label "
                         "propagation (slow on big blobs), 'cucim' = RAPIDS "
                         "union-find (fast; needs cupy+cucim), 'auto' = cucim if "
                         "installed else prop")
    ap.add_argument("--line-thresh", type=int, default=128,
                    help="[bordered] pixels darker than this are grid lines/borders (0..255)")
    ap.add_argument("--seal-ksize", type=int, default=3,
                    help="[bordered] dilate the line mask by this to seal 1px gaps in borders")
    ap.add_argument("--max-area-frac", type=float, default=0.15,
                    help="[bordered] drop cells bigger than this fraction of a tile (the inter-booth background)")
    ap.add_argument("--colors", type=int, default=3, help="how many dominant hue bands to auto-detect")
    ap.add_argument("--hsv-lower", default=None, help="force one HSV lower bound 'H,S,V' (0..179,0..255,0..255)")
    ap.add_argument("--hsv-upper", default=None, help="force one HSV upper bound 'H,S,V'")
    ap.add_argument("--min-area", type=int, default=300, help="hard pixel floor: drop components smaller than this (px^2)")
    ap.add_argument("--min-area-frac", type=float, default=0.0005,
                    help="scaled floor (prod bordered_min_area_frac): drop components smaller than this "
                         "fraction of the CROP area, so small booths survive at any DPI (0 = disable)")
    ap.add_argument("--cc", choices=["gpu", "cpu"], default="gpu",
                    help="colour-pass components backend: gpu label-prop, or cpu cv2 (bordered always uses cv2)")
    ap.add_argument("--max-iters", type=int, default=256, help="label-prop rounds cap (raise for big cells)")
    ap.add_argument("--preview-max", type=int, default=2400,
                    help="longest side of saved preview PNGs (0 = full res); keeps save fast")
    ap.add_argument("--open-ksize", type=int, default=3)
    ap.add_argument("--close-ksize", type=int, default=3,
                    help="colour-mask close kernel (prod dense = 3); lower to 1 if same-colour neighbours fuse")
    ap.add_argument("--merge", choices=["cluster", "assisted"], default="assisted",
                    help="assisted = exact greedy NMS (merged-block pre-filter drops coarse fused blobs); "
                         "cluster = aggressive connected-component grouping")
    ap.add_argument("--iou", type=float, default=0.4, help="merge IoU suppression threshold (prod = 0.4)")
    ap.add_argument("--containment", type=float, default=0.6, help="merge IoS containment threshold (prod = 0.6)")
    # --- full-pipeline stages ported from prod (on by default) ---
    ap.add_argument("--demerge", dest="demerge", action="store_true", default=True,
                    help="drop a colour/geometric box the bordered pass already tiled into >=2 cells")
    ap.add_argument("--no-demerge", dest="demerge", action="store_false")
    ap.add_argument("--big-pass", dest="big_pass", action="store_true", default=True,
                    help="full-image big-region pass to recover halls/stages tiling seam-clips")
    ap.add_argument("--no-big-pass", dest="big_pass", action="store_false")
    ap.add_argument("--big-max-edge", type=int, default=10000,
                    help="[big] downscale the full page so its long edge <= this before the big pass")
    ap.add_argument("--big-min-side-px", type=int, default=700,
                    help="[big] keep only big boxes whose min side >= this (full px)")
    ap.add_argument("--big-min-area-frac", type=float, default=8e-4,
                    help="[big] colour/bordered min-area fraction for the big pass")
    ap.add_argument("--big-max-area-frac", type=float, default=0.5,
                    help="[big] drop big boxes larger than this fraction of the page")
    ap.add_argument("--big-coverage-thresh", type=float, default=0.15,
                    help="[big] crops covering >= this fraction of a big box => it's a hall container (drop)")
    ap.add_argument("--big-max-inner-labels", type=int, default=6,
                    help="[big] a big box holding more distinct label cells than this is a packed grid (drop)")
    ap.add_argument("--text-recover", dest="text_recover", action="store_true", default=True,
                    help="[PDF] synthesise a box at every booth-number token no detected box covers")
    ap.add_argument("--no-text-recover", dest="text_recover", action="store_false")
    ap.add_argument("--label", dest="label", action="store_true", default=True,
                    help="[PDF] attach the PDF text layer to each booth and tag it")
    ap.add_argument("--no-label", dest="label", action="store_false")
    ap.add_argument("--fp-policy", choices=["none", "strict", "shape", "adaptive"],
                    default="adaptive",
                    help="adaptive (prod default): strict when >=15%% of booths are "
                         "boothlike, else shape; strict=only boothlike; shape=trust "
                         "shape-source geometry; none=keep everything")
    # --- Roboflow hall model (same model prod runs alongside booths) ---
    ap.add_argument("--roboflow-hall", dest="roboflow_hall", action="store_true", default=False,
                    help="run the Roboflow hall_detection/6 model (needs ROBOFLOW_HALL_API_KEY env "
                         "+ the `inference` package + network) and build the hall->booth map, "
                         "mirroring prod's hall_with_booth_predict for a full-pipeline comparison")
    ap.add_argument("--hall-conf", type=float, default=0.4, help="[roboflow] hall confidence threshold")
    ap.add_argument("--hall-max-edge", type=int, default=2048,
                    help="[roboflow] downscale the render so its long edge <= this before infer (prod = 2048)")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
