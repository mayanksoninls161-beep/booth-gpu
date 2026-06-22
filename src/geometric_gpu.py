"""
GPU geometric (line-based cell-extraction) booth pass.

This is a CUDA reimplementation of the heavy CV ops in `geometric.py` — the
production "opencv_strict" detector — which in stock OpenCV run entirely on the
CPU and dominate the pipeline's wall-clock:

    cv2.connectedComponentsWithStats   (CPU-only in pip OpenCV; serial)
    cv2.medianBlur(k=31)               (large-kernel, slow)
    cv2.morphologyEx OPEN/CLOSE        (line structuring elements)
    cv2.floodFill                      (inherently serial region grow)
    cv2.Sobel / cv2.cvtColor

How each maps to the GPU (all torch tensors, no CPU round-trip):

  cvtColor BGR->HSV   : the algorithm uses ONLY the V(=max) and S channels,
                        never H, so we compute V and S exactly with elementwise
                        max/min — no approximation. (HSV S matches cv2 to +-1.)
  cvtColor BGR->GRAY  : 0.299R+0.587G+0.114B (cv2 weights), exact.
  connectedComponents : gpu_components.components_stats_gpu (label-propagation
                        max-pool CCL) -> boxes + areas + centroids, vectorised.
  floodFill (border)  : reformulated as "connected components of the traversable
                        mask that touch the image border" — a CCL + isin, fully
                        parallel and equivalent to a flood from every border seed.
  fill_holes          : CCL on the inverted mask, keep interior comps under the
                        hole-area cap (same as the cv2 version, GPU CCL).
  medianBlur(k)       : the EXACT cv2.medianBlur is run on the CPU once per tile
                        and the floor re-uploaded (_median_floor). It is not the
                        bottleneck (CCL is), and a morphological close/open
                        approximation bled bright cells into adjacent darker cells
                        -> 159 small booths silently dropped on IIJS. Bit-exact now.
  Sobel / morphology  : conv2d / rectangular max/avg-pool.

The localized `_subdivide`, tilt detection (HoughLinesP) and `_dedupe_prefer_fine`
stay on the CPU (via geometric.py): they are cheap, rare, or made of many tiny
ops where kernel-launch overhead would erase the GPU win.

Public entry point `detect_array_gpu(bgr, p, source)` matches
`geometric.detect_array` exactly, so it is a drop-in for the runner.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

import geometric  # CPU reference: GeoParams, _scale, _dedupe_prefer_fine, _subdivide, tilt
from geometric import GeoParams
from gpu_components import label_components_gpu, components_stats_gpu


# --------------------------------------------------------------------------- #
# device + channel helpers
# --------------------------------------------------------------------------- #
def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _bgr_to_planes(bgr, device):
    """uint8 HxWx3 BGR -> (B,G,R) float tensors [1,1,H,W] plus V,S,gray.

    Only V(=max) and S are needed downstream (H is never used), so we compute
    them directly and exactly instead of a full HSV conversion.
    """
    t = torch.as_tensor(bgr, device=device).float()          # [H,W,3] BGR
    H, W = t.shape[:2]
    B = t[..., 0].view(1, 1, H, W)
    G = t[..., 1].view(1, 1, H, W)
    R = t[..., 2].view(1, 1, H, W)
    V = torch.maximum(torch.maximum(B, G), R)                 # cv2 HSV V == max
    mn = torch.minimum(torch.minimum(B, G), R)
    diff = V - mn
    S = torch.where(V > 0, torch.round(diff * 255.0 / torch.clamp(V, min=1.0)),
                    torch.zeros_like(V))                      # cv2 HSV S, +-1
    gray = torch.round(0.299 * R + 0.587 * G + 0.114 * B)     # cv2 BGR2GRAY
    return B, G, R, V, S, gray, H, W


# --------------------------------------------------------------------------- #
# morphology primitives (rectangular, border value = 0, matching gpu_ops)
# --------------------------------------------------------------------------- #
def _pad_same(x, kh, kw, value=0.0):
    pt = (kh - 1) // 2; pb = kh - 1 - pt
    pl = (kw - 1) // 2; pr = kw - 1 - pl
    return F.pad(x, (pl, pr, pt, pb), mode="constant", value=value)


def _dilate(mask, kh, kw):
    if kh <= 1 and kw <= 1:
        return mask
    x = _pad_same(mask.float(), kh, kw, 0.0)
    x = F.max_pool2d(x, (kh, kw), stride=1)
    return (x > 0).to(mask.dtype)


def _erode(mask, kh, kw):
    if kh <= 1 and kw <= 1:
        return mask
    x = _pad_same(mask.float(), kh, kw, 0.0)
    s = F.avg_pool2d(x, (kh, kw), stride=1) * (kh * kw)
    return (s >= (kh * kw) - 0.5).to(mask.dtype)


def _open(mask, kh, kw):
    return _dilate(_erode(mask, kh, kw), kh, kw)


def _close(mask, kh, kw):
    return _erode(_dilate(mask, kh, kw), kh, kw)


def _median_floor(gray, k):
    """Exact cv2.medianBlur(k) background floor, computed on CPU and re-uploaded.

    PROD uses medianBlur to estimate a per-pixel background that the line / bright
    cell detectors subtract against. The earlier GPU port approximated it with a
    grayscale morphological close/open, but those BLEED: a k//2-px dilate spreads a
    bright cell's value into an adjacent DARKER-fill cell, so `bg - gray` reads
    large across the *whole* dark cell -> it is treated as a grid line, cut out of
    the foreground, and the booth vanishes. On the IIJS plan this silently dropped
    159 small (~64px) opencv_strict cells that PROD finds with score 1.0. A true
    median ignores thin features without that bleed. medianBlur is NOT the pipeline
    bottleneck (connected-components is), so running the exact cv2 op on CPU and
    uploading the result restores fidelity at negligible cost (~15 ms / tile)."""
    import cv2
    k = int(k) | 1                                   # cv2 requires odd kernel
    g = gray
    if g.dim() == 4:
        g = g[0, 0]
    g_np = torch.clamp(g, 0, 255).to(torch.uint8).cpu().numpy()
    bg = cv2.medianBlur(g_np, k)
    return torch.as_tensor(bg, device=gray.device).to(gray.dtype).view_as(gray)


def _sobel_mag(x):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
                      device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(xp, kx)
    gy = F.conv2d(xp, ky)
    return torch.sqrt(gx * gx + gy * gy)


def _border_touch_mask(binary):
    """Binary mask [1,1,H,W] -> mask of components that touch the image border."""
    lbl = label_components_gpu(binary, connectivity=4)        # [H,W]
    if int((lbl > 0).sum().item()) == 0:
        return torch.zeros_like(binary)
    border = torch.cat([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]])
    border_ids = torch.unique(border[border > 0])
    if border_ids.numel() == 0:
        return torch.zeros_like(binary)
    keep = torch.isin(lbl, border_ids)
    return keep.view_as(binary).to(binary.dtype)


# --------------------------------------------------------------------------- #
# the ported stages
# --------------------------------------------------------------------------- #
def _gray_walkway_gpu(S, V, p, H, W):
    gray = ((S < 30) & (V >= p.gray_walk_v_lo) & (V < p.gray_walk_v_hi)).to(torch.uint8)
    gray = _close(gray, 5, 5)
    st = components_stats_gpu(gray, connectivity=8)
    if st["areas"].numel() == 0:
        return torch.zeros((1, 1, H, W), dtype=torch.uint8, device=S.device)
    big = int(torch.argmax(st["areas"]).item())
    if int(st["areas"][big].item()) < p.gray_walk_frac * H * W:
        return torch.zeros((1, 1, H, W), dtype=torch.uint8, device=S.device)
    return (st["labels"] == (big + 1)).view(1, 1, H, W).to(torch.uint8)


def _cut_lines_gpu(S, gray, p, line_len, H, W, bg=None):
    if bg is None:
        bg = _median_floor(gray, 31)                 # exact cv2.medianBlur(31), dark floor
    darker = bg - gray
    dark_mask = (darker > p.cut_contrast).to(torch.uint8)
    # bright_mask uses bright_contrast=255 in prod -> always empty; skip.
    sob = _sobel_mag(S)
    col_edge = (sob > p.color_edge_sat * 4).to(torch.uint8)
    raw = (dark_mask | col_edge)
    llf = line_len if line_len is not None else p.line_len_frac
    L = max(8, int(llf * min(H, W)))
    horiz = _open(raw, 1, L)                         # cv2 (L,1) == width L, height 1
    vert = _open(raw, L, 1)
    grid = (horiz | vert)
    return _close(grid, p.close_gap, p.close_gap)


def _page_background_gpu(bgr, S, V, walkway, p, H, W):
    """Loose border-flood via CCL. Returns (page_bg_mask[1,1,H,W], bg_val).

    The rare FIXED_RANGE 'tight flood' branch (low-saturation near-empty plans)
    falls back to the exact cv2 implementation."""
    sel = V[S < 25]
    if sel.numel() > 1000:
        bg_val = int(torch.quantile(sel.flatten(), 0.80).item())
    else:
        bg_val = 245
    bg_val = int(np.clip(bg_val, 200, 255))

    dark = (V < p.dark_barrier).to(torch.uint8)
    barriers = (dark | walkway)
    barriers = _dilate(barriers, 3, 3)
    trav = (((V >= bg_val - 60) & (S < 40)).to(torch.uint8)) & (1 - barriers)
    loose = _border_touch_mask(trav)

    sat_frac = float((S > 40).float().mean().item())
    if float((loose > 0).float().mean().item()) > p.tight_flood_pagebg and sat_frac < p.tight_flood_sat:
        # rare path: exact cv2 fallback for the fixed-range flood
        wk_np = (walkway.view(H, W).to(torch.uint8).cpu().numpy() * 255)
        pb_np, bgv = geometric._page_background(bgr, wk_np, p)
        pb = torch.as_tensor((pb_np > 0).astype("uint8"), device=S.device).view(1, 1, H, W)
        return pb, bgv
    return loose, bg_val


def _fill_holes_gpu(mask, max_hole, H, W):
    inv = (1 - mask).to(torch.uint8)
    st = components_stats_gpu(inv, connectivity=4)
    if st["areas"].numel() == 0:
        return mask
    bx = st["boxes"]                                  # x,y,w,h
    x, y, w, h = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]
    touch = (x == 0) | (y == 0) | (x + w >= W) | (y + h >= H)
    fillable = (~touch) & (st["areas"] <= max_hole)
    ids = (torch.nonzero(fillable, as_tuple=False).flatten() + 1)
    if ids.numel() == 0:
        return mask
    fill = torch.isin(st["labels"], ids).view_as(mask).to(mask.dtype)
    return (mask | fill)


def _shape_filter_to_list(st, p, total, H, W, source):
    """Vectorised cv2-stats shape filter -> list of CPU dicts."""
    if st["areas"].numel() == 0:
        return []
    bx = st["boxes"].float()
    x, y, w, h = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]
    area = st["areas"].float()
    cen = st["centroids"]
    min_area = p.min_area_frac * total
    max_area = p.max_area_frac * total
    mnwh = torch.minimum(w, h)
    mxwh = torch.maximum(w, h)
    keep = (area >= min_area) & (area <= max_area) & (mnwh >= p.min_side_px)
    keep &= (x > 2) & (y > 2) & (x + w < W - 2) & (y + h < H - 2)
    keep &= (area / (w * h) >= p.min_rect_score)
    keep &= (mnwh / mxwh >= p.min_aspect)
    idx = torch.nonzero(keep, as_tuple=False).flatten().cpu().numpy()
    bx_c = st["boxes"].cpu().numpy(); area_c = st["areas"].cpu().numpy(); cen_c = cen.cpu().numpy()
    out = []
    for i in idx:
        xi, yi, wi, hi = (int(v) for v in bx_c[i])
        out.append({"bbox": (xi, yi, wi, hi), "area": float(area_c[i]),
                    "centroid": (float(cen_c[i][0]), float(cen_c[i][1])), "source": source})
    return out


def _fg_base_gpu(bgr, S, V, p, H, W):
    """Foreground BEFORE cutting lines — this stage is independent of line_len, so
    it is computed once and reused for every line-length pass. Holds the three
    expensive big-background CCLs (walkway / border-flood / fill_holes)."""
    total = H * W
    walkway = _gray_walkway_gpu(S, V, p, H, W)
    page_bg, _ = _page_background_gpu(bgr, S, V, walkway, p, H, W)
    dark = (V < p.dark_barrier).to(torch.uint8)
    excluded = (page_bg | walkway | dark)
    fg = (1 - excluded).to(torch.uint8)
    return _fill_holes_gpu(fg, 6e-4 * total, H, W)


def _cells_from_base(fg_base, cuts, p, total, H, W, source):
    fg = (fg_base & (1 - cuts)).to(torch.uint8)
    fg = _erode(fg, 3, 3)
    st = components_stats_gpu(fg, connectivity=4)
    return _shape_filter_to_list(st, p, total, H, W, source)


def _extract_booths_gpu(bgr, S, V, gray, p, source, line_len, H, W):
    """Self-contained extract (used by the rare tilt path, which works on a freshly
    rotated crop). The main axis passes use _fg_base_gpu + _cells_from_base."""
    fg_base = _fg_base_gpu(bgr, S, V, p, H, W)
    cuts = _cut_lines_gpu(S, gray, p, line_len, H, W)
    return _cells_from_base(fg_base, cuts, p, H * W, H, W, source)


def _bright_cells_gpu(S, gray, p, line_len, H, W, cuts=None, floor=None):
    total = H * W
    if floor is None:
        k = max(31, (min(H, W) // 20) | 1)
        floor = _median_floor(gray, k)                # exact cv2.medianBlur(k), bright floor
    bright = ((gray - floor) > p.bright_cell_contrast).to(torch.uint8)
    bright = _open(bright, 3, 3)
    if cuts is None:
        cuts = _cut_lines_gpu(S, gray, p, line_len, H, W)
    fg = (bright & (1 - cuts)).to(torch.uint8)
    fg = _fill_holes_gpu(fg, 6e-4 * total, H, W)
    fg = _erode(fg, 3, 3)
    st = components_stats_gpu(fg, connectivity=4)
    return _shape_filter_to_list(st, p, total, H, W, "bright")


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def detect_array_gpu(bgr, p: Optional[GeoParams] = None, source="opencv_strict"):
    """GPU drop-in for geometric.detect_array. Same return shape & coordinates."""
    if p is None:
        p = GeoParams()
    if bgr is None or bgr.size == 0:
        return []
    bgr, scale = geometric._scale(bgr, p)            # cheap CPU resize (exact match)
    device = _device()
    _, _, _, V, S, gray, H, W = _bgr_to_planes(bgr, device)

    total = H * W
    cands = []

    # The walkway / border-flood / fill_holes stages are line_len-independent:
    # compute the foreground base ONCE, then only re-cut lines per length.
    fg_base = _fg_base_gpu(bgr, S, V, p, H, W)
    # medianBlur(31) floor is line_len-independent -> compute the exact cv2 median
    # ONCE and reuse for every cut-line pass (and for bright cells below).
    cut_bg = _median_floor(gray, 31)
    _cut_cache = {}

    def cuts_for(ll):
        key = round(float(ll), 6)
        if key not in _cut_cache:
            _cut_cache[key] = _cut_lines_gpu(S, gray, p, ll, H, W, bg=cut_bg)
        return _cut_cache[key]

    for ll in (p.line_len_frac, p.line_len_frac * 0.55):
        cands += _cells_from_base(fg_base, cuts_for(ll), p, total, H, W, "axis")
    if p.enable_bright:
        cands += _bright_cells_gpu(S, gray, p, p.line_len_frac, H, W,
                                   cuts=cuts_for(p.line_len_frac))

    # tilt detection: rare, HoughLinesP -> keep on CPU via the reference impl
    if p.enable_tilt:
        # tilt uses the default line length, which is already cached
        cuts_np = (cuts_for(p.line_len_frac).view(H, W).to(torch.uint8).cpu().numpy() * 255)
        ang = geometric._dominant_tilt(cuts_np, p)
        if ang is not None and abs(ang) > p.tilt_min_deg:
            import cv2
            rot, M = geometric._rotate(bgr, ang)
            _, _, _, Vr, Sr, grayr, Hr, Wr = _bgr_to_planes(rot, device)
            rc = _extract_booths_gpu(rot, Sr, Vr, grayr, p, "tilt", None, Hr, Wr)
            Minv = cv2.invertAffineTransform(M)
            for c in rc:
                x, y, w, h = c["bbox"]
                pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
                op = (Minv[:, :2] @ pts.T + Minv[:, 2:]).T
                ox1, oy1, ox2, oy2 = op[:, 0].min(), op[:, 1].min(), op[:, 0].max(), op[:, 1].max()
                c["bbox"] = (int(ox1), int(oy1), int(ox2 - ox1), int(oy2 - oy1))
                c["quad"] = [[float(px), float(py)] for px, py in op]
                c["centroid"] = (float(op[:, 0].mean()), float(op[:, 1].mean()))
            cands += rc

    # dedupe + localized subdivide stay on CPU (cheap / many tiny ops)
    cands = geometric._dedupe_prefer_fine(cands)
    if p.enable_subdivide:
        cands = geometric._subdivide(bgr, cands, p)
        cands = geometric._dedupe_prefer_fine(cands)

    inv = 1.0 / scale
    out = []
    for c in cands:
        x, y, w, h = c["bbox"]
        quad = c.get("quad")
        coords = ([[float(px * inv), float(py * inv)] for px, py in quad]
                  if quad is not None else None)
        out.append({"bbox": (int(x * inv), int(y * inv), int(w * inv), int(h * inv)),
                    "area": c["area"] * inv * inv,
                    "centroid": (c["centroid"][0] * inv, c["centroid"][1] * inv),
                    "coords": coords, "source": source})
    return out


# --------------------------------------------------------------------------- #
# fidelity / speed check (run in Colab)
# --------------------------------------------------------------------------- #
def verify_gpu_vs_cpu(bgr, p: Optional[GeoParams] = None, iou_match=0.5):
    """Compare GPU vs CPU geometric on one crop. Returns a dict of match stats."""
    import time
    if p is None:
        p = GeoParams()
    t0 = time.perf_counter(); cpu = geometric.detect_array(bgr, p); t_cpu = time.perf_counter() - t0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter(); gpu = detect_array_gpu(bgr, p)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_gpu = time.perf_counter() - t0

    def iou(a, b):
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        u = aw * ah + bw * bh - inter
        return inter / u if u else 0.0

    matched = 0
    for g in gpu:
        if any(iou(g["bbox"], c["bbox"]) >= iou_match for c in cpu):
            matched += 1
    return {"cpu_boxes": len(cpu), "gpu_boxes": len(gpu),
            "gpu_matched_to_cpu": matched,
            "recall_vs_cpu": matched / max(1, len(cpu)),
            "cpu_ms": round(t_cpu * 1e3, 1), "gpu_ms": round(t_gpu * 1e3, 1),
            "speedup": round(t_cpu / max(1e-6, t_gpu), 2)}
