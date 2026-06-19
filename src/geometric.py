"""
Geometric (line-based cell-extraction) booth pass — ported VERBATIM from prod's
app/pipeline/detectors/booth_detector.py so the GPU runner has the same third
detector the production EnsembleDetector uses (source "opencv_strict").

Prod runs this pass PER TILE inside the ensemble. The only change here is that
`detect_array(bgr, p)` takes an in-memory BGR crop instead of a file path (the
GPU runner already holds every tile as a numpy array), and OCR is dropped —
labels come from the PDF text layer downstream (see text_recover.py), not
tesseract.

Algorithm (unchanged): booths are RECTANGULAR CELLS found by ELIMINATION —
foreground = NOT(page background OR walkway OR dark border), cut along internal
divider lines, eroded, then connected-component labelled and shape-filtered. A
localized `_subdivide` splits any large fused block along its own interior
dividers, which is the piece that recovers small abutting sub-booths.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class GeoParams:
    # working resolution
    target_long_side: int = 2300
    max_upscale: float = 3.0
    # background flood
    dark_barrier: int = 130
    tight_flood_pagebg: float = 0.68
    tight_flood_sat: float = 0.18
    tight_flood_tol: int = 12
    # internal cut lines
    cut_contrast: int = 22
    bright_contrast: int = 255
    line_len_frac: float = 0.03
    color_edge_sat: int = 55
    close_gap: int = 3
    # gray walkway
    gray_walk_v_lo: int = 95
    gray_walk_v_hi: int = 205
    gray_walk_frac: float = 0.015
    # cell geometry filters (fractions of WORKING image area)
    min_area_frac: float = 1e-4
    max_area_frac: float = 0.06
    min_side_px: int = 20
    min_rect_score: float = 0.60
    min_aspect: float = 0.14
    # tilt
    enable_tilt: bool = True
    tilt_min_deg: float = 8.0
    tilt_max_deg: float = 82.0
    # localized sub-booth splitting
    enable_subdivide: bool = True
    subdiv_min_area_frac: float = 1.2e-3
    subdiv_min_side: int = 28
    subdiv_contrast: int = 16
    subdiv_span: float = 0.45
    subdiv_bridge: int = 9
    # bright-cell path
    enable_bright: bool = True
    bright_cell_contrast: int = 10


# --------------------------------------------------------------------------- #
def _scale(bgr, p: GeoParams):
    h, w = bgr.shape[:2]
    long_side = max(h, w)
    if long_side < p.target_long_side:
        s = min(p.max_upscale, p.target_long_side / long_side)
        interp = cv2.INTER_CUBIC
    elif long_side > p.target_long_side * 1.6:
        s = p.target_long_side / long_side
        interp = cv2.INTER_AREA
    else:
        return bgr, 1.0
    return cv2.resize(bgr, (int(round(w * s)), int(round(h * s))), interpolation=interp), s


def _page_background(bgr, walkway, p: GeoParams):
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    unsat_V = V[S < 25]
    bg_val = int(np.percentile(unsat_V, 80)) if unsat_V.size > 1000 else 245
    bg_val = int(np.clip(bg_val, 200, 255))
    dark = (V < p.dark_barrier).astype(np.uint8) * 255
    barriers = cv2.bitwise_or(dark, walkway)
    barriers = cv2.dilate(barriers, np.ones((3, 3), np.uint8), 1)
    trav = ((V >= bg_val - 60) & (S < 40)).astype(np.uint8) * 255
    trav = cv2.bitwise_and(trav, cv2.bitwise_not(barriers))
    flood = trav.copy()
    ff = np.zeros((H + 2, W + 2), np.uint8)
    step = max(1, min(H, W) // 120)
    for x in range(0, W, step):
        for y in (0, H - 1):
            if trav[y, x] and flood[y, x] == 255:
                cv2.floodFill(flood, ff, (x, y), 128)
    for y in range(0, H, step):
        for x in (0, W - 1):
            if trav[y, x] and flood[y, x] == 255:
                cv2.floodFill(flood, ff, (x, y), 128)
    loose = (flood == 128).astype(np.uint8) * 255
    sat_frac = float((S > 40).mean())
    if (loose > 0).mean() > p.tight_flood_pagebg and sat_frac < p.tight_flood_sat:
        ff2 = np.zeros((H + 2, W + 2), np.uint8)
        ff2[1:-1, 1:-1] = (barriers > 0).astype(np.uint8)
        img = bgr.copy()
        flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
        t = p.tight_flood_tol
        for x in range(0, W, step):
            for y in (0, H - 1):
                if ff2[y + 1, x + 1] == 0 and S[y, x] < 40 and V[y, x] >= bg_val - 50:
                    cv2.floodFill(img, ff2, (x, y), 0, (t,) * 3, (t,) * 3, flags)
        for y in range(0, H, step):
            for x in (0, W - 1):
                if ff2[y + 1, x + 1] == 0 and S[y, x] < 40 and V[y, x] >= bg_val - 50:
                    cv2.floodFill(img, ff2, (x, y), 0, (t,) * 3, (t,) * 3, flags)
        return (ff2[1:-1, 1:-1] == 255).astype(np.uint8) * 255, bg_val
    return loose, bg_val


def _cut_lines(bgr, p: GeoParams, line_len=None):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 31)
    darker = cv2.subtract(bg, gray)
    dark_mask = (darker > p.cut_contrast).astype(np.uint8) * 255
    brighter = cv2.subtract(gray, bg)
    bright_mask = (brighter > p.bright_contrast).astype(np.uint8) * 255
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.float32)
    sob = cv2.magnitude(cv2.Sobel(S, cv2.CV_32F, 1, 0, ksize=3),
                        cv2.Sobel(S, cv2.CV_32F, 0, 1, ksize=3))
    col_edge = (sob > p.color_edge_sat * 4).astype(np.uint8) * 255
    raw = cv2.bitwise_or(cv2.bitwise_or(dark_mask, bright_mask), col_edge)
    H, W = gray.shape
    llf = line_len if line_len is not None else p.line_len_frac
    L = max(8, int(llf * min(H, W)))
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))
    horiz = cv2.morphologyEx(raw, cv2.MORPH_OPEN, hk)
    vert = cv2.morphologyEx(raw, cv2.MORPH_OPEN, vk)
    grid = cv2.bitwise_or(horiz, vert)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (p.close_gap, p.close_gap))
    return cv2.morphologyEx(grid, cv2.MORPH_CLOSE, k, iterations=1)


def _gray_walkway(bgr, p: GeoParams):
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    gray = ((S < 30) & (V >= p.gray_walk_v_lo) & (V < p.gray_walk_v_hi)).astype(np.uint8) * 255
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(gray, 8)
    if n <= 1:
        return np.zeros((H, W), np.uint8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[big, cv2.CC_STAT_AREA] < p.gray_walk_frac * H * W:
        return np.zeros((H, W), np.uint8)
    return (lab == big).astype(np.uint8) * 255


def _fill_holes(mask, max_hole):
    H, W = mask.shape
    inv = cv2.bitwise_not(mask)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    out = mask.copy()
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if x == 0 or y == 0 or x + w >= W or y + h >= H:
            continue
        if a <= max_hole:
            out[lab == i] = 255
    return out


def _extract_booths(bgr, p: GeoParams, source="axis", line_len=None):
    H, W = bgr.shape[:2]
    total = H * W
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S, V = hsv[..., 1], hsv[..., 2]
    color_wk = np.zeros((H, W), np.uint8)
    gray_wk = _gray_walkway(bgr, p)
    walkway = cv2.bitwise_or(color_wk, gray_wk)
    page_bg, bg_val = _page_background(bgr, walkway, p)
    cuts = _cut_lines(bgr, p, line_len=line_len)
    dark = (V < p.dark_barrier).astype(np.uint8) * 255
    excluded = cv2.bitwise_or(cv2.bitwise_or(page_bg, walkway), dark)
    fg = cv2.bitwise_not(excluded)
    fg = _fill_holes(fg, 6e-4 * total)
    fg = cv2.bitwise_and(fg, cv2.bitwise_not(cuts))
    fg = cv2.erode(fg, np.ones((3, 3), np.uint8), 1)
    min_area = p.min_area_frac * total
    max_area = p.max_area_frac * total
    n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, 4)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        if min(w, h) < p.min_side_px:
            continue
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
            continue
        rect = area / float(w * h)
        if rect < p.min_rect_score:
            continue
        if min(w, h) / max(w, h) < p.min_aspect:
            continue
        out.append({"bbox": (int(x), int(y), int(w), int(h)), "area": float(area),
                    "centroid": (float(cent[i][0]), float(cent[i][1])), "source": source})
    return out, bg_val


def _bright_cells(bgr, p: GeoParams, line_len=None):
    H, W = bgr.shape[:2]
    total = H * W
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    floor = cv2.medianBlur(gray, max(31, (min(H, W) // 20) | 1))
    bright = (cv2.subtract(gray, floor) > p.bright_cell_contrast).astype(np.uint8) * 255
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cuts = _cut_lines(bgr, p, line_len=line_len)
    fg = cv2.bitwise_and(bright, cv2.bitwise_not(cuts))
    fg = _fill_holes(fg, 6e-4 * total)
    fg = cv2.erode(fg, np.ones((3, 3), np.uint8), 1)
    mn, mx = p.min_area_frac * total, p.max_area_frac * total
    n, lab, stats, cent = cv2.connectedComponentsWithStats(fg, 4)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < mn or a > mx or min(w, h) < p.min_side_px:
            continue
        if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
            continue
        if a / float(w * h) < p.min_rect_score or min(w, h) / max(w, h) < p.min_aspect:
            continue
        out.append({"bbox": (int(x), int(y), int(w), int(h)), "area": float(a),
                    "centroid": (float(cent[i][0]), float(cent[i][1])), "source": "bright"})
    return out


def _iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = aw * ah + bw * bh - inter
    return inter / u if u else 0.0


def _dedupe_prefer_fine(items, iou_t=0.55, cover_t=0.80):
    items = sorted(items, key=lambda c: c["area"])
    kept = []
    for it in items:
        if any(_iou(it["bbox"], k["bbox"]) > iou_t for k in kept):
            continue
        kept.append(it)
    final = []
    kept_sorted = sorted(kept, key=lambda c: c["area"])
    for it in kept_sorted:
        smaller = [k for k in kept_sorted if k["area"] < it["area"] * 0.9]
        x, y, w, h = it["bbox"]
        if smaller and w * h > 0:
            acc = np.zeros((max(1, h // 4), max(1, w // 4)), np.uint8)
            for s in smaller:
                sx, sy, sw, sh = s["bbox"]
                ix1, iy1 = max(x, sx), max(y, sy)
                ix2, iy2 = min(x + w, sx + sw), min(y + h, sy + sh)
                if ix2 > ix1 and iy2 > iy1:
                    acc[(iy1 - y) // 4:(iy2 - y) // 4, (ix1 - x) // 4:(ix2 - x) // 4] = 1
            if acc.mean() > cover_t:
                continue
        final.append(it)
    return final


def _subdivide(bgr, cands, p: GeoParams):
    H, W = bgr.shape[:2]
    total = H * W
    min_area = p.min_area_frac * total
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = []
    for c in cands:
        x, y, w, h = c["bbox"]
        if w * h < p.subdiv_min_area_frac * total or min(w, h) < p.subdiv_min_side:
            out.append(c)
            continue
        sub = gray[y:y + h, x:x + w]
        if sub.size < 100:
            out.append(c)
            continue
        sh, sw = sub.shape
        bg = cv2.medianBlur(sub, max(11, (min(sh, sw) // 4) | 1))
        dark = cv2.subtract(bg, sub)
        lines = (dark > p.subdiv_contrast).astype(np.uint8) * 255
        Lh = max(8, int(p.subdiv_span * sw))
        Lv = max(8, int(p.subdiv_span * sh))
        bridge = p.subdiv_bridge
        hc = cv2.morphologyEx(lines, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (bridge, 1)))
        vc = cv2.morphologyEx(lines, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, bridge)))
        horiz = cv2.morphologyEx(hc, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (Lh, 1)))
        vert = cv2.morphologyEx(vc, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (1, Lv)))
        grid = cv2.bitwise_or(horiz, vert)
        if (grid > 0).mean() < 1e-3:
            out.append(c)
            continue
        cells = cv2.bitwise_not(grid)
        cells = cv2.erode(cells, np.ones((3, 3), np.uint8), 1)
        n, lab, stats, cent = cv2.connectedComponentsWithStats(cells, 4)
        pieces = []
        for i in range(1, n):
            cx, cy, cw, ch, ca = stats[i]
            if ca < min_area or ca < 0.12 * (sw * sh):
                continue
            if min(cw, ch) < p.min_side_px:
                continue
            if ca / float(cw * ch) < p.min_rect_score:
                continue
            pieces.append({"bbox": (int(x + cx), int(y + cy), int(cw), int(ch)),
                           "area": float(ca),
                           "centroid": (x + cent[i][0], y + cent[i][1]),
                           "source": c["source"]})
        if len(pieces) >= 2 and sum(pp["area"] for pp in pieces) > 0.45 * w * h:
            out.extend(pieces)
        else:
            out.append(c)
    return out


def _dominant_tilt(strong, p: GeoParams):
    H, W = strong.shape
    lines = cv2.HoughLinesP(strong, 1, np.pi / 180, threshold=int(0.10 * min(H, W)),
                            minLineLength=int(0.05 * min(H, W)), maxLineGap=8)
    if lines is None:
        return None
    angs = []
    for l in lines[:, 0]:
        a = math.degrees(math.atan2(l[3] - l[1], l[2] - l[0])) % 180
        if a > 90:
            a -= 180
        angs.append(a)
    angs = np.array(angs)
    tilt = angs[(np.abs(angs) > p.tilt_min_deg) & (np.abs(angs) < p.tilt_max_deg)]
    if len(tilt) < max(20, 0.10 * len(angs)):
        return None
    hist, edges = np.histogram(tilt, bins=np.arange(-90, 91, 3))
    b = int(np.argmax(hist))
    if hist[b] < 20:
        return None
    return float((edges[b] + edges[b + 1]) / 2)


def _rotate(img, deg):
    H, W = img.shape[:2]
    c = (W / 2, H / 2)
    M = cv2.getRotationMatrix2D(c, deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(H * sin + W * cos), int(H * cos + W * sin)
    M[0, 2] += (nW - W) / 2
    M[1, 2] += (nH - H) / 2
    return cv2.warpAffine(img, M, (nW, nH), flags=cv2.INTER_LINEAR,
                          borderValue=(255, 255, 255)), M


def detect_array(bgr, p: Optional[GeoParams] = None, source="opencv_strict"):
    """Geometric booth detection on an in-memory BGR crop.

    Returns a list of {"bbox": (x, y, w, h), "area", "centroid", "coords",
    "source"} in INPUT-image pixel coordinates (scale is undone here)."""
    if p is None:
        p = GeoParams()
    if bgr is None or bgr.size == 0:
        return []
    bgr, scale = _scale(bgr, p)
    cands = []
    for ll in (p.line_len_frac, p.line_len_frac * 0.55):
        c, _ = _extract_booths(bgr, p, "axis", line_len=ll)
        cands += c
    if p.enable_bright:
        cands += _bright_cells(bgr, p, line_len=p.line_len_frac)
    if p.enable_tilt:
        ang = _dominant_tilt(_cut_lines(bgr, p), p)
        if ang is not None and abs(ang) > p.tilt_min_deg:
            rot, M = _rotate(bgr, ang)
            rc, _ = _extract_booths(rot, p, "tilt")
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
    cands = _dedupe_prefer_fine(cands)
    if p.enable_subdivide:
        cands = _subdivide(bgr, cands, p)
        cands = _dedupe_prefer_fine(cands)
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
