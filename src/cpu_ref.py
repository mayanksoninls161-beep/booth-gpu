"""
CPU reference implementations — the *exact* primitives the production booth
pipeline uses today, copied here verbatim so the GPU port can be checked for
parity against the real thing (not against a paraphrase of it).

Three hot stages are mirrored:

  1. color mask + morphology   -> `color_mask_cpu`        (cv2.inRange + morphologyEx)
  2. contours/components -> boxes -> `components_boxes_cpu` (cv2.connectedComponentsWithStats)
  3. dedup / merge (NMS)       -> `non_max_suppression`   (verbatim from
                                  app/pipeline/utils/geometry.py)

`bench.py` times each of these against its GPU counterpart and
`tests/test_parity.py` asserts the outputs agree.

cv2 / numpy are imported lazily so the pure-bbox NMS path can be exercised even
in an environment without OpenCV installed.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2  # noqa: F401
    _HAVE_CV2 = True
except Exception:  # pragma: no cover - cv2 optional for bbox-only path
    _HAVE_CV2 = False


# --------------------------------------------------------------------------- #
# Geometry — verbatim from app/pipeline/utils/geometry.py
# --------------------------------------------------------------------------- #
def _poly_area(poly):
    if not _HAVE_CV2:
        raise RuntimeError("polygon path needs OpenCV (cv2) installed")
    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    return float(abs(cv2.contourArea(p)))


def polygon_iou(poly1, poly2):
    a = np.asarray(poly1, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(poly2, dtype=np.float32).reshape(-1, 2)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    try:
        inter, _ = cv2.intersectConvexConvex(a, b)
    except Exception:
        return 0.0
    if inter <= 0:
        return 0.0
    ua = _poly_area(a) + _poly_area(b) - inter
    return float(inter / ua) if ua > 0 else 0.0


def polygon_overlap(poly1, poly2):
    a = np.asarray(poly1, dtype=np.float32).reshape(-1, 2)
    b = np.asarray(poly2, dtype=np.float32).reshape(-1, 2)
    if len(a) < 3 or len(b) < 3:
        return 0.0, 0.0
    try:
        inter, _ = cv2.intersectConvexConvex(a, b)
    except Exception:
        return 0.0, 0.0
    if inter <= 0:
        return 0.0, 0.0
    a1, a2 = _poly_area(a), _poly_area(b)
    union = a1 + a2 - inter
    iou = float(inter / union) if union > 0 else 0.0
    m = min(a1, a2)
    ios = float(inter / m) if m > 0 else 0.0
    return iou, ios


def _bbox_overlap(box1, box2):
    iou = calculate_iou(box1, box2)
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return iou, 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    m = min(a1, a2)
    ios = float(inter / m) if m > 0 else 0.0
    return iou, ios


def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / float(union_area) if union_area > 0 else 0.0


def non_max_suppression(boxes, iou_threshold=0.3, containment_threshold=0.7):
    """VERBATIM copy of the production NMS (greedy, with merged-block pre-filter
    and containment eviction). This is the ground truth the GPU merge is graded
    against. See app/pipeline/utils/geometry.py for the original docstring."""
    if not boxes:
        return []

    def _area(b):
        x1, y1, x2, y2 = b['bbox']
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    valid_boxes = []
    for big in boxes:
        covered, n_inside = 0.0, 0
        is_false_split = False
        for small in boxes:
            if small is big or _area(small) >= _area(big):
                continue
            if 'poly' in small and 'poly' in big:
                _, ios = polygon_overlap(small['poly'], big['poly'])
            else:
                _, ios = _bbox_overlap(small['bbox'], big['bbox'])
            if ios > 0.80:
                w, h = small['bbox'][2] - small['bbox'][0], small['bbox'][3] - small['bbox'][1]
                aspect = max(w, h) / max(1.0, float(min(w, h)))
                if aspect > 2.5 and _area(small) < 15000:
                    is_false_split = True
                covered += ios * _area(small)
                n_inside += 1
        if n_inside >= 2 and covered >= 0.75 * _area(big) and not is_false_split:
            continue  # big is a true merged block, drop it
        valid_boxes.append(big)

    sorted_boxes = sorted(valid_boxes, key=lambda x: (x.get('score', 1.0), _area(x)),
                          reverse=True)
    kept_boxes = []
    for current in sorted_boxes:
        should_keep = True
        evict_idx = None
        for i, kept in enumerate(kept_boxes):
            if 'poly' in current and 'poly' in kept:
                iou, ios = polygon_overlap(current['poly'], kept['poly'])
            else:
                iou, ios = _bbox_overlap(current['bbox'], kept['bbox'])
            if iou > iou_threshold:
                should_keep = False
                break
            if ios > containment_threshold:
                if _area(current) > _area(kept):
                    evict_idx = i
                else:
                    should_keep = False
                    break
        if not should_keep:
            continue
        if evict_idx is not None:
            kept_boxes[evict_idx] = current
        else:
            kept_boxes.append(current)
    return kept_boxes


# --------------------------------------------------------------------------- #
# Stage 1 reference — color mask + morphology (cv2)
# --------------------------------------------------------------------------- #
def color_mask_cpu(img_bgr, lower, upper, open_ksize=3, close_ksize=3):
    """cv2.inRange + morphological open then close. Returns uint8 {0,255} mask.

    `lower`/`upper` are inclusive per-channel bounds in the SAME channel order as
    `img_bgr` (so BGR if you pass a BGR image). Mirrors the production color pass.
    """
    if not _HAVE_CV2:
        raise RuntimeError("color_mask_cpu needs OpenCV (cv2) installed")
    lower = np.asarray(lower, dtype=np.uint8)
    upper = np.asarray(upper, dtype=np.uint8)
    mask = cv2.inRange(img_bgr, lower, upper)
    if open_ksize and open_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_ksize, open_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_ksize and close_ksize > 1:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


# --------------------------------------------------------------------------- #
# Stage 2 reference — connected components -> boxes (cv2)
# --------------------------------------------------------------------------- #
def components_boxes_cpu(mask, min_area=1, connectivity=8):
    """cv2.connectedComponentsWithStats on a {0,255} or {0,1} mask.

    Returns an (M,4) int array of [x1,y1,x2,y2] boxes (background label 0
    excluded), sorted for a stable comparison. This is the moral equivalent of
    `findContours -> boundingRect`, which is the CPU-only serial step the GPU
    label-propagation path replaces.
    """
    if not _HAVE_CV2:
        raise RuntimeError("components_boxes_cpu needs OpenCV (cv2) installed")
    m = (np.asarray(mask) > 0).astype(np.uint8)
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(m, connectivity)
    out = []
    for i in range(1, n):  # skip background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        out.append([int(x), int(y), int(x + w), int(y + h)])
    if not out:
        return np.zeros((0, 4), dtype=np.int64)
    arr = np.asarray(out, dtype=np.int64)
    order = np.lexsort((arr[:, 3], arr[:, 2], arr[:, 1], arr[:, 0]))
    return arr[order]
