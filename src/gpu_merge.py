"""
GPU dedup / merge — the single biggest win in the whole pipeline.

Today's CPU merge is `non_max_suppression` (see cpu_ref.py): a greedy double
loop that calls cv2.intersectConvexConvex for every pair, plus an O(N^2)
merged-block pre-filter. On a dense plan the global pool is ~2400 boxes, so that
is ~6 million polygon-intersection calls done one at a time on the CPU. It is not
the math that is slow — it is doing N^2 of it serially in Python.

The fix is to stop porting the loop and change the *shape* of the computation:

  * all-pairs overlap  ->  ONE vectorised N x N tensor op (iou_ios_matrix_gpu)
  * "which boxes are the same booth" -> connected components on the overlap
    graph, computed by repeated boolean matmul (transitive closure) in
    O(log N) GPU steps (nms_gpu_cluster)

Two GPU entry points, deliberately:

  nms_gpu_assisted  -- EXACT same decisions as the production greedy NMS (same
                       pre-filter, same sort, same containment-eviction), but the
                       6M overlap values are computed once on the GPU instead of
                       6M serial cv2 calls. Use this when output must match prod
                       box-for-box. This is what tests/test_parity.py checks.

  nms_gpu_cluster   -- pure-GPU, O(log N): collapse each connected component of
                       the overlap graph to its best member. Fastest path. It is
                       slightly MORE aggressive than greedy NMS (a whole chain of
                       overlaps becomes one box), so it is offered as the "fast"
                       option and bench.py reports how far it diverges.

Both consume the same input the pipeline already builds at
ensemble_detector.py:239 -> a list of {"bbox":[x1,y1,x2,y2], "score":float}.
The polygon ("poly") field is ignored here (axis-aligned path); the assisted
path therefore matches the production bbox fallback exactly, which is the case
that dominates the dense pool.
"""
from __future__ import annotations

import math

import torch


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def boxes_to_tensor(boxes, device=None, dtype=torch.float64):
    """list[{'bbox':[x1,y1,x2,y2],'score':float}] -> (boxes[N,4], scores[N])."""
    if device is None:
        device = default_device()
    bb = [b["bbox"] for b in boxes]
    scores = [float(b.get("score", 1.0)) for b in boxes]
    bt = torch.tensor(bb, dtype=dtype, device=device)
    st = torch.tensor(scores, dtype=dtype, device=device)
    return bt, st


def iou_ios_matrix_gpu(boxes_t):
    """All-pairs IoU and IoS for axis-aligned boxes, as two N x N tensors.

    boxes_t: [N,4] = [x1,y1,x2,y2]. Returns (iou, ios), each [N,N], on the same
    device/dtype. ios = intersection / min(area_i, area_j) (the nested-box test).
    Matches cpu_ref.calculate_iou / _bbox_overlap value-for-value.
    """
    x1 = boxes_t[:, 0]; y1 = boxes_t[:, 1]; x2 = boxes_t[:, 2]; y2 = boxes_t[:, 3]
    w = (x2 - x1).clamp(min=0); h = (y2 - y1).clamp(min=0)
    area = w * h                                            # [N]

    xx1 = torch.maximum(x1[:, None], x1[None, :])
    yy1 = torch.maximum(y1[:, None], y1[None, :])
    xx2 = torch.minimum(x2[:, None], x2[None, :])
    yy2 = torch.minimum(y2[:, None], y2[None, :])
    iw = (xx2 - xx1).clamp(min=0)
    ih = (yy2 - yy1).clamp(min=0)
    inter = iw * ih                                         # [N,N]

    union = area[:, None] + area[None, :] - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))
    m = torch.minimum(area[:, None], area[None, :])
    ios = torch.where(m > 0, inter / m, torch.zeros_like(inter))
    return iou, ios


def merged_block_keep_mask(ios_t, bt, dtype=torch.float64):
    """Vectorised merged-block pre-filter (the exact rule cpu_ref applies before
    NMS). Returns a boolean [N] keep-mask: drop a big box that is tiled by >=2
    strictly-smaller boxes which together cover >=75% of it, unless one of those
    is a skinny text slice (the false-split guard)."""
    w_t = (bt[:, 2] - bt[:, 0]).clamp(min=0)
    h_t = (bt[:, 3] - bt[:, 1]).clamp(min=0)
    areas_t = w_t * h_t
    smaller = areas_t[:, None] < areas_t[None, :]          # [N,N] s smaller than b
    cover = (ios_t > 0.80) & smaller                       # [N,N]
    n_inside = cover.sum(dim=0)                            # [N] per big box b
    covered = (cover.to(dtype) * (ios_t * areas_t[:, None])).sum(dim=0)  # [N]
    long_side = torch.maximum(w_t, h_t)
    short_side = torch.minimum(w_t, h_t).clamp(min=1.0)
    skinny = ((long_side / short_side) > 2.5) & (areas_t < 15000)        # [N]
    false_split = (cover & skinny[:, None]).any(dim=0)     # [N] per big box b
    drop = (n_inside >= 2) & (covered >= 0.75 * areas_t) & (~false_split)
    return ~drop


# --------------------------------------------------------------------------- #
# Pure-GPU clustering merge (fast path, O(log N))
# --------------------------------------------------------------------------- #
def connected_components_matrix(adj):
    """Connected components of a symmetric boolean adjacency matrix (with self
    loops) via transitive closure by boolean matmul doubling.

    Returns a [N] int tensor: comp[i] = canonical label (the min node index
    reachable from i). Two nodes share a label iff they are in one component.
    O(log N) matmuls.
    """
    n = adj.shape[0]
    R = adj.clone()
    iters = max(1, math.ceil(math.log2(n))) if n > 1 else 1
    for _ in range(iters):
        R_new = ((R.to(torch.float32) @ R.to(torch.float32)) > 0) | R
        if torch.equal(R_new, R):
            break
        R = R_new
    idx = torch.arange(n, device=adj.device)
    big = torch.where(R, idx[None, :].expand(n, n),
                      torch.full((n, n), n, device=adj.device, dtype=idx.dtype))
    return big.min(dim=1).values


def pick_representatives(comp, scores, areas):
    """One winner per component, matching the production sort EXACTLY: highest
    score, then largest area, then lowest index on a true tie.

    A single packed key (score*BIG + area) does NOT work: score in [0,1] and area
    can be the same order of magnitude, so a big low-score box would outrank a
    small high-score box. We instead do a genuine lexicographic argmax in three
    masked scatter-reductions: max score -> among those max area -> among those
    min index. Returns a 1-D tensor of kept box indices.
    """
    uniq, inv = torch.unique(comp, return_inverse=True)      # inv in [0,C)
    C = uniq.numel()
    N = comp.shape[0]
    sc = scores.to(torch.float64)
    ar = areas.to(torch.float64)
    neg = float("-inf")

    # pass 1: max score per component
    best_s = torch.full((C,), neg, device=comp.device, dtype=torch.float64)
    best_s.scatter_reduce_(0, inv, sc, reduce="amax", include_self=True)
    top_s = sc >= best_s[inv]                                # boxes at the group max score

    # pass 2: among top-score boxes, max area
    ar_masked = torch.where(top_s, ar, torch.full_like(ar, neg))
    best_a = torch.full((C,), neg, device=comp.device, dtype=torch.float64)
    best_a.scatter_reduce_(0, inv, ar_masked, reduce="amax", include_self=True)
    top = top_s & (ar >= best_a[inv])

    # pass 3: among those, lowest original index
    idx = torch.arange(N, device=comp.device)
    cand = torch.where(top, idx, torch.full_like(idx, N))
    rep = torch.full((C,), N, device=comp.device, dtype=idx.dtype)
    rep.scatter_reduce_(0, inv, cand, reduce="amin", include_self=True)
    return rep


def nms_gpu_cluster(boxes, iou_threshold=0.3, containment_threshold=0.7,
                    device=None, dtype=torch.float32, prefilter=True):
    """Pure-GPU merge: collapse each overlap-graph component to its best box.

    Fastest path. Still slightly more aggressive than greedy NMS (a chain of
    overlaps becomes ONE box where greedy might keep both ends), so it is the
    "fast/approximate" option. With `prefilter=True` it first drops merged-block
    boxes (same rule as production) so a big block can't bridge several real cells
    into one giant component. Returns the kept boxes (same dict objects).
    """
    if not boxes:
        return []
    if device is None:
        device = default_device()
    bt, scores = boxes_to_tensor(boxes, device=device, dtype=dtype)
    iou, ios = iou_ios_matrix_gpu(bt)
    N = bt.shape[0]

    if prefilter:
        keep_mask = merged_block_keep_mask(ios, bt, dtype=torch.float64)
        sub = keep_mask.nonzero(as_tuple=True)[0]           # surviving indices
    else:
        sub = torch.arange(N, device=device)

    iou_s = iou.index_select(0, sub).index_select(1, sub)
    ios_s = ios.index_select(0, sub).index_select(1, sub)
    M = sub.shape[0]
    eye = torch.eye(M, device=device, dtype=torch.bool)
    adj = (iou_s > iou_threshold) | (ios_s > containment_threshold) | eye
    adj = adj | adj.t()
    comp = connected_components_matrix(adj)

    bt_s = bt.index_select(0, sub)
    scores_s = scores.index_select(0, sub)
    areas_s = (bt_s[:, 2] - bt_s[:, 0]).clamp(min=0) * (bt_s[:, 3] - bt_s[:, 1]).clamp(min=0)
    rep_local = pick_representatives(comp, scores_s, areas_s)   # indices into sub
    kept_global = sub.index_select(0, rep_local)
    kept = torch.sort(kept_global).values.tolist()
    return [boxes[i] for i in kept]


# --------------------------------------------------------------------------- #
# Exact GPU-accelerated greedy NMS (parity path)
# --------------------------------------------------------------------------- #
def nms_gpu_assisted(boxes, iou_threshold=0.3, containment_threshold=0.7,
                     device=None, dtype=torch.float64):
    """Production NMS decisions, GPU-computed overlaps.

    Computes the full IoU/IoS matrices once on the GPU, then runs the *identical*
    greedy algorithm as cpu_ref.non_max_suppression (merged-block pre-filter,
    score-then-area sort, containment eviction) reading values from the matrix
    instead of recomputing them per pair. Output matches the production bbox path
    box-for-box. Returns the kept box dicts.
    """
    if not boxes:
        return []
    if device is None:
        device = default_device()

    bt, scores_t = boxes_to_tensor(boxes, device=device, dtype=dtype)
    iou_t, ios_t = iou_ios_matrix_gpu(bt)
    areas_t = (bt[:, 2] - bt[:, 0]).clamp(min=0) * (bt[:, 3] - bt[:, 1]).clamp(min=0)
    N = bt.shape[0]

    # --- merged-block pre-filter (same rule + helper used by the cluster path) ---
    valid_mask = merged_block_keep_mask(ios_t, bt, dtype=dtype).cpu().numpy()

    # One transfer back: the dense N x N matrices + per-box scalars.
    iou = iou_t.cpu().numpy()
    ios = ios_t.cpu().numpy()
    areas = areas_t.cpu().numpy()
    scores = scores_t.cpu().numpy()

    valid = [i for i in range(N) if valid_mask[i]]

    # --- sort by score desc, then area desc ---
    valid.sort(key=lambda i: (scores[i], areas[i]), reverse=True)

    # --- greedy keep with containment eviction ---
    kept = []  # list of original indices
    for cur in valid:
        should_keep = True
        evict_pos = None
        for pos, ki in enumerate(kept):
            i_iou = iou[cur, ki]
            i_ios = ios[cur, ki]
            if i_iou > iou_threshold:
                should_keep = False
                break
            if i_ios > containment_threshold:
                if areas[cur] > areas[ki]:
                    evict_pos = pos
                else:
                    should_keep = False
                    break
        if not should_keep:
            continue
        if evict_pos is not None:
            kept[evict_pos] = cur
        else:
            kept.append(cur)
    return [boxes[i] for i in kept]
