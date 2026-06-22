"""
GPU connected-components -> bounding boxes. This replaces the single most
GPU-hostile step in the current pipeline: cv2.findContours (a serial boundary
trace) / cv2.connectedComponentsWithStats (CPU-only in stock OpenCV builds).

Algorithm (label propagation, a.k.a. "iterative max"):
  1. Give every foreground pixel a unique label = its flattened index.
     Background = 0.
  2. Repeatedly set each pixel's label to the MAX label among itself and its
     4-/8-connected foreground neighbours. Labels flood outward; after enough
     rounds every pixel in one component carries that component's largest seed.
     Each round is one 3x3 max-pool — fully parallel, no boundary walking.
  3. The set of distinct non-zero labels == the connected components. Reduce
     pixel coordinates per label (scatter min/max) to get [x1,y1,x2,y2] boxes.

We use a FIXED 3x3 propagation kernel and iterate to a fixed point (each round
spreads a label one pixel, so the round count tracks a component's pixel
diameter). For booth-cell masks components are small, so convergence is fast;
`max_iters` caps the worst case. A 3x3 max-pool on millions of pixels is a
single cheap CUDA kernel, so many rounds are still far quicker than a serial CPU
boundary trace. (A large fixed kernel is deliberately NOT used: max-pool cost
grows as k^2, so a 257x257 window would be ~7000x more work per round.)

Everything stays on-device; only the final small box list comes back to host.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Optional cuCIM (RAPIDS) backend: a real GPU union-find connected-components
# (cucim.skimage.measure.label) that converges in a few passes instead of the
# O(component-diameter) rounds the label-propagation path needs. This is the one
# step where label-propagation loses to cv2 (huge background blobs), so cuCIM is
# what makes the GPU geometric pass actually beat the CPU. If cuCIM/CuPy are not
# installed we silently fall back to label propagation (identical results).
#   Colab install:  pip install cupy-cuda12x cucim-cu12
# --------------------------------------------------------------------------- #
try:
    import cupy as _cp
    from cucim.skimage.measure import label as _cucim_label
    _HAS_CUCIM = True
except Exception:
    _HAS_CUCIM = False

_CCL_BACKEND = "prop"   # "prop" (label propagation) | "cucim"


def cucim_available() -> bool:
    return _HAS_CUCIM


def set_ccl_backend(name: str):
    """Choose the GPU connected-components backend: 'prop' or 'cucim'.

    'cucim' silently degrades to 'prop' if RAPIDS cuCIM is not importable, so it
    is always safe to request. Returns the backend actually selected."""
    global _CCL_BACKEND
    _CCL_BACKEND = "cucim" if (name == "cucim" and _HAS_CUCIM) else "prop"
    return _CCL_BACKEND


def get_ccl_backend() -> str:
    return _CCL_BACKEND


def _label_cucim(mask, connectivity):
    """Binary mask [H,W] or [1,1,H,W] CUDA tensor -> int64 label image [H,W].

    Uses cuCIM's GPU CCL via a zero-copy CuPy<->Torch handoff (CUDA array
    interface in, DLPack out). connectivity 4 -> skimage conn 1; 8 -> 2."""
    m = mask
    if m.ndim == 4:
        m = m.view(m.shape[-2], m.shape[-1])
    elif m.ndim != 2:
        m = m.view(*m.shape[-2:])
    m = m.contiguous()
    cm = _cp.asarray(m) != 0                      # zero-copy view of the torch tensor
    conn = 1 if connectivity == 4 else 2
    lab = _cucim_label(cm, connectivity=conn)     # CuPy int label image, 0 = bg
    return torch.from_dlpack(lab.toDlpack()).to(torch.int64)


def _max_pool_neighbors(labels, connectivity=8, ksize=3):
    """One round of 'each pixel takes the max label in its k x k window'."""
    p = ksize // 2
    x = F.pad(labels, (p, p, p, p), mode="constant", value=0)
    if connectivity == 8:
        return F.max_pool2d(x, kernel_size=ksize, stride=1)
    # 4-connectivity: cross-shaped neighbourhood (no diagonals)
    out = labels.clone()
    out[..., 1:, :] = torch.maximum(out[..., 1:, :], labels[..., :-1, :])   # up
    out[..., :-1, :] = torch.maximum(out[..., :-1, :], labels[..., 1:, :])  # down
    out[..., :, 1:] = torch.maximum(out[..., :, 1:], labels[..., :, :-1])   # left
    out[..., :, :-1] = torch.maximum(out[..., :, :-1], labels[..., :, 1:])  # right
    return out


def label_components_gpu(mask, connectivity=8, max_iters=4096, check_every=8):
    """Label a binary mask [H,W] or [1,1,H,W] on GPU.

    Returns an int64 label image [H,W] (0 = background, components numbered with
    arbitrary positive ids — not necessarily contiguous).

    Two T4-critical optimisations vs the naive version:

    * float32, not float64.  The propagation is a max_pool, and max of two ints
      is EXACT in float32 as long as the labels stay below 2^24 (16.7M). On a
      Tesla T4 FP64 runs at ~1/32 of FP32, so the old float64 path was paying a
      ~32x tax for nothing. We seed labels by foreground RANK (1..n_fg) instead
      of flat pixel index (1..H*W), so the largest label is the foreground-pixel
      COUNT — which keeps us in float32 range for any plausible plan. We only
      fall back to float64 if the foreground itself exceeds 16.7M pixels.

    * fewer convergence checks.  torch.equal forces a GPU->CPU sync every round;
      on a multi-hundred-round propagation that stalls the pipeline. We only test
      for the fixed point every `check_every` rounds (a few extra cheap max_pool
      rounds is far cheaper than a sync per round).
    """
    # Fast path: RAPIDS cuCIM union-find CCL (O(few passes) vs O(diameter)).
    if _CCL_BACKEND == "cucim" and _HAS_CUCIM and mask.is_cuda:
        return _label_cucim(mask, connectivity)

    m = mask
    if m.ndim == 2:
        m = m.view(1, 1, *m.shape)
    fg_bool = (m > 0)
    N, _, H, W = fg_bool.shape
    assert N == 1, "label_components_gpu handles one image at a time"

    flat_fg = fg_bool.view(-1)
    n_fg = int(flat_fg.sum().item())          # one sync, unavoidable
    if n_fg == 0:
        return torch.zeros((H, W), dtype=torch.int64, device=fg_bool.device)
    dtype = torch.float32 if n_fg <= (1 << 24) else torch.float64

    # rank seed: cumulative count of fg pixels gives 1..n_fg at fg positions
    ranks = torch.cumsum(flat_fg.to(torch.int64), dim=0)
    seed = (ranks * flat_fg.to(torch.int64)).to(dtype).view(1, 1, H, W)
    fg = fg_bool.to(dtype)                     # foreground gate
    labels = seed

    for i in range(max_iters):
        prop = _max_pool_neighbors(labels, connectivity=connectivity, ksize=3)
        prop = prop * fg                      # never light up background
        if i % check_every == check_every - 1 and torch.equal(prop, labels):
            labels = prop
            break
        labels = prop
    return labels.view(H, W).to(torch.int64)


def components_boxes_gpu(mask, connectivity=8, min_area=1, max_iters=256):
    """Binary mask -> (M,4) int tensor of [x1,y1,x2,y2] boxes on GPU.

    Sorted the same way as cpu_ref.components_boxes_cpu so the two can be
    compared directly. Background label (0) is dropped.
    """
    labels = label_components_gpu(mask, connectivity=connectivity, max_iters=max_iters)
    H, W = labels.shape
    device = labels.device

    flat = labels.view(-1)
    fg = flat > 0
    if fg.sum() == 0:
        return torch.zeros((0, 4), dtype=torch.int64, device=device)

    lab = flat[fg]
    uniq, inv = torch.unique(lab, return_inverse=True)   # inv in [0, M)
    M = uniq.numel()

    ys = torch.arange(H, device=device).view(H, 1).expand(H, W).reshape(-1)[fg]
    xs = torch.arange(W, device=device).view(1, W).expand(H, W).reshape(-1)[fg]
    ys = ys.to(torch.int64); xs = xs.to(torch.int64)

    big = torch.iinfo(torch.int64).max
    x1 = torch.full((M,), big, device=device, dtype=torch.int64)
    y1 = torch.full((M,), big, device=device, dtype=torch.int64)
    x2 = torch.full((M,), -1, device=device, dtype=torch.int64)
    y2 = torch.full((M,), -1, device=device, dtype=torch.int64)

    x1.scatter_reduce_(0, inv, xs, reduce="amin", include_self=True)
    y1.scatter_reduce_(0, inv, ys, reduce="amin", include_self=True)
    x2.scatter_reduce_(0, inv, xs, reduce="amax", include_self=True)
    y2.scatter_reduce_(0, inv, ys, reduce="amax", include_self=True)

    # Pixel maxima are inclusive; +1 to match cv2 stats (x+w == right edge).
    boxes = torch.stack([x1, y1, x2 + 1, y2 + 1], dim=1)

    if min_area > 1:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxes = boxes[areas >= min_area]

    if boxes.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.int64, device=device)

    # Stable multi-key sort by (x1,y1,x2,y2) == np.lexsort in cpu_ref. Sort by the
    # least-significant key first; the last (x1) becomes primary. (A single packed
    # key would overflow float64's exact-int range for large coordinates.)
    M = boxes.shape[0]
    order = torch.arange(M, device=device)
    for col in (3, 2, 1, 0):
        idx = torch.argsort(boxes[order, col], stable=True)
        order = order[idx]
    return boxes[order]


def components_stats_gpu(mask, connectivity=4, max_iters=256):
    """connectedComponentsWithStats equivalent, on GPU.

    Returns a dict with everything the geometric shape-filters need:
      labels   : int64 [H,W] relabelled 1..M contiguous (0 = background)
      boxes    : int64 [M,4]  (x, y, w, h)   -- cv2 stats layout (NOT x2y2)
      areas    : int64 [M]    pixel count per component (cv2 CC_STAT_AREA)
      centroids: float32 [M,2] (cx, cy)       -- cv2 centroids
    Component k (0-based row) carries label (k+1) in `labels`. Empty mask -> M=0.
    """
    raw = label_components_gpu(mask, connectivity=connectivity, max_iters=max_iters)
    H, W = raw.shape
    device = raw.device
    flat = raw.view(-1)
    fg = flat > 0
    if int(fg.sum().item()) == 0:
        return {"labels": torch.zeros((H, W), dtype=torch.int64, device=device),
                "boxes": torch.zeros((0, 4), dtype=torch.int64, device=device),
                "areas": torch.zeros((0,), dtype=torch.int64, device=device),
                "centroids": torch.zeros((0, 2), dtype=torch.float32, device=device)}

    lab = flat[fg]
    uniq, inv = torch.unique(lab, return_inverse=True)      # inv in [0, M)
    M = uniq.numel()

    ys = torch.arange(H, device=device).view(H, 1).expand(H, W).reshape(-1)[fg].to(torch.int64)
    xs = torch.arange(W, device=device).view(1, W).expand(H, W).reshape(-1)[fg].to(torch.int64)

    big = torch.iinfo(torch.int64).max
    x1 = torch.full((M,), big, device=device, dtype=torch.int64)
    y1 = torch.full((M,), big, device=device, dtype=torch.int64)
    x2 = torch.full((M,), -1, device=device, dtype=torch.int64)
    y2 = torch.full((M,), -1, device=device, dtype=torch.int64)
    x1.scatter_reduce_(0, inv, xs, reduce="amin", include_self=True)
    y1.scatter_reduce_(0, inv, ys, reduce="amin", include_self=True)
    x2.scatter_reduce_(0, inv, xs, reduce="amax", include_self=True)
    y2.scatter_reduce_(0, inv, ys, reduce="amax", include_self=True)

    areas = torch.zeros((M,), device=device, dtype=torch.int64)
    areas.scatter_reduce_(0, inv, torch.ones_like(inv), reduce="sum", include_self=True)
    sx = torch.zeros((M,), device=device, dtype=torch.float64)
    sy = torch.zeros((M,), device=device, dtype=torch.float64)
    sx.scatter_reduce_(0, inv, xs.to(torch.float64), reduce="sum", include_self=True)
    sy.scatter_reduce_(0, inv, ys.to(torch.float64), reduce="sum", include_self=True)
    centroids = torch.stack([sx / areas, sy / areas], dim=1).to(torch.float32)

    boxes = torch.stack([x1, y1, x2 - x1 + 1, y2 - y1 + 1], dim=1)  # x,y,w,h

    # Relabel the image to contiguous 1..M so callers can do (labels == k).
    relabel = torch.zeros(int(uniq.max().item()) + 1, dtype=torch.int64, device=device)
    relabel[uniq] = torch.arange(1, M + 1, device=device)
    labels = torch.zeros_like(flat)
    labels[fg] = relabel[lab]
    return {"labels": labels.view(H, W), "boxes": boxes, "areas": areas,
            "centroids": centroids}
