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


def label_components_gpu(mask, connectivity=8, max_iters=4096):
    """Label a binary mask [H,W] or [1,1,H,W] on GPU.

    Returns an int64 label image [H,W] (0 = background, components numbered with
    arbitrary positive ids — not necessarily contiguous).
    """
    m = mask
    if m.ndim == 2:
        m = m.view(1, 1, *m.shape)
    m = (m > 0).to(torch.float64)            # float so max_pool keeps big ints exactly
    N, _, H, W = m.shape
    assert N == 1, "label_components_gpu handles one image at a time"

    device = m.device
    idx = torch.arange(1, H * W + 1, device=device, dtype=torch.float64).view(1, 1, H, W)
    labels = m * idx                          # unique seed per fg pixel, 0 on bg

    fg = m  # foreground gate
    for _ in range(max_iters):
        prop = _max_pool_neighbors(labels, connectivity=connectivity, ksize=3)
        prop = prop * fg                      # never light up background
        if torch.equal(prop, labels):
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
