"""
End-to-end GPU glue: image -> boxes, never leaving the GPU until the final box
list. This wires the three GPU stages together the way the production pipeline
chains its CPU equivalents, and is the thing that proves the "keep tensors
resident on the GPU" rule from the README:

    image (host, once)
       -> upload to GPU                          [gpu_ops.image_to_tensor]
       -> color mask + open/close                [gpu_ops.color_mask_pipeline_gpu]
       -> connected components -> boxes          [gpu_components.components_boxes_gpu]
       -> all-pairs IoU + merge                  [gpu_merge.nms_gpu_*]
       -> download final boxes (host, once)

The only two CPU<->GPU transfers are the initial image upload and the final
small box list. Everything in between is on-device.

This is a *color-cell* detector (the cleanest stage to express purely on GPU),
not the full geometric ensemble — the point is to demonstrate the resident-GPU
dataflow and feed bench.py, not to reproduce every heuristic in booth_detector.
"""
from __future__ import annotations

import torch

from gpu_ops import image_to_tensor, color_mask_pipeline_gpu
from gpu_components import components_boxes_gpu
from gpu_merge import nms_gpu_cluster, nms_gpu_assisted


def detect_color_cells_gpu(
    img_bgr,
    lower,
    upper,
    open_ksize=3,
    close_ksize=3,
    min_area=80,
    connectivity=8,
    merge="cluster",
    iou_threshold=0.3,
    containment_threshold=0.7,
    device=None,
):
    """Detect filled color cells in one image, fully on GPU.

    img_bgr  : HxWx3 uint8 (numpy or tensor), channel order matches lower/upper.
    lower/upper : per-channel inclusive bounds (BGR if img is BGR).
    merge    : "cluster" (fast, pure GPU) | "assisted" (exact greedy) | "none".
    Returns a list of {"bbox":[x1,y1,x2,y2], "score":1.0} dicts.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    img_chw = image_to_tensor(img_bgr, device=device)               # [1,3,H,W]
    mask = color_mask_pipeline_gpu(img_chw, lower, upper,
                                   open_ksize=open_ksize,
                                   close_ksize=close_ksize)          # [1,1,H,W]
    boxes_t = components_boxes_gpu(mask[0, 0], connectivity=connectivity,
                                   min_area=min_area)                # [M,4] on GPU

    boxes = [{"bbox": [float(v) for v in row], "score": 1.0}
             for row in boxes_t.cpu().tolist()]

    if merge == "none" or len(boxes) <= 1:
        return boxes
    if merge == "assisted":
        return nms_gpu_assisted(boxes, iou_threshold, containment_threshold, device=device)
    return nms_gpu_cluster(boxes, iou_threshold, containment_threshold, device=device)
