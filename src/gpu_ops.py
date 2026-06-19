"""
GPU per-pixel ops: color thresholding + binary morphology, as pure torch tensor
operations so they run on CUDA with no CPU round-trips.

Why these belong on the GPU: every output pixel depends only on a small local
neighbourhood of input pixels and on nothing else — there is no data dependency
between pixels, so the work is embarrassingly parallel SIMD. That is exactly the
shape of computation a GPU is built for.

  cv2.inRange      -> per-channel compare + logical-and            (color_mask_gpu)
  cv2.erode        -> min-pool   = -maxpool(-x)                    (erode_gpu)
  cv2.dilate       -> max-pool                                     (dilate_gpu)
  cv2.MORPH_OPEN   -> erode then dilate                           (morph_open_gpu)
  cv2.MORPH_CLOSE  -> dilate then erode                           (morph_close_gpu)

A binary morphology with a rectangular structuring element is *exactly* a
min/max pooling with that kernel and stride 1, which is a single fused CUDA
kernel via torch.nn.functional.max_pool2d.

All functions keep the tensor on whatever device it arrives on. Hand them a CUDA
tensor and nothing ever touches the CPU.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def to_device(x, device=None, dtype=None):
    """Return x as a torch tensor on `device` (defaults to cuda if available)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype)
    return t.to(device)


def image_to_tensor(img, device=None):
    """HxWxC uint8 image (numpy or tensor) -> float tensor [1,C,H,W] on device.

    Channels stay in their original order (pass BGR in, get BGR planes out).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.as_tensor(img)
    if t.ndim == 2:
        t = t.unsqueeze(-1)
    # HWC -> CHW -> NCHW
    t = t.permute(2, 0, 1).unsqueeze(0).contiguous().to(device).float()
    return t


def color_mask_gpu(img_chw, lower, upper):
    """Per-channel inclusive threshold == cv2.inRange.

    img_chw: float tensor [N,C,H,W] (channel order matches `lower`/`upper`).
    lower/upper: length-C sequences.
    Returns a uint8 mask tensor [N,1,H,W] with values in {0,1}.
    """
    device = img_chw.device
    C = img_chw.shape[1]
    lo = torch.as_tensor(lower, dtype=torch.float32, device=device).view(1, C, 1, 1)
    hi = torch.as_tensor(upper, dtype=torch.float32, device=device).view(1, C, 1, 1)
    in_range = (img_chw >= lo) & (img_chw <= hi)      # [N,C,H,W] bool
    mask = in_range.all(dim=1, keepdim=True)          # AND across channels
    return mask.to(torch.uint8)


def _pad_same(x, k):
    p = k // 2
    return F.pad(x, (p, p, p, p), mode="constant", value=0)


def dilate_gpu(mask, ksize=3):
    """Binary dilation with a ksize x ksize rectangular element == max-pool."""
    if ksize <= 1:
        return mask
    x = mask.float()
    x = _pad_same(x, ksize)
    x = F.max_pool2d(x, kernel_size=ksize, stride=1)
    return (x > 0).to(mask.dtype)


def erode_gpu(mask, ksize=3):
    """Binary erosion == min-pool == -maxpool(-x). Element fully covered -> 1."""
    if ksize <= 1:
        return mask
    x = mask.float()
    x = _pad_same(x, ksize)
    # min-pool over the window; a foreground pixel survives only if the WHOLE
    # window was foreground (sum == k*k), matching cv2.erode with a full rect SE.
    s = F.avg_pool2d(x, kernel_size=ksize, stride=1) * (ksize * ksize)
    return (s >= (ksize * ksize) - 0.5).to(mask.dtype)


def morph_open_gpu(mask, ksize=3):
    """Erode then dilate (removes specks). == cv2.MORPH_OPEN with rect SE."""
    if ksize <= 1:
        return mask
    return dilate_gpu(erode_gpu(mask, ksize), ksize)


def morph_close_gpu(mask, ksize=3):
    """Dilate then erode (fills pinholes). == cv2.MORPH_CLOSE with rect SE."""
    if ksize <= 1:
        return mask
    return erode_gpu(dilate_gpu(mask, ksize), ksize)


def color_mask_pipeline_gpu(img_chw, lower, upper, open_ksize=3, close_ksize=3):
    """color_mask -> open -> close, all on GPU. Mirrors cpu_ref.color_mask_cpu.

    Returns uint8 mask [N,1,H,W] in {0,1}. Multiply by 255 for a cv2-style mask.
    """
    mask = color_mask_gpu(img_chw, lower, upper)
    mask = morph_open_gpu(mask, open_ksize)
    mask = morph_close_gpu(mask, close_ksize)
    return mask
