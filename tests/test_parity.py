"""
Parity tests: GPU output must match the CPU reference (the verbatim production
code) within tolerance.

Runs two ways:
    pytest tests/test_parity.py            # CI / dev
    python  tests/test_parity.py           # Colab one-liner (prints PASS/FAIL)

Hard guarantees asserted here:
  * iou_ios_matrix_gpu == cpu_ref.calculate_iou / _bbox_overlap  (numerically)
  * nms_gpu_assisted   == cpu_ref.non_max_suppression            (identical kept set)
  * nms_gpu_cluster    agrees with CPU within a sane band        (it is intentionally
                                                                  a bit more aggressive)
Image-stage parity (color mask + components) is checked only if numpy + cv2 are
installed; otherwise those tests skip cleanly.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

import cpu_ref
from gpu_merge import (
    boxes_to_tensor, iou_ios_matrix_gpu, nms_gpu_assisted, nms_gpu_cluster,
)
from make_fixtures import make_box_pool

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _ids(bs):
    return {id(b) for b in bs}


# --------------------------------------------------------------------------- #
def test_iou_matrix_matches_cpu():
    pool, _ = make_box_pool(n_cells=60, seed=1)
    pool = pool[:80]
    bt, _ = boxes_to_tensor(pool, device=DEVICE, dtype=torch.float64)
    iou, ios = iou_ios_matrix_gpu(bt)
    iou = iou.cpu().tolist(); ios = ios.cpu().tolist()
    bad = 0
    for i in range(len(pool)):
        for j in range(len(pool)):
            ref_iou = cpu_ref.calculate_iou(pool[i]["bbox"], pool[j]["bbox"])
            _, ref_ios = cpu_ref._bbox_overlap(pool[i]["bbox"], pool[j]["bbox"])
            if abs(iou[i][j] - ref_iou) > 1e-9 or abs(ios[i][j] - ref_ios) > 1e-9:
                bad += 1
    assert bad == 0, f"{bad} matrix entries diverged from cpu_ref"
    print("  [ok] iou/ios matrix matches cpu_ref exactly")


def test_assisted_nms_exact_parity():
    pool, _ = make_box_pool(n_cells=400, seed=2)
    cpu_keep = cpu_ref.non_max_suppression(pool, 0.3, 0.7)
    gpu_keep = nms_gpu_assisted(pool, 0.3, 0.7, device=DEVICE)
    assert _ids(cpu_keep) == _ids(gpu_keep), (
        f"assisted NMS diverged: cpu={len(cpu_keep)} gpu={len(gpu_keep)} "
        f"only_cpu={len(_ids(cpu_keep) - _ids(gpu_keep))} "
        f"only_gpu={len(_ids(gpu_keep) - _ids(cpu_keep))}")
    print(f"  [ok] assisted NMS exact parity (kept {len(cpu_keep)})")


def test_cluster_nms_reasonable():
    pool, n_true = make_box_pool(n_cells=400, seed=3)
    cpu_keep = cpu_ref.non_max_suppression(pool, 0.3, 0.7)
    clu_keep = nms_gpu_cluster(pool, 0.3, 0.7, device=DEVICE)
    # cluster collapses each overlap component to one box: count should be in a
    # sane band around the true-cell count, and not wildly off the CPU result.
    assert 0 < len(clu_keep) <= len(pool)
    inter = len(_ids(cpu_keep) & _ids(clu_keep))
    agree = inter / max(1, len(_ids(cpu_keep) | _ids(clu_keep)))
    assert agree >= 0.5, f"cluster agreement too low: {agree:.2f}"
    print(f"  [ok] cluster NMS sane (kept {len(clu_keep)} vs cpu {len(cpu_keep)}, "
          f"agree {agree*100:.0f}%, true~{n_true})")


def test_color_mask_parity():
    try:
        import numpy as np  # noqa: F401
        import cv2  # noqa: F401
    except Exception:
        print("  [skip] color-mask parity (numpy/cv2 not installed)")
        return
    from make_fixtures import make_plan_image
    from gpu_ops import image_to_tensor, color_mask_pipeline_gpu
    out = make_plan_image(n_cells=120, seed=4)
    assert out is not None
    img, lower, upper = out
    cpu_mask = cpu_ref.color_mask_cpu(img, lower, upper)            # {0,255}
    gpu_mask = color_mask_pipeline_gpu(image_to_tensor(img, device=DEVICE),
                                       lower, upper)                # {0,1}
    import numpy as np
    g = (gpu_mask[0, 0].cpu().numpy() > 0).astype(np.uint8) * 255
    agree = float((g == cpu_mask).mean())
    assert agree >= 0.99, f"color mask agreement {agree:.4f} < 0.99"
    print(f"  [ok] color mask parity {agree*100:.2f}% pixels")


def test_components_parity():
    try:
        import numpy as np  # noqa: F401
        import cv2  # noqa: F401
    except Exception:
        print("  [skip] components parity (numpy/cv2 not installed)")
        return
    from make_fixtures import make_plan_image
    from gpu_ops import image_to_tensor, color_mask_pipeline_gpu
    from gpu_components import components_boxes_gpu
    out = make_plan_image(n_cells=120, seed=5)
    img, lower, upper = out
    cpu_mask = cpu_ref.color_mask_cpu(img, lower, upper)
    cpu_boxes = cpu_ref.components_boxes_cpu(cpu_mask, min_area=80).tolist()
    mask = color_mask_pipeline_gpu(image_to_tensor(img, device=DEVICE), lower, upper)
    gpu_boxes = components_boxes_gpu(mask[0, 0], min_area=80).cpu().tolist()
    # every CPU cell should have a GPU box at high IoU
    hit = 0
    for cb in cpu_boxes:
        if any(cpu_ref.calculate_iou(cb, gb) >= 0.9 for gb in gpu_boxes):
            hit += 1
    rate = hit / max(1, len(cpu_boxes))
    assert rate >= 0.95, f"components match rate {rate:.3f} < 0.95"
    print(f"  [ok] components parity {rate*100:.1f}% (cpu {len(cpu_boxes)}, gpu {len(gpu_boxes)})")


def main():
    print(f"device = {DEVICE}")
    tests = [
        test_iou_matrix_matches_cpu,
        test_assisted_nms_exact_parity,
        test_cluster_nms_reasonable,
        test_color_mask_parity,
        test_components_parity,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [ERROR] {t.__name__}: {e!r}")
    print("=" * 50)
    print("ALL PASSED" if failed == 0 else f"{failed} TEST(S) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
