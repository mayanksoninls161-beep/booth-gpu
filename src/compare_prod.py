#!/usr/bin/env python3
"""
Compare a GPU run (run_real.py `..._boxes.json`) against a PROD reference
(`..._booths.json`) at the booth level.

Reports, broken down by prod text_status and source:
  - matched  : prod booth covered by a GPU box at IoU >= --iou (default 0.5)
  - fused    : prod booth contained (IoS >= 0.7) in a LARGER GPU box (rows merged)
  - partial  : some overlap (IoS > 0.2) but no clean match
  - true-miss: NO GPU box there at all  <-- the real detection gap

PROD bbox is xywh, in `kept` (or the top-level list); GPU bbox is xyxy, in `boxes`.
Both are expected at the same render resolution (they are, for the same input).

Usage:
  python src/compare_prod.py PROD_booths.json GPU_boxes.json [--iou 0.5] [--status boothlike]
"""
import argparse
import json


def _load_prod(path):
    d = json.load(open(path))
    rows = d.get("kept") if isinstance(d, dict) else d
    out = []
    for b in rows:
        x, y, w, h = b["bbox"]
        out.append(((x, y, x + w, y + h), b))
    return out


def _load_gpu(path):
    d = json.load(open(path))
    rows = d.get("boxes", d if isinstance(d, list) else [])
    return [(tuple(b["bbox"]), b) for b in rows]


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _ios(small, big):
    ix1, iy1 = max(small[0], big[0]), max(small[1], big[1])
    ix2, iy2 = min(small[2], big[2]), min(small[3], big[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a = (small[2] - small[0]) * (small[3] - small[1])
    return inter / a if a > 0 else 0.0


def _grid_index(boxes, cell=400):
    idx = {}
    for i, (bb, _) in enumerate(boxes):
        for cx in range(int(bb[0]) // cell, int(bb[2]) // cell + 1):
            for cy in range(int(bb[1]) // cell, int(bb[3]) // cell + 1):
                idx.setdefault((cx, cy), []).append(i)
    return idx, cell


def _cands(q, idx, cell):
    s = set()
    for cx in range(int(q[0]) // cell, int(q[2]) // cell + 1):
        for cy in range(int(q[1]) // cell, int(q[3]) // cell + 1):
            s.update(idx.get((cx, cy), []))
    return s


def classify(prod, gpu, iou_thr):
    idx, cell = _grid_index(gpu)
    res = {}
    for bb, meta in prod:
        c = _cands(bb, idx, cell)
        if any(_iou(bb, gpu[j][0]) >= iou_thr for j in c):
            res[id(meta)] = "matched"
            continue
        contained = any(
            _ios(bb, gpu[j][0]) >= 0.7
            and (gpu[j][0][2] - gpu[j][0][0]) * (gpu[j][0][3] - gpu[j][0][1])
            > 1.4 * (bb[2] - bb[0]) * (bb[3] - bb[1])
            for j in c
        )
        if contained:
            res[id(meta)] = "fused"
        elif any(_ios(bb, gpu[j][0]) > 0.2 for j in c):
            res[id(meta)] = "partial"
        else:
            res[id(meta)] = "true-miss"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prod")
    ap.add_argument("gpu")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--status", default=None,
                    help="restrict prod rows to this text_status (e.g. boothlike)")
    args = ap.parse_args()

    prod = _load_prod(args.prod)
    gpu = _load_gpu(args.gpu)
    if args.status:
        prod = [(bb, m) for bb, m in prod if m.get("text_status") == args.status]

    print(f"prod rows: {len(prod)}   gpu boxes: {len(gpu)}   IoU thr: {args.iou}")
    res = classify(prod, gpu, args.iou)

    from collections import Counter
    overall = Counter(res[id(m)] for _, m in prod)
    n = len(prod)
    print("\n=== overall ===")
    for k in ("matched", "fused", "partial", "true-miss"):
        v = overall.get(k, 0)
        print(f"  {k:10s}: {v:5d} ({100*v/max(1,n):5.1f}%)")

    for field in ("text_status", "source"):
        groups = {}
        for bb, m in prod:
            groups.setdefault(m.get(field), Counter())[res[id(m)]] += 1
        print(f"\n=== by prod {field} ===")
        for g, c in sorted(groups.items(), key=lambda kv: -sum(kv[1].values())):
            tot = sum(c.values())
            print(f"  {str(g):14s} n={tot:5d}  matched={c.get('matched',0):5d} "
                  f"fused={c.get('fused',0):4d} partial={c.get('partial',0):4d} "
                  f"true-miss={c.get('true-miss',0):4d}")


if __name__ == "__main__":
    main()
