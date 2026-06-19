"""
Capture the REAL pre-NMS box pool from the production pipeline, so the GPU merge
can be benchmarked / parity-checked against authentic dense-plan data instead of
the synthetic pool from make_fixtures.py.

The merge runs at ensemble_detector.py:239:
    kept = non_max_suppression(nms_in, iou_threshold=self.iou_threshold)
where `nms_in` is a list of {"bbox":[x1,y1,x2,y2], "score":float, "poly"?:...}.

This module monkeypatches `non_max_suppression` to record the LARGEST pool it is
called with (the global dense pool is what we want), then dumps it to JSON when
the process exits.

Run it INSIDE the container, in front of a normal detection run, e.g.:

    /snap/bin/docker exec adaptive-booth-api python3 -c "
    import sys; sys.path.insert(0, '/app')
    import export_real_fixtures as ex      # copy this file into /app first
    ex.install('/data/booth_gpu_fixtures')
    # ... then trigger one real dense-plan detection (your usual CLI/endpoint) ...
    "

Then copy /data/booth_gpu_fixtures/boxes_pool_real.json out of the container into
booth-gpu/fixtures/ and point bench.py at it with --pool.

Only bbox + score are kept (poly is dropped) so the fixture is small, shareable,
and contains no source-image pixels or secrets.
"""
from __future__ import annotations

import atexit
import json
import os

_STATE = {"best": None, "best_n": -1, "out": None, "calls": 0}


def _patch_module(mod):
    if not hasattr(mod, "non_max_suppression"):
        return False
    orig = mod.non_max_suppression

    def wrapper(boxes, *a, **k):
        try:
            n = len(boxes)
            _STATE["calls"] += 1
            if n > _STATE["best_n"]:
                _STATE["best_n"] = n
                _STATE["best"] = [
                    {"bbox": [float(v) for v in b["bbox"]],
                     "score": float(b.get("score", 1.0))}
                    for b in boxes
                ]
        except Exception:
            pass
        return orig(boxes, *a, **k)

    mod.non_max_suppression = wrapper
    return True


def install(out_dir):
    """Monkeypatch every import site of non_max_suppression and arrange a dump."""
    _STATE["out"] = out_dir
    os.makedirs(out_dir, exist_ok=True)
    patched = []
    import importlib
    for name in (
        "app.pipeline.utils.geometry",
        "pipeline.utils.geometry",
        "app.pipeline.detectors.ensemble_detector",
        "pipeline.detectors.ensemble_detector",
        "app.adaptive._detectors",
        "adaptive._detectors",
    ):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        if _patch_module(mod):
            patched.append(name)
    atexit.register(dump)
    print(f"[export_real_fixtures] patched: {patched or 'NONE (check sys.path)'}")
    return patched


def dump():
    if not _STATE["best"] or not _STATE["out"]:
        print("[export_real_fixtures] nothing captured")
        return
    path = os.path.join(_STATE["out"], "boxes_pool_real.json")
    with open(path, "w") as f:
        json.dump({"source": "production non_max_suppression input",
                   "n_calls": _STATE["calls"],
                   "boxes": _STATE["best"]}, f)
    print(f"[export_real_fixtures] wrote {path}: {_STATE['best_n']} boxes "
          f"(largest of {_STATE['calls']} NMS calls)")


if __name__ == "__main__":
    print(__doc__)
