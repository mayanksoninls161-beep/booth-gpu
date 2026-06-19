"""
Roboflow HALL detection — ported from prod's app/main.py (the only Roboflow
model the production hall_with_booth_predict endpoint runs alongside booths;
icon/trail models are unrelated to booths). This lets the GPU repo reproduce the
SAME hall + booth output as prod for an apples-to-apples whole-pipeline compare.

The model (hall_detection/6) is loaded lazily via the `inference` SDK with the
key in ROBOFLOW_HALL_API_KEY. The key is read from the environment and NEVER
printed or written to any output file. Network + the `inference` package are
only required when --roboflow-hall is passed; everything else runs offline.

Usage from run_real:
    halls = detect_halls(render_bgr, conf=0.4, max_edge=2048)
    hall_booth = build_hall_booth_map(halls, booths)   # booths in render px
"""
from __future__ import annotations

import json
import os
from io import BytesIO
from typing import Dict, List

_MODEL = None
_MODEL_ID = "hall_detection/6"
_ENV_KEY = "ROBOFLOW_HALL_API_KEY"


def _get_model():
    """Lazily build + cache the Roboflow hall model. Raises a clear error (never
    echoing the key) if the env var is missing."""
    global _MODEL
    if _MODEL is None:
        key = os.getenv(_ENV_KEY)
        if not key:
            raise RuntimeError(
                f"{_ENV_KEY} is not set in the environment; export it before "
                f"--roboflow-hall (it is read from env, never printed/stored).")
        from inference import get_model
        _MODEL = get_model(model_id=_MODEL_ID, api_key=key)
    return _MODEL


def _serialize_model_output(model_output) -> list:
    """Port of prod serialize_model_output: normalise inference SDK objects to a
    list of plain dicts with a 'predictions' list."""
    if model_output is None:
        return []
    if isinstance(model_output, list) and all(isinstance(x, dict) for x in model_output):
        return model_output
    items = model_output if isinstance(model_output, list) else [model_output]
    out = []
    for item in items:
        done = False
        for method in ("model_dump", "dict"):
            if hasattr(item, method):
                try:
                    out.append(getattr(item, method)())
                    done = True
                    break
                except Exception:
                    continue
        if done:
            continue
        if hasattr(item, "json"):
            try:
                out.append(json.loads(item.json()))
                continue
            except Exception:
                pass
        if isinstance(item, dict):
            out.append(item)
    return out


def detect_halls(render_bgr, conf: float = 0.4, max_edge: int = 2048) -> List[Dict]:
    """Run the Roboflow hall model on a downscaled copy of the render, scale the
    predictions back to FULL render pixel space, and return hall boxes:
        {x1, y1, x2, y2, area, coordinates, confidence, class}
    Coordinates share the booth coordinate system so they can be overlaid."""
    import cv2
    if render_bgr is None:
        return []
    h, w = render_bgr.shape[:2]
    long_edge = max(h, w)
    sf = min(1.0, float(max_edge) / float(long_edge)) if long_edge > 0 else 1.0
    img = (cv2.resize(render_bgr, (max(1, round(w * sf)), max(1, round(h * sf))),
                      interpolation=cv2.INTER_AREA) if sf < 1.0 else render_bgr)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("failed to encode hall raster")
    model = _get_model()
    raw = model.infer(BytesIO(buf.tobytes()), confidence=conf)
    preds = _serialize_model_output(raw)
    inv = (1.0 / sf) if sf > 0 else 1.0
    halls: List[Dict] = []
    for block in preds:
        if not isinstance(block, dict):
            continue
        for p in block.get("predictions", []):
            cx, cy = p.get("x", 0) * inv, p.get("y", 0) * inv
            pw, ph = p.get("width", 0) * inv, p.get("height", 0) * inv
            if pw <= 0 or ph <= 0:
                continue
            x1, y1, x2, y2 = cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2
            halls.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "area": float(pw) * float(ph),
                "coordinates": [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]],
                "confidence": float(p.get("confidence", 0.0)),
                "class": p.get("class", "hall"),
            })
    return halls


def _booth_centroid(b: Dict):
    cen = b.get("centroid")
    if cen and len(cen) >= 2:
        return float(cen[0]), float(cen[1])
    x1, y1, x2, y2 = b["bbox"]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _booth_area(b: Dict) -> float:
    a = b.get("area")
    if isinstance(a, (int, float)) and a > 0:
        return float(a)
    x1, y1, x2, y2 = b["bbox"]
    return abs(x2 - x1) * abs(y2 - y1)


def build_hall_booth_map(halls: List[Dict], booths: List[Dict],
                         max_booth_hall_area_frac: float = 0.5) -> Dict:
    """Group booths by which hall contains their centroid (port of prod
    _build_hall_booth_map). A booth >= max_booth_hall_area_frac of its hall's
    area is too big to be a sub-booth -> falls through to 'Other'."""
    hall_groups = [[] for _ in halls]
    other = []
    for booth in booths:
        cx, cy = _booth_centroid(booth)
        assigned = False
        for idx, hall in enumerate(halls):
            if hall["x1"] <= cx <= hall["x2"] and hall["y1"] <= cy <= hall["y2"]:
                ha = hall.get("area", 0.0)
                if ha > 0 and _booth_area(booth) >= max_booth_hall_area_frac * ha:
                    break
                hall_groups[idx].append(booth)
                assigned = True
                break
        if not assigned:
            other.append(booth)
    result: Dict = {}
    for i, (hall, group) in enumerate(zip(halls, hall_groups), 1):
        result[f"Hall_{i}"] = {"coordinates": hall["coordinates"], "booths": group}
    if other:
        result["Other"] = {"booths": other}
    return result
