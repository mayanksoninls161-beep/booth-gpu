"""
Booth-detection backend for the API's /hall_with_booth_predict endpoint.

This is the GPU analogue of prod's `adaptive_pipeline.run(...)`: it drives the
GPU runner (`src/run_real.py`) on raw bytes and returns
  (payload, render_bgr)
where `payload["booths"]` is the normalized booth list and `render_bgr` is the
page render in the SAME pixel space as the booth coordinates (so the Roboflow
hall boxes can be scaled back onto it).

Detection runs `run_real.py` as a subprocess so the exact, validated CLI
behaviour (tiling, GPU CCL, big-region, OCR/PDF labels, fp-policy) is reused
verbatim; the render is produced in-process with the runner's own loader so the
coordinate spaces match.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _safe_stem(name: str) -> str:
    stem = os.path.splitext(os.path.basename(name or "input"))[0] or "input"
    return "".join(c for c in stem if c.isalnum() or c in ("-", "_", ".")) or "input"


def render_page(path: str, dpi: int, page: int):
    """Render a PDF page (or load an image) to a BGR array in the same pixel
    space the runner uses for boxes. Reuses run_real.load_image so DPI / page
    handling is identical to detection."""
    import run_real
    return run_real.load_image(path, dpi=dpi, page=page)


def normalize_booths(boxes: List[Dict]) -> List[Dict]:
    """Shape GPU boxes (xyxy bbox + label/source/text_status) into the production
    booth schema the API consumer expects: guarantee `area` and `centroid`, and
    expose the PDF/OCR label as `name`. Original keys are retained."""
    out = []
    for b in boxes:
        b = dict(b)
        bb = b.get("bbox")
        if isinstance(bb, (list, tuple)) and len(bb) >= 4:
            x1, y1, x2, y2 = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
            if not isinstance(b.get("area"), (int, float)) or not b["area"]:
                b["area"] = abs(x2 - x1) * abs(y2 - y1)
            b.setdefault("centroid", [(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            b.setdefault("coordinates",
                         [[x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]])
        lab = b.get("label") or ""
        lab = lab.strip() if isinstance(lab, str) else ""
        if lab:
            b["name"] = lab
        out.append(b)
    return out


def run_booths(src_bytes: bytes, src_name: str, is_pdf: bool, page: int = 0,
               fp_policy: Optional[str] = None, dpi: int = 250,
               ocr: str = "auto", extra_args: Optional[List[str]] = None,
               timeout: int = 1800) -> Tuple[Dict, "object"]:
    """Run the GPU pipeline on raw bytes; return (payload, render_bgr).

    payload = {count, booths, image, stages, params, source_json}
    """
    workdir = tempfile.mkdtemp(prefix="booth_api_")
    try:
        stem = _safe_stem(src_name)
        ext = ".pdf" if is_pdf else (os.path.splitext(src_name or "")[1] or ".png")
        in_path = os.path.join(workdir, stem + ext)
        with open(in_path, "wb") as fh:
            fh.write(src_bytes)
        outdir = os.path.join(workdir, "out")

        cmd = [sys.executable, os.path.join(_SRC, "run_real.py"),
               "--input", in_path, "--outdir", outdir,
               "--page", str(page), "--dpi", str(dpi), "--ocr", ocr]
        if fp_policy:
            cmd += ["--fp-policy", fp_policy]
        if extra_args:
            cmd += list(extra_args)

        proc = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True,
                              timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"booth detection failed (exit {proc.returncode}):\n"
                f"{proc.stderr[-2000:]}")

        json_path = os.path.join(outdir, f"{stem}_boxes.json")
        if not os.path.exists(json_path):
            raise RuntimeError(f"no output JSON produced; stdout tail:\n"
                               f"{proc.stdout[-2000:]}")
        with open(json_path) as fh:
            data = json.load(fh)

        booths = normalize_booths(data.get("boxes", []))
        payload = {
            "count": len(booths),
            "booths": booths,
            "image": data.get("image"),
            "stages": data.get("stages"),
            "params": data.get("params"),
        }
        # Render in the same coord space as the boxes for hall alignment.
        render_bgr = render_page(in_path, dpi=dpi, page=page)
        return payload, render_bgr
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
