"""
PDF text-layer use — labeling + recovery — ported from prod's
app/adaptive/labeling.py, but reading the text layer with PyMuPDF (fitz) instead
of pypdfium2, to match the GPU runner's existing PDF rasteriser (load_image uses
fitz). fitz's text coordinates are already top-left origin in PDF points, so —
unlike the pdfium path in prod — NO y-flip is needed; we only scale by dpi/72.

Two jobs:
  1. label_booths(booths, text_items): attach the PDF text under each box and
     tag it (boothlike|text|facility|empty) — exactly prod's spatial assignment.
  2. recover_missing(booths, text_items): every booth-NUMBER token (RE_BOOTH,
     short) whose centre lands in NO detected box is a booth the geometry passes
     missed; synthesise a small box at the token so it is not lost. This is the
     "are we using the texts?" recall safety net.
"""
from __future__ import annotations

import re
from typing import Dict, List

RE_AREA = re.compile(r"\d{1,4}\s*(?:sq\.?\s*m(?:tr)?|sqm|m2|m²)\b", re.I)
RE_BOOTH = re.compile(
    r"\b(?:[A-Z]{1,3}[-\s]?\d{2,4}|\d{1,2}[A-Z]{1,2}[-\s]?\d{1,4})[A-Z]?\b")
RE_COMPANY = re.compile(
    r"\b(?:PVT|LTD|LLP|INC|LLC|EXPORTS?|IMPEX|INDUSTR|ENTERPRIS|"
    r"INTERNATIONAL|TRADERS?|OVERSEAS|LIFECARE|TECHNOLOG)\b", re.I)
RE_FACILITY = re.compile(
    r"\b(?:TOILET|LIFT|DRINKING|WATER|CARGO|SERVICE|FHC|RWP|JC|HUB|LV|"
    r"STAIR|ENTRY|EXIT|GATE|RAMP|PANTRY|FIRE|ELECTRIC|DG|AHU|DUCT|SHAFT)\b", re.I)


def is_boothlike(text: str) -> bool:
    if RE_AREA.search(text) or RE_COMPANY.search(text):
        return True
    return bool(RE_BOOTH.search(text)) and len(text.split()) <= 8


def tag_from_label(label: str) -> str:
    if not label:
        return "empty"
    if is_boothlike(label):
        return "boothlike"
    if RE_FACILITY.search(label):
        return "facility"
    return "text"


def extract_text_items_pdf_fitz(pdf_path: str, dpi: int, page_index: int = 0) -> List[Dict]:
    """PDF text spans -> [{text, bbox_px(x0,y0,x1,y1), center_px}] in render-pixel
    coordinates at `dpi`. fitz origin is top-left (same as the pixmap), so the
    mapping is a pure dpi/72 scale with no y-flip."""
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    scale = dpi / 72.0
    items: List[Dict] = []
    data = page.get_text("dict")
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = (span.get("text") or "").strip()
                if not txt:
                    continue
                x0, y0, x1, y1 = span["bbox"]
                x0 *= scale; y0 *= scale; x1 *= scale; y1 *= scale
                items.append({"text": txt, "bbox_px": (x0, y0, x1, y1),
                              "center_px": ((x0 + x1) / 2.0, (y0 + y1) / 2.0)})
    doc.close()
    return items


def _rect_area(r):
    x0, y0, x1, y1 = r
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _xywh(b):
    """Booth bbox -> (x, y, w, h) regardless of xyxy/xywh storage."""
    bb = b["bbox"]
    x, y, a, c = bb
    # run_real stores xyxy [x1,y1,x2,y2]; prod stores xywh. Disambiguate: if the
    # 3rd/4th values look like x2,y2 (>= x,y) treat as xyxy.
    if b.get("_xyxy", True) and a >= x and c >= y:
        return x, y, a - x, c - y
    return x, y, a, c


def _overlap_area(xywh, r):
    x, y, w, h = xywh
    x0, y0, x1, y1 = r
    ix = max(0.0, min(x + w, x1) - max(x, x0))
    iy = max(0.0, min(y + h, y1) - max(y, y0))
    return ix * iy


def label_booths(booths: List[Dict], text_items: List[Dict]) -> None:
    """Attach text to booths and tag each. Booth bbox is xyxy here (run_real)."""
    owners: Dict[int, List[Dict]] = {id(b): [] for b in booths}
    bx = {id(b): _xywh(b) for b in booths}
    for ti in text_items:
        cx, cy = ti["center_px"]
        containing = [b for b in booths
                      if bx[id(b)][0] <= cx <= bx[id(b)][0] + bx[id(b)][2]
                      and bx[id(b)][1] <= cy <= bx[id(b)][1] + bx[id(b)][3]]
        if containing:
            for b in containing:
                owners[id(b)].append(ti)
        else:
            ra = _rect_area(ti["bbox_px"]) or 1.0
            best, best_ov = None, 0.0
            for b in booths:
                ov = _overlap_area(bx[id(b)], ti["bbox_px"])
                if ov > best_ov:
                    best, best_ov = b, ov
            if best is not None and best_ov >= 0.30 * ra:
                owners[id(best)].append(ti)
    for bo in booths:
        inside = owners[id(bo)]
        inside.sort(key=lambda ti: (round(ti["center_px"][1] / 8.0), ti["center_px"][0]))
        label = re.sub(r"\s+", " ", " ".join(ti["text"] for ti in inside)).strip()
        bo["label"] = label
        bo["n_text"] = len(inside)
        bo["text_status"] = tag_from_label(label)


def recover_missing(booths: List[Dict], text_items: List[Dict],
                    default_side: float = 60.0) -> int:
    """Synthesise a box at every booth-NUMBER token whose centre lands in no
    detected booth. Returns the number recovered (and appends them to `booths`).
    The synthesised box is sized to the token rect, padded out a little so it
    reads as a real cell. Mutates `booths`."""
    bx = {id(b): _xywh(b) for b in booths}

    def covered(cx, cy):
        for b in booths:
            x, y, w, h = bx[id(b)]
            if x <= cx <= x + w and y <= cy <= y + h:
                return True
        return False

    recovered = 0
    for ti in text_items:
        if not is_boothlike(ti["text"]):
            continue
        cx, cy = ti["center_px"]
        if covered(cx, cy):
            continue
        x0, y0, x1, y1 = ti["bbox_px"]
        tw, th = (x1 - x0), (y1 - y0)
        # pad the token rect to a plausible booth footprint
        pw = max(default_side, tw * 1.6)
        ph = max(default_side, th * 2.2)
        nx1 = cx - pw / 2.0; ny1 = cy - ph / 2.0
        nx2 = cx + pw / 2.0; ny2 = cy + ph / 2.0
        nb = {"bbox": [nx1, ny1, nx2, ny2], "score": 0.5, "source": "text_recovered",
              "label": ti["text"], "text_status": tag_from_label(ti["text"]),
              "n_text": 1}
        booths.append(nb)
        bx[id(nb)] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
        recovered += 1
    return recovered
