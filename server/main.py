"""
Booth-GPU API — same endpoints as the production "main" API, with the
hall+booth endpoint running OUR GPU booth pipeline (src/run_real.py) instead of
the CPU single-pass detector:

  POST /predict                  icons + trails (Roboflow)            [unchanged]
  POST /debug_predict            trail debug (Roboflow)               [unchanged]
  POST /hall_with_booth_predict  Roboflow hall detection  +  GPU booth pipeline
                                 (tiling + GPU CCL + big-region + PDF/OCR labels).
                                 Accepts image_url AND/OR pdf_url; when BOTH are
                                 given everything is computed from the PDF. [GPU]
  GET  /health

S3 writeback is hard-OFF (constraint). Roboflow models load lazily on first use
so the app boots even without keys; each endpoint only needs its own Roboflow
key when it is actually called.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "src")
for _p in (_HERE, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from io import BytesIO

import cv2
import httpx
import imagehash
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Security, UploadFile, status
from fastapi.security import APIKeyHeader
from PIL import Image

# Support modules (copied verbatim from prod app/pipeline).
from logging_setup import setup_logging
from trail_merger import merge_trails
from image_hash_checker import (
    content_type_for_bytes,
    load_hash_db,
    parse_s3_url,
    read_image_metadata,
    save_hash_db,
    upload_bytes_to_s3,
    write_image_metadata,
)

# GPU booth pipeline + Roboflow hall model (from src/).
import booth_backend
import roboflow_hall

load_dotenv()
os.environ.setdefault("ONNXRUNTIME_EXECUTION_PROVIDERS", "CPUExecutionProvider")

_LOG_PATH = setup_logging()
logger = logging.getLogger(__name__)
logger.info("Booth-GPU API starting; deep logs at %s", _LOG_PATH)

# ── AUTH ──────────────────────────────────────────────────────────
_API_KEY = os.getenv("AUTHENTICATION_API_KEY")
if not _API_KEY:
    raise ValueError("AUTHENTICATION_API_KEY environment variable is not set")

_api_key_header = APIKeyHeader(name="Authentication-API-Key", auto_error=False)


async def require_api_key(key: str = Security(_api_key_header)):
    if key != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key or Add API key in the Header",
        )


app = FastAPI(title="Booth-GPU API")

# ── ROBOFLOW MODELS (lazy: download/connect on first use) ─────────
_MODEL_SPECS = {
    "icon": ("ROBOFLOW_ICON_API_KEY", "plotmymap_synthetic/2"),
    "trail": ("ROBOFLOW_TRAIL_API_KEY", "plotourmap-trails-largedataset/3"),
}
_models: dict = {}


def _get_rf_model(kind: str):
    if kind not in _models:
        env_name, model_id = _MODEL_SPECS[kind]
        key = os.getenv(env_name)
        if not key:
            raise HTTPException(status_code=500, detail=f"{env_name} is not set")
        from inference import get_model
        logger.info("Loading Roboflow model %s (%s)", kind, model_id)
        _models[kind] = get_model(model_id=model_id, api_key=key)
    return _models[kind]


http_client = httpx.AsyncClient(timeout=60)

# ── CONFIG ────────────────────────────────────────────────────────
TRAIL_MERGE_CONFIG = {"cross_cluster_merge": True}
HASH_CHECK_MODE = "always_run"  # always_run | skip_if_present | reject_if_present
S3_WRITEBACK_ENABLED = os.getenv("S3_WRITEBACK_ENABLED", "false").lower() == "true"
HALL_RASTER_MAX_EDGE = int(os.getenv("HALL_RASTER_MAX_EDGE", "2048"))
BOOTH_DPI = int(os.getenv("BOOTH_DPI", "250"))
BOOTH_FP_POLICY = os.getenv("BOOTH_FP_POLICY", "shape")

PERSIST_ENABLED = os.getenv("PERSIST_EXECUTIONS", "true").lower() == "true"
PERSIST_IN_DIR = os.getenv("PERSIST_IN_DIR", "/data/in")
PERSIST_OUT_DIR = os.getenv("PERSIST_OUT_DIR", "/data/out")
_VALID_POLICIES = {"none", "strict", "shape", "adaptive"}

_hash_db_lock = asyncio.Lock()


def _persist_execution(endpoint, input_bytes, input_name, result) -> None:
    if not PERSIST_ENABLED:
        return
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe = "".join(c for c in (input_name or "input")
                       if c.isalnum() or c in ("-", "_", ".")) or "input"
        base = f"{ts}_{endpoint}_{safe}"
        os.makedirs(PERSIST_IN_DIR, exist_ok=True)
        os.makedirs(PERSIST_OUT_DIR, exist_ok=True)
        if input_bytes is not None:
            with open(os.path.join(PERSIST_IN_DIR, base), "wb") as fh:
                fh.write(input_bytes)
        with open(os.path.join(PERSIST_OUT_DIR, base + ".json"), "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, default=str, indent=2)
    except Exception:
        logger.exception("Failed to persist execution for %s", endpoint)


# ── HELPERS (copied from prod) ────────────────────────────────────
def _aws_creds_present() -> bool:
    return bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))


def _s3_download_bytes(bucket: str, key: str, region: str) -> bytes:
    import boto3
    region = region or os.getenv("AWS_REGION") or None
    s3 = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


async def fetch_image(url: str) -> BytesIO:
    loop = asyncio.get_event_loop()
    url = (url or "").strip()
    if url and "://" not in url:
        url = "https://" + url.lstrip("/")
    parsed = parse_s3_url(url) if _aws_creds_present() else None
    if parsed:
        bucket, key, region = parsed
        try:
            data = await loop.run_in_executor(
                None, _s3_download_bytes, bucket, key, region)
            return BytesIO(data)
        except Exception as e:
            logger.warning("S3 fetch failed for %s/%s (%s); falling back to HTTPS",
                           bucket, key, e)
    try:
        async with http_client.stream("GET", url) as response:
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch image")
            buf = BytesIO()
            async for chunk in response.aiter_bytes():
                buf.write(chunk)
            buf.seek(0)
            return buf
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image fetch failed: {e}")


async def run_model_async(model, image_data, confidence: float):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: model.infer(image_data, confidence=confidence))


def compute_hash_from_bytes(raw_bytes: bytes) -> str:
    img = Image.open(BytesIO(raw_bytes)).convert("RGB")
    return str(imagehash.phash(img))


async def get_or_compute_hash(raw_bytes: bytes, extra: dict = None):
    loop = asyncio.get_event_loop()
    existing_hash, existing_date = await loop.run_in_executor(
        None, read_image_metadata, raw_bytes)
    if existing_hash and existing_date:
        return existing_hash, existing_date, raw_bytes, False
    phash = await loop.run_in_executor(None, compute_hash_from_bytes, raw_bytes)
    updated_bytes = await loop.run_in_executor(
        None, lambda: write_image_metadata(raw_bytes, phash, None, extra))
    _, written_date = await loop.run_in_executor(
        None, read_image_metadata, updated_bytes)
    was_written = updated_bytes is not raw_bytes and len(updated_bytes) != len(raw_bytes)
    return phash, written_date, updated_bytes, was_written


async def check_and_register_hash(img_hash: str) -> bool:
    loop = asyncio.get_event_loop()
    async with _hash_db_lock:
        hash_db = await loop.run_in_executor(None, load_hash_db)
        if img_hash in hash_db:
            return True
        hash_db.add(img_hash)
        await loop.run_in_executor(None, save_hash_db, hash_db)
        return False


async def writeback_to_s3(image_url: str, modified_bytes: bytes) -> dict:
    if not S3_WRITEBACK_ENABLED:
        return {"attempted": False, "reason": "writeback disabled"}
    parsed = parse_s3_url(image_url)
    if not parsed:
        return {"attempted": False, "reason": "URL not resolvable to S3"}
    bucket, key, region = parsed
    ctype = content_type_for_bytes(modified_bytes)
    loop = asyncio.get_event_loop()
    upload_result = await loop.run_in_executor(
        None, upload_bytes_to_s3, bucket, key, modified_bytes, region, ctype)
    response = {"attempted": True, "success": upload_result["success"],
                "bucket": bucket, "key": key, "region": region,
                "content_type": ctype}
    if not upload_result["success"]:
        response["error"] = upload_result["error"]
        response["error_code"] = upload_result["error_code"]
    return response


def _deep_serialize(obj):
    if isinstance(obj, dict):
        return {k: _deep_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_serialize(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return {k: _deep_serialize(v) for k, v in obj.__dict__.items()
                if not k.startswith("_")}
    return obj


def serialize_model_output(model_output) -> list:
    if model_output is None:
        return []
    if isinstance(model_output, list) and all(isinstance(x, dict) for x in model_output):
        return model_output
    items = model_output if isinstance(model_output, list) else [model_output]
    serialized = []
    for item in items:
        for method in ["model_dump", "dict"]:
            if hasattr(item, method):
                try:
                    serialized.append(getattr(item, method)())
                    break
                except Exception:
                    continue
        else:
            if hasattr(item, "json"):
                try:
                    serialized.append(json.loads(item.json()))
                    continue
                except Exception:
                    pass
            if hasattr(item, "__dict__"):
                try:
                    serialized.append(_deep_serialize(item.__dict__))
                    continue
                except Exception:
                    pass
            if isinstance(item, dict):
                serialized.append(item)
            else:
                logger.warning(f"Could not serialize: {type(item)}")
    return serialized


def extract_trail_data(serialized_output: list):
    if not serialized_output:
        return {}, []
    block = serialized_output[0] if isinstance(serialized_output, list) else serialized_output
    if isinstance(block, dict):
        preds = block.get("predictions", [])
    else:
        preds = []
        block = {}
    return block, preds


# ── INPUT COLLECTION (copied from prod) ───────────────────────────
def _looks_like_pdf(b: bytes) -> bool:
    return bool(b) and b[:4] == b"%PDF"


async def _collect_inputs(image_url, pdf_url):
    pdf_bytes = pdf_name = img_bytes = img_name = None
    if pdf_url:
        pdf_bytes = (await fetch_image(pdf_url)).getvalue()
        pdf_name = os.path.basename(pdf_url.split("?")[0]) or "input.pdf"
    if image_url and pdf_bytes is None:
        b = (await fetch_image(image_url)).getvalue()
        nm = os.path.basename(image_url.split("?")[0]) or "input"
        if _looks_like_pdf(b):
            pdf_bytes, pdf_name = b, nm
        else:
            img_bytes, img_name = b, nm
    return pdf_bytes, pdf_name, img_bytes, img_name


async def _hash_fields(src_bytes: bytes, is_pdf: bool, render_bgr):
    loop = asyncio.get_event_loop()
    img_hash = None
    img_date = None
    try:
        if not is_pdf:
            img_hash, img_date, _bytes, _w = await get_or_compute_hash(
                src_bytes, extra={"type": "Indoor"})
        elif render_bgr is not None:
            def _ph():
                rgb = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2RGB)
                return str(imagehash.phash(Image.fromarray(rgb)))
            img_hash = await loop.run_in_executor(None, _ph)
    except Exception:
        logger.exception("hash computation failed; falling back to sha1")
    if not img_hash:
        img_hash = hashlib.sha1(src_bytes).hexdigest()
    is_present = await check_and_register_hash(img_hash)
    return img_hash, img_date, ("present" if is_present else "absent")


# ── /debug_predict (unchanged from prod) ──────────────────────────
@app.post("/debug_predict")
async def debug_predict(
    file: UploadFile = File(None),
    image_url: str = Form(None),
    _: None = Security(require_api_key),
):
    if not file and not image_url:
        raise HTTPException(status_code=400, detail="Provide file or image_url")
    try:
        if file:
            image_bytes = BytesIO(await file.read())
        else:
            image_bytes = await fetch_image(image_url)
        trail_output_raw = await run_model_async(
            _get_rf_model("trail"), BytesIO(image_bytes.getvalue()), 0.25)
        serialized = serialize_model_output(trail_output_raw)
        trail_block, preds = extract_trail_data(serialized)
        image_info = trail_block.get("image", {})
        result = {
            "raw_type": str(type(trail_output_raw)),
            "serialized_keys": list(trail_block.keys()) if trail_block else [],
            "image_info": image_info,
            "js_w": image_info.get("width"),
            "js_h": image_info.get("height"),
            "prediction_count": len(preds),
            "predictions_with_points": sum(1 for p in preds if p.get("points")),
            "first_pred_keys": list(preds[0].keys()) if preds else [],
            "first_pred_point_count": len(preds[0]["points"]) if preds and preds[0].get("points") else 0,
            "sample_point": preds[0]["points"][0] if preds and preds[0].get("points") else None,
        }
        _persist_execution("debug_predict", image_bytes.getvalue(),
                           (file.filename if file else os.path.basename(image_url.split("?")[0])), result)
        return result
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


# ── /predict (icons + trails, unchanged from prod) ────────────────
@app.post("/predict")
async def predict(
    file: UploadFile = File(None),
    image_url: str = Form(None),
    _: None = Security(require_api_key),
):
    if not file and not image_url:
        raise HTTPException(status_code=400, detail="Provide file or image_url")
    try:
        if file:
            image_bytes = BytesIO(await file.read())
        else:
            image_bytes = await fetch_image(image_url)
        raw_bytes = image_bytes.getvalue()

        img_hash, img_date, raw_bytes, was_written = await get_or_compute_hash(raw_bytes)
        s3_result = {"attempted": False, "reason": "no metadata change"}
        if was_written and image_url:
            s3_result = await writeback_to_s3(image_url, raw_bytes)

        is_present = await check_and_register_hash(img_hash)
        hash_status = "present" if is_present else "absent"

        if is_present and HASH_CHECK_MODE == "reject_if_present":
            raise HTTPException(status_code=409, detail={
                "message": "Duplicate image — already processed.",
                "hash": img_hash, "date": img_date, "hash_status": hash_status})
        if is_present and HASH_CHECK_MODE == "skip_if_present":
            return {"hash": img_hash, "date": img_date, "hash_status": hash_status,
                    "s3_writeback": s3_result, "icon_output": [], "trail_output": [],
                    "skipped": True}

        trail_task = run_model_async(_get_rf_model("trail"), BytesIO(raw_bytes), 0.25)
        icon_task = run_model_async(_get_rf_model("icon"), BytesIO(raw_bytes), 0.30)
        trail_output_raw, icon_output_raw = await asyncio.gather(trail_task, icon_task)

        trail_serialized = serialize_model_output(trail_output_raw)
        icon_serialized = serialize_model_output(icon_output_raw)
        trail_block, trail_predictions = extract_trail_data(trail_serialized)
        icon_block, _ = extract_trail_data(icon_serialized)

        map_image = Image.open(BytesIO(raw_bytes)).convert("RGB")
        merged_predictions = merge_trails(predictions=trail_predictions,
                                          map_image=map_image,
                                          config=TRAIL_MERGE_CONFIG,
                                          trail_block=trail_block)
        merged_trail_block = dict(trail_block)
        merged_trail_block["predictions"] = merged_predictions

        result = {"hash": img_hash, "date": img_date, "hash_status": hash_status,
                  "s3_writeback": s3_result, "icon_output": [icon_block],
                  "trail_output": [merged_trail_block]}
        _persist_execution("predict", raw_bytes,
                           (file.filename if file else os.path.basename(image_url.split("?")[0])), result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


# ── /hall_with_booth_predict (Roboflow hall + GPU booth pipeline) ─
@app.post("/hall_with_booth_predict")
async def hall_with_booth_predict(
    image_url: str = Form(None),
    pdf_url: str = Form(None),
    page: int = Form(0),
    fp_policy: str = Form(None),
    hall_conf: float = Form(0.50),
    _: None = Security(require_api_key),
):
    pdf_bytes, pdf_name, img_bytes, img_name = await _collect_inputs(image_url, pdf_url)
    if pdf_bytes is None and img_bytes is None:
        raise HTTPException(status_code=400, detail="Provide image_url and/or pdf_url")

    use_pdf = pdf_bytes is not None
    src_bytes = pdf_bytes if use_pdf else img_bytes
    src_name = pdf_name if use_pdf else img_name
    fp = fp_policy if (fp_policy in _VALID_POLICIES) else BOOTH_FP_POLICY

    try:
        loop = asyncio.get_event_loop()
        # 1. GPU booth detection (long pole) — booths + page render.
        payload, render_bgr = await loop.run_in_executor(
            None, booth_backend.run_booths, src_bytes, src_name, use_pdf,
            page, fp, BOOTH_DPI)
        booths = payload["booths"]

        # 2. Roboflow hall detection on the downscaled render, scaled back.
        halls = await loop.run_in_executor(
            None, roboflow_hall.detect_halls, render_bgr,
            float(hall_conf), HALL_RASTER_MAX_EDGE)
        hall_predictions = [{
            "predictions": [{
                "x": (h["x1"] + h["x2"]) / 2.0, "y": (h["y1"] + h["y2"]) / 2.0,
                "width": h["x2"] - h["x1"], "height": h["y2"] - h["y1"],
                "confidence": h.get("confidence", 0.0), "class": h.get("class", "hall"),
                "points": [{"x": p[0], "y": p[1]} for p in h.get("coordinates", [])],
            } for h in halls],
            "image": {"width": (payload.get("image") or {}).get("w"),
                      "height": (payload.get("image") or {}).get("h")},
        }] if halls else []

        hall_booth_map = roboflow_hall.build_hall_booth_map(halls, booths)
        booth_detections = {"count": len(booths), "booths": booths}

        img_hash, img_date, hash_status = await _hash_fields(
            src_bytes, use_pdf, render_bgr)

        result = {
            "hash": img_hash,
            "date": img_date,
            "hash_status": hash_status,
            "s3_writeback": {"attempted": False, "reason": "writeback disabled"},
            "hall_predictions": hall_predictions,
            "booth_detections": booth_detections,
            "hall_booth_map": hall_booth_map,
        }
        _persist_execution("hall_with_booth_predict", src_bytes, src_name, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Hall + booth prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "gpu", "log_path": _LOG_PATH}
