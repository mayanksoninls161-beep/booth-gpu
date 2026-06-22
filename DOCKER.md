# Booth-GPU on a GPU server (Docker)

Serves the **same API as production**, with the booth endpoint running the GPU
pipeline:

| Method | Endpoint | What |
|---|---|---|
| POST | `/predict` | icons + trails (Roboflow) — copied from prod |
| POST | `/debug_predict` | trail debug (Roboflow) — copied from prod |
| POST | `/hall_with_booth_predict` | Roboflow hall + **GPU** booth pipeline |
| GET | `/health` | liveness |

All POSTs require header `Authentication-API-Key: <AUTHENTICATION_API_KEY>`.

## Prerequisites (host)
- NVIDIA GPU + driver + **NVIDIA Container Toolkit**. Verify:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
  ```

## Build & run the API
```bash
git clone <repo-url> booth-gpu && cd booth-gpu
cp .env.example .env          # fill in AUTHENTICATION_API_KEY + Roboflow keys
mkdir -p data models

docker compose up -d --build api
curl -fsS localhost:8000/health
```

### Call the booth endpoint
```bash
# PDF floor plan (best recall with shape policy):
curl -s -X POST localhost:8000/hall_with_booth_predict \
  -H "Authentication-API-Key: $AUTHENTICATION_API_KEY" \
  -F "pdf_url=https://.../IIJS_2017.pdf" \
  -F "fp_policy=shape"

# Raster image (OCR fires automatically when there's no PDF text layer):
curl -s -X POST localhost:8000/hall_with_booth_predict \
  -H "Authentication-API-Key: $AUTHENTICATION_API_KEY" \
  -F "image_url=https://.../PU-TECH-2027.jpg" \
  -F "fp_policy=strict"
```
Response shape matches prod: `{hash, date, hash_status, s3_writeback,
hall_predictions, booth_detections:{count,booths}, hall_booth_map}`.
Hall boxes, booths, and the hall→booth map all share the render pixel space.

Form fields: `image_url`, `pdf_url` (PDF wins when both), `page` (default 0),
`fp_policy` (none/strict/shape/adaptive; default `BOOTH_FP_POLICY=shape`),
`hall_conf` (default 0.50).

## One-off CLI detection (no API / no Roboflow key)
```bash
# put a plan in ./data/in, then:
docker compose run --rm booth-cli --input /data/in/plan.pdf --fp-policy shape --outdir /data/out
```

## Config (env / .env)
| Var | Default | Meaning |
|---|---|---|
| `AUTHENTICATION_API_KEY` | (required) | API key clients must send. |
| `ROBOFLOW_ICON_API_KEY` | — | `/predict` icons. |
| `ROBOFLOW_TRAIL_API_KEY` | — | `/predict`, `/debug_predict` trails. |
| `ROBOFLOW_HALL_API_KEY` | — | `/hall_with_booth_predict` halls. |
| `BOOTH_FP_POLICY` | `shape` | Default keep-policy for the booth pipeline. |
| `BOOTH_DPI` | `250` | PDF render DPI. |
| `HALL_RASTER_MAX_EDGE` | `2048` | Downscale cap before hall inference. |
| `S3_WRITEBACK_ENABLED` | `false` | Keep OFF (constraint). Private-S3 *reads* still work with AWS creds. |

## Notes
- Roboflow models load lazily on first use, so the API boots even without keys;
  the booth pipeline needs no Roboflow key at all.
- `./models` + the `easyocr-cache`/`model-cache` volumes persist downloaded
  weights across runs (first OCR / Roboflow call downloads them once).
- Booth detection runs `src/run_real.py` as a subprocess per request (exact CLI
  behaviour); expect a few seconds of process/model warm-up on top of detection.
- Never commit the real `.env` (it's gitignored).
