# Running booth-gpu on a GPU server (Docker)

## Prerequisites (host)
- NVIDIA GPU + driver, and the **NVIDIA Container Toolkit** installed.
- Verify GPU is visible to Docker:
  ```bash
  docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
  ```

## Build
```bash
git clone https://github.com/mayanksoninls161-beep/booth-gpu.git
cd booth-gpu
docker compose build          # or: docker build -t booth-gpu:latest .
```
The image bundles CUDA 12.1 + torch/torchvision, OpenCV, PyMuPDF (PDF render),
EasyOCR (raster/scanned labeling), and cupy + cuCIM (fast RAPIDS CCL).

## Run (recommended: `docker compose up`)
```bash
mkdir -p data out models
cp /path/to/your_plan.pdf data/

# builds the image the first time, then runs the pipeline on $INPUT:
INPUT=/data/your_plan.pdf docker compose up --build

# later runs (image already built) — just:
INPUT=/data/another_plan.jpg docker compose up
```
Knobs are env vars (all optional except INPUT):
| Env | Default | Meaning |
|---|---|---|
| `INPUT` | (required) | Path inside the container, e.g. `/data/plan.pdf`. |
| `FP_POLICY` | `shape` | `adaptive` / `shape` / `strict` / `none`. |
| `OCR` | `auto` | `auto` / `on` / `off`. |
| `EXTRA_ARGS` | (none) | Any extra run_real flags, e.g. `"--dpi 300 --workers 8"`. |

Examples:
```bash
INPUT=/data/PU-TECH-2027.jpg FP_POLICY=strict docker compose up
INPUT=/data/IIJS_2017.pdf EXTRA_ARGS="--dpi 300" docker compose up
```
Outputs (`*_boxes.json`, `*_boxes.png`, `*_mask.png`) land in `./out`.
`./models` persists EasyOCR weights (downloaded once on first OCR run).

## One-off run via compose (no long-lived service)
```bash
docker compose run --rm booth --input /data/plan.pdf --fp-policy shape --outdir /data/out
```

## Without compose (plain docker)
```bash
docker run --rm --gpus all \
  -v "$PWD/data:/data" -v "$PWD/out:/data/out" -v "$PWD/models:/models" \
  booth-gpu:latest --input /data/plan.pdf --fp-policy shape --outdir /data/out
```

## Useful flags
| Flag | Use |
|---|---|
| `--fp-policy {adaptive,shape,strict,none}` | Keep-policy. `shape` = best recall on dense plans. |
| `--ocr {auto,on,off}` | EasyOCR fallback (auto: only when no PDF text layer). |
| `--ccl {auto,cucim,prop}` | `cucim` = fast GPU connected-components (default auto). |
| `--geo-backend {auto,gpu,cpu}` | GPU geometric pass. |
| `--dpi N` | PDF render DPI (default 250). |
| `--workers N` | CPU tile workers (CPU geo backend only). |

## Notes
- `ROBOFLOW_HALL_API_KEY` is read from the host env (compose passes it through);
  only needed with `--roboflow-hall`. Never bake it into the image.
- Keep AWS/S3 writeback OFF; the container only reads/writes the mounted `/data`.
