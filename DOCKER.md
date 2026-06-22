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

## Run
Put a floor plan in `./data`, then:
```bash
# Vector PDF — dense plan, best recall:
docker compose run --rm booth --input /data/IIJS_2017.pdf --fp-policy shape --outdir /data/out

# Raster image / flattened PDF — OCR fires automatically (--ocr auto is default):
docker compose run --rm booth --input /data/PU-TECH-2027.jpg --fp-policy strict --outdir /data/out
```
Outputs (`*_boxes.json`, `*_boxes.png`, `*_mask.png`) land in `./data/out`.

`./models` is mounted so EasyOCR's downloaded weights persist between runs
(first OCR run downloads them once).

## Without compose (plain docker)
```bash
docker run --rm --gpus all \
  -v "$PWD/data:/data" -v "$PWD/models:/models" \
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
