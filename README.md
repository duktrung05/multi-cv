# ReaS IAS VAS — REAL / AI_EDITED

The active project detects AI face manipulation in images using DGM4.
Labels: `REAL` (original) and `AI_EDITED` (SimSwap or StyleCLIP manipulation).
The repository directory name is unchanged. Product inspection code is archived
under `docs/legacy_product` and is no longer connected to training or the app.

## Model and current status

Frozen pretrained DINOv3 ViT-S/16, CLS + mean-patch features, a trainable two-class
linear head, Cross-Entropy loss and softmax scores. All active entrypoints use
`configs/dgm4_binary.yml`. Old product/moderation checkpoints are rejected.

The 10,000-image dataset exists, but its audit found cross-split near-duplicate
source images requiring resampling. Full training enforces the audit gate.
Pretrained and trained weights are not bundled. No real training or accuracy
claim is implied by the synthetic/unit tests. The app fails clearly if a trained
checkpoint is missing rather than returning random predictions.

## Install and audit

Run from this directory with Python 3.12:

```powershell
python -m pip install -r requirements.txt
python tools/dataset/validate_dgm4.py
python -m unittest discover -s tests -p 'test_dgm4*.py' -v
```

See `outputs/dgm4_binary/audit/audit.json` and `visual_review.json`.
Resolve confirmed cross-split duplicates and audit again before full training.
See [server training guide](docs/DGM4_SERVER_TRAIN.md) for pretrained setup,
GPU configuration, feature caching, resume and evaluation.

## Train and evaluate

```powershell
python train.py -c configs/dgm4_binary.yml
python train.py -c configs/dgm4_binary.yml -u test_only=true eval_split=test resume=outputs/dgm4_binary/baseline/best.pth
```

`train_ias.py` delegates to the same DGM4 trainer. Use validation to select the
model; use test only for final evaluation. Training writes `best.pth`, `last.pth`,
metrics and history. Copy the selected `best.pth` to `ckpts/dgm4_binary.pth`
for deployment, or set `CLASSIFICATION_CKPT` to its path for the API.

## Predict and serve

```powershell
python infer_dgm4.py --image path/to/image.jpg --weights ckpts/dgm4_binary.pth
python gradio_app.py --weights ckpts/dgm4_binary.pth
python main.py
```

Settings are in `.env.example`; `.env` loads automatically for the API, with
process environment taking precedence. Main settings: `CLASSIFICATION_CONFIG`,
`CLASSIFICATION_CKPT`, `MODEL_DEVICE`, `APP_HOST`, `APP_PORT`.

API docs: http://127.0.0.1:8126/docs
- `POST /classify/image`: multipart `file`; response `ai_result` contains
  `label`, `score`, `scores`, `ai_edited_threshold`, `task_domain`.
- `POST /classify/video`: URL payload; samples frames and returns the frame with
  highest AI_EDITED score. This is a heuristic, not a validated video detector.
- Asynchronous `/jobs` endpoints retain their job envelope; result details now
  use `classification`, replacing the former `inspection` field.
- Old `/inspect/*` endpoints are removed. Hidden `/test/classify/*` aliases remain.

Scores are model outputs, not calibrated proof of authenticity. Classification
covers whole images; no edit mask or localization is produced. Performance on
other editing tools or fully generated images must be measured separately.

## Docker

```powershell
docker compose up --build
docker compose -f compose.yaml -f compose.gpu.yaml up --build
docker compose -f compose.train.yaml run --rm trainer
```

API containers require trained weights under `ckpts/`; GPU containers require a
compatible host driver/runtime. Docker training setup is described in the server
guide. Local images, weights, outputs and `.env` are excluded from Git/build context.
