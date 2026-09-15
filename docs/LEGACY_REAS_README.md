## ReaS IAS/VAS — 18-class image classification (DINOv3 + RTDETR)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c)](https://pytorch.org/)
[![TorchVision](https://img.shields.io/badge/TorchVision-0.20.1-5c6bc0)](https://pytorch.org/vision/stable/index.html)
[![Task](https://img.shields.io/badge/Task-Image%20Classification-1f6feb)](#)
[![Labels](https://img.shields.io/badge/Classes-18-success)](#dataset)

Train/evaluate an **18-class** ReaS classification model using a DEIMv2-style classifier with a **DINOv3** backbone. The repo also includes utilities to **rebuild ImageFolder datasets from zipped JP-labeled sources**, and evaluation tooling with per-class accuracy + macro F1.

### Highlights

- **Deterministic class index order** via `ReasImageFolderClassification` (prevents accidental label reordering).
- **Mispacked zip tolerant dataset builder** (`--mapped-source auto`) to prefer the folder name inside a zip when zip filename is wrong.
- **Imbalance-friendly training**:
  - Focal loss (`ClassificationFocalLoss`) and weighted CE (`ClassificationWeightedCrossEntropy`)
  - Macro-F1 reporting in validation + ReduceLROnPlateau monitoring Macro-F1
  - Stronger augmentation for hard classes (e.g. `NUDE`, `ANAL`, `SEXY`)

### Repo layout (quick map)

- **Training entrypoint**: `train.py`
- **Main config**: `configs/deimv2/reas_dinov3_vit_s.yml`
- **Dataset**: `engine/data/dataset/reas_imagefolder_classification.py`
- **Classification solver**: `engine/solver/clas_solver.py`
- **Criteria (losses)**: `engine/solver/classification_criterion.py`
- **Dataset prep**: `tools/dataset/prepare_reas_imagefolder_from_zip.py`
- **Evaluation script**: `test/test_model.py`

## Setup

Create an environment and install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Notes:
- This repo expects **PyTorch `2.5.1`** + **TorchVision `0.20.1`** (see `requirements.txt`).
- CUDA is optional but recommended for training.

## Dataset

The training config expects an **ImageFolder** structure:

```text
data/reas_imagefolder/
  train/<CLASS>/*.jpg
  valid/<CLASS>/*.jpg
  test/<CLASS>/*.jpg
  labels_map.json
  manifest_train.csv
  manifest_valid.csv
  manifest_test.csv
```

Class ordering is configured via YAML using `train_dataloader.dataset.class_list`
(and `val_dataloader.dataset.class_list`).

### Build `data/reas_imagefolder` from zipped JP sources

If you have the source zips under `data/2026AIモデル教育用画像/`:

```bash
python3 tools/dataset/prepare_reas_imagefolder_from_zip.py \
  --outer-zip "data/2026AIモデル教育用画像" \
  --output-root data/reas_imagefolder \
  --make-splits \
  --seed 42 \
  --label-mode mapped \
  --mapped-source auto \
  --labels-map tools/dataset/reas_labels_map.json
```

What this does:
- Extracts images into `train/valid/test` (default ratio `0.8/0.1/0.1`)
- Writes `data/reas_imagefolder/labels_map.json` + split manifests
- Uses `--mapped-source auto` to handle **mispacked zips** (e.g. zip name doesn’t match the top-level folder inside)

## Auto-labeling & Label Studio

Tooling to label raw images with the `classify` API and push images + labels into a
Label Studio project for review. All scripts live in `tools/dataset/`.

### 1. Classify a folder → annotations CSV

`classify_folder.py` calls the `/classify` API for each image and streams rows to a CSV
(`image,body_parts,orig_path`, where `body_parts` is a Label-Studio `{"choices": [...]}` blob).

```bash
python tools/dataset/classify_folder.py \
  --folder raw_data/FEMALE_GENITALS \
  --api http://10.0.65.25:8005/classify \
  --categories MALE_GENITALIA FEMALE_GENITALIA ANUS \
  --out labels/FEMALE_GENITALS_annotations.csv \
  --default-label FEMALE_GENITALS \
  --label-map FEMALE_GENITALIA=FEMALE_GENITALS ANUS=ANAL MALE_GENITALIA=MALE_GENITALS
```

- `--default-label` always adds a folder-wide label to every image's choices.
- `--label-map API=CLASS` normalizes API label names to class names (bare `--label-map`
  uses the built-in genitals map). Rows are flushed per image (resumable, parallel-safe).

### 2. Configure the Label Studio server (`.env`)

Scripts read `.env` at the repo root:

```bash
LABEL_STUDIO_URL=http://10.0.64.77:8081/projects/18   # base URL + project id
LABEL_STUDIO_USERNAME=you@example.com                 # username/password login, or…
LABEL_STUDIO_PASSWORD=secret
# LABEL_STUDIO_API_KEY=<token>                         # …a token (if legacy tokens enabled)
```

### 3. Test connectivity + upload one image

```bash
python tools/dataset/test_labelstudio_connection.py
```

Checks `/version`, logs in (session auth), verifies the project, then uploads **one**
labelled image so you can eyeball it before a full run. Add `--no-upload` for checks only.

### 4. Upload images + labels to a project

`upload_to_labelstudio.py` uploads each image and attaches its `choices` as predictions
(or `--as annotations`). The Choices tag name must match the project config (default
`--from-name body_parts`).

```bash
python tools/dataset/upload_to_labelstudio.py \
  --project 18 --labels-dir labels --as predictions \
  --batch-id reas_row20_24_v1 \
  --label-map FEMALE_GENITALS=FEMALE_GENITALIA ANAL=FEMALE_GENITALIA ANUS=FEMALE_GENITALIA \
              MALE_GENITALS=MALE_SEXUAL MALE_GENITALIA=MALE_SEXUAL ADULT=EXPLICIT
```

- **Input selection** — `--labels-dir labels` uploads every `*.csv` in a folder, or
  `--annotations <file> [<file> ...]` uploads one or more specific files:
  ```bash
  # single annotation file
  python tools/dataset/upload_to_labelstudio.py \
    --project 18 --annotations labels/ANAL_annotations.csv --as predictions
  ```
- **Test with one image first** — add `--max-images 1` (and a throwaway `--batch-id`) to
  upload a single task, eyeball it in the UI, then delete the batch (step 5).
- `--image-mode upload` (default) hosts the file on the server; `local-storage` / `url`
  reference it instead.
- `--ensure-label-config` rewrites the project labeling config from the CSV labels.
- **Duplicate images are skipped by content.** Images whose bytes were already uploaded
  (same picture saved under a different filename) are detected via an MD5 hash and skipped,
  even across runs (hashes persist in `./.ls_hashes_p<project>.txt`). The run summary reports
  `dup_content`. Disable with `--no-dedupe-content`.
- **Per-file manifests.** Each annotation CSV gets its own manifest at
  `manifests/<csv-stem>_p<project>.csv` recording every `task_id` it created, so you can
  later delete exactly the tasks from one annotation file (step 5). A `--batch-id` marker is
  also stamped into every task. Use `--manifest-dir` to change the folder, or `--manifest
  <file>` to force a single combined manifest for all inputs instead.

### 5. Delete an uploaded batch

`delete_labelstudio_batch.py` removes tasks created by an upload run. Dry-run by default;
pass `--yes` to actually delete.

```bash
# delete exactly the tasks from ONE annotation file (its per-file manifest)
python tools/dataset/delete_labelstudio_batch.py --manifest manifests/ANAL_annotations_p18.csv        # dry-run
python tools/dataset/delete_labelstudio_batch.py --manifest manifests/ANAL_annotations_p18.csv --yes  # delete

# fallback (no manifest): scan the project by the in-task batch marker
python tools/dataset/delete_labelstudio_batch.py --project 18 --batch-id reas_row20_24_v1 --yes
```

### 6. De-duplicate a project

`dedupe_labelstudio_tasks.py` finds tasks whose images are byte-identical (same picture
uploaded under different filenames) and keeps one task per group, deleting the rest. It only
touches tool-created tasks (those with a `_source_path` marker and a readable local file);
foreign tasks are left alone. Dry-run by default; pass `--yes` to delete.

```bash
# dry run: list duplicate groups and what would be removed
python tools/dataset/dedupe_labelstudio_tasks.py --project 18
# delete redundant duplicates, keeping the reviewed copy of each
python tools/dataset/dedupe_labelstudio_tasks.py --project 18 --yes
```

`--prefer reviewed` (default) keeps the task with the most annotations so review work is
never lost; use `--prefer newest`/`oldest` to keep by task id instead.

### 7. Pull reviewed annotations back to a CSV

`pull_reviewed_p18.py` (at the **repo root**, not `tools/dataset/`) exports the human-reviewed
labels from the project and writes them back to `labels/REVIEWED_annotations.csv` in the same
`image,body_parts,orig_path` format as the other label CSVs — a drop-in for
`generate_annotations.py` / training. It reuses the auth + URL helpers from
`tools/dataset/test_labelstudio_connection.py`, so the same `.env` (step 2) applies.

```bash
# straight pull (path-based matching only) -> labels/REVIEWED_annotations.csv
python pull_reviewed_p18.py

# full pull: also recover images whose upload filename was truncated, by content hash
python pull_reviewed_p18.py --recover-hash
```

What it does, end to end:

1. **Auth + export.** Logs in via `.env`, then `GET /api/projects/18/export` (JSON,
   `download_all_tasks=false` → only tasks that have annotations).
2. **Keep only *reviewed* annotations.** This project runs Label Studio **Community** (no
   accept/reject review workflow — `reviews`/`ground_truth`/`last_action` are all empty), so
   "reviewed" = **submitted by a human reviewer**, not by the `classify_api` service account.
   Each annotation's `completed_by` id is mapped to an email via `GET /api/users`; only
   annotations by a reviewer are kept. Default reviewer is `LABEL_STUDIO_USERNAME` from
   `.env`; override/add with `--reviewer <email>` (repeatable).
3. **Match each task to a local `raw_data/` image**, in priority order:
   - **`_source_path`** — the absolute path stamped into `task.data` at upload by
     `upload_to_labelstudio.py` (exact, preferred).
   - **filename stem** — for tasks uploaded without that marker, the Label Studio upload name
     (`<uuid8>-<stem>_<hash>.<ext>`) is de-uuid'd and matched against the same sanitized stem
     of `raw_data` basenames; used only when it resolves to exactly **one** file.
   - **content MD5** (only with `--recover-hash`) — for anything still unmatched, the image
     bytes are downloaded from the server and matched by MD5 against `raw_data`. This recovers
     uploads whose unique filename portion was **truncated** by Label Studio (e.g. many
     distinct `api=load...&img_id=X.jpg` files that collapsed to one
     `TEXT_api_load_img_admin_token_...` name). Byte-identical, so the match is exact.
4. **Write the CSV.** One row per matched task; every row has a real on-disk `orig_path` and a
   non-empty `{"choices": [...]}`. The run summary reports the match breakdown
   (`_source_path` / `filename` / `md5` / `md5-dup`) and any leftover unmatched tasks.

Useful flags:

- `--recover-hash` — enable the MD5 recovery pass (needs to download each still-unmatched
  image once). Raw-data hashes are cached in `.raw_md5_cache_p18.json` (keyed by size + mtime),
  so reruns don't re-hash unchanged files.
- `--reviewer <email>` — count a different/additional account as a reviewer (repeatable).
- `--keep-unmatched` — still emit a row for tasks with no local match (blank `orig_path`,
  `image` set to `task_<id>`), instead of dropping them.
- `--project`, `--out`, `--raw-dir`, `--md5-cache` — override the defaults
  (`18`, `labels/REVIEWED_annotations.csv`, `raw_data`, `.raw_md5_cache_p18.json`).

> **Duplicate-upload caveat.** If the same image was uploaded as several separate tasks and the
> reviewer labeled the copies slightly differently, the CSV will contain multiple rows for that
> `orig_path` with differing choices. `generate_annotations.py` merges by basename and takes the
> **union** of choices, so they collapse harmlessly at training-prep time.


## Training

Default training (ReaS classification):

```bash
python3 train.py -c configs/deimv2/reas_dinov3_vit_s.yml
```

### What gets printed / saved

At startup, `train.py` prints a **readable config summary** and writes a full resolved config to:

- `./outputs/resolved_config.yml` (or the `output_dir` you set)

Model weights (lightweight):
- `last.pth`
- `best.pth`

TensorBoard logs:
- `./outputs/summary/<run_name>/`

## Evaluation

Evaluate a checkpoint on an ImageFolder split and write a report:

```bash
python3 test/test_model.py \
  --config configs/deimv2/reas_dinov3_vit_s.yml \
  --weights outputs/summary/<run_name>/best.pth \
  --data_dir data/reas_imagefolder/test \
  --device cuda:0 \
  --out_dir outputs/test
```

The report includes:
- `overall_acc`
- `macro_f1`
- per-class accuracy
- top confusions per true class
- simple forward speed stats (optional warmup)

## Configuration tips

- **Focal loss / weighted CE**:
  - See `engine/solver/classification_criterion.py`
  - In `configs/deimv2/reas_dinov3_vit_s.yml` you can switch `criterion` between:
    - `ClassificationFocalLoss`
    - `ClassificationWeightedCrossEntropy`
    - `ClassificationCrossEntropy`
- **ReduceLROnPlateau monitor**:
  - `ClasSolver` steps Plateau using `macro_f1` (fallback to `acc`).
- **Hard-class augmentation**:
  - Configure in `configs/base/dataloader_cls.yml`:
    - `hard_classes: [NUDE, PRIVATE_PART, SEDUCTIVE, PORN]`
    - `hard_transforms: { type: Compose, ops: [...] }`

## Troubleshooting

- **`ReduceLROnPlateau.__init__() got an unexpected keyword argument 'T_max'`**
  - This can happen when YAML merging leaves stale keys from another scheduler.
  - The factory now filters unknown kwargs (see `engine/core/workspace.py`), but you should still keep configs clean.

- **`TypeError: 'dict' object is not callable` for `hard_transforms`**
  - Ensure `hard_transforms` is built via injection (dataset supports it).

- **Class mismatch between checkpoint and dataset**
  - `test/test_model.py` will error if number of dataset folders differs from checkpoint head classes.
  - Rebuild dataset or retrain so they match.