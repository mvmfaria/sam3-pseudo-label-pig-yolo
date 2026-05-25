# SAM3-Assisted Training of Lightweight YOLO Models for Precision Pig Farming

Knowledge distillation experiment using SAM3 as a teacher model to generate pseudo-labels for training compact YOLOv8 detection models on the PigLife dataset.

## Requirements

| | |
|---|---|
| **Python** | 3.10+ |
| **GPU** | NVIDIA CUDA (≥ 8 GB VRAM recommended) |
| **Disk** | ≥ 20 GB free (dataset is ~16 GB) |
| **Accounts** | PigLife dataset access + HuggingFace token for SAM3 |

## Setup

**1. Install dependencies**

```bash
uv sync
```

**2. Configure environment**

Copy `.env.example` to `.env` and fill in the two required values:

```bash
cp .env.example .env
```

| Variable | Where to get it |
|---|---|
| `PIGLIFE_URL` | Request access at [AIFARMS Data Portal](https://data.aifarms.org/view/piglife) |
| `HF_TOKEN` | [HuggingFace settings → Access Tokens](https://huggingface.co/settings/tokens) |

---

## Pipeline

Each step is an `invoke` task. Run them in order or use `uv run inv all` to execute the full pipeline automatically.

### Step 1 — Build dataset

Downloads the PigLife zip, extracts images, sanitizes annotations, splits into train/val/test, and converts to YOLO format.

```bash
uv run inv build
```

Output: `datasets/piglife/yolo/human/`

---

### Step 2 — Generate SAM3 pseudo-labels

Runs SAM3 (zero-shot, text prompt `"pig"`) on all images to produce pseudo-annotations, then converts them to YOLO format.

```bash
uv run inv label
```

Output: `datasets/piglife/coco/sam3/annotations/` and `datasets/piglife/yolo/sam3/`

> This step requires a GPU and a HuggingFace token with access to `facebook/sam3`.

---

### Step 3 — Train YOLOv8 models

Trains YOLOv8n, YOLOv8s, and YOLOv8m on both annotation sources (human and SAM3). Resumes automatically from the last checkpoint if interrupted.

```bash
uv run inv train
```

To train a single source:

```bash
uv run inv train --source human
uv run inv train --source sam3
```

Output: `runs/{source}/{model}/weights/best.pt`

---

### Step 4 — Evaluate models

Runs YOLO validation on the test split for every trained model and saves COCO-format prediction JSONs.

```bash
uv run inv evaluate
```

To evaluate a single source:

```bash
uv run inv evaluate --source human
uv run inv evaluate --source sam3
```

Output: `runs/{source}/{model}/predictions.json`

---

### Step 5 — Compute metrics and benchmarks

Calculates COCO accuracy metrics (mAP 50-95, mAP 50, mAP 75) and latency benchmarks (forward-pass and full pipeline) for all models.

```bash
uv run inv metrics
```

Output: `reports/output/metrics/`

---

### Step 6 — Generate report tables

Produces publication-ready LaTeX tables from the computed metrics.

```bash
uv run inv report
```

Output: `reports/output/latex/`

---

### Run everything at once

```bash
uv run inv all
```

---

## Extra tools

**Interactive prediction gallery** — visualize ground truth vs. model predictions on test images:

```bash
uv run streamlit run extra/gallery.py
```

**List all available tasks:**

```bash
uv run inv --list
```
