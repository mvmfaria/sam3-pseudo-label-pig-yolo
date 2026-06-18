# SAM3-Assisted Training of Lightweight YOLO Models for Precision Pig Farming

<table width="100%"><tr>
<td align="center" width="50%"><a href="reports/report.pdf"><img src="img/cover.png" height="300"/></a></td>
<td align="center" width="50%"><img src="img/8ed1955d-7fa5-409b-89bf-d51c32f075d3.png" height="300"/></td>
</tr></table>

[![arXiv](https://img.shields.io/badge/arXiv-2605.25860-b31b1b.svg)](https://doi.org/10.48550/arXiv.2605.25860)

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

## Extra Datasets Experiments

You can also run experiments with the **BamaPig2D** and **FaroPigSeg** datasets. These datasets are converted from their original formats (COCO and YOLO Seg) to YOLO Detection format automatically.

**1. Prepare the datasets**

Ensure you have downloaded and unzipped them into `datasets/BamaPig2D` and `datasets/FaroPigSeg` as described in the "Expanding datasets experiments" section below. Then run:

```bash
uv run inv setup_extra
```

**2. Train models**

```bash
uv run inv train --source bama
uv run inv train --source faro
```

**3. Evaluate models**

```bash
uv run inv evaluate --source bama
uv run inv evaluate --source faro
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

---

## Expanding datasets experiments

### FaroPigSeg

```bash
cd zips/
wget https://data.chalearnlap.cvc.uab.cat/FaroPig/FaroPigSeg.zip
```

After the download complete
```bash
unzip FaroPigSeg.zip -d ../datasets/
```

### BamaPig2D

```bash
cd zips/
gdown 'https://drive.google.com/file/d/1yWBtNpYpkUdGKDqUAE7ya5m_fwinn0HN/view?usp=sharing'
```

After the download complete
```bash
unzip BamaPig2D.zip -d ../datasets/
```

Setup both datasets to YOLO's format
```
```bash
uv run python -m invoke setup-bama-and-faro
```

## Citation

If you use this work, please cite:

```bibtex
@article{faria2026sam3,
  title={SAM3-Assisted Training of Lightweight YOLO Models for Precision Pig Farming},
  author={Faria, Marcos Vinicius Mendes and Pereira, Thiago Borges and Condotta, Isabella C.F.S. and Paix{\~a}o, Thiago Meireles and Boldt, Francisco de Assis},
  journal={arXiv preprint arXiv:2605.25860},
  year={2026},
  doi={10.48550/arXiv.2605.25860}
}
```
