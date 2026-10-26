# Hugging Face Artifacts

Staging area for the two Hugging Face repos published from this project:

* **Dataset** — [`marcosvmf/sam3-piglife-labels`](https://huggingface.co/datasets/marcosvmf/sam3-piglife-labels)
* **Model** — [`marcosvmf/yolov8-piglife-sam3`](https://huggingface.co/marcosvmf/yolov8-piglife-sam3)

## Layout

```
huggingface/
├── dataset/
│   ├── README.md        # dataset card — upload as README.md at the root of the dataset repo
│   └── data/             # train/val/test CSVs referenced by the card's `configs.data_files`
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── model/
│   └── README.md        # model card — upload as README.md at the root of the model repo
├── template.md           # generic, fill-in-the-blanks card for future SAM3/YOLO HF repos
├── scripts/
│   └── generate_csv.py   # rebuilds sam3_labels.csv from the YOLO .txt labels in datasets/piglife/yolo/sam3/labels/
├── sam3_labels.csv        # flat per-box CSV (all splits, with a `split` column) — intermediate output of generate_csv.py
└── sam3_labels.zip        # raw YOLO label export (.txt files + .cache) used as the source for generate_csv.py
```

`huggingface/dataset/data/*.csv` are `sam3_labels.csv` split by `split` and with that column dropped — they are what the dataset repo actually serves to the HF dataset viewer.

## Regenerating the CSVs

```bash
uv run python huggingface/scripts/generate_csv.py
```

This reads the YOLO label `.txt` files from `datasets/piglife/yolo/sam3/labels/{train,val,test}/` and writes `huggingface/sam3_labels.csv`. Splitting that file into `huggingface/dataset/data/{train,val,test}.csv` (dropping the `split` column) is currently a manual step.

## Uploading

```bash
huggingface-cli upload marcosvmf/sam3-piglife-labels huggingface/dataset . --repo-type dataset
huggingface-cli upload marcosvmf/yolov8-piglife-sam3 huggingface/model . --repo-type model
```

The model repo additionally needs the `yolov8{n,s,m}-pig.pt` weights and a `cover.png`, referenced by both cards but not stored in this folder.
