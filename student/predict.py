from pathlib import Path
from ultralytics import YOLO
import argparse
import torch

BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_PIGLIFE = BASE_DIR / "datasets" / "piglife" / "yolo"
DATASETS_EXTRA = BASE_DIR / "datasets"
RUNS_ROOT = BASE_DIR / "runs"

MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]


def predict(source: str):
    # Try piglife structure first (e.g., datasets/piglife/yolo/human/dataset.yaml)
    data_config = DATASETS_PIGLIFE / source / "dataset.yaml"
    if not data_config.exists():
        # Try extra datasets structure (e.g., datasets/bama/yolo/dataset.yaml)
        data_config = DATASETS_EXTRA / source / "yolo" / "dataset.yaml"
    
    if not data_config.exists():
        print(f"[error] Could not find dataset.yaml for source: {source}")
        return

    for model_name in MODELS:
        run = RUNS_ROOT / source / model_name.replace(".pt", "")
        best_pt = run / "weights" / "best.pt"

        if not (run / "done.flag").exists():
            print(f"[skip] {source}/{model_name} not trained yet")
            continue

        if not best_pt.exists():
            print(f"[skip] {source}/{model_name} best.pt not found")
            continue

        print(f"[predict] {source}/{model_name}")
        model = YOLO(str(best_pt))
        model.val(
            data=str(data_config),
            split="test",
            verbose=False,
            save_json=True,
            project=str(RUNS_ROOT / source),
            name=model_name.replace(".pt", ""),
            exist_ok=True,
        )

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=None,
        help="Annotation source (e.g., human, sam3, bama, faro). Default: human, sam3.",
    )
    args = parser.parse_args()

    sources = [args.source] if args.source else ["human", "sam3"]
    for source in sources:
        predict(source)
