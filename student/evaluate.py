from pathlib import Path
from ultralytics import YOLO
import argparse
import torch

DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "piglife" / "yolo"
RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"

MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]


def evaluate(source: str):
    for model_name in MODELS:
        run = RUNS_ROOT / source / model_name.replace(".pt", "")
        best_pt = run / "weights" / "best.pt"

        if not (run / "done.flag").exists():
            print(f"[skip] {source}/{model_name} not trained yet")
            continue

        print(f"[eval] {source}/{model_name}")
        model = YOLO(str(best_pt))
        model.val(
            data=str(DATASETS_ROOT / source / "dataset.yaml"),
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
        choices=["human", "sam3"],
        default=None,
        help="Annotation source (default: evaluate both)",
    )
    args = parser.parse_args()

    sources = [args.source] if args.source else ["human", "sam3"]
    for source in sources:
        evaluate(source)
