from pathlib import Path
from ultralytics import YOLO
import argparse
import torch

DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "piglife" / "yolo"
RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"

MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]


def evaluate(source: str, device: str = None):
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    for model_name in MODELS:
        run = RUNS_ROOT / source / model_name.replace(".pt", "")
        best_pt = run / "weights" / "best.pt"

        if not (run / "done.flag").exists():
            print(f"[skip] {source}/{model_name} not trained yet")
            continue

        print(f"[eval] {source}/{model_name} on {device}")
        try:
            model = YOLO(str(best_pt))
            model.val(
                data=str(DATASETS_ROOT / source / "dataset.yaml"),
                split="test",
                verbose=False,
                save_json=True,
                project=str(RUNS_ROOT / source),
                name=model_name.replace(".pt", ""),
                exist_ok=True,
                device=device,
            )
        except Exception as e:
            if "out of memory" in str(e).lower() and device != "cpu":
                print(f"[warn] GPU OOM encountered on {device}. Retrying {source}/{model_name} on CPU...")
                torch.cuda.empty_cache()
                model = YOLO(str(best_pt))
                model.val(
                    data=str(DATASETS_ROOT / source / "dataset.yaml"),
                    split="test",
                    verbose=False,
                    save_json=True,
                    project=str(RUNS_ROOT / source),
                    name=model_name.replace(".pt", ""),
                    exist_ok=True,
                    device="cpu",
                )
            else:
                raise e

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["human", "sam3", "dino", "yoloworld"],
        default=None,
        help="Annotation source (default: evaluate all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on (e.g. 'cpu', 'cuda:0')",
    )
    args = parser.parse_args()

    sources = [args.source] if args.source else ["human", "sam3", "dino", "yoloworld"]
    for source in sources:
        evaluate(source, device=args.device)
