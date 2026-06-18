from pathlib import Path
from ultralytics import YOLO
import argparse
import torch

DATASETS_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "piglife" / "yolo"
RUNS_ROOT = Path(__file__).resolve().parents[1] / "runs"

MODELS = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]

EPOCHS = 100
IMGSZ = 640
BATCH = 4
WORKERS = 2


def run_dir(source: str, model_name: str) -> Path:
    return RUNS_ROOT / source / model_name.replace(".pt", "")


def train(source: str):
    # Try piglife structure first (e.g., datasets/piglife/yolo/human/dataset.yaml)
    data_config = DATASETS_ROOT / source / "dataset.yaml"
    if not data_config.exists():
        # Try extra datasets structure (e.g., datasets/bama/yolo/dataset.yaml)
        data_config = DATASETS_ROOT.parents[1] / source / "yolo" / "dataset.yaml"
    
    if not data_config.exists():
        raise FileNotFoundError(f"Could not find dataset.yaml for source: {source} at {data_config}")

    data_config = str(data_config)

    for model_name in MODELS:
        run = run_dir(source, model_name)
        done_flag = run / "done.flag"

        if done_flag.exists():
            print(f"[skip] {source}/{model_name} already complete")
            continue

        last_pt = run / "weights" / "last.pt"
        if last_pt.exists():
            print(f"[resume] {source}/{model_name} from {last_pt}")
            model = YOLO(str(last_pt))
            model.train(resume=True)
        else:
            print(f"[train] {source}/{model_name} from scratch")
            model = YOLO(model_name)
            model.train(
                data=data_config,
                epochs=EPOCHS,
                imgsz=IMGSZ,
                batch=BATCH,
                workers=WORKERS,
                project=str(RUNS_ROOT / source),
                name=model_name.replace(".pt", ""),
                exist_ok=True,
                verbose=False,
            )

        run.mkdir(parents=True, exist_ok=True)
        done_flag.touch()
        print(f"[done] {source}/{model_name}")

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
        train(source)
