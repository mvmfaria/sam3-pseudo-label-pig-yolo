import json
import os
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = str(Path(__file__).resolve().parents[2])
GT_PATH = f"{ROOT}/datasets/piglife/coco/human/annotations/instances_test.json"
OUTPUT_DIR = f"{ROOT}/reports/output/metrics"

YOLO_MODELS = ["yolov8n", "yolov8s", "yolov8m"]
ANNOTATION_SOURCES = ["human", "sam3"]


def calculate_coco_metrics(ground_truth_path, predictions_path, output_path, model_name, trained):
    with open(ground_truth_path, 'r') as f:
        gt_data = json.load(f)

    filename_to_id = {
        Path(img['file_name']).stem: img['id'] for img in gt_data['images']
    }

    with open(predictions_path) as f:
        preds_data = json.load(f)

    if isinstance(preds_data, dict) and "annotations" in preds_data:
        id_to_filename = {
            img["id"]: Path(img["file_name"]).stem for img in preds_data.get("images", [])
        }
        flat_preds = []
        for ann in preds_data.get("annotations", []):
            flat_preds.append({
                "image_id": id_to_filename.get(ann["image_id"], ann["image_id"]),
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "score": ann.get("score", 1.0)
            })
        preds_data = flat_preds

    final_preds = []
    for pred in preds_data:
        if 'file_name' in pred:
            filename = Path(pred['file_name']).stem
        else:
            filename = Path(str(pred['image_id'])).stem
            
        if filename in filename_to_id:
            pred['image_id'] = filename_to_id[filename]
            final_preds.append(pred)

    coco_gt = COCO(ground_truth_path)
    coco_dt = coco_gt.loadRes(final_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = sorted({p['image_id'] for p in final_preds})
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    metrics = {
        "Model": f"{model_name}_{trained}",
        "mAP_50-95": stats[0],
        "mAP_50": stats[1],
        "mAP_75": stats[2],
        "AP_Medium": stats[4],
        "AP_Large": stats[5],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    # 1. PigLife (Zero-shot)
    calculate_coco_metrics(
        ground_truth_path=f"{ROOT}/datasets/piglife/coco/human/annotations/instances_test.json",
        predictions_path=f"{ROOT}/teacher/predictions.json",
        output_path=f"{OUTPUT_DIR}/sam3_zero_shot_performance.json",
        model_name="sam3",
        trained="zero_shot",
    )

    # Bama (Zero-shot)
    calculate_coco_metrics(
        ground_truth_path=f"{ROOT}/datasets/bama/BamaPig2D/annotations/eval_pig_cocostyle.json",
        predictions_path=f"{ROOT}/datasets/sam3-bama/coco/annotations/predictions_test.json",
        output_path=f"{OUTPUT_DIR}/sam3_bama_zero_shot_performance.json",
        model_name="sam3",
        trained="zero_shot",
    )

    # Faro (Zero-shot)
    calculate_coco_metrics(
        ground_truth_path=f"{ROOT}/datasets/faro/coco/annotations/instances_test.json",
        predictions_path=f"{ROOT}/datasets/sam3-faro/coco/annotations/predictions_test.json",
        output_path=f"{OUTPUT_DIR}/sam3_faro_zero_shot_performance.json",
        model_name="sam3",
        trained="zero_shot",
    )

    # 2. YOLO models on various datasets
    datasets_config = [
        {
            "gt_path": f"{ROOT}/datasets/piglife/coco/human/annotations/instances_test.json",
            "sources": ["human", "sam3"]
        },
        {
            "gt_path": f"{ROOT}/datasets/bama/BamaPig2D/annotations/eval_pig_cocostyle.json",
            "sources": ["bama", "sam3-bama"]
        },
        {
            "gt_path": f"{ROOT}/datasets/faro/coco/annotations/instances_test.json",
            "sources": ["faro", "sam3-faro"]
        }
    ]

    for config in datasets_config:
        gt_path = config["gt_path"]
        for source in config["sources"]:
            for model in YOLO_MODELS:
                pred_path = f"{ROOT}/runs/{source}/{model}/predictions.json"
                if not os.path.exists(pred_path):
                    print(f"Skipping: {pred_path} (not found)")
                    continue
                calculate_coco_metrics(
                    ground_truth_path=gt_path,
                    predictions_path=pred_path,
                    output_path=f"{OUTPUT_DIR}/{model}_{source}_performance.json",
                    model_name=model,
                    trained=source,
                )

