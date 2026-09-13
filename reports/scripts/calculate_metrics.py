import json
import os
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = str(Path(__file__).resolve().parents[2])
GT_PATH = f"{ROOT}/datasets/piglife/coco/human/annotations/instances_test.json"
OUTPUT_DIR = f"{ROOT}/reports/output/metrics"

YOLO_MODELS = ["yolov8n", "yolov8s", "yolov8m"]
ANNOTATION_SOURCES = ["human", "sam3", "dino", "yoloworld"]


def calculate_coco_metrics(ground_truth_path, predictions_path, output_path, model_name, trained):
    with open(ground_truth_path, 'r') as f:
        gt_data = json.load(f)

    filename_to_id = {
        Path(img['file_name']).stem: img['id'] for img in gt_data['images']
    }

    with open(predictions_path) as f:
        preds_data = json.load(f)

    final_preds = []
    for pred in preds_data:
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
    # SAM3 zero-shot baseline
    sam3_preds = f"{ROOT}/teacher/predictions.json"
    if os.path.exists(sam3_preds):
        calculate_coco_metrics(
            ground_truth_path=GT_PATH,
            predictions_path=sam3_preds,
            output_path=f"{OUTPUT_DIR}/sam3_zero_shot_performance.json",
            model_name="sam3",
            trained="zero_shot",
        )

    # Grounding DINO zero-shot baseline
    dino_test_json = f"{ROOT}/datasets/piglife/coco/dino/annotations/instances_test.json"
    dino_preds_json = f"{ROOT}/teacher/predictions_dino.json"
    if os.path.exists(dino_test_json):
        with open(dino_test_json) as f:
            dino_test = json.load(f)
        id_to_filename = {img['id']: Path(img['file_name']).stem for img in dino_test['images']}
        dino_preds = [
            {
                'image_id': id_to_filename[ann['image_id']],
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'score': 1.0,
            }
            for ann in dino_test['annotations']
        ]
        with open(dino_preds_json, 'w') as f:
            json.dump(dino_preds, f, indent=2)

        calculate_coco_metrics(
            ground_truth_path=GT_PATH,
            predictions_path=dino_preds_json,
            output_path=f"{OUTPUT_DIR}/dino_zero_shot_performance.json",
            model_name="dino",
            trained="zero_shot",
        )

    # YOLO-World zero-shot baseline
    yoloworld_preds = f"{ROOT}/teacher/predictions_yoloworld.json"
    if os.path.exists(yoloworld_preds):
        calculate_coco_metrics(
            ground_truth_path=GT_PATH,
            predictions_path=yoloworld_preds,
            output_path=f"{OUTPUT_DIR}/yoloworld_zero_shot_performance.json",
            model_name="yoloworld",
            trained="zero_shot",
        )

    # YOLO models
    for source in ANNOTATION_SOURCES:
        for model in YOLO_MODELS:
            preds_file = f"{ROOT}/runs/{source}/{model}/predictions.json"
            if os.path.exists(preds_file):
                calculate_coco_metrics(
                    ground_truth_path=GT_PATH,
                    predictions_path=preds_file,
                    output_path=f"{OUTPUT_DIR}/{model}_{source}_performance.json",
                    model_name=model,
                    trained=source,
                )
