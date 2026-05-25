import contextlib
import io
import json
import os
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = str(Path(__file__).resolve().parents[2])
GT_PATH = f"{ROOT}/datasets/piglife/coco/human/annotations/instances_test.json"
OUTPUT_DIR = f"{ROOT}/reports/output/metrics"

# Each inner list is a group; filenames are matched by prefix
TARGET_GROUPS = [
    ["1050s1132a1110s3003-3s5001"],
    ["1050s1132a1110s3003-5s5001"],
    ["1050s1132a1110s3004", "1050s1132a1112"],
    ["1010s1120", "1010s1121"],
    ["1060s1112", "1060a1000s1110"],
    ["1020s1120", "1020s1220"],
    ["1040s1132"],
    ["1020s1121", "1020s1221"],
]


def evaluate_per_group(ground_truth_path, predictions_path, output_path):
    with open(ground_truth_path, 'r') as f:
        gt_data = json.load(f)

    filename_to_id = {Path(img['file_name']).stem: img['id'] for img in gt_data['images']}

    with open(predictions_path) as f:
        preds_data = json.load(f)

    final_preds = []
    for pred in preds_data:
        filename = Path(str(pred['image_id'])).stem
        if filename in filename_to_id:
            pred['image_id'] = filename_to_id[filename]
            final_preds.append(pred)

    coco_gt = COCO(ground_truth_path)

    if not final_preds:
        print("No valid predictions matched GT images.")
        return

    coco_dt = coco_gt.loadRes(final_preds)

    group_buckets = {i: [] for i in range(len(TARGET_GROUPS))}
    for img_id in coco_gt.getImgIds():
        img_info = coco_gt.loadImgs(img_id)[0]
        filename = Path(img_info['file_name']).stem
        for i, prefixes in enumerate(TARGET_GROUPS):
            if any(filename.startswith(p) for p in prefixes):
                group_buckets[i].append(img_id)
                break

    results = []
    for i, img_ids in group_buckets.items():
        group_label = f"Group {i + 1}"
        if not img_ids:
            print(f"Skipping {group_label}: no images found.")
            continue

        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.imgIds = img_ids

        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

        stats = coco_eval.stats
        results.append({
            "group_name": group_label,
            "filter_ids": ", ".join(TARGET_GROUPS[i]),
            "image_count": len(img_ids),
            "mAP_50-95": round(stats[0], 4),
            "mAP_50": round(stats[1], 4),
            "mAP_75": round(stats[2], 4),
            "AP_Medium": round(stats[4], 4),
            "AP_Large": round(stats[5], 4),
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Saved: {output_path}")
    for r in results:
        print(f"  {r['group_name']:8s}  images={r['image_count']:4d}  mAP={r['mAP_50-95']:.3f}")


if __name__ == "__main__":
    evaluate_per_group(
        ground_truth_path=GT_PATH,
        predictions_path=f"{ROOT}/teacher/predictions.json",
        output_path=f"{OUTPUT_DIR}/sam3_zero_shot_groups.json",
    )
