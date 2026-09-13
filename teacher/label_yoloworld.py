import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from ultralytics import YOLOWorld

CLASS_PROMPT = "pig"
CLASS_ID = 1
CONFIDENCE_THRESHOLD = 0.05
ZERO_SHOT_CONF_THRESHOLD = 0.01

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_ROOT = BASE_DIR / "datasets" / "piglife" / "yolo" / "human"
OUTPUT_ROOT = BASE_DIR / "datasets" / "piglife" / "coco" / "yoloworld" / "annotations"
TEACHER_DIR = BASE_DIR / "teacher"


def generate_predictions(subset_name, model):
    image_dir = SOURCE_ROOT / "images" / subset_name
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"supercategory": CLASS_PROMPT, "id": CLASS_ID, "name": CLASS_PROMPT}]
    }

    test_zero_shot_predictions = []

    img_id_counter = 1
    ann_id_counter = 1

    for img_name in tqdm(image_files, desc=f"YOLO-World labeling {subset_name}"):
        img_path = image_dir / img_name
        
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        coco_data["images"].append({
            "height": img_height,
            "width": img_width,
            "id": img_id_counter,
            "file_name": img_name
        })

        results = model.predict(
            source=str(img_path),
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        pred = results[0]
        boxes = pred.boxes.xyxy.cpu().numpy()
        scores = pred.boxes.conf.cpu().numpy()

        for box, score in zip(boxes, scores):
            x_min, y_min, x_max, y_max = box
            width = max(0.0, float(x_max - x_min))
            height = max(0.0, float(y_max - y_min))
            area = width * height

            coco_data["annotations"].append({
                "iscrowd": 0,
                "image_id": img_id_counter,
                "bbox": [round(float(x_min), 2), round(float(y_min), 2), round(float(width), 2), round(float(height), 2)],
                "category_id": CLASS_ID,
                "id": ann_id_counter,
                "area": round(float(area), 2),
                "segmentation": []
            })
            ann_id_counter += 1

        if subset_name == "test":
            if ZERO_SHOT_CONF_THRESHOLD != CONFIDENCE_THRESHOLD:
                zero_results = model.predict(
                    source=str(img_path),
                    conf=ZERO_SHOT_CONF_THRESHOLD,
                    verbose=False,
                    device="cuda:0" if torch.cuda.is_available() else "cpu"
                )
                z_boxes = zero_results[0].boxes.xyxy.cpu().numpy()
                z_scores = zero_results[0].boxes.conf.cpu().numpy()
            else:
                z_boxes, z_scores = boxes, scores

            for z_box, z_score in zip(z_boxes, z_scores):
                zx1, zy1, zx2, zy2 = z_box
                zw = max(0.0, float(zx2 - zx1))
                zh = max(0.0, float(zy2 - zy1))
                test_zero_shot_predictions.append({
                    "image_id": Path(img_name).stem,
                    "category_id": CLASS_ID,
                    "bbox": [round(float(zx1), 2), round(float(zy1), 2), round(float(zw), 2), round(float(zh), 2)],
                    "score": round(float(z_score), 4)
                })

        img_id_counter += 1

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    output_file = OUTPUT_ROOT / f"instances_{subset_name}.json"
    with open(output_file, "w") as f:
        json.dump(coco_data, f, indent=2)
    print(f"Saved: {output_file} ({len(coco_data['annotations'])} annotations across {len(coco_data['images'])} images)")

    if subset_name == "test":
        preds_file = TEACHER_DIR / "predictions_yoloworld.json"
        with open(preds_file, "w") as f:
            json.dump(test_zero_shot_predictions, f, indent=2)
        print(f"Saved zero-shot test predictions: {preds_file} ({len(test_zero_shot_predictions)} detections)")


if __name__ == "__main__":
    print("Loading YOLO-World model (yolov8x-worldv2.pt)...")
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.set_classes([CLASS_PROMPT])
    print(f"Set class prompt: ['{CLASS_PROMPT}']")

    subsets = ["train", "val", "test"]
    for subset in subsets:
        generate_predictions(subset, model)
