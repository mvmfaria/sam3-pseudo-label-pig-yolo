import os
import torch
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
import json
from pathlib import Path
import argparse

load_dotenv()
token = os.getenv("HF_TOKEN")

from transformers import Sam3Processor, Sam3Model

CLASS_PROMPT = "pig"
CLASS_ID = 1
CONFIDENCE_THRESHOLD = 0.4

BASE_DIR = Path(__file__).resolve().parent.parent

def generate_predictions(subset_name, model, processor, device, image_root, output_root):
    image_dir = image_root / subset_name
    if not image_dir.exists():
        print(f"[skip] Subset {subset_name} not found in {image_root}")
        return

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"supercategory": CLASS_PROMPT, "id": CLASS_ID, "name": CLASS_PROMPT}]
    }

    img_id_counter = 1
    ann_id_counter = 1

    for img_name in tqdm(image_files, desc=f"SAM3 {subset_name}"):
        img_path = image_dir / img_name
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size
        
        inputs = processor(images=image, text=CLASS_PROMPT, return_tensors="pt").to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            outputs = model(**inputs)
        
        results = processor.post_process_instance_segmentation(
            outputs, 
            threshold=CONFIDENCE_THRESHOLD, 
            target_sizes=inputs.get("original_sizes").tolist()
        )

        prediction = results[0] if isinstance(results, list) else results
        boxes_tensor = prediction.get("boxes") if isinstance(prediction, dict) else None
        scores_tensor = prediction.get("scores") if isinstance(prediction, dict) else None

        if boxes_tensor is None or scores_tensor is None:
            boxes = []
            scores = []
        else:
            boxes = boxes_tensor.float().cpu().numpy()
            scores = scores_tensor.float().cpu().numpy()

        coco_data["images"].append({
            "height": img_height,
            "width": img_width,
            "id": img_id_counter,
            "file_name": img_name
        })

        for box, score in zip(boxes, scores):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
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
            
        img_id_counter += 1

    os.makedirs(output_root, exist_ok=True)
    output_file = output_root / f"instances_{subset_name}.json"
    with open(output_file, "w") as f:
        json.dump(coco_data, f, indent=2)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="Root directory for images (should contain train/val/test subdirs)")
    parser.add_argument("--output-dir", required=True, help="Directory to save COCO annotations")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Sam3Model.from_pretrained(
        "facebook/sam3", 
        torch_dtype=torch.bfloat16
    ).to(device)
    
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    image_root = Path(args.image_dir)
    output_root = Path(args.output_dir)

    subsets = ["train", "val", "test"]
    for subset in subsets:
        generate_predictions(subset, model, processor, device, image_root, output_root)