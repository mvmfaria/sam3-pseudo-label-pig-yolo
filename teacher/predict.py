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


def generate_predictions(model, processor, device, image_dir, output_file):
    if not image_dir.exists():
        print(f"[error] Image directory {image_dir} not found.")
        return

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    coco_preds = []

    for img_name in tqdm(image_files, desc="SAM3 Inference"):
        img_path = image_dir / img_name
        image = Image.open(img_path).convert("RGB")
        
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

        for box, score in zip(boxes, scores):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min

            coco_preds.append({
                "image_id": Path(img_name).stem,
                "category_id": CLASS_ID,
                "bbox": [round(float(x_min), 2), round(float(y_min), 2), round(float(width), 2), round(float(height), 2)],
                "score": round(float(score), 4)
            })

    os.makedirs(output_file.parent, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(coco_preds, f, indent=2)
    print(f"[predict] Predictions written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, help="Directory containing images to run inference on")
    parser.add_argument("--output-file", required=True, help="Path to save predictions JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Sam3Model.from_pretrained(
        "facebook/sam3", 
        torch_dtype=torch.bfloat16
    ).to(device)
    
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    generate_predictions(model, processor, device, Path(args.image_dir), Path(args.output_file))
