import os
import torch
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
import json
from pathlib import Path
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

load_dotenv()

# We query with "pig." because Grounding DINO is highly sensitive to format:
# labels should end with a period.
CLASS_PROMPT = "pig."
CLASS_ID = 1
CONFIDENCE_THRESHOLD = 0.4

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_ROOT = BASE_DIR / "datasets" / "piglife" / "yolo" / "human"
OUTPUT_ROOT = BASE_DIR / "datasets" / "piglife" / "coco" / "dino" / "annotations"

def generate_predictions(subset_name, model, processor, device):
    image_dir = SOURCE_ROOT / "images" / subset_name
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"supercategory": "pig", "id": CLASS_ID, "name": "pig"}]
    }

    img_id_counter = 1
    ann_id_counter = 1

    for img_name in tqdm(image_files):
        img_path = image_dir / img_name
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size
        
        # Grounding DINO inputs
        inputs = processor(images=image, text=CLASS_PROMPT, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process detections
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=CONFIDENCE_THRESHOLD,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )

        prediction = results[0]
        boxes_tensor = prediction.get("boxes")
        scores_tensor = prediction.get("scores")
        labels_list = prediction.get("labels")

        if boxes_tensor is None or len(boxes_tensor) == 0:
            boxes = []
            scores = []
            labels = []
        else:
            boxes = boxes_tensor.float().cpu().numpy()
            scores = scores_tensor.float().cpu().numpy()
            labels = labels_list

        coco_data["images"].append({
            "height": img_height,
            "width": img_width,
            "id": img_id_counter,
            "file_name": img_name
        })

        for box, score, label in zip(boxes, scores, labels):
            # Only record if the detected label corresponds to our target class
            if "pig" in label.lower():
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

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    output_file = OUTPUT_ROOT / f"instances_{subset_name}.json"
    with open(output_file, "w") as f:
        json.dump(coco_data, f, indent=2)
    
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Using the recommended base model
    model_id = "IDEA-Research/grounding-dino-base" 
    
    print(f"Loading Grounding DINO from Hugging Face model: {model_id}...")
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    subsets = ["train", "val", "test"]
    for subset in subsets:
        print(f"Generating labels for {subset} split...")
        generate_predictions(subset, model, processor, device)
