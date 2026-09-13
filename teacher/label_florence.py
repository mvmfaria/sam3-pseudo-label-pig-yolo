import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

TASK_PROMPT = "<CAPTION_TO_PHRASE_GROUNDING>"
CLASS_PROMPT = "pig"
FULL_PROMPT = f"{TASK_PROMPT}{CLASS_PROMPT}"
CLASS_ID = 1

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_ROOT = BASE_DIR / "datasets" / "piglife" / "yolo" / "human"
OUTPUT_ROOT = BASE_DIR / "datasets" / "piglife" / "coco" / "florence" / "annotations"
TEACHER_DIR = BASE_DIR / "teacher"


def generate_predictions(subset_name, model, processor, device):
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

    for img_name in tqdm(image_files, desc=f"Florence-2 labeling {subset_name}"):
        img_path = image_dir / img_name
        
        with Image.open(img_path) as orig_img:
            image = orig_img.convert("RGB")
            img_width, img_height = image.size

        coco_data["images"].append({
            "height": img_height,
            "width": img_width,
            "id": img_id_counter,
            "file_name": img_name
        })

        # Resize to 768x768 required by DaViT square feature map
        resized_image = image.resize((768, 768))

        inputs = processor(text=FULL_PROMPT, images=resized_image, return_tensors="pt").to(device, torch.float16)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=2,
                use_cache=False
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text, 
            task=TASK_PROMPT, 
            image_size=(img_width, img_height)
        )

        res = parsed.get(TASK_PROMPT, {})
        bboxes = res.get("bboxes", [])

        for bbox in bboxes:
            if len(bbox) != 4:
                continue
            x_min, y_min, x_max, y_max = bbox
            width = max(0.0, float(x_max - x_min))
            height = max(0.0, float(y_max - y_min))
            area = width * height

            # Skip degenerated or near-zero boxes
            if width <= 2 or height <= 2:
                continue

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
                test_zero_shot_predictions.append({
                    "image_id": Path(img_name).stem,
                    "category_id": CLASS_ID,
                    "bbox": [round(float(x_min), 2), round(float(y_min), 2), round(float(width), 2), round(float(height), 2)],
                    "score": 1.0
                })

        img_id_counter += 1

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    output_file = OUTPUT_ROOT / f"instances_{subset_name}.json"
    with open(output_file, "w") as f:
        json.dump(coco_data, f, indent=2)
    print(f"Saved: {output_file} ({len(coco_data['annotations'])} annotations across {len(coco_data['images'])} images)")

    if subset_name == "test":
        preds_file = TEACHER_DIR / "predictions_florence.json"
        with open(preds_file, "w") as f:
            json.dump(test_zero_shot_predictions, f, indent=2)
        print(f"Saved zero-shot test predictions: {preds_file} ({len(test_zero_shot_predictions)} detections)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Florence-2 zero-shot pseudo-label generator")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu', etc.)")
    parser.add_argument("--model-id", type=str, default="microsoft/Florence-2-large", help="Hugging Face model ID")
    args = parser.parse_args()

    model_id = args.model_id
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_id} on {device}...")

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(device)
    except Exception as e:
        if "out of memory" in str(e).lower() and device != "cpu":
            print(f"[warn] GPU VRAM insufficient for {model_id}. Switching to CPU...")
            torch.cuda.empty_cache()
            device = "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                attn_implementation="eager"
            ).to(device)
        else:
            raise e

    # Tie shared weights to embed_tokens and lm_head
    shared_w = model.language_model.model.shared.weight
    model.language_model.model.encoder.embed_tokens.weight.data = shared_w
    model.language_model.model.decoder.embed_tokens.weight.data = shared_w
    model.language_model.lm_head.weight.data = shared_w
    model.eval()

    print(f"Model initialized successfully on {device}!")

    subsets = ["train", "val", "test"]
    for subset in subsets:
        generate_predictions(subset, model, processor, device)
