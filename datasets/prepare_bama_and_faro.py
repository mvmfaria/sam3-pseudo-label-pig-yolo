import json
import os
from pathlib import Path
from ultralytics.data.converter import convert_coco
import shutil

BASE_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = BASE_DIR / "datasets"

def prepare_bama():
    print("[bama] Preparing BamaPig2D...")
    
    # Check / move raw dataset from old path if it exists
    bama_raw = DATASETS_DIR / "bama" / "BamaPig2D"
    old_bama_raw = DATASETS_DIR / "BamaPig2D"
    if old_bama_raw.exists() and not bama_raw.exists():
        print(f"[bama] Moving raw BamaPig2D to {bama_raw}...")
        bama_raw.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_bama_raw), str(bama_raw))

    bama_root = DATASETS_DIR / "bama"
    bama_yolo = bama_root / "yolo"

    # Clean up existing yolo directory to avoid incrementing (without deleting raw folder)
    if bama_yolo.exists():
        shutil.rmtree(bama_yolo)
    
    # Temporary coco structure for conversion (outside bama_yolo)
    temp_coco_base = bama_root / "temp_coco"
    temp_coco_ann = temp_coco_base / "annotations"
    temp_coco_ann.mkdir(parents=True, exist_ok=True)
    
    # Copy annotations to standard names for ultralytics converter
    shutil.copy(bama_raw / "annotations" / "train_pig_cocostyle.json", temp_coco_ann / "instances_train.json")
    shutil.copy(bama_raw / "annotations" / "eval_pig_cocostyle.json", temp_coco_ann / "instances_val.json")
    shutil.copy(bama_raw / "annotations" / "eval_pig_cocostyle.json", temp_coco_ann / "instances_test.json")
    
    # Convert COCO to YOLO (Detection only)
    convert_coco(
        labels_dir=str(temp_coco_ann),
        save_dir=str(bama_yolo),
        use_segments=False,
        cls91to80=False
    )
    
    # Symlink images
    src_img_dir = bama_raw / "images"
    for split in ["train", "val", "test"]:
        dst_img_dir = bama_yolo / "images" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        # In BamaPig2D, all images are in one folder
        # The converter creates label files, we only link images that have labels
        label_dir = bama_yolo / "labels" / split
        for label_file in label_dir.glob("*.txt"):
            img_name = label_file.stem + ".png" # Bama uses .png
            src_img = src_img_dir / img_name
            dst_img = dst_img_dir / img_name
            if src_img.exists() and not dst_img.exists():
                os.symlink(os.path.relpath(src_img, dst_img.parent), dst_img)

    # Write dataset.yaml
    yaml_content = f"""path: {bama_yolo.resolve()}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['pig']
"""
    (bama_yolo / "dataset.yaml").write_text(yaml_content)
    
    # Cleanup temp coco
    shutil.rmtree(temp_coco_base)
    print("[bama] Done.")

def seg_to_det(seg_path, det_path):
    if not seg_path.exists(): return
    det_path.parent.mkdir(parents=True, exist_ok=True)
    with open(seg_path, 'r') as f:
        lines = f.readlines()
    
    det_lines = []
    for line in lines:
        parts = list(map(float, line.strip().split()))
        if len(parts) < 5: continue
        cls = int(parts[0])
        coords = parts[1:]
        xs = coords[0::2]
        ys = coords[1::2]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        # YOLO detection format: cls cx cy w h
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin
        det_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    
    with open(det_path, 'w') as f:
        f.write("\n".join(det_lines))

def prepare_faro():
    print("[faro] Preparing FaroPigSeg...")
    
    # Check / move raw dataset from old path if it exists
    faro_raw = DATASETS_DIR / "faro" / "FaroPigSeg"
    old_faro_raw = DATASETS_DIR / "FaroPigSeg"
    if old_faro_raw.exists() and not faro_raw.exists():
        print(f"[faro] Moving raw FaroPigSeg to {faro_raw}...")
        faro_raw.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_faro_raw), str(faro_raw))

    faro_yolo = DATASETS_DIR / "faro" / "yolo"
    
    if faro_yolo.exists():
        shutil.rmtree(faro_yolo)
    
    for split in ["train", "val", "test"]:
        # Convert labels
        src_label_dir = faro_raw / split / "labels"
        dst_label_dir = faro_yolo / "labels" / split
        for seg_file in src_label_dir.glob("*.txt"):
            seg_to_det(seg_file, dst_label_dir / seg_file.name)
        
        # Symlink images
        src_img_dir = faro_raw / split / "images"
        dst_img_dir = faro_yolo / "images" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        for img_file in src_img_dir.glob("*"):
            dst_img = dst_img_dir / img_file.name
            if not dst_img.exists():
                os.symlink(os.path.relpath(img_file, dst_img.parent), dst_img)

    # Write dataset.yaml
    yaml_content = f"""path: {faro_yolo.resolve()}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['pig']
"""
    (faro_yolo / "dataset.yaml").write_text(yaml_content)
    print("[faro] Done.")

def fix_sam3_symlinks():
    # Update sam3-bama and sam3-faro yolo images to link relatively to raw images
    for dataset, sam3_dataset in [("bama", "sam3-bama"), ("faro", "sam3-faro")]:
        sam3_images = DATASETS_DIR / sam3_dataset / "yolo" / "images"
        src_images = DATASETS_DIR / dataset / "yolo" / "images"
        if not sam3_images.exists():
            continue
            
        print(f"[{sam3_dataset}] Recreating symlinks to raw images...")
        shutil.rmtree(sam3_images)
        sam3_images.mkdir(parents=True, exist_ok=True)
        
        for split in ["train", "val", "test"]:
            src_split = src_images / split
            dst_split = sam3_images / split
            if not src_split.exists():
                continue
            dst_split.mkdir(parents=True, exist_ok=True)
            for img_file in src_split.glob("*"):
                raw_img = img_file.resolve()
                dst_img = dst_split / img_file.name
                os.symlink(os.path.relpath(raw_img, dst_split), dst_img)
    print("[sam3-symlinks] Done.")

if __name__ == "__main__":
    prepare_bama()
    prepare_faro()
    fix_sam3_symlinks()
