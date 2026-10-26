import csv
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]

base_dir = PROJECT_DIR / 'datasets/piglife/yolo/sam3/labels'
csv_path = PROJECT_DIR / 'huggingface/sam3_labels.csv'

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['split', 'filename', 'class_id', 'x_center', 'y_center', 'width', 'height'])

    for split in ['train', 'val', 'test']:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue
        
        for txt_file in split_dir.glob('*.txt'):
            with open(txt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = parts[0]
                        x, y, w, h = parts[1:5]
                        writer.writerow([split, txt_file.name, class_id, x, y, w, h])
print("CSV generated successfully!")
