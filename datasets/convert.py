from ultralytics.data.converter import convert_coco
import argparse
import json
import os
from pathlib import Path


def write_dataset_yaml(save_dir: str, annotations_dir: str):
    save_path = Path(save_dir).resolve()

    # Extract class names from any available annotation file
    ann_dir = Path(annotations_dir)
    categories = []
    for ann_file in ann_dir.glob("instances_*.json"):
        with open(ann_file) as f:
            data = json.load(f)
        categories = sorted(data["categories"], key=lambda c: c["id"])
        break

    splits = [s for s in ["train", "val", "test"] if (save_path / "images" / s).is_dir()]

    lines = [f"path: {save_path}"]
    for split in splits:
        lines.append(f"{split}: images/{split}")
    lines.append("")
    lines.append(f"nc: {len(categories)}")
    lines.append(f"names: [{', '.join(c['name'] for c in categories)}]")
    lines.append("")

    yaml_path = save_path / "dataset.yaml"
    yaml_path.write_text("\n".join(lines))
    print(f"[yaml] written to {yaml_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--hardlink-images-from",
        type=str,
        help="Create hardlinks for images from another source instead of copying",
    )

    args = parser.parse_args()

    annotations_dir = f"datasets/piglife/coco/{args.source}/annotations/"
    save_dir = f"datasets/piglife/yolo/{args.source}/"

    convert_coco(
        labels_dir=annotations_dir,
        save_dir=save_dir,
        use_segments=False,
        cls91to80=False
    )

    write_dataset_yaml(save_dir, annotations_dir)

    if args.hardlink_images_from:
        src_images = os.path.abspath(f"datasets/piglife/yolo/{args.hardlink_images_from}/images")
        dst_images = f"datasets/piglife/yolo/{args.source}/images"

        if os.path.islink(dst_images):
            os.unlink(dst_images)

        for split in os.listdir(src_images):
            src_split = os.path.join(src_images, split)
            dst_split = os.path.join(dst_images, split)
            if not os.path.isdir(src_split):
                continue
            os.makedirs(dst_split, exist_ok=True)
            for filename in os.listdir(src_split):
                src_file = os.path.join(src_split, filename)
                dst_file = os.path.join(dst_split, filename)
                if os.path.isfile(src_file) and not os.path.exists(dst_file):
                    os.link(src_file, dst_file)

if __name__ == "__main__":
    main()