from ultralytics.data.converter import convert_coco
import argparse
import os

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--symlink-images-from",
        type=str,
    )

    args = parser.parse_args()

    convert_coco(
        labels_dir=f"datasets/piglife/coco/{args.source}/annotations/",
        save_dir=f"datasets/piglife/yolo/{args.source}/",
        use_segments=False,
        cls91to80=False
    )

    if args.symlink_images_from:
        target = os.path.abspath(f"datasets/piglife/yolo/{args.symlink_images_from}/images")
        link = f"datasets/piglife/yolo/{args.source}/images"
        if os.path.isdir(link) and not os.path.islink(link):
            os.rmdir(link)
        os.symlink(target, link)

if __name__ == "__main__":
    main()