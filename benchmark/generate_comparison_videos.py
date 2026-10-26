import argparse
import glob
import os
import re
import time
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Color schemes (BGR)
COLORS = {
    "human": (255, 140, 0),    # Deep Sky Blue / Amber
    "sam3": (50, 205, 50),     # Lime Green
    "dino": (0, 165, 255),     # Orange
}

DISPLAY_NAMES = {
    "human": "Human",
    "sam3": "SAM3",
    "dino": "DINO",
}


def scan_scenes(split_dirs):
    """Group image frames by scene prefix and sort them by frame number."""
    scenes = defaultdict(list)
    for s_dir in split_dirs:
        images = glob.glob(os.path.join(s_dir, "*.jpg"))
        for p in images:
            fname = os.path.basename(p)
            m = re.match(r"(.+)-(\d+)\.jpg", fname)
            if m:
                prefix, frame_num = m.group(1), int(m.group(2))
                scenes[prefix].append((frame_num, p))
    for prefix in scenes:
        scenes[prefix].sort(key=lambda x: x[0])
    return scenes


def draw_header(img, title, subtitle="", color=(255, 255, 255), bg_color=(20, 20, 20)):
    """Draw a sleek header bar with title and subtitle."""
    h, w = img.shape[:2]
    header_h = int(h * 0.09)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    # Accent color line below header
    cv2.line(img, (0, header_h), (w, header_h), color, 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = max(0.55, w / 1800.0)
    sub_scale = max(0.45, w / 2200.0)

    cv2.putText(
        img,
        title,
        (15, int(header_h * 0.58)),
        font,
        title_scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if subtitle:
        text_size = cv2.getTextSize(subtitle, font, sub_scale, 1)[0]
        cv2.putText(
            img,
            subtitle,
            (w - text_size[0] - 15, int(header_h * 0.58)),
            font,
            sub_scale,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_info_footer(img, scene_id, frame_idx, total_frames, count, fps, color=(255, 255, 255)):
    """Draw a semi-transparent info footer bar."""
    h, w = img.shape[:2]
    footer_h = int(h * 0.06)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - footer_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, w / 2400.0)

    left_text = f"Scene: {scene_id} | Frame {frame_idx + 1}/{total_frames}"
    right_text = f"Detected: {count} pigs | Model Inference: {fps:.1f} FPS"

    cv2.putText(
        img,
        left_text,
        (15, h - int(footer_h * 0.35)),
        font,
        scale,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    text_size = cv2.getTextSize(right_text, font, scale, 1)[0]
    cv2.putText(
        img,
        right_text,
        (w - text_size[0] - 15, h - int(footer_h * 0.35)),
        font,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def render_detections(img, results, source_name, conf_thresh=0.25):
    """Render bounding boxes and labels onto the image frame."""
    canvas = img.copy()
    color = COLORS.get(source_name, (0, 255, 0))
    boxes = results[0].boxes

    pigs_count = 0
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_thresh:
                continue
            pigs_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Draw label pill
            label = f"pig {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

            cv2.rectangle(
                canvas,
                (x1, y1 - th - baseline - 4),
                (x1 + tw + 6, y1),
                color,
                -1,
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 3, y1 - baseline - 2),
                font,
                font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return canvas, pigs_count


def build_comparison_grid(frames_dict, target_w=1920, target_h=1080):
    """
    Combine 3 annotated frames (Human, SAM3, DINO) into a side-by-side (1x3) comparative view.
    """
    keys = ["human", "sam3", "dino"]
    sub_w = target_w // 3
    sub_h = target_h

    resized_panels = []
    for k in keys:
        panel = frames_dict[k]
        panel_resized = cv2.resize(panel, (sub_w, sub_h), interpolation=cv2.INTER_AREA)
        # Add subtle vertical separator
        cv2.line(panel_resized, (sub_w - 1, 0), (sub_w - 1, sub_h), (60, 60, 60), 2)
        resized_panels.append(panel_resized)

    grid = np.hstack(resized_panels)
    return grid


def generate_videos(
    model_variant="yolov8n",
    split="test",
    fps=8,
    out_width=1280,
    out_height=720,
    conf_thresh=0.3,
    max_scenes=None,
    output_dir="reports/videos",
):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device}")

    # 1. Load models
    sources = ["human", "sam3", "dino"]
    models = {}
    for s in sources:
        weights_path = f"runs/{s}/{model_variant}/weights/best.pt"
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        print(f"Loading {s.upper()} model ({model_variant}) from {weights_path}...")
        models[s] = YOLO(weights_path)
        models[s].to(device)

    # 2. Collect image frames across scenes
    if split == "all":
        split_dirs = ["datasets/piglife/raw/Image/test", "datasets/piglife/raw/Image/train"]
    else:
        split_dirs = [f"datasets/piglife/raw/Image/{split}"]

    for d in split_dirs:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Directory not found: {d}")

    scenes = scan_scenes(split_dirs)
    sorted_scene_keys = sorted(scenes.keys())
    if max_scenes and max_scenes > 0:
        sorted_scene_keys = sorted_scene_keys[:max_scenes]

    total_scenes = len(sorted_scene_keys)
    total_frames_count = sum(len(scenes[k]) for k in sorted_scene_keys)
    print(f"Processing {total_scenes} scenes and {total_frames_count} total frames from split '{split}'.")

    # Flatten ordered frames list across all selected scenes
    all_frame_items = []
    for scene_id in sorted_scene_keys:
        frame_list = scenes[scene_id]
        for f_idx, (f_num, f_path) in enumerate(frame_list):
            all_frame_items.append((scene_id, f_idx, len(frame_list), f_path))

    # 3. Setup Video Writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writers = {}
    # Individual video writers
    for s in sources:
        out_path = os.path.join(output_dir, f"{model_variant}_{s}_{split}.mp4")
        writers[s] = (
            cv2.VideoWriter(out_path, fourcc, fps, (out_width, out_height)),
            out_path,
        )

    # Side-by-Side 1x3 Comparison video writer (1920x720)
    grid_w = 1920
    grid_h = 720
    grid_path = os.path.join(output_dir, f"{model_variant}_comparison_side_by_side_{split}.mp4")
    grid_writer = (cv2.VideoWriter(grid_path, fourcc, fps, (grid_w, grid_h)), grid_path)

    print(f"\nRendering {len(all_frame_items)} frames at {fps} FPS...")
    start_total_time = time.time()

    for idx, (scene_id, f_idx, scene_len, f_path) in enumerate(all_frame_items):
        raw_img = cv2.imread(f_path)
        if raw_img is None:
            continue

        annotated_frames = {}
        counts = {}
        model_fps = {}

        # Run inference for each model
        for s in sources:
            t0 = time.time()
            results = models[s](raw_img, verbose=False, conf=conf_thresh)
            t1 = time.time()
            inf_fps = 1.0 / max(t1 - t0, 1e-5)
            model_fps[s] = inf_fps

            # Render individual frame
            rendered, count = render_detections(
                raw_img, results, s, conf_thresh=conf_thresh
            )
            counts[s] = count

            # Add Header Banner
            title_text = f"{model_variant.upper()} trained with PigLife labeled by {DISPLAY_NAMES[s]}"
            sub_text = f"{counts[s]} Pigs Detected"
            draw_header(rendered, title_text, sub_text, color=COLORS[s])

            # Add Footer
            draw_info_footer(
                rendered, scene_id, f_idx, scene_len, counts[s], inf_fps, color=COLORS[s]
            )

            # Resize to individual video target size
            resized_ind = cv2.resize(rendered, (out_width, out_height))
            annotated_frames[s] = resized_ind
            writers[s][0].write(resized_ind)

        # Build & Write Side-by-Side Grid
        grid_frame = build_comparison_grid(annotated_frames, target_w=grid_w, target_h=grid_h)
        grid_writer[0].write(grid_frame)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(all_frame_items):
            elapsed = time.time() - start_total_time
            proc_fps = (idx + 1) / elapsed
            print(
                f"  Progress: [{idx + 1}/{len(all_frame_items)}] frames rendered "
                f"({proc_fps:.1f} fps) - Elapsed: {elapsed:.1f}s"
            )

    # Release writers
    for s in sources:
        writers[s][0].release()
        size_mb = os.path.getsize(writers[s][1]) / (1024 * 1024)
        print(f" Saved: {writers[s][1]} ({size_mb:.2f} MB)")

    grid_writer[0].release()
    grid_size_mb = os.path.getsize(grid_writer[1]) / (1024 * 1024)
    print(f" Saved: {grid_writer[1]} ({grid_size_mb:.2f} MB)")

    total_time = time.time() - start_total_time
    print(f"\n All videos successfully generated in {total_time:.2f} seconds!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate YOLOv8 comparison videos for PigLife."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m"],
        help="YOLO model variant",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train", "all"],
        help="Dataset split to render ('test', 'train', or 'all')",
    )
    parser.add_argument("--fps", type=int, default=8, help="Video playback FPS")
    parser.add_argument("--width", type=int, default=1280, help="Individual video width")
    parser.add_argument("--height", type=int, default=720, help="Individual video height")
    parser.add_argument(
        "--conf", type=float, default=0.3, help="Confidence threshold"
    )
    parser.add_argument(
        "--max-scenes", type=int, default=None, help="Limit number of scenes to render"
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports/videos", help="Output directory"
    )

    args = parser.parse_args()
    generate_videos(
        model_variant=args.model,
        split=args.split,
        fps=args.fps,
        out_width=args.width,
        out_height=args.height,
        conf_thresh=args.conf,
        max_scenes=args.max_scenes,
        output_dir=args.output_dir,
    )
