import streamlit as st
import os
import cv2
import json
import numpy as np
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = str(_ROOT / "datasets" / "piglife" / "yolo" / "human" / "images" / "test")
GT_DIR  = str(_ROOT / "datasets" / "piglife" / "yolo" / "human" / "labels" / "test")
PRED_JSON = str(_ROOT / "teacher" / "predictions.json")

# Define the groups logic
TARGET_GROUPS = [
    ["1050s1132a1110s3003-3s5001"],
    ["1050s1132a1110s3003-5s5001"],
    ["1050s1132a1110s3004", "1050s1132a1112"],
    ["1010s1120", "1010s1121"],
    ["1060s1112", "1060a1000s1110"],
    ["1020s1120", "1020s1220"],
    ["1040s1132"],
    ["1020s1121", "1020s1221"]
]

TH_SMALL = 32 * 32
TH_MEDIUM = 96 * 96

st.set_page_config(layout="wide", page_title="PigLife Gallery")

@st.cache_data
def load_data():
    try:
        image_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png'))])
        
        with open(PRED_JSON, 'r') as f:
            raw_preds = json.load(f)
        
        pred_map = {}
        for p in raw_preds:
            if p['image_id'] not in pred_map: 
                pred_map[p['image_id']] = []
            pred_map[p['image_id']].append(p)
            
        return image_files, pred_map
    except FileNotFoundError:
        st.error(f"Could not find data in {BASE_PATH}. Please check paths.")
        return [], {}

image_files, pred_map = load_data()

def get_color_by_size(width, height, is_gt=True):
    """
    Returns BGR color tuple based on box area (w * h).
    """
    area = width * height
    
    if area < TH_SMALL:
        return (255, 0, 255) if is_gt else (255, 0, 0)
    elif area < TH_MEDIUM:
        return (0, 255, 255) if is_gt else (255, 255, 0)
    else:
        return (0, 0, 255) if is_gt else (0, 255, 0)

def process_image(img_name, show_gt, show_pred):
    """
    Reads image, draws boxes colored by size, returns RGB array.
    """
    img_path = os.path.join(IMG_DIR, img_name)
    image = cv2.imread(img_path)
    if image is None:
        return np.zeros((100,100,3), dtype=np.uint8) # Return black square on error

    h_img, w_img, _ = image.shape
    img_id = os.path.splitext(img_name)[0]

    if show_gt:
        gt_path = os.path.join(GT_DIR, img_id + ".txt")
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 5:
                        xc, yc, bw_norm, bh_norm = parts[1:5]
                        
                        w_px = bw_norm * w_img
                        h_px = bh_norm * h_img
                        
                        x1 = int((xc - bw_norm/2) * w_img)
                        y1 = int((yc - bh_norm/2) * h_img)
                        x2 = int((xc + bw_norm/2) * w_img)
                        y2 = int((yc + bh_norm/2) * h_img)
                        
                        color = get_color_by_size(w_px, h_px, is_gt=True)
                        
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    if show_pred:
        preds = pred_map.get(img_id, [])
        for p in preds:
            bbox = p.get('bbox', [])
            if len(bbox) == 4:
                x_min, y_min, w_px, h_px = bbox
                
                x = int(x_min)
                y = int(y_min)
                w = int(w_px)
                h = int(h_px)
                
                color = get_color_by_size(w, h, is_gt=False)
                
                cv2.rectangle(image, (x-2, y-2), (x+w+2, y+h+2), color, 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

st.title("🐷 PigLife Gallery Explorer")

with st.sidebar:
    st.header("Visual Settings")
    show_gt = st.checkbox("Show Labels", value=True)
    show_pred = st.checkbox("Show Preds", value=True)
    
    st.markdown("""
    **Color Legend (Size)**
    * **Large (>96²):** :red[GT Red] | :green[Pred Green]
    * **Medium (32²-96²):** :orange[GT Yellow] | :blue[Pred Cyan]
    * **Small (<32²):** :violet[GT Magenta] | :blue[Pred Blue]
    """)
    
    st.divider()
    
    st.header("Group Filters")
    st.write("Select groups to include in gallery:")
    
    selected_prefixes = []
    
    for i, group_ids in enumerate(TARGET_GROUPS, 1):
        label = f"Group {i}"
        if st.checkbox(label, value=False):
            selected_prefixes.extend(group_ids)

    st.divider()
    max_imgs = st.slider("Max Images to Display", min_value=10, max_value=200, value=30, step=10, help="Limit images to prevent browser lag")

if not selected_prefixes:
    st.info("👈 Please select at least one Group from the sidebar to view the gallery.")
    st.stop()

filtered_files = [
    f for f in image_files 
    if any(f.startswith(prefix) for prefix in selected_prefixes)
]

display_files = filtered_files[:max_imgs]

st.write(f"### Found {len(filtered_files)} images (Showing top {len(display_files)})")

cols = st.columns(3)

for idx, img_name in enumerate(display_files):
    col_idx = idx % 3
    
    with cols[col_idx]:
        final_img = process_image(img_name, show_gt, show_pred)
        
        st.image(final_img, use_container_width=True)
        st.caption(f"**{img_name}**")
        st.divider()

if len(filtered_files) > max_imgs:
    st.warning(f"⚠️ Only showing the first {max_imgs} images. Increase the limit in the sidebar to see more.")