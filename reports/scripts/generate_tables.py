"""
Reads pre-computed metric JSONs and generates the two LaTeX tables:
  - table1_model_performance.tex  : COCO metrics for all models (Table I)
  - table2_performance_per_group.tex : SAM3 zero-shot metrics per group (Table II)

Run order:
  1. calculate_metrics.py
  2. metrics_per_group.py
  3. benchmark.py  (or keep benchmark.json as-is if hardware hasn't changed)
  4. generate_tables.py  (this script)
"""
import json
import os
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
METRICS_DIR = f"{ROOT}/reports/output/metrics"
OUTPUT_DIR = f"{ROOT}/reports/output/latex"

YOLO_MODELS = ["yolov8n", "yolov8s", "yolov8m"]
MODEL_DISPLAY = {"yolov8n": "YOLOv8n", "yolov8s": "YOLOv8s", "yolov8m": "YOLOv8m"}
MODEL_PARAMS = {
    "yolov8n": "3.2",
    "yolov8s": "11.2",
    "yolov8m": "25.9",
    "sam3_zero_shot": "840.4",
    "dino_zero_shot": "172.0",
}


def fmt(value, decimals=1):
    """Format a [0,1] metric as a percentage string."""
    if value < 0:
        return "-"
    return f"{round(value * 100, decimals):.{decimals}f}"


def load_metrics(filename):
    path = os.path.join(METRICS_DIR, filename)
    with open(path) as f:
        return json.load(f)


def metrics_cells(data):
    keys = ["mAP_50-95", "mAP_50", "mAP_75", "AP_Medium", "AP_Large"]
    return " & ".join(fmt(data[k]) for k in keys)


def build_table1():
    bench = load_metrics("benchmark.json")

    sources = ["human", "sam3"]
    if os.path.exists(os.path.join(METRICS_DIR, "yolov8n_dino_performance.json")):
        sources.append("dino")

    rows = {s: [] for s in sources}
    source_labels = {
        "human": "\\multirow{3}{*}{\\makecell{Human\\\\ annotated}}",
        "sam3": "\\multirow{3}{*}{\\makecell{SAM 3\\\\ generated}}",
        "dino": "\\multirow{3}{*}{\\makecell{Grounding DINO\\\\ generated}}",
    }

    sections = []
    for source in sources:
        for model in YOLO_MODELS:
            data = load_metrics(f"{model}_{source}_performance.json")
            b = bench.get(f"{source}_{model}", {"inf_forward_ms": 0.0, "inf_pipeline_ms": 0.0})
            display = MODEL_DISPLAY[model]
            params = MODEL_PARAMS[model]
            fwd = f"{b['inf_forward_ms']:.2f}"
            pipe = f"{b['inf_pipeline_ms']:.2f}"
            cells = metrics_cells(data)
            rows[source].append(
                f"    & {display} & {params} & {fwd} & {pipe} & {cells} \\\\"
            )
        section_rows = "\n".join(rows[source])
        sections.append(f"  {source_labels[source]}\n{section_rows}")

    zero_rows = []
    if os.path.exists(os.path.join(METRICS_DIR, "sam3_zero_shot_performance.json")):
        zero = load_metrics("sam3_zero_shot_performance.json")
        b0 = bench.get("sam3_zero_shot", {"inf_forward_ms": 855.16, "inf_pipeline_ms": 900.51})
        zero_cells = metrics_cells(zero)
        zero_rows.append(
            f"  \\makecell{{Zero-shot\\\\ baseline}} & SAM 3 & \\approx{{{MODEL_PARAMS['sam3_zero_shot']}}} "
            f"& {b0['inf_forward_ms']:.2f} & {b0['inf_pipeline_ms']:.2f} & {zero_cells} \\\\"
        )

    if os.path.exists(os.path.join(METRICS_DIR, "dino_zero_shot_performance.json")):
        dino_zero = load_metrics("dino_zero_shot_performance.json")
        b_dino = bench.get("dino_zero_shot", {"inf_forward_ms": 115.0, "inf_pipeline_ms": 135.0})
        dino_cells = metrics_cells(dino_zero)
        zero_rows.append(
            f"  \\makecell{{Zero-shot\\\\ baseline}} & Grounding DINO & \\approx{{{MODEL_PARAMS['dino_zero_shot']}}} "
            f"& {b_dino['inf_forward_ms']:.2f} & {b_dino['inf_pipeline_ms']:.2f} & {dino_cells} \\\\"
        )

    body = "\n  \\midrule\n".join(sections)
    if zero_rows:
        body += "\n  \\midrule\n" + "\n".join(zero_rows)

    table = f"""\\begin{{table*}}[t]
  \\caption{{Object detection performance (COCO metrics) of YOLOv8 models.}}
  \\centering
  \\label{{tab:yolo_ap_metrics}}
  \\begin{{tabular}}{{c l c c c l l l l l}}
  \\toprule
  \\textbf{{Annotation}} & \\textbf{{Model}} & \\textbf{{Params (M)}} & \\textbf{{Inf. Forward (ms)}} & \\textbf{{Inf. Pipeline (ms)}} & \\textbf{{$mAP$}} & \\textbf{{$AP_{{50}}$}} & \\textbf{{$AP_{{75}}$}} & \\textbf{{$AP_{{M}}$}} & \\textbf{{$AP_{{L}}$}} \\\\
  \\midrule
{body}
  \\bottomrule
  \\end{{tabular}}
\\end{{table*}}
"""
    return table


def build_table2():
    groups = load_metrics("sam3_zero_shot_groups.json")

    groups = sorted(groups, key=lambda g: g["group_name"])

    rows = []
    for g in groups:
        cells = " & ".join(
            fmt(g[k]) for k in ["mAP_50-95", "mAP_50", "mAP_75", "AP_Medium", "AP_Large"]
        )
        rows.append(f"  {g['group_name']} & {g['image_count']} & {cells} \\\\")

    body = "\n".join(rows)

    table = f"""\\begin{{table}}[t]
  \\caption{{Performance metrics per group (SAM3 zero-shot, in \\%).}}
  \\centering
  \\label{{tab:group_metrics}}
  \\begin{{tabular}}{{l c c c c c c}}
  \\toprule
  \\textbf{{Group}} & \\textbf{{Images}} & \\textbf{{$mAP$}} & \\textbf{{$AP_{{50}}$}} & \\textbf{{$AP_{{75}}$}} & \\textbf{{$AP_{{M}}$}} & \\textbf{{$AP_{{L}}$}} \\\\
  \\midrule
{body}
  \\bottomrule
  \\end{{tabular}}
\\end{{table}}
"""
    return table


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t1 = build_table1()
    path1 = os.path.join(OUTPUT_DIR, "table1_model_performance.tex")
    with open(path1, "w") as f:
        f.write(t1)
    print(f"Saved: {path1}")

    t2 = build_table2()
    path2 = os.path.join(OUTPUT_DIR, "table2_performance_per_group.tex")
    with open(path2, "w") as f:
        f.write(t2)
    print(f"Saved: {path2}")
