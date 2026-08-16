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

    rows = {"human": [], "sam3": []}
    for source in ["human", "sam3"]:
        for model in YOLO_MODELS:
            data = load_metrics(f"{model}_{source}_performance.json")
            b = bench[f"{source}_{model}"]
            display = MODEL_DISPLAY[model]
            params = MODEL_PARAMS[model]
            fwd = f"{b['inf_forward_ms']:.2f}"
            pipe = f"{b['inf_pipeline_ms']:.2f}"
            cells = metrics_cells(data)
            rows[source].append(
                f"    & {display} & {params} & {fwd} & {pipe} & {cells} \\\\"
            )

    zero = load_metrics("sam3_zero_shot_performance.json")
    b0 = bench["sam3_zero_shot"]
    zero_cells = metrics_cells(zero)
    zero_row = (
        f"  \\makecell{{Zero-shot\\\\ baseline}} & SAM 3 & \\approxm{{{MODEL_PARAMS['sam3_zero_shot']}}} "
        f"& {b0['inf_forward_ms']:.2f} & {b0['inf_pipeline_ms']:.2f} & {zero_cells} \\\\"
    )

    human_rows = "\n".join(rows["human"])
    sam3_rows = "\n".join(rows["sam3"])

    table = f"""\\begin{{table*}}[t]
  \\caption{{Object detection performance (COCO metrics) of YOLOv8 models.}}
  \\centering
  \\label{{tab:yolo_ap_metrics}}
  \\begin{{tabular}}{{c l c c c l l l l l}}
  \\toprule
  \\textbf{{Annotation}} & \\textbf{{Model}} & \\textbf{{Params (M)}} & \\textbf{{Inf. Forward (ms)}} & \\textbf{{Inf. Pipeline (ms)}} & \\textbf{{$mAP$}} & \\textbf{{$AP_{{50}}$}} & \\textbf{{$AP_{{75}}$}} & \\textbf{{$AP_{{M}}$}} & \\textbf{{$AP_{{L}}$}} \\\\
  \\midrule
  \\multirow{{3}}{{*}}{{\\makecell{{Human\\\\ annotated}}}}
{human_rows}
  \\midrule
  \\multirow{{3}}{{*}}{{\\makecell{{SAM 3\\\\ generated}}}}
{sam3_rows}
  \\midrule
  {zero_row}
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
