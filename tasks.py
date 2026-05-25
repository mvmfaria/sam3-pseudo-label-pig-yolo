from pathlib import Path
from dotenv import load_dotenv
from invoke import task
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import os
import shutil

console = Console()

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "datasets"
PIGLIFE_DIR = DATASET_DIR / "piglife"
ZIP_DIR = PIGLIFE_DIR / "zip"
RAW_DIR = PIGLIFE_DIR / "raw"

load_dotenv()
PIGLIFE_URL = os.getenv("PIGLIFE_URL")

def ensure_directories():
    for d in [ZIP_DIR, RAW_DIR, PIGLIFE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

@task
def download(c):
    """Download the PigLife dataset zip file."""
    ensure_directories()

    if not PIGLIFE_URL or PIGLIFE_URL == "YOUR_LINK_HERE":
        console.print("[bold red]Error:[/bold red] PIGLIFE_URL not defined in .env")
        raise ValueError("Please set PIGLIFE_URL in your .env file.")

    zip_file = ZIP_DIR / "piglife.zip"
    if not zip_file.exists():
        console.print(Panel(f"[bold #ff5f03]Starting dataset download...[/]\n[dim]{PIGLIFE_URL}[/dim]", title="Download", border_style="#13294c"))
        c.run(
            f'wget -q --show-progress --progress=bar:force:noscroll -O "{zip_file}" "{PIGLIFE_URL}"',
            pty=True,
        )
    else:
        console.print("[green]✔[/green] Dataset zip already exists. Skipping download.")

@task(pre=[download])
def setup(c):
    """Unzip, sanitize, split and convert the human-annotated dataset to YOLO format."""
    ensure_directories()
    piglife_zip = ZIP_DIR / "piglife.zip"
    images_dir = RAW_DIR / "Image"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:

        if not any(RAW_DIR.iterdir()):
            progress.add_task(description="Extracting main dataset...", total=None)
            c.run(f'unzip -q "{piglife_zip}" -d "{RAW_DIR}" -x "Video/*"')

        images_dir.mkdir(parents=True, exist_ok=True)
        for zname in ("train.zip", "test.zip"):
            zip_path = images_dir / zname
            target_folder = images_dir / zname.replace(".zip", "")

            if zip_path.exists() and not target_folder.exists():
                progress.add_task(description=f"Unzipping {zname}...", total=None)
                c.run(f'unzip -q "{zip_path}" -d "{images_dir}"')

        c.run(f'unzip -q "{RAW_DIR / "Names.zip"}" -d "{RAW_DIR}"')

    console.print("[white]Data processing pipeline:[/white]")

    steps = [
        ("Sanitizing filenames",            "sanitize.py",  ""),
        ("Splitting dataset (train/val)",   "split.py",     ""),
        ("Converting COCO → YOLO (human)", "convert.py",   "--source human"),
        ("Organizing directory structure",  "organize.py",  ""),
    ]

    for desc, script, args in steps:
        cmd = f'uv run python "{PROJECT_DIR}/datasets/{script}"'
        if args:
            cmd += f" {args}"
        with console.status(f"[bold white]{desc}...[/bold white]"):
            c.run(cmd, hide=True)

            if script == "split.py":
                anno_path = PIGLIFE_DIR / "coco" / "human" / "annotations"
                anno_path.mkdir(parents=True, exist_ok=True)

                source_json = images_dir / "pig_coco_test.json"
                dest_json = anno_path / "instances_test.json"

                if source_json.exists():
                    shutil.copy2(source_json, dest_json)

            console.print(f"  [green]✔[/green] {desc} completed.")

    macosx_dir = images_dir / "__MACOSX"
    if macosx_dir.exists():
        shutil.rmtree(macosx_dir)

@task(pre=[setup])
def build(c):
    """Complete dataset build — download, extract, convert to YOLO format."""
    console.print("[white]YOLO structure generated at:[/white] datasets/piglife/yolo")


@task
def label(c):
    """Generate SAM3 pseudo-labels for train/val/test and convert to YOLO format."""
    console.print("[white]Teacher pipeline:[/white]")

    console.print("  Running SAM3 on all images (this takes a while)...")
    c.run(f'uv run python "{PROJECT_DIR}/teacher/label.py"')
    console.print("  [green]✔[/green] SAM3 annotations generated.")

    with console.status("[bold white]Converting SAM3 annotations → YOLO...[/bold white]"):
        c.run(
            f'uv run python "{PROJECT_DIR}/datasets/convert.py"'
            f' --source sam3 --hardlink-images-from human',
            hide=True,
        )
    console.print("  [green]✔[/green] SAM3 YOLO conversion complete.")


@task
def train(c, source=None):
    """Train YOLOv8 (n/s/m) models. Use --source human|sam3 to train one variant."""
    cmd = f'uv run python "{PROJECT_DIR}/student/train.py"'
    if source:
        cmd += f" --source {source}"
    c.run(cmd)


@task
def evaluate(c, source=None):
    """Evaluate trained YOLOv8 models on the test set. Use --source human|sam3 for one variant."""
    cmd = f'uv run python "{PROJECT_DIR}/student/evaluate.py"'
    if source:
        cmd += f" --source {source}"
    c.run(cmd)


@task
def metrics(c):
    """Compute COCO accuracy metrics and latency benchmarks for all models."""
    console.print("[white]Metrics pipeline:[/white]")

    steps = [
        ("Calculating COCO metrics",          "reports/scripts/calculate_metrics.py"),
        ("Calculating per-group metrics",     "reports/scripts/metrics_per_group.py"),
        ("Running latency benchmarks",        "reports/scripts/benchmark.py"),
    ]

    for desc, script in steps:
        with console.status(f"[bold white]{desc}...[/bold white]"):
            c.run(f'uv run python "{PROJECT_DIR}/{script}"', hide=True)
        console.print(f"  [green]✔[/green] {desc} completed.")


@task
def report(c):
    """Generate LaTeX tables from pre-computed metrics (run after `metrics`)."""
    with console.status("[bold white]Generating LaTeX tables...[/bold white]"):
        c.run(f'uv run python "{PROJECT_DIR}/reports/scripts/generate_tables.py"', hide=True)
    console.print("  [green]✔[/green] Tables written to reports/output/latex/")


@task
def all(c, source=None):
    """Run the complete pipeline end-to-end: dataset → label → train → evaluate → metrics → report."""
    build(c)
    label(c)
    train(c, source=source)
    evaluate(c, source=source)
    metrics(c)
    report(c)
    console.print(Panel("[bold green]Full pipeline complete![/bold green]", border_style="green"))
