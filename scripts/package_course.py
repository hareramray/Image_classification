"""Package clean lessons and successful example runs without large local caches."""
from pathlib import Path
import importlib.metadata
import json
import platform
import zipfile

ROOT = Path(__file__).resolve().parents[1]
records = {}
for report in sorted((ROOT / "validation").glob("*.json"), key=lambda p: p.stat().st_mtime):
    payload = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        continue
    for row in payload:
        if not isinstance(row, dict) or not row.get("execution_requested"):
            continue
        device = "cpu" if "_cpu_" in report.name else "gpu"
        records[(row["notebook"], row["mode"], device)] = {**row, "device": device, "report": report.name}

notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
assert len(notebooks) == 13
assert all(records[(p.name, "smoke", "gpu")]["execution"] == "passed" for p in notebooks)
assert all(records[(p.name, "quick", "gpu")]["execution"] == "passed" for p in notebooks[2:])
assert records[("08_training_optimization.ipynb", "smoke", "cpu")]["execution"] == "passed"
for path in notebooks:
    nb = json.loads(path.read_text(encoding="utf-8"))
    assert all(not c.get("outputs") and c.get("execution_count") is None
               for c in nb["cells"] if c["cell_type"] == "code")

summary = {
    "date": "2026-10-15", "timezone": "Asia/Calcutta", "python": platform.python_version(),
    "packages": {name: importlib.metadata.version(name) for name in (
        "torch", "torchvision", "jupyterlab", "ipykernel", "numpy", "matplotlib",
        "pandas", "scikit-learn", "pillow", "nbformat", "nbclient")},
    "checks": sorted(records.values(), key=lambda r: (r["notebook"], r["mode"], r["device"]))}
(ROOT / "validation" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

files = [ROOT / name for name in ("README.md", "DEMONSTRATION_GUIDE.md", "VALIDATION.md",
         "requirements.txt", "launch_jupyter.ps1", "course_utils.py", "predict_image.py", "authoring.py", ".gitignore")]
files += notebooks + sorted((ROOT / "scripts").glob("*.py"))
files += sorted((ROOT / "validation").glob("*.json"))
for i, path in enumerate(notebooks):
    mode = "smoke" if i < 2 else "quick"
    example = ROOT / "validation" / "executed" / f"{mode}_gpu_{path.name}"
    assert example.is_file(), example
    files.append(example)

target = ROOT / "image_classification_course.zip"
with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, "image_classification_course/" + path.relative_to(ROOT).as_posix())
print(f"Packaged {len(files)} files, including 13 clean lessons and 13 executed examples: {target}")
