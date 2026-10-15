"""Validate notebook structure and optionally execute each in a fresh kernel.

Usage: python scripts/validate_notebooks.py --execute --mode smoke
       python scripts/validate_notebooks.py --execute --mode quick --pattern 04_*
"""
from pathlib import Path
import argparse
import ast
import json
import os
import time
import traceback
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=["smoke", "quick"], default="smoke")
    parser.add_argument("--pattern", default="*.ipynb")
    parser.add_argument("--cpu", action="store_true", help="Hide CUDA for fallback verification")
    args = parser.parse_args()
    paths = sorted((ROOT / "notebooks").glob(args.pattern))
    if not paths:
        raise SystemExit("No notebooks matched.")
    validation_dir = ROOT / "validation"
    output_dir = validation_dir / "executed"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    env = os.environ.copy()
    env["COURSE_SMOKE"] = "1" if args.mode == "smoke" else "0"
    if args.cpu:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    report_name = (f"{args.mode}_{'cpu' if args.cpu else 'gpu'}_" if args.execute else "structure_") + args.pattern.replace("*", "all").replace(".ipynb", "") + ".json"
    for path in paths:
        start = time.perf_counter()
        nb = nbformat.read(path, as_version=4)
        item = {"notebook": path.name, "mode": args.mode, "execution_requested": args.execute}
        try:
            nbformat.validate(nb)
            for index, cell in enumerate(nb.cells):
                if cell.cell_type == "code":
                    ast.parse(cell.source, filename=f"{path.name}:cell{index}")
            item["structure"] = "passed"
            if args.execute:
                print(f"RUN {path.name} ({args.mode})", flush=True)
                client = NotebookClient(nb, timeout=600, kernel_name="image-classification",
                                        resources={"metadata": {"path": str(ROOT / "notebooks")}})
                client.execute(env=env)
                nbformat.write(nb, output_dir / f"{args.mode}_{'cpu' if args.cpu else 'gpu'}_{path.name}")
                item["execution"] = "passed"
                item["output_cells"] = sum(bool(c.get("outputs")) for c in nb.cells)
            print(f"PASS {path.name}", flush=True)
        except Exception:
            item["execution" if args.execute else "structure"] = "failed"
            item["error"] = traceback.format_exc()
            nbformat.write(nb, output_dir / f"FAILED_{args.mode}_{path.name}")
            print(item["error"], flush=True)
        item["seconds"] = round(time.perf_counter() - start, 2)
        report.append(item)
        (validation_dir / report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    failures = sum("failed" in (row.get("structure"), row.get("execution")) for row in report)
    print(f"{len(report) - failures}/{len(report)} passed; report: {report_name}", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
