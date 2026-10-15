"""Small stdlib-only writer used to maintain the teaching notebooks."""
from pathlib import Path
import json
import textwrap
import uuid

ROOT = Path(__file__).resolve().parent


def md(source):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}


def code(source):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [],
            "source": textwrap.dedent(source).strip() + "\n"}


def write_notebook(filename, cells):
    target = ROOT / "notebooks" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Image Classification (PyTorch)",
                       "language": "python", "name": "image-classification"},
        "language_info": {"name": "python", "version": "3.13"}},
        "nbformat": 4, "nbformat_minor": 5}
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(target.name)


bootstrap = '''
from pathlib import Path
import sys

# Works when Jupyter starts in the project root or in notebooks/.
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "course_utils.py").exists()), None)
if ROOT is None:
    raise RuntimeError("Open this notebook from the complete course folder; course_utils.py is required.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import torch
import course_utils as cu

DEVICE = cu.configure(seed=42)
SMOKE = cu.SMOKE  # COURSE_SMOKE=1: validation only; too small to judge model quality.
QUICK = True     # Set False for a longer learning experiment, then Restart Kernel and Run All.
'''
