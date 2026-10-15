# Practical Image Classification with PyTorch

**13 Jupyter notebooks, from your first tensor to advanced classification workflows.**
Designed for step-by-step live demonstrations on Windows with an NVIDIA GPU.
Each lesson contains explanations, runnable code, visualizations, discussion prompts,
exercises, and suggested answers. You can run every notebook independently.

## Start on this computer

The project environment is already prepared in `.venv`, with the Jupyter kernel
**Image Classification (PyTorch)** registered inside that environment. It reuses the
existing CUDA-enabled PyTorch installation instead of downloading another copy.

Open PowerShell in this folder and run:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

Open [notebooks/00_start_here_and_gpu_setup.ipynb](notebooks/00_start_here_and_gpu_setup.ipynb).
Select **Image Classification (PyTorch)** as the kernel, then use **Shift+Enter**
to teach cell by cell. Use **Kernel → Restart Kernel and Run All Cells** before a class.
The optional [launch_jupyter.ps1](launch_jupyter.ps1) launches the same environment.

The portable `image_classification_course.zip` includes the notebooks, supporting code,
guides, and successful example outputs. It omits `.venv`, datasets, and model weights;
use the recreation instructions below when opening the ZIP on another computer.

## Learning sequence

Teaching times include explanation and exercises; they are planning estimates,
not measured execution-time promises. First-time downloads add time.

| Notebook | Level | What you demonstrate | Suggested teaching time |
|---|---|---|---|
| [00 · Start here and GPU setup](notebooks/00_start_here_and_gpu_setup.ipynb) | Beginner | Jupyter, GPU check, tensors, logits, loss, gradients | 30–45 min |
| [01 · Images, datasets, and splits](notebooks/01_images_datasets_and_splits.ipynb) | Beginner | Pixels, channels, batches, normalization, leakage-free splits | 40–60 min |
| [02 · Linear classifier](notebooks/02_linear_classifier.ipynb) | Beginner | First trained classifier, explicit training loop, baseline | 45–60 min |
| [03 · Neural networks](notebooks/03_neural_networks.ipynb) | Beginner–intermediate | Hidden layers, activations, gradients, learning curves | 45–60 min |
| [04 · Convolutional networks](notebooks/04_convolutional_networks.ipynb) | Intermediate | Convolution, pooling, CNN training, feature maps | 45–60 min |
| [05 · Augmentation and evaluation](notebooks/05_augmentation_and_evaluation.ipynb) | Intermediate | Controlled augmentation comparison, confusion matrix, precision/recall/F1 | 60–90 min |
| [06 · Transfer learning](notebooks/06_transfer_learning.ipynb) | Intermediate | Pretrained ResNet18, frozen backbone, fine-tuning, BatchNorm | 60–90 min |
| [07 · Custom image dataset](notebooks/07_custom_image_dataset.ipynb) | Intermediate | ImageFolder, local shapes demo, custom photos, duplicate audits | 60–90 min |
| [08 · Training optimization](notebooks/08_training_optimization.ipynb) | Advanced | Validation-only tuning, AMP, accumulation, clipping, checkpoint/resume | 60–90 min |
| [09 · Explainability](notebooks/09_explainability.ipynb) | Advanced | Saliency, Grad-CAM, occlusion, explanation limitations | 45–60 min |
| [10 · Vision transformers](notebooks/10_vision_transformers.ipynb) | Advanced | Patches, tokens, attention, tiny ViT training, CNN comparison | 60–90 min |
| [11 · Robustness and calibration](notebooks/11_robustness_and_calibration.ipynb) | Advanced | Separate calibration split, temperature scaling, reliability, corruptions | 60–90 min |
| [12 · Deployment and capstone](notebooks/12_deployment_and_capstone.ipynb) | Advanced | Model bundle, standalone prediction, export, latency, project rubric | 60–90 min |

This is a progressive practical course through advanced topics. Expertise requires
repeated experiments with real, representative datasets; short demos are not benchmark results.
See [DEMONSTRATION_GUIDE.md](DEMONSTRATION_GUIDE.md) for teaching schedules and class preparation.

## Workload, data, and downloads

- `QUICK=True` is the default: a few thousand training images and a few epochs in most lessons.
  Transfer learning uses a smaller subset and batch size 16 to fit modest GPU memory.
- Set `QUICK=False` in a lesson's setup cell for its longer configuration, then restart and run all.
- `COURSE_SMOKE=1` is only for tiny execution checks. Its metrics are not meaningful performance evidence.
- Fashion-MNIST downloads roughly 30 MB. CIFAR-10 downloads roughly 170 MB and expands on disk.
  ResNet18 pretrained weights add roughly 45 MB. These are cached after first use.
- Lesson 07 generates a local shapes dataset by default. It requires no user-provided photographs.
  It also explains how to use your own image folders.
- Lesson 10 trains a small ViT from scratch by default. Its optional ImageNet-pretrained ViT
  demonstration is disabled initially and requires a separate download of more than 300 MB
  plus a supplied local RGB image.
- Allow several GB for the environment, cached datasets, checkpoints, and optional weights.

Data are stored in `data/`, results in `artifacts/`, and pretrained weights in the standard
PyTorch user cache. Preserve these caches for an offline presentation. The notebooks do not
require paid services, credentials, or a preceding lesson's saved model.

## Recreate the environment

On this machine, the base Python already has a compatible `torch`/`torchvision` installation:

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name image-classification --display-name "Image Classification (PyTorch)"
.\.venv\Scripts\python.exe -m jupyterlab
```

On another computer, create a clean virtual environment with `python -m venv .venv`.
Install the matching PyTorch and torchvision pair using the command produced by the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/)
for that operating system and supported CUDA runtime. Run that pip command through
`.\.venv\Scripts\python.exe -m pip`, then install `requirements.txt` and register the kernel as above.
`requirements.txt` intentionally does not replace an existing PyTorch installation.
The CUDA wheel includes the runtime used by these examples; compiling CUDA extensions
is outside this course.

Recorded preparation environment: Python 3.13.15, PyTorch 2.11.0+cu128,
torchvision 0.26.0+cu128, NVIDIA GeForce RTX 5050 Laptop GPU (about 8 GB), driver 577.13.
These describe this workspace, not a claim that every other version/GPU combination is supported.

## Practical troubleshooting

| Symptom | Action |
|---|---|
| `ModuleNotFoundError` | Check the selected kernel and `sys.executable`; install through that interpreter. |
| `course_utils` cannot be found | Keep the complete project together and start Jupyter in this root folder. |
| CUDA is unavailable | Run notebook 00, inspect the driver with `nvidia-smi`, and check the PyTorch CUDA build. CPU fallback works. |
| Triton warns that the CUDA Toolkit is missing | This machine's optional Triton package can print that warning even when ordinary PyTorch CUDA training succeeds. These lessons do not use Triton compilation; verify computation in notebook 00. |
| CUDA out of memory | Restart the kernel, close other GPU workloads, reduce batch size, and keep `QUICK=True`. |
| Windows DataLoader hangs | Keep `num_workers=0`, as in the supplied loader helper. |
| Download fails | Retry with internet access before class; retain the original downloaded data cache. |
| Plot or metric differs | Restart and run all; seeds reduce variation but do not promise equality across devices and software versions. |
| Very low smoke accuracy | Expected: smoke mode verifies execution, not learning quality. |
| A cell depends on an old variable | Restart the kernel and run cells in order. |

## Validate or maintain the course

```powershell
# Validate notebook JSON structure and Python syntax.
.\.venv\Scripts\python.exe scripts/validate_notebooks.py

# Execute independent fresh kernels with tiny datasets on the available device.
.\.venv\Scripts\python.exe scripts/validate_notebooks.py --execute --mode smoke

# Execute the normal quick configuration for one lesson.
.\.venv\Scripts\python.exe scripts/validate_notebooks.py --execute --mode quick --pattern "04_*"

# Verify CPU fallback for one lesson.
.\.venv\Scripts\python.exe scripts/validate_notebooks.py --execute --mode smoke --cpu --pattern "08_*"
```

Reports go in `validation/`; executed copies with plots go in `validation/executed/`.
The teaching notebooks remain output-free so learners can generate the results themselves.
Read [VALIDATION.md](VALIDATION.md) for what was actually checked in this workspace.

The optional `scripts/build_*.py` files generate the notebooks using `authoring.py`.
They are maintenance sources, not part of the learner workflow. Running a builder
replaces its corresponding notebooks, including any manual edits and outputs.

## Reuse a trained classifier

After running lesson 12 in normal quick mode:

```powershell
.\.venv\Scripts\python.exe predict_image.py --bundle artifacts/12_bundle_demo --image artifacts/12_bundle_demo/example.png --device cpu
```

This predicts Fashion-MNIST-like clothing images only. Its bundle stores class names,
preprocessing, weights, and limitations. Follow the lesson's capstone to adapt a classifier
to your own representative data instead of treating arbitrary photos as supported inputs.

## Primary references

The lessons link to their relevant primary sources. The core references are
[PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html),
[torchvision models and weights](https://docs.pytorch.org/vision/stable/models.html),
[transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html),
[automatic mixed precision](https://docs.pytorch.org/docs/stable/notes/amp_examples.html),
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), and
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
