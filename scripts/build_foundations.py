"""Regenerate lessons 00, 01 and 12. Learners should edit the notebooks directly."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from authoring import md, code, write_notebook, bootstrap


write_notebook("00_start_here_and_gpu_setup.ipynb", [
md('''
# 00 · Start here: Jupyter, tensors, and your GPU

**Level:** first contact with PyTorch · **Teaching time:** 30–45 minutes.

By the end you will run notebook cells, verify your GPU, read image tensor shapes,
calculate class probabilities, and watch automatic differentiation work.
You need basic Python variables, lists, functions, and loops; no deep learning experience is assumed.

This course contains **13 independent notebooks**. Work through 00–12 in order when learning;
later you can run any single lesson from a clean kernel for a demonstration.
Keep the whole folder together: `course_utils.py` contains shared data and plotting helpers.

**Before class:** use the PowerShell setup in the root README, select the
**Image Classification (PyTorch)** kernel, and run notebooks 01 and 06 once to cache their downloads.
'''),
md('''
## Step 1 — Learn the notebook controls

Click a cell and press **Shift+Enter** to run it and move down. Markdown cells contain explanations;
code cells execute Python in one shared kernel. An `[*]` prompt means a cell is still running.
Use **Kernel → Restart Kernel and Run All Cells** to verify that a lesson works from top to bottom.
Running cells out of order can leave old values in memory, so restart after changing the configuration.

**Demonstrate:** change the greeting below, run it, then explain that Python keeps its value until restart.
'''),
code('''
greeting = "Welcome to practical image classification!"
print(greeting)
'''),
md('''
## Step 2 — Check Python and the selected kernel

The interpreter path should point to this course's `.venv` when using the supplied launcher.
Installing a package into one Python does not make it available to every Jupyter kernel.
'''),
code('''
import sys
import platform
print("Python:", sys.version)
print("Interpreter:", sys.executable)
print("Platform:", platform.platform())
'''),
code(bootstrap),
md('''
## Step 3 — Prove the GPU can perform an operation

CUDA is the interface PyTorch uses to run on an NVIDIA GPU. A CPU tensor must be moved to the
GPU before a GPU model can process it. The model and its input must be on the same device.
`nvidia-smi` reports the driver's CUDA compatibility; `torch.version.cuda` reports the runtime
used to build PyTorch. They do not have to show the same version.

The code below also works on a CPU, but the device label tells you which hardware actually ran it.
'''),
code('''
print("CUDA available:", torch.cuda.is_available())
print("PyTorch CUDA runtime:", torch.version.cuda)
a = torch.randn(128, 128, device=DEVICE)
b = a @ a.T
if DEVICE.type == "cuda":
    torch.cuda.synchronize()
assert b.shape == (128, 128) and torch.isfinite(b).all()
print("Matrix multiplication succeeded on", b.device)
'''),
md('''
## Step 4 — Read the course map

| Lesson | Practical result |
|---|---|
| 00 | Working environment and tensor basics |
| 01 | Visualized images, batches, and leakage-free splits |
| 02 | A trained linear image classifier |
| 03 | A multilayer neural network and a full training loop |
| 04 | A CNN and visualized feature maps |
| 05 | Augmentation comparison, confusion matrix, and error analysis |
| 06 | Pretrained ResNet18: feature extraction and fine-tuning |
| 07 | Your own image-folder dataset, with a runnable shapes example |
| 08 | Validation-only tuning, mixed precision, and resumable training |
| 09 | Saliency, Grad-CAM, and occlusion explanations |
| 10 | Patch embeddings and a trained small vision transformer |
| 11 | Calibration and controlled distribution-shift experiments |
| 12 | Saved model bundle, image prediction, export, and capstone |

These notebooks introduce advanced workflows; becoming expert also requires repeated experiments
on representative real data. Short demo training is for understanding behavior, not claiming benchmark results.
'''),
md('''
## Step 5 — A tensor is an array with a shape and a data type

An image can be represented as numbers. PyTorch vision models typically receive
`[batch, channels, height, width]`, abbreviated **NCHW**. RGB has three channels; grayscale has one.
Floating-point inputs carry pixel values; integer class labels identify categories.
'''),
code('''
single_image = torch.rand(1, 28, 28)  # one grayscale image, values from 0 to 1
batch = torch.stack([single_image, single_image])
labels = torch.tensor([3, 7], dtype=torch.long)
print("Single image CHW:", single_image.shape)
print("Batch NCHW:", batch.shape, batch.dtype)
print("Labels N:", labels.shape, labels.dtype)
print("Flattened batch:", batch.flatten(start_dim=1).shape)
plt.imshow(single_image[0], cmap="gray", vmin=0, vmax=1)
plt.title("Random pixels: useful for shapes, not classification")
plt.axis("off")
plt.show()
'''),
md('''
## Step 6 — From scores to class probabilities

A classifier returns one score, or **logit**, per class. `softmax` turns those scores into positive
numbers that add to one. `argmax` chooses the class with the greatest score.
These are model probabilities, not a guarantee that a prediction is correct.
'''),
code('''
logits = torch.tensor([[2.0, 0.5, -1.0]])
probabilities = logits.softmax(dim=1)
names = ["shirt", "shoe", "bag"]
print("Logits:", logits)
print("Probabilities:", probabilities)
print("Sum:", probabilities.sum(1))
print("Predicted class:", names[logits.argmax(1).item()])
plt.bar(names, probabilities[0].numpy())
plt.ylim(0, 1)
plt.ylabel("Model probability")
plt.show()
'''),
md('''
## Step 7 — Measure a mistake with cross-entropy

Cross-entropy penalizes giving low probability to the correct class. For one example it is
`-log(probability of the true class)`. PyTorch's `CrossEntropyLoss` accepts **raw logits** and
integer labels; it performs the needed normalization internally. Do not add softmax before it.
'''),
code('''
loss_fn = torch.nn.CrossEntropyLoss()
target = torch.tensor([0])
loss = loss_fn(logits, target)
manual_loss = -torch.log(probabilities[0, target.item()])
assert torch.allclose(loss, manual_loss)
print("Loss when shirt is correct:", loss.item())
print("Loss when bag is correct:", loss_fn(logits, torch.tensor([2])).item())
'''),
md('''
## Step 8 — See the gradient that makes learning possible

Training adjusts weights to lower a loss. A gradient tells us how a small weight change affects that
loss. `backward()` computes gradients; the optimizer later applies an update.
For this scalar example the derivative of `(w - 3)^2` is `2 * (w - 3)`.
'''),
code('''
w = torch.tensor(1.0, requires_grad=True)
toy_loss = (w - 3) ** 2
toy_loss.backward()
print("Weight:", w.item(), "loss:", toy_loss.item(), "gradient:", w.grad.item())
assert w.grad.item() == -4.0
with torch.no_grad():
    w -= 0.1 * w.grad
print("Updated weight:", w.item(), "new loss:", ((w - 3) ** 2).item())
'''),
md('''
## Step 9 — Know the three modes used throughout the course

| Situation | Model mode | Gradient recording |
|---|---|---|
| Training | `model.train()` | Enabled |
| Validation and ordinary prediction | `model.eval()` | `torch.no_grad()` or `torch.inference_mode()` |
| Gradient-based explanations | `model.eval()` | Enabled |

`eval()` changes layers such as dropout and BatchNorm; it does **not** disable gradients.
`no_grad()` prevents recording gradients; it does **not** change training mode.
'''),
md('''
## Step 10 — Choose demonstration settings

Each training notebook begins with small settings suitable for a live lesson. `QUICK=True` uses
subsets and a few epochs (full passes through training examples). Set `QUICK=False` for a longer run.
The separate `COURSE_SMOKE=1` environment variable is used by the validation script to check tiny
runs. Smoke results are not meaningful model-quality measurements.

Data are cached under `data/`, generated models under `artifacts/`, and downloaded pretrained
weights in PyTorch's user cache. Downloads need internet the first time.
No notebook needs an API key or a paid service.
'''),
md('''
## Exercises and live questions

1. Make a batch of four RGB images of size 32×32. Predict its shape before printing it.
2. Change the correct class in Step 7. Why does the loss change even though the prediction does not?
3. Set the learning rate in Step 8 to `1.1`. Does this update still improve the loss?

<details><summary>Suggested answers</summary>

1. `torch.rand(4, 3, 32, 32)` has shape `[4, 3, 32, 32]`.
2. Loss compares logits with the supplied target; the prediction uses only the logits.
3. The updated weight is `5.4`, so the loss becomes `5.76`, larger than `4.0`. A larger step is not always better.

</details>

**Ready to move on:** you can explain a batch shape, a logit, a class label, and why we need gradients.

**Troubleshooting:** restart the kernel after installation; reduce batch size on CUDA out-of-memory;
use `num_workers=0` in Windows notebooks; check `sys.executable` when an import fails.
See the README for installation and the demonstration guide for a teaching schedule.

**Primary references:** [PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html),
[local installation](https://pytorch.org/get-started/locally/),
[autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html).
''')])


write_notebook("01_images_datasets_and_splits.ipynb", [
md('''
# 01 · Images, datasets, and trustworthy splits

**Level:** beginner · **Prerequisite:** lesson 00 · **Teaching time:** 40–60 minutes.

Build the data pipeline before building a model. You will inspect real clothing images,
understand pixel/channel/batch dimensions, distinguish training from validation and test data,
and calculate normalization statistics from training examples only.

Dataset: **Fashion-MNIST**, 28×28 grayscale images in ten clothing classes.
Its original training and test partitions have 60,000 and 10,000 examples.
First run downloads roughly 30 MB; cached runs work offline.
'''),
code(bootstrap),
md('''
## Step 1 — A color image is three grids

Start with a generated RGB image so every channel has an obvious meaning. Pillow and plotting
usually use **HWC**; PyTorch vision models use **CHW**, plus a batch axis for multiple examples.
'''),
code('''
from PIL import Image
from torchvision import transforms

rgb = np.zeros((32, 48, 3), dtype=np.uint8)
rgb[:, :16, 0] = 255
rgb[:, 16:32, 1] = 255
rgb[:, 32:, 2] = 255
rgb_image = Image.fromarray(rgb)
tensor_image = transforms.ToTensor()(rgb_image)
print("NumPy HWC:", rgb.shape, rgb.dtype)
print("PyTorch CHW:", tensor_image.shape, tensor_image.dtype)
print("Pixel range:", tensor_image.min().item(), tensor_image.max().item())
fig, axes = plt.subplots(1, 4, figsize=(11, 3))
axes[0].imshow(rgb_image)
axes[0].set_title("RGB combined")
for channel, name, ax in zip(tensor_image, ["Red", "Green", "Blue"], axes[1:]):
    ax.imshow(channel, cmap="gray", vmin=0, vmax=1)
    ax.set_title(name + " channel")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
'''),
md('''
## Step 2 — Set the workload explicitly

We use small, reproducible, class-balanced subsets for fast teaching. The subset is useful for
learning but is not the complete benchmark. Download size stays the same even when training
uses a subset: torchvision first caches the original dataset.
'''),
code('''
TRAIN_LIMIT = 120 if SMOKE else (3000 if QUICK else None)
VAL_LIMIT = 60 if SMOKE else (800 if QUICK else None)
TEST_LIMIT = 60 if SMOKE else (1000 if QUICK else None)
train_ds, val_ds, test_ds, class_names = cu.fashion_data(
    train_limit=TRAIN_LIMIT, val_limit=VAL_LIMIT, test_limit=TEST_LIMIT)
print(dict(train=len(train_ds), validation=len(val_ds), test=len(test_ds)))
print(dict(enumerate(class_names)))
'''),
md('''
## Step 3 — Inspect images and labels together

Always look at the data before training. Ask whether labels are plausible, the images are readable,
and the task is possible from the available pixels. In this dataset a shirt and a T-shirt may look
similar at 28×28 resolution.

**Demonstrate:** ask the audience to classify three images before revealing their labels.
'''),
code('''
cu.show_images(train_ds, class_names, n=12)
image, label = train_ds[0]
print("Image shape:", tuple(image.shape), "label ID:", label, "name:", class_names[label])
assert image.shape == (1, 28, 28)
assert image.dtype == torch.float32
assert 0 <= image.min() <= image.max() <= 1
'''),
md('''
## Step 4 — Zoom in until an image becomes numbers

An 8×8 crop below is a small matrix of brightness values. The model receives numbers, not a file name
or a human description. `ToTensor()` scales ordinary uint8 image values from `[0, 255]` to `[0, 1]`.
'''),
code('''
crop = image[0, 10:18, 10:18]
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(crop, cmap="gray", vmin=0, vmax=1)
for row in range(8):
    for col in range(8):
        ax.text(col, row, f"{crop[row, col]:.1f}", ha="center", va="center",
                color="black" if crop[row, col] > 0.5 else "white", fontsize=8)
ax.set_title("An 8×8 crop: brightness as numbers")
plt.show()
'''),
md('''
## Step 5 — Separate learning, choosing, and reporting

| Split | Allowed use |
|---|---|
| Training | Fit model weights and data statistics |
| Validation | Choose hyperparameters, model, and stopping epoch |
| Test | Report performance after those choices are fixed |

Our helper splits only the original training partition into training and validation, stratified by
class. A stratified split keeps class coverage. The official test partition is separate.
Different transforms wrap separate dataset objects, so changing training augmentation cannot
accidentally augment validation images.

For real data, split by **person, product, recording session, or time** when images share a source.
Random image-level splitting can leak near-duplicate or same-subject information.
'''),
code('''
train_indices = set(train_ds.indices)
val_indices = set(val_ds.indices)
assert train_indices.isdisjoint(val_indices)
assert train_ds.dataset.train and val_ds.dataset.train
assert not test_ds.dataset.train
print("Train/validation source indices are disjoint.")
print("Test comes from the official held-out partition.")
# Index 5 in the test partition is NOT the same source image as index 5 in training.
'''),
md('''
## Step 6 — Check class balance and a no-learning baseline

Accuracy is the fraction of correct predictions. A model that always predicts the most common
class establishes a useful baseline. On strongly imbalanced data it can look impressive while
never recognizing a minority class. We inspect training and validation counts here, leaving test
performance for later modeling lessons.
'''),
code('''
def subset_labels(subset):
    return np.asarray(subset.dataset.targets)[np.asarray(subset.indices)]

train_labels = subset_labels(train_ds)
val_labels = subset_labels(val_ds)
train_counts = np.bincount(train_labels, minlength=len(class_names))
val_counts = np.bincount(val_labels, minlength=len(class_names))
majority_class = int(train_counts.argmax())
print("Majority-class validation accuracy:", np.mean(val_labels == majority_class))
x = np.arange(len(class_names))
plt.figure(figsize=(11, 4))
plt.bar(x - .2, train_counts, .4, label="Train")
plt.bar(x + .2, val_counts, .4, label="Validation")
plt.xticks(x, class_names, rotation=45, ha="right")
plt.ylabel("Example count")
plt.legend()
plt.tight_layout()
plt.show()
'''),
md('''
## Step 7 — Turn single examples into batches

A `Dataset` retrieves one `(image, label)` pair. A `DataLoader` groups those pairs into batches,
optionally shuffling the order each epoch. Training is shuffled; evaluation usually is not.
We set `num_workers=0` to keep the examples reliable inside Windows notebooks.
'''),
code('''
train_loader = cu.loader(train_ds, batch_size=32, shuffle=True)
images, labels = next(iter(train_loader))
print("Images NCHW:", tuple(images.shape))
print("Labels N:", tuple(labels.shape), labels.dtype)
print("Batches per epoch:", len(train_loader))
print("First label names:", [class_names[i] for i in labels[:6].tolist()])
assert images.shape[0] == labels.shape[0]
assert labels.dtype == torch.long
'''),
md('''
## Step 8 — Calculate normalization from training data only

Scaling to `[0, 1]` and standardizing to `(pixel - mean) / std` are different operations.
Estimate mean and standard deviation from training pixels, then reuse the same values on
validation, test, and future input images. Here we use a bounded sample for demonstration.
For a pretrained model, follow its published preprocessing instead (lesson 06).
'''),
code('''
sample_loader = cu.loader(train_ds, batch_size=128, shuffle=False)
pixel_sum = torch.zeros(1, dtype=torch.float64)
pixel_square_sum = torch.zeros(1, dtype=torch.float64)
pixel_count = 0
for batch_index, (x, _) in enumerate(sample_loader):
    x = x.double()
    pixel_sum += x.sum(dim=(0, 2, 3))
    pixel_square_sum += x.square().sum(dim=(0, 2, 3))
    pixel_count += x.shape[0] * x.shape[2] * x.shape[3]
    if batch_index == 9:  # at most 1,280 training images
        break
mean = pixel_sum / pixel_count
std = (pixel_square_sum / pixel_count - mean.square()).clamp_min(1e-12).sqrt()
print("Training-sample mean:", mean.tolist(), "std:", std.tolist())
normalizer = transforms.Normalize(mean.tolist(), std.tolist())
normalized = normalizer(image)
reconstructed = normalized * std.float().view(-1, 1, 1) + mean.float().view(-1, 1, 1)
assert torch.allclose(image, reconstructed, atol=1e-6)
print("Normalized range:", normalized.min().item(), normalized.max().item())
'''),
md('''
## Step 9 — See how augmentation changes a training example

Augmentation simulates plausible changes without changing the label. It belongs in the training
pipeline. Validation and test preprocessing should be deterministic so comparisons are meaningful.
An augmentation that destroys the true label is harmful; ask what transformations make sense for
your own domain before choosing them.
'''),
code('''
augment = transforms.RandomAffine(degrees=12, translate=(.08, .08))
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    candidate = image if i == 0 else augment(image)
    ax.imshow(candidate[0], cmap="gray", vmin=0, vmax=1)
    ax.set_title("Original" if i == 0 else f"Variation {i}")
    ax.axis("off")
plt.tight_layout()
plt.show()
'''),
md('''
## Step 10 — Inspect the helper instead of treating it as magic

The course keeps repeated plumbing in `course_utils.py`. Early model lessons write the training
loop explicitly. Read the split implementation below and identify where the random seed is used.
'''),
code('''
import inspect
print(inspect.getsource(cu.split_indices))
'''),
md('''
## Exercises and live questions

1. Change batch size from 32 to 7. What shape does the last batch have?
2. Replace the augmentation with a vertical flip. Is the result plausible for clothing photos?
3. Your dataset has ten photos of each person. How should you split it?
4. Why does normalization use training statistics rather than all available images?

<details><summary>Suggested answers</summary>

1. Inspect `list(cu.loader(train_ds, batch_size=7))[-1][0].shape`; the final batch may contain fewer than seven examples.
2. Usually not representative of the expected input; augmentation depends on the task.
3. Keep every photo of one person in one partition, using a group-aware split.
4. Otherwise the training pipeline uses information from held-out data and no longer represents a clean evaluation.

</details>

**Ready to move on:** describe one image, one batch, and the separate jobs of all three splits.

**Primary references:** [Fashion-MNIST project](https://github.com/zalandoresearch/fashion-mnist),
[datasets and dataloaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html),
[torchvision transforms](https://docs.pytorch.org/vision/stable/transforms.html).
''')])


write_notebook("12_deployment_and_capstone.ipynb", [
md('''
# 12 · From notebook to reusable classifier

**Level:** advanced workflow · **Prerequisites:** training/evaluation, lesson 11's confidence caveats.
**Teaching time:** 60–90 minutes; training/download time depends on hardware.

Train a compact Fashion-MNIST classifier, package its weights **and preprocessing**, reload it in a
fresh object, predict a saved image using a standalone script, export a fixed-shape program, and
measure latency. Finish with a capstone plan for your own dataset.

This is a **local deployment demonstration**, not an internet service. It teaches artifact contracts,
prediction correctness, and measurement; a production service also needs monitoring and operational testing.
No previous notebook's checkpoint is required.
'''),
code(bootstrap),
md('''
## Step 1 — Define the task and input contract

This model recognizes the ten Fashion-MNIST clothing categories in grayscale 28×28 images.
It does not recognize arbitrary objects in full-size photos. Store the class order, resize operation,
color conversion, and normalization with the weights: a correct network with incorrect preprocessing
can still produce incorrect predictions.
'''),
code('''
import json
import copy
import time
from PIL import Image
from torchvision import transforms

TRAIN_LIMIT = 128 if SMOKE else (4000 if QUICK else None)
VAL_LIMIT = 48 if SMOKE else (800 if QUICK else None)
TEST_LIMIT = 48 if SMOKE else (1000 if QUICK else None)
EPOCHS = 1 if SMOKE else (3 if QUICK else 12)
train_ds, val_ds, test_ds, class_names = cu.fashion_data(
    train_limit=TRAIN_LIMIT, val_limit=VAL_LIMIT, test_limit=TEST_LIMIT)
train_loader = cu.loader(train_ds, batch_size=64, shuffle=True)
val_loader = cu.loader(val_ds, batch_size=64)
test_loader = cu.loader(test_ds, batch_size=64)
print("Class order:", class_names)
'''),
md('''
## Step 2 — Train and freeze the final model choice

The helper restores the epoch with lowest validation cross-entropy. After choosing the model,
evaluate on test once. Do not change hyperparameters in response to the test score.
The quick subset and short run are a teaching experiment, not a benchmark submission.
'''),
code('''
cu.seed_everything(42)
model = cu.SmallCNN(num_classes=len(class_names)).to(DEVICE)
history = cu.fit(model, train_loader, val_loader, epochs=EPOCHS, device=DEVICE)
cu.plot_history(history)
test_metrics = cu.evaluate(model, test_loader, DEVICE)
print("Final test metrics:", test_metrics)
'''),
md('''
## Step 3 — Create a versioned model bundle

Save a `state_dict` of tensors rather than a pickled model object. Record architecture and preprocessing
in JSON. We load with `weights_only=True` later. Use only model artifacts from sources you trust.
The bundle name distinguishes smoke checks from demonstration models so a tiny validation run
does not replace the model you trained for class.
'''),
code('''
bundle_dir = cu.ARTIFACTS / ("12_bundle_smoke" if SMOKE else "12_bundle_demo")
bundle_dir.mkdir(parents=True, exist_ok=True)
cpu_model = copy.deepcopy(model).cpu().eval()
torch.save(cpu_model.state_dict(), bundle_dir / "weights.pt")
metadata = {
    "schema_version": 1,
    "architecture": "SmallCNN",
    "in_channels": 1,
    "class_names": class_names,
    "preprocessing": {
        "color_mode": "L", "size_hw": [28, 28], "interpolation": "bilinear",
        "scale": "uint8_to_0_1", "mean": [0.0], "std": [1.0]
    },
    "dataset": "FashionMNIST",
    "training_examples": len(train_ds), "epochs_run": EPOCHS,
    "seed": 42, "smoke_only": SMOKE,
    "torch_version": str(torch.__version__),
    "test_metrics": test_metrics,
    "intended_use": "Educational classification of Fashion-MNIST-like clothing images",
    "limitations": ["Short demo training", "Not validated on ordinary clothing photographs",
                   "Softmax confidence is uncalibrated", "No unknown-object class"]
}
(bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print("Saved bundle:", bundle_dir)
'''),
md('''
## Step 4 — Reload from disk and verify numerical parity

Construct a new model using only the documented architecture and metadata. This catches a
wrong class count, missing state, or different architecture before deployment.
Use a tolerance for floating-point comparisons instead of assuming bit-for-bit equality across devices.
'''),
code('''
from predict_image import load_bundle, predict_image

loaded_model, loaded_metadata, preprocess = load_bundle(bundle_dir, device="cpu")
sample_images, sample_labels = next(iter(test_loader))
with torch.inference_mode():
    before = cpu_model(sample_images)
    after = loaded_model(sample_images)
torch.testing.assert_close(before, after)
print("Reloaded model produces matching logits.")
'''),
md('''
## Step 5 — Predict one actual image file

Save a test image as PNG, then pass its file path through the same public prediction function the
command-line script uses. The example already matches the training distribution; supplying an
unrelated photograph is a distribution-shift experiment, not a supported use case.
'''),
code('''
example_image, example_label = test_ds[0]
example_path = bundle_dir / "example.png"
transforms.ToPILImage()(example_image).save(example_path)
result = predict_image(bundle_dir, example_path, device="cpu", top_k=3)
print(json.dumps(result, indent=2))
plt.imshow(Image.open(example_path), cmap="gray")
plt.title(f"True: {class_names[example_label]} | Predicted: {result['predictions'][0]['class_name']}")
plt.axis("off")
plt.show()
'''),
md('''
## Step 6 — Verify the file preprocessing contract

The saved PNG should reproduce the original tensor within quantization tolerance. This round-trip
test checks the complete input path, including conversion, resizing, and normalization.
'''),
code('''
with Image.open(example_path) as pil_image:
    recovered_tensor = preprocess(pil_image)
torch.testing.assert_close(recovered_tensor, example_image, atol=1/255 + 1e-7, rtol=0)
assert result["predictions"][0]["class_index"] == int(after[0].argmax())
assert 0 <= result["predictions"][0]["probability"] <= 1
print("Image preprocessing and top prediction agree.")
'''),
md('''
## Step 7 — Run the standalone command-line predictor

This starts another Python process, so it cannot accidentally use a model left in the notebook's
memory. The script depends on this course's `course_utils.py`, PyTorch, torchvision, NumPy, and Pillow.
It accepts CPU or CUDA, validates the supported metadata schema, and returns JSON.
'''),
code('''
import subprocess
command = [sys.executable, str(ROOT / "predict_image.py"),
           "--bundle", str(bundle_dir), "--image", str(example_path), "--device", "cpu"]
completed = subprocess.run(command, check=True, capture_output=True, text=True)
cli_result = json.loads(completed.stdout)
assert cli_result["predictions"][0]["class_index"] == result["predictions"][0]["class_index"]
print(completed.stdout)
'''),
md('''
## Step 8 — Export a fixed-shape inference program

`torch.export` captures a model program with input constraints. This teaching artifact accepts
**one float32 grayscale 28×28 image**, shape `[1, 1, 28, 28]`; larger batches are outside this contract.
It still needs a compatible PyTorch runtime. Export is not automatically a speed optimization or a
standalone executable. Other deployment targets may require additional conversion and testing.
'''),
code('''
example_input = sample_images[:1].cpu().contiguous()
exported = torch.export.export(cpu_model, (example_input,))
export_path = bundle_dir / "classifier.pt2"
torch.export.save(exported, export_path)
restored_program = torch.export.load(export_path)
with torch.inference_mode():
    eager_output = cpu_model(example_input)
    exported_output = restored_program.module()(example_input)
torch.testing.assert_close(eager_output, exported_output, rtol=1e-4, atol=1e-5)
print("Export saved and parity checked:", export_path)
'''),
md('''
## Step 9 — Measure model latency honestly

Warm up first, then synchronize CUDA around timing because GPU operations are asynchronous.
Report batch size and whether loading/preprocessing/device transfer are included. This experiment
measures only a model forward pass with the input already on its target device. A small model at
batch size one can be faster on CPU; measure instead of assuming a GPU always wins.
'''),
code('''
def latency_ms(candidate, x, repeats=20):
    candidate.eval()
    timings = []
    with torch.inference_mode():
        for _ in range(5):
            candidate(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        for _ in range(repeats):
            start = time.perf_counter()
            candidate(x)
            if x.is_cuda:
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.percentile(timings, 50)),
            "p95_ms": float(np.percentile(timings, 95)), "batch_size": len(x)}

latencies = {"cpu": latency_ms(cpu_model, example_input)}
if DEVICE.type == "cuda":
    latencies["cuda"] = latency_ms(model, example_input.to(DEVICE))
print(json.dumps(latencies, indent=2))
(bundle_dir / "latency.json").write_text(json.dumps(latencies, indent=2), encoding="utf-8")
'''),
md('''
## Step 10 — Review the model card before reusing the artifact

The metadata contains intended use, class order, training settings, test metrics, and limitations.
For a real project also record dataset provenance and permission, group/time splitting decisions,
per-class results, calibration, failure cases, the input contract, and retraining triggers.
Monitor input quality and class-specific performance after deployment; a high original test score
does not establish performance under future distribution shift.
'''),
code('''
print((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
print("Bundle files:", [p.name for p in sorted(bundle_dir.iterdir())])
'''),
md('''
## Capstone — Build and demonstrate a classifier for your own task

Choose a small task with clearly defined classes, such as three types of household objects.
Start from lesson 07's folder pipeline, then adapt lesson 06's pretrained ResNet18.

| Milestone | Evidence to show in your demonstration |
|---|---|
| Define the task | Class definitions, intended use, representative example images |
| Audit the data | Counts, duplicates, corrupt images, group-aware or time-aware split rationale |
| Establish a baseline | Majority-class score and a simple trained model |
| Improve the model | One controlled validation comparison at a time |
| Diagnose errors | Confusion matrix, minority-class recall, representative mistakes |
| Assess reliability | Calibration split, confidence diagram, plausible corruption tests |
| Package prediction | Weights, label order, preprocessing, standalone image prediction |
| Final report | Once-only held-out test results and concrete limitations |

**Suggested 10-minute demonstration:** explain the problem (1 min), inspect data (2 min), run a
short training segment (2 min), compare results and mistakes (2 min), predict a new image (2 min),
and discuss one limitation (1 min). Keep a previously trained model for presentation continuity.

**Success criteria:** another person can restart the kernel, run every cell, reproduce your pipeline,
and understand what your model can and cannot classify.
'''),
md('''
## Exercises

1. Deliberately change the class order in a copied metadata file. Why can numerical parity still pass
   while the named prediction becomes wrong?
2. Try a batch of two images with the exported program. Explain its shape constraint.
3. Add a rule that flags images of the wrong aspect ratio. Is this enough to detect unknown classes?
4. For a real application, measure end-to-end latency from image-file read to returned JSON.

<details><summary>Suggested answers</summary>

1. Tensor values stay identical, but their class-index interpretation changes. Validate label mapping as part of the bundle.
2. The export example fixes batch size to one. Dynamic shapes need an explicitly designed and tested export contract.
3. No. Input validation catches malformed inputs; unknown-class detection is a separate problem requiring its own evaluation.
4. Start the timer before image loading and stop after prediction formatting; include warm-up policy and hardware in the report.

</details>

**Primary references:** [saving and loading models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html),
[torch.export](https://docs.pytorch.org/docs/stable/export.html),
[benchmarking recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html).
''')])
