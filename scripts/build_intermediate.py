"""Author the independent intermediate image-classification course notebooks."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from authoring import md, code, write_notebook, bootstrap


def transfer_learning():
    cells = [
        md("""
# 06 · Transfer learning: reuse a pretrained vision model

**Level:** intermediate. **Before this:** understand a CNN training loop (notebooks 03–05).
**Practical outcome:** adapt ResNet-18 to ten CIFAR-10 classes, first train its classifier,
then fine-tune its last residual block, and select a model using validation data.

Run each cell in order. Every notebook can start from a fresh kernel. `QUICK=True` uses
small teaching subsets, so results are demonstrations, not benchmark scores. An internet
connection is needed for the first CIFAR-10 and pretrained-weight download. On an 8 GB
GPU, start with batch size 16; reduce it to 8 if another app occupies GPU memory.
"""),
        md("""
## Step 1 · Configure the session

The setup prints the active device. CUDA uses your NVIDIA GPU. CPU fallback works but
224-pixel images take longer. Keep `QUICK=True` for a live demonstration; use `False`
for a larger practice run. `COURSE_SMOKE=1` is a tiny pipeline check, not an accuracy run.
"""),
        code(bootstrap),
        code("""
import copy
import pandas as pd
from torchvision import models, transforms
from torch import nn

cu.seed_everything(42)
BATCH_SIZE = 16
HEAD_EPOCHS = 1 if SMOKE else (2 if QUICK else 5)
FINE_TUNE_EPOCHS = 1 if SMOKE else (2 if QUICK else 5)
limits = (32, 16, 16) if SMOKE else ((800, 200, 300) if QUICK else (8000, 1500, 2000))
print("Dataset limits:", limits, "| Epochs:", HEAD_EPOCHS, "+", FINE_TUNE_EPOCHS)
"""),
        md("""
## Step 2 · Use the preprocessing expected by the weights

ImageNet-pretrained ResNet expects RGB inputs with its training normalization.
The weight object's `transforms()` supplies deterministic validation/inference
preprocessing. Training uses a random crop and horizontal flip with the same mean
and standard deviation. CIFAR-10 starts at 32 × 32; resizing provides the expected
input shape, but cannot recover detail that was never in the original image.
"""),
        code("""
weights = models.ResNet18_Weights.DEFAULT
eval_transform = weights.transforms()
MEAN = list(eval_transform.mean)
STD = list(eval_transform.std)
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
print("Pretrained weights:", weights)
print("Inference preprocessing:", eval_transform)
"""),
        md("""
## Step 3 · Split before comparing models

Training and validation are disjoint subsets of CIFAR-10's training partition.
The official test partition stays out of the model-selection loop. Training gets
random augmentation; validation and test get the deterministic weight preprocessing.
"""),
        code("""
train_ds, val_ds, test_ds, class_names = cu.cifar_data(
    train_transform=train_transform, eval_transform=eval_transform,
    train_limit=limits[0], val_limit=limits[1], test_limit=limits[2], seed=42,
)
train_dl = cu.loader(train_ds, batch_size=BATCH_SIZE, shuffle=True, seed=42)
val_dl = cu.loader(val_ds, batch_size=BATCH_SIZE)
test_dl = cu.loader(test_ds, batch_size=BATCH_SIZE)
print("Train / validation / test:", len(train_ds), len(val_ds), len(test_ds))
cu.show_images(val_ds, class_names, n=8, mean=MEAN, std=STD)
"""),
        md("""
## Step 4 · Inspect the network and replace its classifier

The original classifier produces 1,000 ImageNet logits. Our replacement produces
ten CIFAR-10 logits. A logit is an unnormalized class score; cross-entropy expects
these raw scores. The rest of the network supplies reusable image features.
The first model construction downloads the pretrained weights into PyTorch's cache.
"""),
        code("""
model = models.resnet18(weights=weights)
for parameter in model.parameters():
    parameter.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(DEVICE)

def parameter_counts(network):
    return {
        "total": sum(p.numel() for p in network.parameters()),
        "trainable": sum(p.numel() for p in network.parameters() if p.requires_grad),
    }

print(model.fc)
print(parameter_counts(model))
xb, yb = next(iter(val_dl))
model.eval()
with torch.inference_mode():
    print("Input:", tuple(xb.shape), "→ logits:", tuple(model(xb.to(DEVICE)).shape))
"""),
        md("""
## Step 5 · Freeze behavior as well as gradients

`requires_grad=False` prevents weight gradients; it does **not** stop BatchNorm
running-statistic updates. `model.train()` normally enables those updates. Our loop
holds every BatchNorm layer in evaluation mode during both phases. This is a
deliberate policy for a small dataset; the classifier and selected convolution
weights can still learn. `eval()` does not turn gradient computation off.
"""),
        code("""
def hold_batchnorm_statistics(network):
    for module in network.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

def transfer_epoch(network, dataloader, optimizer):
    network.train()
    hold_batchnorm_statistics(network)
    total_loss, correct, count = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = network(images)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        n = labels.numel()
        total_loss += loss.item() * n
        correct += (logits.argmax(1) == labels).sum().item()
        count += n
    return {"loss": total_loss / count, "accuracy": correct / count}

def run_phase(network, optimizer, epochs, title):
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = transfer_epoch(network, train_dl, optimizer)
        val_metrics = cu.evaluate(network, val_dl, DEVICE)
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"], "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
        })
        print(f"{title} {epoch}/{epochs}: val_loss={val_metrics['loss']:.4f}, "
              f"val_accuracy={val_metrics['accuracy']:.3f}")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
    network.load_state_dict(best_state)
    return history, best_state
"""),
        md("""
## Step 6 · Train only the new head

Only the classifier parameters enter the optimizer. Inspect the trainable parameter
count and BatchNorm check below while explaining why this phase is inexpensive.
We retain the epoch with the lowest validation loss, which may be earlier than the
last epoch. Do not choose settings from test accuracy.
"""),
        code("""
head_optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
bn_before = model.bn1.running_mean.detach().clone()
head_history, head_state = run_phase(model, head_optimizer, HEAD_EPOCHS, "Frozen backbone")
assert torch.equal(bn_before, model.bn1.running_mean), "Frozen BatchNorm statistics changed."
head_validation = cu.evaluate(model, val_dl, DEVICE)
cu.plot_history(head_history)
"""),
        md("""
## Step 7 · Fine-tune the last residual block

Continue from the best head-only state. Unfreeze `layer4`, keep BatchNorm affine
parameters and running statistics frozen, and use a smaller learning rate for the
pretrained convolutions. The new classifier gets a somewhat larger rate. Recreate
the optimizer so the newly trainable parameters are included.
"""),
        code("""
for parameter in model.layer4.parameters():
    parameter.requires_grad = True
for module in model.modules():
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        for parameter in module.parameters():
            parameter.requires_grad = False

fine_optimizer = torch.optim.AdamW([
    {"params": [p for p in model.layer4.parameters() if p.requires_grad], "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 1e-4},
], weight_decay=1e-4)
print(parameter_counts(model))
fine_history, fine_state = run_phase(model, fine_optimizer, FINE_TUNE_EPOCHS, "Fine-tune layer4")
fine_validation = cu.evaluate(model, val_dl, DEVICE)
cu.plot_history(fine_history)
"""),
        md("""
## Step 8 · Compare on validation and select once

Fine-tuning does not guarantee improvement, especially with a very small training
subset. The rule below selects the lower validation loss, using the frozen-backbone
model on a tie. This is a sequential experiment: fine-tuning starts from phase one's
best state, rather than being an independent from-scratch trial.
"""),
        code("""
comparison = pd.DataFrame([
    {"phase": "head_only", **head_validation},
    {"phase": "fine_tuned", **fine_validation},
]).set_index("phase")
display(comparison)
selected_phase = "fine_tuned" if fine_validation["loss"] < head_validation["loss"] else "head_only"
model.load_state_dict(fine_state if selected_phase == "fine_tuned" else head_state)
print("Selected using validation loss:", selected_phase)
comparison["accuracy"].plot.bar(ylim=(0, 1), title="Validation accuracy (selection uses loss)")
plt.ylabel("Accuracy")
plt.xticks(rotation=0)
plt.show()
"""),
        md("""
## Step 9 · Open the held-out test set

The experiment is now fixed. Evaluate the chosen model once, then inspect several
predictions. Softmax scores rank the model's classes; they are not a guarantee of
calibrated real-world confidence. Repeatedly tuning after viewing this result would
turn the test set into another validation set.
"""),
        code("""
test_metrics = cu.evaluate(model, test_dl, DEVICE)
print({"selected_phase": selected_phase, "test": test_metrics})
images, labels = next(iter(test_dl))
model.eval()
with torch.inference_mode():
    probabilities = model(images.to(DEVICE)).softmax(1).cpu()
fig, axes = plt.subplots(1, min(6, len(images)), figsize=(15, 3))
for i, ax in enumerate(np.atleast_1d(axes)):
    rgb = images[i] * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
    predicted = probabilities[i].argmax().item()
    ax.imshow(rgb.permute(1, 2, 0).clamp(0, 1))
    ax.set_title(f"True: {class_names[labels[i].item()]}\\nPred: {class_names[predicted]}", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()
"""),
        md("""
## Step 10 · Save the model together with its meaning

Weights alone do not record the class order or input transformation. Save these
alongside the architecture and selection information. The reload example constructs
the network without downloading pretrained weights again: our saved state already
contains all its parameters and buffers.
"""),
        code("""
checkpoint_path = cu.ARTIFACTS / "06_resnet18_cifar10.pt"
torch.save({
    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    "architecture": "resnet18", "class_names": list(class_names),
    "mean": MEAN, "std": STD, "resize": 256, "center_crop": 224,
    "selected_phase": selected_phase,
}, checkpoint_path)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
restored = models.resnet18(weights=None)
restored.fc = nn.Linear(restored.fc.in_features, len(checkpoint["class_names"]))
restored.load_state_dict(checkpoint["model_state"])
restored = restored.to(DEVICE).eval()
with torch.inference_mode():
    original_logits = model.eval()(images[:2].to(DEVICE))
    restored_logits = restored(images[:2].to(DEVICE))
torch.testing.assert_close(original_logits, restored_logits)
print("Saved and verified:", checkpoint_path)
"""),
        md("""
## Step 11 · Demonstrate an actual prediction on your own image

Set `IMAGE_PATH` to a local image to try this cell. Predictions are restricted to the
ten CIFAR-10 categories; a portrait or document still receives one of those labels.
Use the same deterministic preprocessing used for validation. The `None` default
keeps Run All independent of your personal files.
"""),
        code("""
from PIL import Image

IMAGE_PATH = None  # Example: Path(r"C:\\Users\\YourName\\Pictures\\dog.jpg")
if IMAGE_PATH is None:
    print("Optional: set IMAGE_PATH to classify a local image.")
else:
    with Image.open(IMAGE_PATH) as source:
        rgb_image = source.convert("RGB")
    with torch.inference_mode():
        scores = restored(eval_transform(rgb_image).unsqueeze(0).to(DEVICE)).softmax(1)[0].cpu()
    values, indices = scores.topk(3)
    display(pd.DataFrame({
        "class": [class_names[i] for i in indices.tolist()],
        "softmax_score": values.tolist(),
    }))
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.show()
"""),
        md("""
## Step 12 · Pause, explain, and experiment

1. Why can BatchNorm change when all its parameters have `requires_grad=False`?
2. Print the names of the trainable parameters in each phase.
3. In a **new validation experiment**, try fine-tuning `layer3` too. Compare validation
   loss and training time. Keep the existing test result out of this decision.
4. Explain what information belongs in an inference artifact besides weights.

<details><summary>Hints / solutions</summary>

BatchNorm's running mean and variance are buffers, not gradient-updated parameters.
Use evaluation mode to hold those statistics fixed. For inspection:

```python
[name for name, p in model.named_parameters() if p.requires_grad]
```

To include `layer3`, unfreeze its convolution weights, keep BatchNorm frozen, and add
those parameters to a **new** optimizer with a small learning rate. Save architecture,
class order, normalization, resize/crop policy, and the selected trained state.
Choose final settings on validation, then use a fresh held-out set for a new final claim.

</details>

**References:** [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html),
[ResNet-18 weights and preprocessing](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html),
[BatchNorm behavior](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html).
"""),
    ]
    write_notebook("06_transfer_learning.ipynb", cells)


def custom_image_dataset():
    cells = [
        md("""
# 07 · Classify images from your own folders

**Level:** intermediate. **Before this:** CNNs, augmentation, and validation.
**Practical outcome:** build an `ImageFolder` pipeline with class checks, stratified
splits, duplicate-content leakage checks, a trained classifier, and a reusable artifact.

This notebook runs without your personal dataset. Its default data is an explicitly
**synthetic** set of colored circles, squares, and triangles generated locally with
Pillow. It teaches the workflow; synthetic accuracy does not establish performance
on photographs. Point `DATASET_ROOT` at your own folders when you are ready.
"""),
        md("""
## Step 1 · Configure the session

Run from a fresh kernel. The default is a short demonstration. All generated images
stay inside `data/generated_shapes_v1`; your own files are only read. On Windows,
the course loader uses zero worker processes to avoid notebook multiprocessing issues.
"""),
        code(bootstrap),
        code("""
import hashlib
import json
from collections import defaultdict
from PIL import Image, ImageDraw
from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import pandas as pd

cu.seed_everything(42)
EPOCHS = 1 if SMOKE else (3 if QUICK else 10)
BATCH_SIZE = 32
GENERATED_ROOT = cu.DATA_DIR / "generated_shapes_v1"
DATASET_ROOT = GENERATED_ROOT  # Change to Path(r"D:\\my_images") for your own classes.
"""),
        md("""
## Step 2 · Understand the required folder layout

`ImageFolder` derives labels from class directory names. Put original images in
folders like this; this notebook creates index-based splits without moving files:

```text
my_images/
  cat/    photo_001.jpg, photo_002.jpg, ...
  dog/    photo_101.jpg, photo_102.jpg, ...
  horse/  photo_201.jpg, photo_202.jpg, ...
```

Use at least ten distinct images per class for this tiny teaching split, and many
more for a useful real classifier. Do not mix an existing `train/val/test` directory
tree into this layout. If images belong to the same patient, animal, product, video,
or capture session, split by that group **before** augmentation; random image splits
can leak related images even when file content differs.

For the default shapes, every class uses the same background-color range and the
same foreground-color range; colors vary per image rather than encoding the class.
The generator is idempotent: it skips existing filenames. Editing its seed or colors
and rerunning does **not** modify images already on disk. To study a changed background,
use the separate, in-memory shift dataset in the final exercise and keep the trained
model fixed. A newly generated source dataset would need an unused versioned folder.
"""),
        code("""
def generate_shapes(root, per_class=160, seed=2026):
    # This helper may write only to the dedicated generated-data directory.
    root = Path(root).resolve()
    assert root == GENERATED_ROOT.resolve(), "Generator is restricted to its dedicated folder."
    names = ["circle", "square", "triangle"]
    created = 0
    for class_id, name in enumerate(names):
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            path = folder / f"shape_{i:04d}.png"
            if path.exists():
                continue  # Idempotent: never overwrite an existing image.
            rng = np.random.default_rng(seed + class_id * 10000 + i)
            background = tuple(int(v) for v in rng.integers(215, 256, size=3))
            foreground = tuple(int(v) for v in rng.integers(20, 175, size=3))
            image = Image.new("RGB", (64, 64), background)
            draw = ImageDraw.Draw(image)
            size = int(rng.integers(22, 43))
            x, y = [int(v) for v in rng.integers(4, 60 - size, size=2)]
            box = (x, y, x + size, y + size)
            if name == "circle":
                draw.ellipse(box, fill=foreground)
            elif name == "square":
                draw.rectangle(box, fill=foreground)
            else:
                draw.polygon([(x + size // 2, y), (x, y + size), (x + size, y + size)],
                             fill=foreground)
            image.save(path)
            created += 1
    return created

if Path(DATASET_ROOT).resolve() == GENERATED_ROOT.resolve():
    print("New synthetic images:", generate_shapes(GENERATED_ROOT))
else:
    print("Using your image folders; no images will be generated here.")
assert Path(DATASET_ROOT).is_dir(), f"Dataset folder does not exist: {DATASET_ROOT}"
print("Dataset root:", Path(DATASET_ROOT).resolve())
"""),
        md("""
## Step 3 · Inspect the label map and class balance

Label IDs follow folder-name ordering. Never assume that class `0` means your first
handwritten label. Save and reuse `class_to_idx` at inference. Inspect the counts to
spot missing classes or strong imbalance before training.
"""),
        code("""
catalog = datasets.ImageFolder(DATASET_ROOT)
class_names = catalog.classes
labels = np.asarray(catalog.targets)
counts = pd.Series(labels).value_counts().sort_index()
assert len(class_names) >= 2, "Classification needs at least two class folders."
assert counts.min() >= 10, "Use at least 10 distinct images per class for this demonstration."
print("Class map:", catalog.class_to_idx)
count_table = pd.DataFrame({"class": class_names, "images": counts.values})
display(count_table)
count_table.plot.bar(x="class", y="images", legend=False, title="Images per class")
plt.ylabel("Count")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
"""),
        md("""
## Step 4 · Validate files and fingerprint decoded pixels

An image extension does not guarantee that the image is readable. Open every image
once and hash its RGB pixels plus dimensions. This catches identical decoded images
even if filenames or metadata differ. It does **not** catch resized, recompressed,
cropped, or nearly identical images; group-aware review is still needed for those.
Large real datasets should run this audit once and cache the resulting manifest.
"""),
        code("""
def pixel_fingerprint(path):
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        rgb.load()  # Fully decode now, so corrupt files fail during the audit.
        digest = hashlib.sha256()
        digest.update(str(rgb.size).encode("utf-8"))
        digest.update(rgb.tobytes())
        return digest.hexdigest()

fingerprints = []
broken = []
for path, target in catalog.samples:
    try:
        fingerprints.append(pixel_fingerprint(path))
    except (OSError, ValueError) as error:
        broken.append((path, str(error)))
        fingerprints.append(None)
if broken:
    display(pd.DataFrame(broken, columns=["path", "error"]).head(20))
    raise ValueError("Fix or remove the unreadable images from the dataset, then rerun.")

duplicate_groups = defaultdict(list)
for index, fingerprint in enumerate(fingerprints):
    duplicate_groups[fingerprint].append(index)
duplicate_groups = {key: value for key, value in duplicate_groups.items() if len(value) > 1}
print("Readable images:", len(catalog), "| Duplicate-content groups:", len(duplicate_groups))
conflicting = [ids for ids in duplicate_groups.values() if len(set(labels[ids])) > 1]
if conflicting:
    raise ValueError("Identical image content has conflicting labels. Review these groups: "
                     + str(conflicting[:5]))
"""),
        md("""
## Step 5 · Create reproducible, stratified index splits

Reserve 20% for test and then 20% of the remainder for validation: approximately
64% training, 16% validation, and 20% test. Stratification preserves class coverage.
The file order and split seed form a reproducibility contract. Duplicate images
across splits trigger a clear failure in the next step; remove duplicates or keep
each duplicate/group together before using a real dataset.
"""),
        code("""
all_indices = np.arange(len(catalog))
try:
    train_val_idx, test_idx = train_test_split(
        all_indices, test_size=0.20, random_state=42, stratify=labels,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.20, random_state=42, stratify=labels[train_val_idx],
    )
except ValueError as error:
    raise ValueError("The split needs more images per class. Inspect the class counts.") from error

splits = {"train": train_idx, "validation": val_idx, "test": test_idx}
for name, indices in splits.items():
    assert set(labels[indices]) == set(range(len(class_names))), f"Missing class in {name}."
assert not (set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx))
display(pd.DataFrame({
    name: np.bincount(labels[indices], minlength=len(class_names))
    for name, indices in splits.items()
}, index=class_names))
"""),
        md("""
## Step 6 · Verify that content does not cross a split boundary

Index disjointness alone is insufficient: two different paths can contain the same
picture. Compare content fingerprints for every split pair. This check runs on the
full selected folder before any smoke-test reduction, so a tiny run cannot hide a
duplicate elsewhere in the dataset.
"""),
        code("""
split_hashes = {name: {fingerprints[int(i)] for i in indices} for name, indices in splits.items()}
leakage = {}
for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
    shared = split_hashes[left] & split_hashes[right]
    if shared:
        leakage[f"{left} ↔ {right}"] = len(shared)
if leakage:
    raise ValueError(f"Duplicate image content crosses splits: {leakage}. "
                     "Deduplicate or split by content/group before training.")
print("Passed: split indices are disjoint and no exact decoded image crosses splits.")

# Keep all classes represented when reducing to a tiny pipeline check.
if SMOKE:
    splits = {
        name: np.concatenate([indices[labels[indices] == label][:4]
                              for label in range(len(class_names))])
        for name, indices in splits.items()
    }
train_idx, val_idx, test_idx = (splits[name] for name in ["train", "validation", "test"])
"""),
        md("""
## Step 7 · Keep train and evaluation transformations separate

Each `Subset` points to its own `ImageFolder` view. Sharing one mutable dataset and
then changing `.transform` would also change the validation transform. Here training
gets a random flip; evaluation is deterministic. Review whether a flip preserves
your real labels before using it for letters, medical images, or directional objects.
"""),
        code("""
MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
eval_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])
train_view = datasets.ImageFolder(DATASET_ROOT, transform=train_transform)
eval_view = datasets.ImageFolder(DATASET_ROOT, transform=eval_transform)
assert train_view.class_to_idx == eval_view.class_to_idx == catalog.class_to_idx
assert train_view.samples == eval_view.samples == catalog.samples
train_ds = Subset(train_view, train_idx.tolist())
val_ds = Subset(eval_view, val_idx.tolist())
test_ds = Subset(eval_view, test_idx.tolist())
train_dl = cu.loader(train_ds, batch_size=BATCH_SIZE, shuffle=True, seed=42)
val_dl = cu.loader(val_ds, batch_size=BATCH_SIZE)
test_dl = cu.loader(test_ds, batch_size=BATCH_SIZE)
print("Train / validation / test:", len(train_ds), len(val_ds), len(test_ds))
cu.show_images(val_ds, class_names, n=9, mean=MEAN, std=STD)
"""),
        md("""
## Step 8 · Save a split manifest

An index alone stops being meaningful if files are added or reordered. A manifest
records relative paths, labels, and fingerprints for this run. It also makes the
demonstration auditable. The manifest is saved under `artifacts`, not in your dataset.
This cell replaces the notebook's previous manifest when you rerun it.
"""),
        code("""
dataset_root_resolved = Path(DATASET_ROOT).resolve()
manifest = {
    "seed": 42,
    "dataset_root": str(dataset_root_resolved),
    "class_to_idx": catalog.class_to_idx,
    "splits": {
        name: [{
            "path": str(Path(catalog.samples[int(i)][0]).resolve().relative_to(dataset_root_resolved)),
            "label": int(labels[i]), "pixel_sha256": fingerprints[int(i)],
        } for i in indices]
        for name, indices in splits.items()
    },
}
manifest_path = cu.ARTIFACTS / "07_split_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("Saved:", manifest_path)
"""),
        md("""
## Step 9 · Train a classifier matched to your class count

This compact CNN learns from scratch and accepts three-channel images. The final
layer has one output per folder. `fit` restores the epoch with the best validation
loss. Tiny synthetic datasets may produce noisy curves; use them to explain the
training mechanics, not to promise real-data accuracy.
"""),
        code("""
cu.seed_everything(42)
model = cu.SmallCNN(num_classes=len(class_names), in_channels=3).to(DEVICE)
history = cu.fit(model, train_dl, val_dl, epochs=EPOCHS, device=DEVICE, lr=1e-3)
cu.plot_history(history)
print("Selected-model validation:", cu.evaluate(model, val_dl, DEVICE))
"""),
        md("""
## Step 10 · Evaluate once and inspect per-class errors

After fixing the training procedure, evaluate on test data. A confusion matrix shows
which class pairs are confused; overall accuracy alone can hide this. A class may
receive no predictions in a tiny run, so the report uses `zero_division=0` explicitly.
"""),
        code("""
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

test_metrics = cu.evaluate(model, test_dl, DEVICE)
test_logits, test_labels = cu.predict(model, test_dl, DEVICE)
test_predictions = test_logits.argmax(1)
print("Held-out test:", test_metrics)
print(classification_report(test_labels.numpy(), test_predictions.numpy(),
                            labels=list(range(len(class_names))),
                            target_names=class_names, zero_division=0))
ConfusionMatrixDisplay.from_predictions(
    test_labels.numpy(), test_predictions.numpy(), labels=list(range(len(class_names))),
    display_labels=class_names, cmap="Blues", xticks_rotation=30,
)
plt.title("Held-out test confusion matrix")
plt.tight_layout()
plt.show()
"""),
        md("""
## Step 11 · Package and reload a prediction

Save class names and the class-to-index map in the same artifact as the state dict.
Reload on a newly constructed model and classify one image using the deterministic
transform. Rebuilding a label map from a different folder at inference can silently
assign the wrong human-readable name to a correct numeric prediction.
"""),
        code("""
model_path = cu.ARTIFACTS / "07_custom_classifier.pt"
torch.save({
    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    "architecture": "SmallCNN", "in_channels": 3,
    "class_names": list(class_names), "class_to_idx": catalog.class_to_idx,
    "image_size": [64, 64], "mean": list(MEAN), "std": list(STD),
}, model_path)
saved = torch.load(model_path, map_location="cpu", weights_only=True)
loaded = cu.SmallCNN(num_classes=len(saved["class_names"]), in_channels=3).to(DEVICE)
loaded.load_state_dict(saved["model_state"])
loaded.eval()

image_path = Path(catalog.samples[int(test_idx[0])][0])
with Image.open(image_path) as source:
    image = source.convert("RGB")
with torch.inference_mode():
    probability = loaded(eval_transform(image).unsqueeze(0).to(DEVICE)).softmax(1)[0].cpu()
predicted_id = probability.argmax().item()
print("Image:", image_path.name)
print("Prediction:", saved["class_names"][predicted_id], "| softmax score:", float(probability.max()))
plt.imshow(image)
plt.axis("off")
plt.show()
print("Saved:", model_path)
"""),
        md("""
## Step 12 · Demonstration challenges

1. For the default synthetic dataset, create a **separate shifted test view** that
   recolors background pixels, using the example below. Keep the trained model fixed
   and select the new color before measuring. Does original validation performance
   predict performance on these shifted images? Editing `generate_shapes` and rerunning
   will not change existing files because the generator skips existing filenames.
2. In a copy of your dataset, duplicate an image into a different class. Which audit
   stops the run, and why should it stop before training?
3. Adapt this pipeline to your own two-class collection. Inspect every class mapping,
   remove duplicate images, and record any patient/session/product grouping.
4. Replace this CNN with notebook 06's pretrained model. Use its preprocessing and
   compare **validation** results before opening a new final test set.

<details><summary>Hints / solutions</summary>

Background changes create a distribution shift. This wrapper creates shifted examples
in memory and leaves both the original dataset and model unchanged. It applies only
to our synthetic shapes: their top-left pixel is always background, and background
and foreground color ranges do not overlap. It is not a segmentation method for
photographs. The tensors use this notebook's `Normalize(0.5, 0.5)` preprocessing.

```python
class BackgroundShift(torch.utils.data.Dataset):
    def __init__(self, base_dataset, rgb=(0.25, 0.35, 0.45)):
        self.base_dataset = base_dataset
        self.new_color = 2 * torch.tensor(rgb, dtype=torch.float32) - 1

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        background = image[:, 0, 0]
        mask = (image - background[:, None, None]).abs().amax(dim=0) < 1e-6
        shifted = image.clone()
        shifted[:, mask] = self.new_color[:, None]
        return shifted, label

assert Path(DATASET_ROOT).resolve() == GENERATED_ROOT.resolve()
shifted_test_ds = BackgroundShift(test_ds)
cu.show_images(shifted_test_ds, class_names, n=9, mean=MEAN, std=STD)
print(cu.evaluate(model, cu.loader(shifted_test_ds), DEVICE))
```

Predeclare this study's color; choosing the color or retraining after inspecting its
test outcomes needs a fresh final evaluation set. The paired shifted examples measure
this specific intervention, not performance on independently collected photographs.
Identical pixels with different labels trigger the
conflicting-label audit. Exact content shared across splits triggers the leakage
audit. For grouped data, use group-aware splits (for example `GroupShuffleSplit`)
and then verify class coverage. A class count is not enough to detect group leakage.

```python
# Inspect the saved labels before deploying:
print(saved["class_to_idx"])
assert saved["class_names"][0] == min(saved["class_to_idx"], key=saved["class_to_idx"].get)
```

</details>

**References:** [torchvision ImageFolder](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html),
[PyTorch datasets and loaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html),
[scikit-learn stratified splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html),
[group-aware splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html).
"""),
    ]
    write_notebook("07_custom_image_dataset.ipynb", cells)


def training_optimization():
    cells = [
        md("""
# 08 · Train efficiently, compare fairly, and resume safely

**Level:** advanced intermediate. **Before this:** implement and explain a training loop.
**Practical outcome:** run a small validation-only learning-rate comparison, then train
with mixed precision, gradient accumulation, clipping, scheduling, early stopping,
and complete resumable checkpoints.

The dataset is Fashion-MNIST and the model is a compact CNN. Short runs make each
mechanism visible; they do not establish optimal hyperparameters. Every component is
shown in notebook cells so you can pause, inspect state, and explain what changes.
An 8 GB NVIDIA GPU is ample for these defaults; CPU fallback disables mixed precision.
"""),
        md("""
## Step 1 · Choose a small, repeatable experiment

Use a fresh kernel and run sequentially. Seeds reduce accidental differences, but
do not promise bitwise-identical results across PyTorch versions or hardware. The
tiny smoke mode tests the pipeline and performs a checkpoint reload without an extra
resumed training epoch. Normal runs perform that extra epoch when patience allows.
"""),
        code(bootstrap),
        code("""
import copy
import random
import time
from itertools import islice
import pandas as pd
from torch import nn

SEED = 42
TUNING_EPOCHS = 1 if SMOKE else (2 if QUICK else 4)
INITIAL_EPOCHS = 1 if SMOKE else (3 if QUICK else 8)
RESUME_EXTRA_EPOCHS = 0 if SMOKE else 1
PATIENCE = 3
MICRO_BATCH_SIZE = 16
ACCUMULATION_STEPS = 4
limits = (64, 32, 32) if SMOKE else ((1600, 400, 400) if QUICK else (10000, 2000, 2000))
train_ds, val_ds, test_ds, class_names = cu.fashion_data(
    train_limit=limits[0], val_limit=limits[1], test_limit=limits[2], seed=SEED,
)
val_dl = cu.loader(val_ds, batch_size=64)
test_dl = cu.loader(test_ds, batch_size=64)
print("Train / validation / test:", len(train_ds), len(val_ds), len(test_ds))
print("Nominal effective batch:", MICRO_BATCH_SIZE * ACCUMULATION_STEPS)
"""),
        md("""
## Step 2 · Compare two choices using validation only

Keep architecture, split, initialization seed, batch size, and epoch budget fixed.
Rebuild both the model and shuffled loader for each candidate. The table records the
validation result of each run's best epoch, because `fit` restores that state.
This is a small teaching search, not evidence that one learning rate is universally
better. A serious study repeats multiple seeds and reports variability.
"""),
        code("""
candidates = [
    {"lr": 1e-3, "weight_decay": 1e-4},
    {"lr": 3e-3, "weight_decay": 1e-4},
]
trial_rows, trial_histories = [], {}
for trial_id, config in enumerate(candidates):
    cu.seed_everything(SEED)
    candidate_model = cu.SmallCNN(num_classes=len(class_names), in_channels=1).to(DEVICE)
    candidate_train_dl = cu.loader(train_ds, batch_size=64, shuffle=True, seed=SEED)
    history = cu.fit(candidate_model, candidate_train_dl, val_dl,
                     epochs=TUNING_EPOCHS, device=DEVICE, **config)
    metrics = cu.evaluate(candidate_model, val_dl, DEVICE)
    trial_rows.append({"trial": trial_id, **config,
                       "val_loss": metrics["loss"], "val_accuracy": metrics["accuracy"]})
    trial_histories[trial_id] = history
    del candidate_model

trial_results = pd.DataFrame(trial_rows).sort_values(["val_loss", "trial"])
display(trial_results)
chosen_id = int(trial_results.iloc[0]["trial"])
chosen_config = dict(candidates[chosen_id])
print("Selected using validation loss:", chosen_config)
"""),
        md("""
## Step 3 · Read the curves before adding machinery

Plot validation loss for both learning rates. A lower final loss is not the same as
the best epoch's loss. Large oscillations, divergence, and widening train-validation
gaps answer different questions. If you alter settings after inspecting the curves,
record the new validation experiment and leave the test set untouched.
"""),
        code("""
for trial_id, history in trial_histories.items():
    plt.plot([row["epoch"] for row in history], [row["val_loss"] for row in history],
             marker="o", label=f"lr={candidates[trial_id]['lr']}")
plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.title("Same initialization, split, and budget")
plt.legend()
plt.grid(alpha=0.2)
plt.show()
"""),
        md("""
## Step 4 · Understand the optimization recipe

| Mechanism | What it changes | What to watch |
|---|---|---|
| CUDA autocast + gradient scaling | Some operations use lower precision | Numerical stability and actual measured speed |
| Gradient accumulation | Several small forward/backward passes make one update | Correct normalization of the final partial window |
| Gradient clipping | Limits the unscaled gradient norm | Clip only after accumulation and unscaling |
| Learning-rate scheduler | Changes the rate between epochs | Save and resume its state |
| Early stopping | Stops after repeated validation non-improvement | Restore the best checkpoint for final inference |

We start a **new** run with the selected learning rate. Accumulation and mixed
precision change the training procedure, so do not present this run as a controlled
speed-only comparison with Step 2. BatchNorm and stochastic layers can also make
microbatch training differ from one large physical batch.
"""),
        code("""
AMP_ENABLED = DEVICE.type == "cuda"
run_config = {
    **chosen_config, "micro_batch_size": MICRO_BATCH_SIZE,
    "accumulation_steps": ACCUMULATION_STEPS,
    "amp_enabled": AMP_ENABLED, "seed": SEED, "patience": PATIENCE,
    "max_grad_norm": 1.0,
}

def make_components(config):
    cu.seed_everything(config["seed"])
    network = cu.SmallCNN(num_classes=len(class_names), in_channels=1).to(DEVICE)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    scaler = torch.amp.GradScaler("cuda", enabled=config["amp_enabled"])
    return network, optimizer, scheduler, scaler

model, optimizer, scheduler, scaler = make_components(run_config)
print("CUDA mixed precision:", AMP_ENABLED)
print("Starting learning rate:", optimizer.param_groups[0]["lr"])
"""),
        md("""
## Step 5 · Accumulate a sample-weighted effective batch

A final accumulation window may have fewer microbatches, and its last batch may have
fewer samples. Dividing every mean loss by a fixed accumulation count underweights
that window. For a window of `N` samples, each microbatch mean gets weight `n / N`.
The accumulated gradient is then the gradient of the mean loss across that window.

Only one microbatch is moved to the GPU at a time. Keep gradients scaled until the
whole window is complete, then unscale once, clip, step, update the scaler, and clear
gradients. Sample-weighted normalization does not make BatchNorm or dropout exactly
equivalent to processing a large physical batch.
"""),
        code("""
def optimized_epoch(network, dataloader, optimizer, scaler, config):
    network.train()
    total_loss, correct, count, updates = 0.0, 0, 0, 0
    iterator = iter(dataloader)
    while True:
        # These are CPU batches; GPU activations exist for just one at a time.
        window = list(islice(iterator, config["accumulation_steps"]))
        if not window:
            break
        window_samples = sum(labels.numel() for _, labels in window)
        optimizer.zero_grad(set_to_none=True)
        for images, labels in window:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            n = labels.numel()
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16,
                                enabled=config["amp_enabled"]):
                logits = network(images)
                mean_loss = nn.functional.cross_entropy(logits, labels)
                contribution = mean_loss * (n / window_samples)
            scaler.scale(contribution).backward()
            total_loss += mean_loss.detach().item() * n
            correct += (logits.detach().argmax(1) == labels).sum().item()
            count += n

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(network.parameters(), config["max_grad_norm"])
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # A reduced scale indicates overflow and an optimizer update that was skipped.
        if scaler.get_scale() >= old_scale:
            updates += 1
        optimizer.zero_grad(set_to_none=True)
    return {"loss": total_loss / count, "accuracy": correct / count, "updates": updates}
"""),
        md("""
## Step 6 · Check the partial-window arithmetic directly

This tiny linear-model check compares a physical batch of eleven examples with
microbatches of sizes 4, 4, and 3. It isolates the normalization rule without dropout,
BatchNorm, or mixed-precision rounding. Both gradients should agree within tolerance.
This is a mechanics check, not a model-accuracy test.
"""),
        code("""
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(123)
    reference = nn.Linear(5, 3)
    accumulated = copy.deepcopy(reference)
    x_demo, y_demo = torch.randn(11, 5), torch.randint(0, 3, (11,))
    nn.functional.cross_entropy(reference(x_demo), y_demo).backward()
    for start, end in [(0, 4), (4, 8), (8, 11)]:
        mean_loss = nn.functional.cross_entropy(accumulated(x_demo[start:end]), y_demo[start:end])
        (mean_loss * ((end - start) / len(y_demo))).backward()
    for full_parameter, micro_parameter in zip(reference.parameters(), accumulated.parameters()):
        torch.testing.assert_close(full_parameter.grad, micro_parameter.grad, rtol=1e-5, atol=1e-7)
print("Passed: partial-window gradients match a physical batch for this deterministic model.")
"""),
        md("""
## Step 7 · Capture the states needed to resume

Inference needs model weights and preprocessing. Training continuation also needs
optimizer momentum, scheduler position, AMP scale, epoch, stopping counters, history,
and random-generator states. The helper stores NumPy's RNG array as a tensor so the
checkpoint can be reloaded with `weights_only=True`. This notebook saves only its
own checkpoints and uses no arbitrary-object unpickling.

We create the shuffled loader from `seed + epoch` for each epoch. Thus epoch-boundary
resume recovers its sample order without storing a partially consumed iterator.
Mid-epoch resume and arbitrary worker-prefetch state are outside this demonstration.
"""),
        code("""
LAST_PATH = cu.ARTIFACTS / "08_last_checkpoint.pt"
BEST_PATH = cu.ARTIFACTS / "08_best_checkpoint.pt"

def capture_rng():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": torch.tensor(numpy_state[1].astype(np.int64)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]), "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if DEVICE.type == "cuda" else [],
    }

def restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state["name"], numpy_state["keys"].cpu().numpy().astype(np.uint32),
                         numpy_state["position"], numpy_state["has_gauss"], numpy_state["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if DEVICE.type == "cuda" and state["cuda"]:
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda"]])

def checkpoint_payload(network, optimizer, scheduler, scaler, training_state, config):
    return {
        "model_state": network.state_dict(), "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
        "training_state": training_state, "config": config, "rng": capture_rng(),
        "class_names": list(class_names), "architecture": "SmallCNN", "in_channels": 1,
        "preprocessing": "FashionMNIST grayscale 28x28, ToTensor(), no normalization",
    }
"""),
        md("""
## Step 8 · Schedule, validate, save, and stop

Validation uses full precision and no gradients. Step the scheduler after an epoch
with an actual optimizer update; save its updated state. `last` is for continuing
training, while `best` is for inference. The best checkpoint may precede the most
recent checkpoint. Early stopping tracks **validation loss**, not training loss.

CUDA kernels are asynchronous, so synchronize before starting and stopping the
wall-clock timer. The displayed epoch time includes training and validation but
excludes checkpoint writing; state that scope when reporting a measurement.
"""),
        code("""
def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

def train_until(network, optimizer, scheduler, scaler, state, config, stop_epoch):
    for epoch in range(state["next_epoch"], stop_epoch):
        if state["bad_epochs"] >= config["patience"]:
            print("Patience already exhausted; keeping the best validation checkpoint.")
            break
        train_dl = cu.loader(train_ds, batch_size=config["micro_batch_size"],
                             shuffle=True, seed=config["seed"] + epoch)
        current_lr = optimizer.param_groups[0]["lr"]
        synchronize()
        start = time.perf_counter()
        train_metrics = optimized_epoch(network, train_dl, optimizer, scaler, config)
        val_metrics = cu.evaluate(network, val_dl, DEVICE)
        synchronize()
        elapsed = time.perf_counter() - start
        if train_metrics["updates"]:
            scheduler.step()

        improved = val_metrics["loss"] < state["best_loss"]
        if improved:
            state["best_loss"] = val_metrics["loss"]
            state["bad_epochs"] = 0
        else:
            state["bad_epochs"] += 1
        state["next_epoch"] = epoch + 1
        state["history"].append({
            "epoch": epoch + 1, "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"], "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"], "lr": current_lr,
            "seconds": elapsed, "optimizer_updates": train_metrics["updates"],
        })
        payload = checkpoint_payload(network, optimizer, scheduler, scaler, state, config)
        torch.save(payload, LAST_PATH)
        if improved:
            torch.save(payload, BEST_PATH)
        print(f"Epoch {epoch + 1}: val_loss={val_metrics['loss']:.4f}, "
              f"lr={current_lr:.2g}, seconds={elapsed:.2f}, "
              f"bad_epochs={state['bad_epochs']}")
        if state["bad_epochs"] >= config["patience"]:
            print("Early stopping triggered.")
            break
    return state

state = {"next_epoch": 0, "best_loss": float("inf"), "bad_epochs": 0, "history": []}
state = train_until(model, optimizer, scheduler, scaler, state, run_config, INITIAL_EPOCHS)
print("Last checkpoint:", LAST_PATH)
print("Best checkpoint:", BEST_PATH)
"""),
        md("""
## Step 9 · Simulate an interruption and restore all components

Build new objects, load their saved states, and restore random state **after** object
construction (initialization consumes random numbers). For this teaching run, resume
on the same hardware and with unchanged data/configuration. Moving between CPU and
CUDA requires a deliberate AMP policy change and is not an exact continuation.
The assertions verify that reload reproduces the saved weights and scheduling state.
"""),
        code("""
checkpoint = torch.load(LAST_PATH, map_location=DEVICE, weights_only=True)
resume_config = checkpoint["config"]
assert resume_config["amp_enabled"] == (DEVICE.type == "cuda"), "Resume on the same device type."
resumed_model, resumed_optimizer, resumed_scheduler, resumed_scaler = make_components(resume_config)
resumed_model.load_state_dict(checkpoint["model_state"])
resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
resumed_scheduler.load_state_dict(checkpoint["scheduler_state"])
resumed_scaler.load_state_dict(checkpoint["scaler_state"])
resumed_state = checkpoint["training_state"]
restore_rng(checkpoint["rng"])

for name, tensor in model.state_dict().items():
    torch.testing.assert_close(tensor, resumed_model.state_dict()[name], rtol=0, atol=0)
assert resumed_scheduler.state_dict() == scheduler.state_dict()
assert resumed_scaler.state_dict() == scaler.state_dict()
assert resumed_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
print("Restored next epoch:", resumed_state["next_epoch"] + 1)
print("Restored LR:", resumed_optimizer.param_groups[0]["lr"])
print("Restored bad-epoch counter:", resumed_state["bad_epochs"])
del model, optimizer, scheduler, scaler  # Release the original run's GPU objects.
"""),
        md("""
## Step 10 · Continue without resetting patience or the schedule

The next call adds one epoch in a normal run, provided patience has not already been
exhausted. It preserves history and counters. If early stopping already fired, the
cell reports that and performs no further optimization. Smoke mode reloads only.
Do not restart from the best weights while retaining the last optimizer state:
those states come from different points in the training trajectory.
"""),
        code("""
resume_stop = resumed_state["next_epoch"] + RESUME_EXTRA_EPOCHS
resumed_state = train_until(resumed_model, resumed_optimizer, resumed_scheduler, resumed_scaler,
                            resumed_state, resume_config, resume_stop)
history_table = pd.DataFrame(resumed_state["history"])
display(history_table)
cu.plot_history(resumed_state["history"])
history_table.plot(x="epoch", y="lr", marker="o", legend=False, title="Learning rate used each epoch")
plt.ylabel("Learning rate")
plt.show()
"""),
        md("""
## Step 11 · Restore the best validation model and evaluate test once

The final inference model comes from `best`, not automatically from the last epoch.
The test set has not influenced the learning-rate comparison, early stopping, or
checkpoint selection. Its small teaching subset has substantial sampling uncertainty.
Do not interpret one short-run score as a production performance guarantee.
"""),
        code("""
best_checkpoint = torch.load(BEST_PATH, map_location=DEVICE, weights_only=True)
resumed_model.load_state_dict(best_checkpoint["model_state"])
resumed_model.eval()
print("Best epoch:", best_checkpoint["training_state"]["next_epoch"])
print("Best validation loss:", best_checkpoint["training_state"]["best_loss"])
print("Held-out test:", cu.evaluate(resumed_model, test_dl, DEVICE))
"""),
        md("""
## Step 12 · Measure inference instead of assuming a speedup

Warm up, then time repeated forward passes on one already-transferred batch.
Synchronize CUDA around the measurement. These numbers measure **model forward time**,
excluding image decoding, disk I/O, data transfer, and application overhead. Compare
FP32 and AMP on your actual GPU; a small CNN can be too small to benefit from AMP.
No result is prefilled, and a speedup is not guaranteed.
"""),
        code("""
benchmark_images, _ = next(iter(val_dl))
benchmark_images = benchmark_images.to(DEVICE)

def benchmark_forward(network, images, use_amp, repeats):
    network.eval()
    with torch.inference_mode():
        for _ in range(3):
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                _ = network(images)
        synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                _ = network(images)
        synchronize()
    return 1000 * (time.perf_counter() - start) / repeats

repeats = 2 if SMOKE else 30
timings = [{"mode": "FP32", "ms_per_batch": benchmark_forward(resumed_model, benchmark_images, False, repeats)}]
if AMP_ENABLED:
    timings.append({"mode": "CUDA AMP", "ms_per_batch": benchmark_forward(resumed_model, benchmark_images, True, repeats)})
display(pd.DataFrame(timings))
print("Batch size:", len(benchmark_images), "| Device:", DEVICE)
if DEVICE.type == "cuda":
    print("Current allocated GPU memory (MiB):", round(torch.cuda.memory_allocated() / 2**20, 1))
"""),
        md("""
## Step 13 · Demonstration challenges

1. Set `MICRO_BATCH_SIZE=15` and `ACCUMULATION_STEPS=3`. Explain the final window's
   weights, then rerun the training procedure as a new validation experiment.
2. Print `scaler.get_scale()` during a CUDA run. Why must the scale remain fixed
   while accumulating one effective batch?
3. Inspect the checkpoint keys. Which states would be missing if you saved only
   `model.state_dict()`?
4. Vary the patience and scheduler period in a new run. Record why each change was
   made and keep the final test set outside this process.

<details><summary>Hints / solutions</summary>

Use each microbatch's sample count divided by the window's actual sample count.
Mixing gradients created with different scale factors makes their sum inconsistent.
Before clipping, unscale once after the full window; update the scaler after the
optimizer step. Resume needs optimizer, scheduler, scaler, random state, epoch,
history, class/preprocessing information, and early-stopping counters.

```python
print(checkpoint.keys())
print(checkpoint["training_state"]["next_epoch"])
print(checkpoint["scheduler_state"])
```

For a stronger continuation check, compare an uninterrupted deterministic CPU run
against an epoch-boundary save/reload run with the same data and epoch seeds. Full
cross-hardware bitwise identity is not promised by this notebook.

</details>

**References:** [PyTorch AMP examples: accumulation and clipping](https://docs.pytorch.org/docs/stable/notes/amp_examples.html),
[AMP performance recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html),
[Saving a general checkpoint](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html),
[Reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html).
"""),
    ]
    write_notebook("08_training_optimization.ipynb", cells)


if __name__ == "__main__":
    transfer_learning()
    custom_image_dataset()
    training_optimization()
