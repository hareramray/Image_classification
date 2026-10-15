"""Readable shared plumbing; early lessons implement the training loop explicitly.

Data limits select reproducible stratified subsets. Validation examples come only
from the original training partition. The official test partition stays separate.
"""
from pathlib import Path
import copy
import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
SMOKE = os.environ.get("COURSE_SMOKE", "0") == "1"
DATA_DIR.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)


def seed_everything(seed=42):
    """Repeatable random seeds; exact equality across devices is not promised."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure(seed=42):
    seed_everything(seed)
    # Small demos often lose time to excessive CPU thread overhead.
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__} | device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"{props.name} | {props.total_memory / 2**30:.1f} GiB VRAM")
    else:
        print("CPU fallback active. Use QUICK=True; see README for GPU setup.")
    if SMOKE:
        print("SMOKE MODE: tiny execution check, not an accuracy experiment.")
    return device


def loader(dataset, batch_size=64, shuffle=False, seed=42):
    """num_workers=0 avoids Windows notebook multiprocessing surprises."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed))


def _stratified_limit(indices, targets, limit, seed):
    indices = np.asarray(indices, dtype=np.int64)
    if limit is None or limit >= len(indices):
        return indices.tolist()
    if limit < 1:
        raise ValueError("Dataset limits must be positive or None.")
    # Round-robin over shuffled per-class queues yields balanced teaching subsets.
    rng = np.random.default_rng(seed)
    queues = [rng.permutation(indices[targets[indices] == c]).tolist()
              for c in np.unique(targets[indices])]
    chosen = []
    while len(chosen) < limit:
        for queue in queues:
            if queue and len(chosen) < limit:
                chosen.append(queue.pop())
    rng.shuffle(chosen)
    return chosen


def split_indices(targets, val_fraction=0.2, seed=42, limit=None):
    """Stratified train/validation split, optionally limit training afterwards."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one.")
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for label in np.unique(targets):
        idx = rng.permutation(np.flatnonzero(targets == label))
        if len(idx) < 2:
            raise ValueError("Each class needs at least two examples.")
        n_val = min(len(idx) - 1, max(1, round(len(idx) * val_fraction)))
        val.extend(idx[:n_val].tolist())
        train.extend(idx[n_val:].tolist())
    rng.shuffle(train)
    rng.shuffle(val)
    return _stratified_limit(train, targets, limit, seed + 1), val


def _image_data(dataset_cls, train_transform=None, eval_transform=None,
                train_limit=None, val_limit=None, test_limit=None, seed=42):
    train_transform = train_transform if train_transform is not None else transforms.ToTensor()
    eval_transform = eval_transform if eval_transform is not None else transforms.ToTensor()
    raw_train = dataset_cls(DATA_DIR, train=True, download=True, transform=train_transform)
    eval_train = dataset_cls(DATA_DIR, train=True, download=False, transform=eval_transform)
    raw_test = dataset_cls(DATA_DIR, train=False, download=True, transform=eval_transform)
    targets = np.asarray(raw_train.targets)
    train_idx, val_idx = split_indices(targets, seed=seed)
    train_idx = _stratified_limit(train_idx, targets, train_limit, seed + 2)
    val_idx = _stratified_limit(val_idx, targets, val_limit, seed + 3)
    test_idx = _stratified_limit(np.arange(len(raw_test)), np.asarray(raw_test.targets), test_limit, seed + 4)
    assert set(train_idx).isdisjoint(val_idx)
    return (Subset(raw_train, train_idx), Subset(eval_train, val_idx),
            Subset(raw_test, test_idx), list(raw_train.classes))


def fashion_data(train_transform=None, eval_transform=None, train_limit=None,
                 val_limit=None, test_limit=None, seed=42):
    return _image_data(datasets.FashionMNIST, train_transform, eval_transform,
                       train_limit, val_limit, test_limit, seed)


def cifar_data(train_transform=None, eval_transform=None, train_limit=None,
               val_limit=None, test_limit=None, seed=42):
    return _image_data(datasets.CIFAR10, train_transform, eval_transform,
                       train_limit, val_limit, test_limit, seed)


class SmallCNN(nn.Module):
    """Small adaptive classifier: accepts grayscale or RGB images >= 8x8."""
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 4 * 4, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def train_epoch(model, data_loader, optimizer, device, criterion=None):
    criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    model.train()
    total_loss = correct = count = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    if count == 0:
        raise ValueError("Empty training loader.")
    return {"loss": total_loss / count, "accuracy": correct / count}


@torch.inference_mode()
def evaluate(model, data_loader, device, criterion=None):
    criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    model.eval()
    total_loss = correct = count = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * len(labels)
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    if count == 0:
        raise ValueError("Empty evaluation loader.")
    return {"loss": total_loss / count, "accuracy": correct / count}


def fit(model, train_loader, val_loader, epochs, device, lr=1e-3,
        weight_decay=0., label_smoothing=0.):
    """Ordinary AdamW loop. Restore best validation CE; never inspect test here."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    history, best_state, best_loss = [], None, float("inf")
    for epoch in range(1, epochs + 1):
        train = train_epoch(model, train_loader, optimizer, device, train_criterion)
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": train["loss"], "val_loss": val["loss"],
               "train_accuracy": train["accuracy"], "val_accuracy": val["accuracy"]}
        history.append(row)
        print(f"Epoch {epoch:02d} | train loss {train['loss']:.4f} acc {train['accuracy']:.3f}"
              f" | val loss {val['loss']:.4f} acc {val['accuracy']:.3f}")
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


@torch.inference_mode()
def predict(model, data_loader, device):
    model.eval()
    logits, labels = [], []
    for images, target in data_loader:
        logits.append(model(images.to(device)).cpu())
        labels.append(target.cpu())
    return torch.cat(logits), torch.cat(labels)


def plot_history(history):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "val"):
            ax.plot([h["epoch"] for h in history], [h[f"{split}_{metric}"] for h in history],
                    marker="o", label=split)
        ax.set(xlabel="Epoch", ylabel=metric.title(), title=f"Learning curve: {metric}")
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    plt.show()


def show_images(dataset, class_names, n=12, mean=None, std=None):
    import matplotlib.pyplot as plt
    n = min(n, len(dataset))
    rows = (n + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(10, 2.5 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= n:
            continue
        image, label = dataset[i]
        image = image.detach().cpu().clone() if torch.is_tensor(image) else transforms.ToTensor()(image)
        if mean is not None and std is not None:
            image = image * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
        arr = image.clamp(0, 1).permute(1, 2, 0).numpy()
        ax.imshow(arr[..., 0] if arr.shape[-1] == 1 else arr, cmap="gray" if arr.shape[-1] == 1 else None)
        ax.set_title(str(class_names[int(label)]), fontsize=10)
    fig.tight_layout()
    plt.show()
