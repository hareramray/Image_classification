"""Build the advanced, independently runnable image-classification lessons."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from authoring import md, code, write_notebook, bootstrap


explainability = [
md(r"""
# 09 · Explain a CNN prediction

**Level:** advanced. **Prerequisites:** tensors, CNNs, a training loop, and validation.
**Live demonstration:** approximately 30–45 minutes of teaching, plus training/download time.

By the end you can compute input saliency, implement Grad-CAM, test an explanation with
occlusion, and explain why a colorful map is not proof that a model is correct.
This notebook trains its own small Fashion-MNIST model; no earlier checkpoint is needed.
The default uses 4,000 training examples. Set `QUICK=False` for a larger experiment.
Accuracy depends on training and is intentionally not pre-filled.
"""),
md("""
## 1. Prepare the environment

Run the setup notebook first to install dependencies. Then use **Restart Kernel and Run All**.
The next cell finds the project folder and selects CUDA when it is available.
"""),
code(bootstrap),
code("""
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

cu.seed_everything(42)
N_TRAIN = 128 if SMOKE else (4000 if QUICK else 20000)
N_VAL = 48 if SMOKE else (800 if QUICK else 3000)
N_TEST = 48 if SMOKE else (1000 if QUICK else 3000)
EPOCHS = 1 if SMOKE else (3 if QUICK else 8)
"""),
md("""
## 2. Keep input coordinates easy to interpret

`ToTensor` maps grayscale pixels to `[0, 1]`, so zero really means a black pixel in this
lesson. Normalized inputs would require undoing normalization for display and choosing
an appropriate occlusion value. The training and evaluation transforms are identical here.
"""),
code("""
train_ds, val_ds, test_ds, class_names = cu.fashion_data(
    train_transform=transforms.ToTensor(), eval_transform=transforms.ToTensor(),
    train_limit=N_TRAIN, val_limit=N_VAL, test_limit=N_TEST, seed=42,
)
train_loader = cu.loader(train_ds, batch_size=64, shuffle=True)
val_loader = cu.loader(val_ds)
test_loader = cu.loader(test_ds)
print('Training / validation / test:', len(train_ds), len(val_ds), len(test_ds))
print(class_names)
"""),
md("""
## 3. Train a model to investigate

The helper uses cross-entropy on logits and restores the checkpoint with the lowest
validation loss. We will explain its decisions, including any mistakes.

**Ask the audience:** does a correct prediction imply that the model used the right evidence?
"""),
code("""
model = cu.SmallCNN(num_classes=len(class_names), in_channels=1).to(DEVICE)
history = cu.fit(model, train_loader, val_loader, EPOCHS, DEVICE, lr=1e-3)
cu.plot_history(history)
model.eval()
"""),
md("""
## 4. Select a correct prediction and a mistake when available

We inspect held-out examples for this fixed-model diagnostic lesson. Once test examples
influence a model revision, use a fresh test set for the revised model's final evaluation.
Small smoke runs might contain no correct examples, or no mistakes: the selection handles both.
"""),
code("""
logits, labels = cu.predict(model, test_loader, DEVICE)
probabilities = logits.softmax(dim=1)
predictions = probabilities.argmax(dim=1)
correct_ids = torch.where(predictions == labels)[0]
wrong_ids = torch.where(predictions != labels)[0]
selected = []
if len(correct_ids):
    selected.append(int(correct_ids[0]))
if len(wrong_ids):
    selected.append(int(wrong_ids[0]))
if not selected:
    selected = [0]
print(f'Test accuracy: {(predictions == labels).float().mean():.3f}')
print('Selected dataset indices:', selected)
for idx in selected:
    print(idx, 'true:', class_names[int(labels[idx])],
          'predicted:', class_names[int(predictions[idx])],
          'confidence:', round(float(probabilities[idx].max()), 3))
"""),
md(r"""
## 5. Input saliency: sensitivity of one logit to the pixels

For target class $c$, calculate $|\partial z_c/\partial x|$.
This measures a local derivative, not what caused the real-world object to belong to a class.
We explain a **logit**, avoiding an additional softmax derivative.

`model.eval()` turns off training behavior such as dropout. It does **not** disable gradients.
Explanation code must not run inside `torch.inference_mode()`; this function explicitly enables gradients.
"""),
code("""
def normalize_map(values):
    values = values.detach().float().cpu()
    lo, hi = values.min(), values.max()
    return (values - lo) / (hi - lo).clamp_min(1e-8)

def input_saliency(model, image, target_class):
    model.eval()
    with torch.enable_grad():
        x = image.unsqueeze(0).to(DEVICE).detach().requires_grad_(True)
        score = model(x)[0, int(target_class)]
        gradient, = torch.autograd.grad(score, x)
    # max across channels also supports RGB images
    return normalize_map(gradient[0].abs().amax(dim=0))

sample_idx = selected[0]
sample, true_class = test_ds[sample_idx]
target_class = int(predictions[sample_idx])
saliency = input_saliency(model, sample, target_class)
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
axes[0].imshow(sample[0], cmap='gray', vmin=0, vmax=1)
axes[0].set_title(f'Input: {class_names[true_class]}')
axes[1].imshow(saliency, cmap='magma', vmin=0, vmax=1)
axes[1].set_title(f'Sensitivity: {class_names[target_class]}')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
"""),
md(r"""
## 6. Grad-CAM: explain with spatial feature maps

For activations $A^k$ in a convolutional layer, average the target-logit gradients over
height and width to obtain channel weights $\alpha_k$. Then form
$\operatorname{ReLU}(\sum_k \alpha_k A^k)$ and resize it for display.
The original feature grid is coarse; interpolation does not add localization precision.
This implements the core method from the [Grad-CAM paper](https://arxiv.org/abs/1610.02391).

A forward hook saves the activations without detaching them. `autograd.grad` obtains
their gradients without accumulating parameter gradients. `finally` removes the hook,
including when an error occurs.
"""),
code("""
def grad_cam(model, image, target_class, target_layer):
    model.eval()
    captured = {}

    def capture_output(module, inputs, output):
        captured['activation'] = output

    handle = target_layer.register_forward_hook(capture_output)
    try:
        with torch.enable_grad():
            x = image.unsqueeze(0).to(DEVICE).detach().requires_grad_(True)
            score = model(x)[0, int(target_class)]
            activation = captured['activation']
            gradients, = torch.autograd.grad(score, activation)
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            coarse = (weights * activation).sum(dim=1, keepdim=True).relu()
            resized = F.interpolate(coarse, size=image.shape[-2:],
                                    mode='bilinear', align_corners=False)
        return normalize_map(resized[0, 0]), tuple(activation.shape[-2:])
    finally:
        handle.remove()

convolution_layers = [m for m in model.features.modules() if isinstance(m, nn.Conv2d)]
target_layer = convolution_layers[-1]
cam, grid_size = grad_cam(model, sample, target_class, target_layer)
print('Target layer:', target_layer)
print('Original explanation grid:', grid_size, 'Displayed image:', sample.shape[-2:])
"""),
md("""
## 7. Compare explanations for correct and incorrect predictions

The map is conditioned on the requested class. A confident mistake can still produce
a convincing heatmap. Use the title, true class, and predicted class when presenting it.
A flat all-zero Grad-CAM map can occur and is displayed as such, not silently replaced.
"""),
code("""
fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3.3 * len(selected)), squeeze=False)
for row, idx in enumerate(selected):
    image, true_id = test_ds[idx]
    predicted_id = int(predictions[idx])
    local_saliency = input_saliency(model, image, predicted_id)
    local_cam, _ = grad_cam(model, image, predicted_id, target_layer)
    axes[row, 0].imshow(image[0], cmap='gray', vmin=0, vmax=1)
    axes[row, 0].set_title(f'True: {class_names[true_id]}\\nPred: {class_names[predicted_id]}')
    axes[row, 1].imshow(local_saliency, cmap='magma', vmin=0, vmax=1)
    axes[row, 1].set_title('Absolute input gradient')
    axes[row, 2].imshow(image[0], cmap='gray', vmin=0, vmax=1)
    axes[row, 2].imshow(local_cam, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    axes[row, 2].set_title('Grad-CAM for predicted class')
    for ax in axes[row]:
        ax.axis('off')
plt.tight_layout()
cu.ARTIFACTS.mkdir(parents=True, exist_ok=True)
attribution_path = cu.ARTIFACTS / '09_attributions.png'
fig.savefig(attribution_path, dpi=140, bbox_inches='tight')
plt.show()
print('Saved:', attribution_path)
"""),
md("""
## 8. Change the question by changing the target class

For a mistake, compare the predicted class with the true class. Otherwise use the
second-highest scoring class. The same pixels can support different model questions.
"""),
code("""
sample_idx = int(wrong_ids[0]) if len(wrong_ids) else selected[0]
sample, true_class = test_ds[sample_idx]
target_class = int(predictions[sample_idx])
alternative_class = (int(true_class) if target_class != int(true_class)
                     else int(probabilities[sample_idx].topk(2).indices[1]))
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
for ax, class_id in zip(axes, [target_class, alternative_class]):
    heatmap, _ = grad_cam(model, sample, class_id, target_layer)
    ax.imshow(sample[0], cmap='gray', vmin=0, vmax=1)
    ax.imshow(heatmap, cmap='jet', alpha=0.45, vmin=0, vmax=1)
    ax.set_title('Target: ' + class_names[class_id])
    ax.axis('off')
plt.tight_layout()
plt.show()
"""),
md("""
## 9. Occlusion: change pixels and measure the score change

Replace a small patch with black and measure **original target logit minus occluded
target logit**. Positive values mean the intervention reduced that score. Negative values
mean it increased the score. Unlike the gradient methods, this needs only forward passes.

Occlusion is a controlled input intervention, but the square mask may itself be unusual
for the data. Conclusions depend on patch size and replacement value.
"""),
code("""
@torch.no_grad()
def occlusion_map(model, image, target_class, patch_size=7, stride=7, fill=0.0):
    model.eval()
    x = image.unsqueeze(0).to(DEVICE)
    baseline = model(x)[0, int(target_class)]
    h, w = image.shape[-2:]
    changes = torch.zeros(h, w, device=DEVICE)
    counts = torch.zeros_like(changes)
    for top in range(0, h, stride):
        for left in range(0, w, stride):
            bottom, right = min(top + patch_size, h), min(left + patch_size, w)
            altered = x.clone()
            altered[:, :, top:bottom, left:right] = fill
            delta = baseline - model(altered)[0, int(target_class)]
            changes[top:bottom, left:right] += delta
            counts[top:bottom, left:right] += 1
    return (changes / counts.clamp_min(1)).cpu()

occlusion = occlusion_map(model, sample, target_class)
limit = max(float(occlusion.abs().max()), 1e-8)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(occlusion, cmap='coolwarm', vmin=-limit, vmax=limit)
ax.set_title('Target-logit drop after black-patch occlusion')
ax.axis('off')
fig.colorbar(im, ax=ax, label='Original score - occluded score')
plt.tight_layout()
plt.show()
"""),
md("""
## 10. A sanity check: compare trained and random weights

An explanation intended to describe a learned model should be examined for sensitivity
to its parameters. Keep the same image and target class, then compare the trained model
with an independently initialized model. One image is a demonstration, not a formal
validation of an explanation method.
"""),
code("""
cu.seed_everything(123)
random_model = cu.SmallCNN(num_classes=len(class_names), in_channels=1).to(DEVICE).eval()
random_layer = [m for m in random_model.features.modules() if isinstance(m, nn.Conv2d)][-1]
trained_map, _ = grad_cam(model, sample, target_class, target_layer)
random_map, _ = grad_cam(random_model, sample, target_class, random_layer)
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
for ax, heatmap, title in zip(axes, [trained_map, random_map], ['Trained model', 'Random model']):
    ax.imshow(heatmap, cmap='magma', vmin=0, vmax=1)
    ax.set_title(title + ': same target class')
    ax.axis('off')
plt.tight_layout()
plt.show()
del random_model
"""),
md("""
## 11. Practical exercises

1. Run occlusion with `patch_size=4, stride=4`, then with `fill=0.5`. Explain what changes.
2. Use the first convolutional layer for Grad-CAM. Compare spatial resolution and meaning.
3. Find a high-confidence mistake and inspect both the predicted and true class.
4. Record one observation the maps support and one claim they cannot establish.

<details><summary>Suggested solutions and discussion</summary>

Call `occlusion_map(model, sample, target_class, patch_size=4, stride=4, fill=0.5)`.
Smaller patches perform more forward passes and measure a different intervention.
Call `grad_cam(model, sample, target_class, convolution_layers[0])` for the earlier layer.
For exercise 3, sort `probabilities.max(1).values[wrong_ids]` in descending order;
skip it explicitly if there are no errors. A map may indicate score sensitivity to an
image region; it does not establish causal reasoning, fairness, or deployment safety.

</details>

**Demonstration checkpoint:** show an input, state the target class, compute an attribution,
change the target, and test a region by occlusion. Explain why the methods may disagree.
"""),
md("""
## References and next lesson

- [Grad-CAM original paper](https://arxiv.org/abs/1610.02391)
- [Captum attribution-method comparison](https://captum.ai/docs/algorithms_comparison_matrix)
- [PyTorch autograd documentation](https://docs.pytorch.org/docs/stable/autograd.html)
- [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292)

Next: [10 · Vision transformers](10_vision_transformers.ipynb).
"""),
]


vision_transformers = [
md(r"""
# 10 · Build and train a tiny vision transformer

**Level:** advanced. **Prerequisites:** CNNs, tensor shapes, training, and validation.
**Live demonstration:** approximately 45–60 minutes, plus training/download time.

Turn an image into patches, project patches into tokens, add positions and a class token,
train a transformer classifier, and compare it with a CNN using the same data split.
Everything needed for the default experiment trains locally; no pretrained download is required.
The tiny model is a teaching architecture, not a reproduction of a large benchmark ViT.
"""),
md("""
## 1. Configure a small experiment

Run the environment setup notebook first. CUDA is used when available; CPU is supported.
`QUICK=True` keeps this lesson small enough for a live class. Larger training runs should
be prepared before the presentation, and results should be labeled with their actual configuration.
"""),
code(bootstrap),
code("""
import math
import time
from torch import nn
from torchvision import transforms

cu.seed_everything(42)
N_TRAIN = 128 if SMOKE else (4000 if QUICK else 20000)
N_VAL = 48 if SMOKE else (800 if QUICK else 3000)
N_TEST = 48 if SMOKE else (1000 if QUICK else 3000)
EPOCHS = 1 if SMOKE else (3 if QUICK else 10)
IMAGE_SIZE = 28
PATCH_SIZE = 7
if not isinstance(PATCH_SIZE, int) or PATCH_SIZE <= 0 or IMAGE_SIZE % PATCH_SIZE:
    raise ValueError('PATCH_SIZE must be a positive integer that divides IMAGE_SIZE=28')
GRID_SIDE = IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES = GRID_SIDE ** 2
EMBED_DIM = 64
HEADS = 4
LAYERS = 2
"""),
md("""
## 2. Use one split for both models

The CNN and transformer will see the same training, validation, and test examples.
We compare a short shared training budget, not fully tuned architectures.
The test set stays untouched until the comparison protocol is fixed.
"""),
code("""
train_ds, val_ds, test_ds, class_names = cu.fashion_data(
    train_transform=transforms.ToTensor(), eval_transform=transforms.ToTensor(),
    train_limit=N_TRAIN, val_limit=N_VAL, test_limit=N_TEST, seed=42,
)
val_loader = cu.loader(val_ds)
example_image, example_label = train_ds[0]
print('Image shape [channels, height, width]:', tuple(example_image.shape))
print('Class:', class_names[example_label])
print('Split sizes:', len(train_ds), len(val_ds), len(test_ds))
"""),
md(r"""
## 3. Cut the image into patches

With the default settings, a $28\times28$ image with $7\times7$ non-overlapping patches produces
$(28/7)^2=16$ patches. Each grayscale patch contains $1\times7\times7=49$ values.
`unfold` exposes a sliding-window view; here the stride equals the window size.
The plot and shape checks adapt when `PATCH_SIZE` changes to another divisor of 28.

**Ask the audience:** what happens to the number of tokens if the patch side is halved?
For divisible image sizes, there are four times as many patches.
"""),
code("""
patch_grid = example_image.unfold(1, PATCH_SIZE, PATCH_SIZE).unfold(2, PATCH_SIZE, PATCH_SIZE)
# [C, rows, columns, patch_h, patch_w] -> [number_of_patches, flattened_patch]
patches = patch_grid.permute(1, 2, 0, 3, 4).reshape(-1, PATCH_SIZE * PATCH_SIZE)
print('Patch grid:', tuple(patch_grid.shape))
print('Flattened patches:', tuple(patches.shape))
figure_side = min(10, max(3.5, GRID_SIDE * 1.2))
fig, axes = plt.subplots(GRID_SIDE, GRID_SIDE, figsize=(figure_side, figure_side), squeeze=False)
for index, ax in enumerate(axes.flat):
    ax.imshow(patches[index].reshape(PATCH_SIZE, PATCH_SIZE), cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'Patch {index}', fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.show()
"""),
md("""
## 4. Project patches into learned token vectors

A convolution with `kernel_size=patch_size` and `stride=patch_size` implements a shared
linear patch projection efficiently. Its output is flattened into `[batch, tokens, embedding]`.
Each token here has 64 learned features instead of 49 raw pixel values.
"""),
code("""
patch_projection = nn.Conv2d(1, EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
batch = example_image.unsqueeze(0)
projected = patch_projection(batch)
tokens = projected.flatten(2).transpose(1, 2)
print('Projected feature grid:', tuple(projected.shape))
print('Patch tokens [B, N, D]:', tuple(tokens.shape))
assert tokens.shape == (1, NUM_PATCHES, EMBED_DIM)
"""),
md(r"""
## 5. Understand a single attention head

Learned projections produce queries $Q$, keys $K$, and values $V$.
Weights are $\operatorname{softmax}(QK^T/\sqrt{d_k})$; multiplying by $V$ combines
information across tokens. The following randomly initialized head demonstrates shape
and normalization only. Its weights are **not** a trained explanation.
"""),
code("""
head_dim = EMBED_DIM // HEADS
q_projection = nn.Linear(EMBED_DIM, head_dim)
k_projection = nn.Linear(EMBED_DIM, head_dim)
v_projection = nn.Linear(EMBED_DIM, head_dim)
q, k, v = q_projection(tokens), k_projection(tokens), v_projection(tokens)
attention = (q @ k.transpose(-2, -1) / math.sqrt(head_dim)).softmax(dim=-1)
mixed = attention @ v
print('Q, K, V:', tuple(q.shape), tuple(k.shape), tuple(v.shape))
print('Attention:', tuple(attention.shape), 'Mixed values:', tuple(mixed.shape))
print('Attention row sums:', attention[0].sum(-1).detach())
plt.figure(figsize=(5, 4))
plt.imshow(attention[0].detach(), cmap='viridis')
plt.xlabel('Key patch')
plt.ylabel('Query patch')
plt.title('One untrained attention head')
plt.colorbar(label='Attention weight')
plt.tight_layout()
plt.show()
"""),
md("""
## 6. Assemble the tiny ViT

The learned class token collects information for classification. Learned position
embeddings identify patch locations. Each encoder block contains multi-head self-attention,
a feed-forward network, residual connections, and layer normalization.

We construct blocks independently in a `ModuleList`, so each block gets its own random
initialization. `batch_first=True` preserves `[B, N, D]`; `norm_first=True` uses pre-normalization.
The classifier returns logits and must not apply softmax before cross-entropy.
"""),
code("""
class TinyViT(nn.Module):
    def __init__(self, image_size=28, patch_size=7, in_channels=1,
                 num_classes=10, embed_dim=64, heads=4, depth=2, dropout=0.1):
        super().__init__()
        if not isinstance(patch_size, int) or patch_size <= 0 or image_size % patch_size:
            raise ValueError('image_size must be divisible by patch_size')
        if embed_dim % heads:
            raise ValueError('embed_dim must be divisible by heads')
        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4,
                dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, x):
        if tuple(x.shape[1:]) != (self.in_channels, self.image_size, self.image_size):
            raise ValueError('Unexpected channel count or image dimensions')
        patch_tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        sequence = torch.cat([cls_tokens, patch_tokens], dim=1)
        sequence = self.dropout(sequence + self.position)
        for block in self.blocks:
            sequence = block(sequence)
        return self.head(self.norm(sequence)[:, 0])

cu.seed_everything(42)
vit = TinyViT(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
              heads=HEADS, depth=LAYERS).to(DEVICE)
with torch.no_grad():
    output = vit(torch.stack([train_ds[0][0], train_ds[1][0]]).to(DEVICE))
print('Logits shape:', tuple(output.shape))
print('Sequence length including class token:', vit.num_patches + 1)
print('Trainable parameters:', sum(p.numel() for p in vit.parameters() if p.requires_grad))
assert output.shape == (2, len(class_names))
"""),
md("""
## 7. Train the transformer

Both models use the same seed, training split, batch size, epochs, learning rate, and
weight decay. This is a controlled classroom comparison, not an equal-compute benchmark.
The validation loss selects the checkpoint separately for each model.
"""),
code("""
def synchronize_device():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()

# Reset before initialization: the preceding shape demo must not change the experiment.
cu.seed_everything(42)
vit = TinyViT(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM,
              heads=HEADS, depth=LAYERS).to(DEVICE)
vit_loader = cu.loader(train_ds, batch_size=64, shuffle=True, seed=42)
synchronize_device()
start = time.perf_counter()
vit_history = cu.fit(vit, vit_loader, val_loader, EPOCHS, DEVICE,
                     lr=1e-3, weight_decay=1e-4)
synchronize_device()
vit_seconds = time.perf_counter() - start
cu.plot_history(vit_history)
print(f'Transformer training time: {vit_seconds:.1f} seconds')
"""),
md("""
## 8. Train the CNN baseline

Convolutions encode locality and shared filters directly. A transformer has different
inductive biases and can need more data or stronger pretraining. A larger architecture
is not guaranteed to win a short, small-data experiment.
"""),
code("""
cu.seed_everything(42)
cnn = cu.SmallCNN(num_classes=len(class_names), in_channels=1).to(DEVICE)
cnn_loader = cu.loader(train_ds, batch_size=64, shuffle=True, seed=42)
synchronize_device()
start = time.perf_counter()
cnn_history = cu.fit(cnn, cnn_loader, val_loader, EPOCHS, DEVICE,
                     lr=1e-3, weight_decay=1e-4)
synchronize_device()
cnn_seconds = time.perf_counter() - start
cu.plot_history(cnn_history)
print(f'CNN training time: {cnn_seconds:.1f} seconds')
"""),
md("""
## 9. Compare validation results before opening the test set

Report parameter counts and observed training time alongside accuracy. Time depends on
hardware, software, other processes, and warm-up. These measurements are descriptive.
The two models are predeclared comparisons; do not change their configuration after seeing test results.
"""),
code("""
import pandas as pd

models = {'Tiny ViT': vit, 'Small CNN': cnn}
training_times = {'Tiny ViT': vit_seconds, 'Small CNN': cnn_seconds}
validation_rows = []
for name, candidate in models.items():
    metrics = cu.evaluate(candidate, val_loader, DEVICE)
    validation_rows.append({
        'model': name,
        'parameters': sum(p.numel() for p in candidate.parameters()),
        'training_seconds': training_times[name],
        'validation_loss': metrics['loss'],
        'validation_accuracy': metrics['accuracy'],
    })
validation_table = pd.DataFrame(validation_rows)
display(validation_table.round(4))
"""),
md("""
## 10. Evaluate both frozen models once

No model selection or hyperparameter changes follow this step. A single seed and a small
test subset do not establish a general ranking of CNNs and transformers.
"""),
code("""
test_loader = cu.loader(test_ds)
test_rows = []
test_predictions = {}
for name, candidate in models.items():
    test_logits, test_labels = cu.predict(candidate, test_loader, DEVICE)
    test_predictions[name] = test_logits.argmax(dim=1)
    test_rows.append({
        'model': name,
        'test_loss': float(nn.functional.cross_entropy(test_logits, test_labels)),
        'test_accuracy': float((test_logits.argmax(1) == test_labels).float().mean()),
    })
test_table = pd.DataFrame(test_rows)
display(test_table.round(4))
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for idx, ax in enumerate(axes.flat):
    image, label = test_ds[idx]
    ax.imshow(image[0], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'True: {class_names[label]}\\n'
                 f'ViT: {class_names[int(test_predictions["Tiny ViT"][idx])]}\\n'
                 f'CNN: {class_names[int(test_predictions["Small CNN"][idx])]}', fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.show()
cu.ARTIFACTS.mkdir(parents=True, exist_ok=True)
validation_table.merge(test_table, on='model').to_csv(cu.ARTIFACTS / '10_vit_cnn_comparison.csv', index=False)
"""),
md("""
## 11. Optional: run an ImageNet-pretrained ViT on your own photograph

This cell is disabled by default. Set `RUN_PRETRAINED=True` and provide an existing photo path.
The first enabled run downloads approximately 330 MB of weights. Use batch size one on
an 8 GB GPU. These weights predict **ImageNet classes**, not the ten Fashion-MNIST classes.
The exact pretrained weights provide their own resizing, cropping, and normalization
through `weights.transforms()`; do not reuse this lesson's bare `ToTensor` transform.
The official [ViT-B/16 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html)
describes the weights and preprocessing.
"""),
code("""
RUN_PRETRAINED = False
PHOTO_PATH = Path('my_photo.jpg')  # Replace with an absolute path to your RGB photograph.

if RUN_PRETRAINED and not SMOKE:
    from PIL import Image
    from torchvision.models import vit_b_16, ViT_B_16_Weights

    if not PHOTO_PATH.is_file():
        raise FileNotFoundError(f'Choose an existing photo first: {PHOTO_PATH.resolve()}')
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    pretrained = vit_b_16(weights=weights).to(DEVICE).eval()
    photo = Image.open(PHOTO_PATH).convert('RGB')
    photo_tensor = weights.transforms()(photo).unsqueeze(0).to(DEVICE)
    print('Preprocessed shape:', tuple(photo_tensor.shape))
    with torch.inference_mode():
        confidence, indices = pretrained(photo_tensor).softmax(1)[0].topk(5)
    for value, index in zip(confidence.cpu(), indices.cpu()):
        print(f'{weights.meta["categories"][int(index)]:25s} {float(value):.3f}')
    plt.figure(figsize=(5, 4))
    plt.imshow(photo)
    plt.axis('off')
    plt.show()
    del pretrained
else:
    print('Optional pretrained photo inference skipped.')
"""),
md("""
## 12. Exercises and expert extensions

1. Compare `PATCH_SIZE=4`, `7`, and `14` on validation. Change the setting in Step 1,
   restart the kernel, and run only through Step 9 for each candidate. Choose the final
   configuration before opening a fresh final test set; keep test results out of selection.
2. Remove position embeddings and repeat a validation experiment. What information is lost?
3. Compare two training-set sizes while keeping the validation subset fixed.
4. For a custom dataset, describe how to replace the pretrained classification head and
   avoid leaking related photographs across splits.

<details><summary>Suggested solutions and discussion</summary>

For a 28-pixel image, patch sizes 4, 7, and 14 produce 49, 16, and 4 patches, respectively;
add one class token in every case. Attention matrices scale quadratically with sequence
length. Smaller patches can therefore increase memory and computation substantially.
Without positions, the class-token output cannot directly distinguish permutations of
the patch tokens in this architecture. For pretrained torchvision ViT-B/16, replace
`pretrained.heads.head` with `nn.Linear(pretrained.heads.head.in_features, num_classes)`;
the replacement is untrained and needs supervised training. Start with a frozen backbone,
then consider unfreezing using a small learning rate. Split by subject or source when
images from the same source are related, and validate before final testing.

</details>

**Demonstration checkpoint:** explain `[B, NUM_PATCHES, EMBED_DIM]`, then the added class
token, then `[B, 10]`. With the defaults these are `[B, 16, 64]`, `[B, 17, 64]`, and `[B, 10]`;
train both models and describe the limits of the comparison using the observed results.
"""),
md("""
## References and next lesson

- [Original Vision Transformer paper](https://arxiv.org/abs/2010.11929)
- [PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)
- [Torchvision ViT-B/16 weights and transforms](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html)

Next: [11 · Robustness and calibration](11_robustness_and_calibration.ipynb).
"""),
]


robustness = [
md(r"""
# 11 · Confidence, calibration, and robustness

**Level:** expert foundations. **Prerequisites:** validation/test separation, logits, softmax,
cross-entropy, and a trained CNN. **Live demonstration:** approximately 45–60 minutes.

Train a classifier, calibrate its confidence on a dedicated held-out subset, and evaluate
both clean and corrupted images. You will implement temperature scaling, negative
log-likelihood (NLL), expected calibration error (ECE), and reliability diagrams.
This notebook trains its own model and runs independently of earlier notebooks.
"""),
md("""
## 1. Start with a fixed experiment plan

Calibration asks whether predicted confidence agrees with observed correctness over groups
of examples. Robustness asks how behavior changes when inputs change. They are related
but different: calibrating a model does not make it resistant to blur or noise.
"""),
code(bootstrap),
code("""
import json
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur

cu.seed_everything(42)
N_TRAIN = 128 if SMOKE else (4000 if QUICK else 20000)
N_VAL = 48 if SMOKE else (1600 if QUICK else 5000)
N_TEST = 48 if SMOKE else (1000 if QUICK else 3000)
EPOCHS = 1 if SMOKE else (3 if QUICK else 8)
N_BINS = 5 if SMOKE else 10
# Decide these stress tests now, before looking at test outcomes.
CORRUPTION_PLAN = [('gaussian_noise', 0.10), ('gaussian_noise', 0.25),
                   ('gaussian_blur', 0.75), ('gaussian_blur', 1.50)]
"""),
md("""
## 2. Reserve separate data for model selection and calibration

| Subset | Allowed use |
|---|---|
| Training | Update network parameters |
| Tuning validation | Select the model checkpoint and architecture |
| Calibration | Fit one temperature after model selection |
| Test | Final clean evaluation and the predeclared corruption study |

We split the original validation subset in two using a fixed random seed. For a real
project, use stratification or source-aware grouping where needed and reserve enough
calibration examples. A smoke run verifies execution only; its metrics are not evidence.
"""),
code("""
train_ds, validation_ds, test_ds, class_names = cu.fashion_data(
    train_transform=transforms.ToTensor(), eval_transform=transforms.ToTensor(),
    train_limit=N_TRAIN, val_limit=N_VAL, test_limit=N_TEST, seed=42,
)
n_tuning = len(validation_ds) // 2
tuning_ds, calibration_ds = random_split(
    validation_ds, [n_tuning, len(validation_ds) - n_tuning],
    generator=torch.Generator().manual_seed(2026),
)
assert set(tuning_ds.indices).isdisjoint(set(calibration_ds.indices))
train_loader = cu.loader(train_ds, batch_size=64, shuffle=True, seed=42)
tuning_loader = cu.loader(tuning_ds)
calibration_loader = cu.loader(calibration_ds)
print('Train / tune / calibrate / test:', len(train_ds), len(tuning_ds), len(calibration_ds), len(test_ds))
calibration_counts = torch.bincount(torch.tensor([int(y) for _, y in calibration_ds]),
                                   minlength=len(class_names))
display(pd.DataFrame({'class': class_names, 'calibration_examples': calibration_counts.numpy()}))
"""),
md("""
## 3. Train and freeze the selected classifier

Only the tuning subset controls checkpoint selection. After this cell, the model weights
are fixed. The next optimization will change one confidence-scaling parameter, not the CNN.
"""),
code("""
model = cu.SmallCNN(num_classes=len(class_names), in_channels=1).to(DEVICE)
history = cu.fit(model, train_loader, tuning_loader, EPOCHS, DEVICE, lr=1e-3)
cu.plot_history(history)
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
calibration_logits, calibration_labels = cu.predict(model, calibration_loader, DEVICE)
print('Calibration logits shape:', tuple(calibration_logits.shape))
"""),
md(r"""
## 4. Define NLL and expected calibration error

NLL is cross-entropy applied to the logits and true class labels: lower is better.
ECE groups examples into confidence bins and averages each bin's absolute confidence–accuracy
gap, weighted by its population. ECE depends on binning and sample size; a low value alone
does not prove reliable probabilities. Empty bins contribute zero and are omitted from plots.

We use top-label calibration: confidence is the largest predicted probability and correctness
is whether that class matches the label. This does not evaluate every class probability separately.
"""),
code("""
def calibration_statistics(logits, labels, temperature=1.0, n_bins=10):
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError('Temperature must be strictly positive')
    scaled = logits.detach().float().cpu() / temperature
    labels = labels.detach().long().cpu()
    probabilities = scaled.softmax(dim=1)
    confidence, prediction = probabilities.max(dim=1)
    correct = prediction.eq(labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    rows = []
    ece = 0.0
    for bin_index in range(n_bins):
        # Include zero in the first bin and one in the last.
        mask = ((confidence >= edges[bin_index]) if bin_index == 0
                else (confidence > edges[bin_index])) & (confidence <= edges[bin_index + 1])
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else float('nan')
        accuracy = float(correct[mask].mean()) if count else float('nan')
        if count:
            ece += count / len(labels) * abs(mean_confidence - accuracy)
        rows.append({'bin': bin_index, 'count': count,
                     'confidence': mean_confidence, 'accuracy': accuracy})
    metrics = {
        'accuracy': float(correct.mean()),
        'nll': float(F.cross_entropy(scaled, labels)),
        'ece': ece,
        'mean_confidence': float(confidence.mean()),
    }
    return metrics, pd.DataFrame(rows)

uncalibrated_metrics, uncalibrated_bins = calibration_statistics(
    calibration_logits, calibration_labels, n_bins=N_BINS,
)
print('Before calibration, on the calibration subset:', uncalibrated_metrics)
"""),
md("""
## 5. Draw a reliability diagram before fitting

Points on the diagonal have matching mean confidence and accuracy. Below the diagonal,
confidence exceeds observed accuracy. The companion count plot shows how many examples
support each point. Sparse bins should not drive strong conclusions.
"""),
code("""
def plot_reliability(binned, title):
    populated = binned[binned['count'] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot([0, 1], [0, 1], '--', color='gray', label='Perfect group calibration')
    axes[0].plot(populated['confidence'], populated['accuracy'], 'o-', label='Observed bins')
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean confidence', ylabel='Accuracy', title=title)
    axes[0].legend(fontsize=8)
    axes[1].bar(binned['bin'], binned['count'])
    axes[1].set(xlabel='Confidence-bin index', ylabel='Examples', title='Support per bin')
    plt.tight_layout()
    plt.show()

plot_reliability(uncalibrated_bins, 'Calibration subset: before scaling')
"""),
md(r"""
## 6. Fit a positive temperature using calibration data only

Temperature scaling replaces logits $z$ with $z/T$ before softmax. Parameterize
$T=\exp(\log T)$ so it stays positive. Optimize calibration-set NLL and retain the best
temperature observed, including the baseline $T=1$.

Here we use a bounded numerical search range for `log_temperature` to keep the classroom
optimization stable. $T>1$ softens probabilities; $T<1$ sharpens them. Since positive
scaling preserves logit ordering, the predicted class stays the same.
The method is described in [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599).
"""),
code("""
# Detached CPU float64 logits make this one-scalar optimization inexpensive and stable.
# clone outside inference_mode: predict() returns inference tensors, which cannot
# be saved for backward by the temperature-fitting loss.
fit_logits = calibration_logits.detach().double().cpu().clone()
fit_labels = calibration_labels.detach().long().cpu().clone()
log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
temperature_optimizer = torch.optim.Adam([log_temperature], lr=0.05)
best_log_temperature = log_temperature.detach().clone()
best_nll = float(F.cross_entropy(fit_logits, fit_labels))
temperature_trace = []

for step in range(20 if SMOKE else 150):
    temperature_optimizer.zero_grad()
    loss = F.cross_entropy(fit_logits / log_temperature.exp(), fit_labels)
    loss.backward()
    temperature_optimizer.step()
    with torch.no_grad():
        log_temperature.clamp_(-4.0, 4.0)
        candidate_nll = float(F.cross_entropy(fit_logits / log_temperature.exp(), fit_labels))
        temperature_trace.append(candidate_nll)
        if np.isfinite(candidate_nll) and candidate_nll < best_nll:
            best_nll = candidate_nll
            best_log_temperature = log_temperature.detach().clone()

TEMPERATURE = float(best_log_temperature.exp())
print(f'Frozen temperature: {TEMPERATURE:.4f}')
print(f'Calibration NLL before: {uncalibrated_metrics["nll"]:.4f}; after: {best_nll:.4f}')
assert TEMPERATURE > 0 and np.isfinite(TEMPERATURE)
assert torch.equal(calibration_logits.argmax(1), (calibration_logits / TEMPERATURE).argmax(1))
plt.figure(figsize=(6, 3))
plt.plot(temperature_trace)
plt.xlabel('Optimization step')
plt.ylabel('Calibration NLL')
plt.title('Fitting one temperature')
plt.tight_layout()
plt.show()
"""),
md("""
## 7. Inspect the fitted calibration subset

This is an optimization diagnostic, not an unbiased estimate of calibration performance:
the temperature was fitted on these same examples. NLL is the optimization objective;
ECE can increase even when NLL decreases.
"""),
code("""
calibrated_metrics, calibrated_bins = calibration_statistics(
    calibration_logits, calibration_labels, TEMPERATURE, N_BINS,
)
display(pd.DataFrame([uncalibrated_metrics, calibrated_metrics],
                     index=['Before', 'After']).round(4))
plot_reliability(calibrated_bins, 'Calibration subset: after fitting')
"""),
md("""
## 8. Define deterministic corruption tests before final evaluation

Noise adds seeded Gaussian pixel perturbations and clips back to `[0, 1]`.
Blur applies a fixed Gaussian kernel. Each dataset index receives the same corruption
on reruns, allowing paired comparisons. We will use the temperature learned on clean
calibration data unchanged under corruption.

These synthetic tests cover two specific changes. They are neither an adversarial-attack
evaluation nor a guarantee for real camera, population, or acquisition shifts.
"""),
code("""
class CorruptedImages(Dataset):
    def __init__(self, dataset, kind, severity, seed=2026):
        self.dataset = dataset
        self.kind = kind
        self.severity = float(severity)
        self.seed = seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.kind == 'gaussian_noise':
            generator = torch.Generator().manual_seed(self.seed + int(index))
            noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
            corrupted = (image + self.severity * noise).clamp(0, 1)
        elif self.kind == 'gaussian_blur':
            corrupted = gaussian_blur(image, kernel_size=[7, 7], sigma=[self.severity, self.severity])
        else:
            raise ValueError(f'Unknown corruption: {self.kind}')
        return corrupted, label

print('Frozen stress-test plan:', CORRUPTION_PLAN)
print('Frozen temperature:', TEMPERATURE)
"""),
md("""
## 9. Open the clean test set once

All training, model selection, calibration, binning, and corruption choices are now fixed.
The same logits give before/after metrics, so there is no retraining or second prediction
pass needed for temperature scaling. Accuracy must agree; confidence, NLL, and ECE can change.
"""),
code("""
test_logits, test_labels = cu.predict(model, cu.loader(test_ds), DEVICE)
test_before, test_bins_before = calibration_statistics(test_logits, test_labels, 1.0, N_BINS)
test_after, test_bins_after = calibration_statistics(test_logits, test_labels, TEMPERATURE, N_BINS)
assert abs(test_before['accuracy'] - test_after['accuracy']) < 1e-7
display(pd.DataFrame([test_before, test_after], index=['Uncalibrated test', 'Calibrated test']).round(4))
plot_reliability(test_bins_before, 'Clean test: before scaling')
plot_reliability(test_bins_after, 'Clean test: after scaling')
"""),
md("""
## 10. Evaluate the predeclared stress tests

Report the original and calibrated model on exactly the same corrupted examples.
Accuracy can fall under corruption while confidence stays high. Clean-data calibration
does not guarantee calibration under distribution shift. Do not choose a new temperature
based on these test outcomes.
"""),
code("""
stress_rows = []
for scaling, metrics in [('raw', test_before), ('calibrated', test_after)]:
    stress_rows.append({'condition': 'clean', 'severity': 0.0, 'scaling': scaling, **metrics})
for kind, severity in CORRUPTION_PLAN:
    corrupted_ds = CorruptedImages(test_ds, kind, severity)
    corrupted_logits, corrupted_labels = cu.predict(model, cu.loader(corrupted_ds), DEVICE)
    assert torch.equal(corrupted_labels, test_labels)
    for scaling, temperature in [('raw', 1.0), ('calibrated', TEMPERATURE)]:
        metrics, _ = calibration_statistics(corrupted_logits, corrupted_labels, temperature, N_BINS)
        stress_rows.append({'condition': kind, 'severity': severity, 'scaling': scaling, **metrics})
stress_table = pd.DataFrame(stress_rows)
display(stress_table.round(4))
"""),
md("""
## 11. Show the input changes and confidence gap

Seeing the corruption alongside the measurements helps explain what the test actually covers.
The confidence–accuracy comparison is an aggregate view; consult NLL, ECE, and reliability
diagrams together instead of reducing trustworthiness to one number.
"""),
code("""
fig, axes = plt.subplots(1, 5, figsize=(13, 3))
clean_image, label = test_ds[0]
axes[0].imshow(clean_image[0], cmap='gray', vmin=0, vmax=1)
axes[0].set_title('Clean')
for ax, (kind, severity) in zip(axes[1:], CORRUPTION_PLAN):
    altered, _ = CorruptedImages(test_ds, kind, severity)[0]
    ax.imshow(altered[0], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'{kind.replace("gaussian_", "")}\\nseverity={severity}')
for ax in axes:
    ax.axis('off')
plt.suptitle('True class: ' + class_names[label])
plt.tight_layout()
plt.show()

summary = stress_table[stress_table['scaling'] == 'calibrated'].copy()
summary['label'] = summary.apply(lambda row: f'{row["condition"]} {row["severity"]:g}', axis=1)
fig, ax = plt.subplots(figsize=(10, 4))
positions = np.arange(len(summary))
ax.bar(positions - 0.18, summary['accuracy'], width=0.36, label='Accuracy')
ax.bar(positions + 0.18, summary['mean_confidence'], width=0.36, label='Mean calibrated confidence')
ax.set_xticks(positions, summary['label'], rotation=20, ha='right')
ax.set_ylim(0, 1)
ax.legend()
ax.set_title('A clean-data temperature under input corruption')
plt.tight_layout()
plt.show()
"""),
md("""
## 12. Save the experiment evidence

The metadata records the fitted temperature, bin count, and evaluation protocol. In an
application, calibration metadata belongs to the exact checkpoint and preprocessing
pipeline it was fitted for. Replacing model weights requires reassessing calibration.
"""),
code("""
cu.ARTIFACTS.mkdir(parents=True, exist_ok=True)
results_path = cu.ARTIFACTS / '11_robustness_results.csv'
metadata_path = cu.ARTIFACTS / '11_calibration_metadata.json'
checkpoint_path = cu.ARTIFACTS / '11_calibrated_cnn.pt'
stress_table.to_csv(results_path, index=False)
torch.save({'model_state_dict': {name: value.detach().cpu() for name, value in model.state_dict().items()},
            'temperature': TEMPERATURE, 'class_names': class_names,
            'architecture': 'SmallCNN', 'in_channels': 1,
            'preprocessing': 'grayscale 28x28, ToTensor, no normalization'}, checkpoint_path)
metadata = {
    'temperature': TEMPERATURE, 'ece_bins': N_BINS,
    'train_examples': len(train_ds), 'tuning_examples': len(tuning_ds),
    'calibration_examples': len(calibration_ds), 'test_examples': len(test_ds),
    'seed': 42, 'validation_partition_seed': 2026, 'corruption_seed': 2026,
    'corruptions': CORRUPTION_PLAN,
    'checkpoint': checkpoint_path.name,
    'temperature_fit_data': 'calibration subset only',
}
metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print('Saved:', results_path, metadata_path, checkpoint_path, sep='\\n')
"""),
md("""
## 13. Exercises and expert discussion

1. Recompute ECE with 5, 10, and 20 bins on the cached test logits. Explain why estimates
   differ without selecting a bin count to make the model look better.
2. Before another experiment, reserve a separate corrupted calibration subset and compare
   clean-only versus corruption-aware temperature fitting on a fresh test set.
3. Design an abstention policy: use validation/calibration data to choose a confidence
   threshold, then report both coverage and accuracy on accepted test examples.
4. Propose a real-world shift that these noise and blur experiments do not represent.

<details><summary>Suggested solutions and discussion</summary>

For exercise 1, call `calibration_statistics(test_logits, test_labels, TEMPERATURE, bins)`
for each predeclared bin count. More bins use fewer examples per bin and can make ECE
noisier. NLL does not depend on binning. For abstention, coverage is the fraction accepted;
accuracy is computed only among accepted examples, with the zero-accepted case handled
explicitly. A threshold selected on test data leaks evaluation information. Camera changes,
different object populations, mislabeled data, and new classes need additional evaluation;
temperature scaling cannot recover information the classifier never learned.

</details>

**Demonstration checkpoint:** draw the four data subsets, fit temperature without touching
test labels, show unchanged accuracy, and explain why corruption can invalidate confidence.
"""),
md("""
## References

- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
- [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261)
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [Torchvision Gaussian blur](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.functional.gaussian_blur.html)

Continue to the deployment and capstone lessons to package a trained model and build a
complete image-classification demonstration with your own data.
"""),
]


if __name__ == '__main__':
    write_notebook('09_explainability.ipynb', explainability)
    write_notebook('10_vision_transformers.ipynb', vision_transformers)
    write_notebook('11_robustness_and_calibration.ipynb', robustness)
