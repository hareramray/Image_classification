"""Author lessons 02-05. Run this file from any working directory."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from authoring import bootstrap, code, md, write_notebook


def linear_classifier():
    cells = [
        md(r"""
        # 02 - Your first image classifier: a linear model

        **Goal:** train a model yourself, understand every training step, and explain its predictions.
        You will turn a 28 x 28 image into ten class scores, calculate a loss, inspect a gradient,
        and keep the model that performs best on validation data.

        **Before you start:** complete the environment and tensor/data lessons (00 and 01).
        Know Python variables, loops, functions, and lists. No previous machine learning is needed.
        This notebook also runs independently: choose **Kernel > Restart Kernel and Run All Cells**.

        **Live demonstration:** allow 45-60 minutes including discussion. Training on the default
        subset is much shorter; download speed, hardware, and package startup affect runtime.
        `QUICK=True` uses a small real dataset. Smoke mode is only a plumbing check, not an experiment.

        **Audience prediction:** can a weighted sum of pixels distinguish a sandal from a shirt?
        Write down your prediction before training. Accuracy is measured here, never promised.
        """),
        code(bootstrap),
        md("""
        ## 1. Choose an experiment size

        An **epoch** is one pass through the training subset. A **batch** is a small group of images
        processed together. Start small so you can see a complete experiment during a demonstration.
        Change `QUICK` in the initialization cell above, then rerun the notebook, for a longer experiment.
        """),
        code("""
        from copy import deepcopy
        from torch import nn
        from torchvision import transforms

        SEED = 42
        cu.seed_everything(SEED)
        TRAIN_N = 96 if SMOKE else (3000 if QUICK else 48000)
        VAL_N = 48 if SMOKE else (800 if QUICK else 12000)
        TEST_N = 48 if SMOKE else (1000 if QUICK else 10000)
        EPOCHS = 1 if SMOKE else (4 if QUICK else 15)
        BATCH_SIZE = 64
        LR = 0.05
        print({"device": str(DEVICE), "train": TRAIN_N, "epochs": EPOCHS,
               "quick": QUICK, "smoke": SMOKE})
        """),
        md("""
        ## 2. Load three separate datasets

        FashionMNIST contains small grayscale clothing images. A label is an integer from 0 to 9.
        Training images teach the model; validation images choose its checkpoint; test images
        estimate performance only after that choice is finished.

        `ToTensor` converts a PIL image into a float tensor with shape `[1, 28, 28]` and values in
        `[0, 1]`. `Normalize((0.5,), (0.5,))` applies `(pixel - 0.5) / 0.5`, giving `[-1, 1]`.
        These fixed constants are an explicit scaling choice, not statistics calculated from test data.
        The helper makes a reproducible train/validation split from the original training set.
        """),
        code("""
        preprocessing = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        train_ds, val_ds, test_ds, class_names = cu.fashion_data(
            train_transform=preprocessing, eval_transform=preprocessing,
            train_limit=TRAIN_N, val_limit=VAL_N, test_limit=TEST_N, seed=SEED,
        )
        train_loader = cu.loader(train_ds, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
        val_loader = cu.loader(val_ds, batch_size=BATCH_SIZE)
        test_loader = cu.loader(test_ds, batch_size=BATCH_SIZE)
        print("Train / validation / test:", len(train_ds), len(val_ds), len(test_ds))
        print(dict(enumerate(class_names)))
        """),
        md("""
        ## 3. Inspect a batch before building a model

        PyTorch image batches use **NCHW**: batch size, channels, height, width.
        Labels use `torch.int64` because each number is a class index.
        Always inspect real inputs before training: a mistaken shape or label can spoil the entire run.

        **Ask the audience:** why is there one channel here, but usually three in a color photograph?
        """),
        code("""
        images, labels = next(iter(train_loader))
        print("Images:", images.shape, images.dtype)
        print("Labels:", labels.shape, labels.dtype)
        print("Range:", float(images.min()), float(images.max()))
        assert images.shape[1:] == (1, 28, 28)
        assert labels.dtype == torch.long
        cu.show_images(train_ds, class_names, n=12, mean=(0.5,), std=(0.5,))
        """),
        md(r"""
        ## 4. Convert pixels into ten scores

        A linear classifier flattens `[N, 1, 28, 28]` into `[N, 784]`. For each class it learns
        784 pixel weights and one bias. Its calculation is `scores = pixels @ weights.T + bias`.
        The ten output numbers are **logits**: unrestricted scores, not probabilities.

        The model has `(784 + 1) * 10 = 7,850` trainable numbers. This is multiclass logistic
        regression when trained with cross-entropy, even though the PyTorch layer is called `Linear`.
        """),
        code("""
        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10)).to(DEVICE)
        logits = model(images.to(DEVICE))
        print(model)
        print("Batch output:", logits.shape)
        print("First image's logits:", logits[0].detach().cpu().numpy().round(2))
        print("Parameters:", sum(p.numel() for p in model.parameters()))
        assert logits.shape == (len(images), 10)
        """),
        md("""
        ## 5. Understand the loss and the prediction

        `CrossEntropyLoss` accepts raw logits and integer labels. It combines a numerically stable
        log-softmax with negative log likelihood. **Do not add softmax before this loss.**
        Lower loss means that the model assigns higher probability to the correct labels overall.

        For display, softmax turns logits into values that sum to one. `argmax` chooses the largest
        score, and gives the same class before or after softmax. A large softmax value alone does not
        establish that the prediction is trustworthy or that the model is well calibrated.
        """),
        code("""
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels.to(DEVICE))
        probabilities = logits.detach().softmax(dim=1).cpu()
        predicted = logits.detach().argmax(dim=1).cpu()
        print("Untrained batch loss:", round(loss.item(), 4))
        print("Probability sum:", probabilities[0].sum().item())
        print("Prediction:", class_names[predicted[0].item()])
        print("Actual:", class_names[labels[0].item()])
        assert torch.equal(predicted, probabilities.argmax(dim=1))
        """),
        md("""
        ## 6. Perform exactly one learning step

        Follow these five actions aloud: clear old gradients, run the model, calculate the loss,
        calculate gradients with `backward`, and update weights with `step`.
        A gradient tells us how a small weight change affects the current loss.

        We use a **copy** for this demonstration so the main model still starts from its original
        weights. A single update is not expected to solve the task or reduce every future batch loss.
        """),
        code("""
        probe = deepcopy(model)
        probe_optimizer = torch.optim.SGD(probe.parameters(), lr=LR)
        x, y = images.to(DEVICE), labels.to(DEVICE)
        before_weights = probe[1].weight.detach().clone()

        probe_optimizer.zero_grad(set_to_none=True)
        probe_logits = probe(x)
        probe_loss = criterion(probe_logits, y)
        probe_loss.backward()
        print("Gradient norm:", probe[1].weight.grad.norm().item())
        probe_optimizer.step()

        print("Weight change norm:", (probe[1].weight - before_weights).norm().item())
        with torch.no_grad():
            print("Same batch loss before / after:", probe_loss.item(), criterion(probe(x), y).item())
        del probe, probe_optimizer
        """),
        md("""
        ## 7. Establish a simple validation baseline

        Before investing in a neural network, ask how well a trivial rule performs.
        This baseline always predicts the most common **training** class. We assess it on validation
        labels, without using test labels or changing the training set.
        These teaching subsets are stratified, so class counts are equal or nearly equal.
        This deliberately simple baseline gives you a reference point before learning starts.
        """),
        code("""
        train_labels = torch.tensor([train_ds[i][1] for i in range(len(train_ds))])
        majority_class = torch.bincount(train_labels, minlength=10).argmax().item()
        val_labels = torch.tensor([val_ds[i][1] for i in range(len(val_ds))])
        baseline_accuracy = (val_labels == majority_class).float().mean().item()
        print("Always predict:", class_names[majority_class])
        print(f"Validation baseline accuracy: {baseline_accuracy:.1%}")
        """),
        md("""
        ## 8. Train with a fully visible loop

        `model.train()` enables training behavior; `model.eval()` enables evaluation behavior.
        This linear network has no dropout, but the distinction matters in later models.
        `torch.no_grad()` stops gradient recording during validation. Validation never calls
        `backward()` or `optimizer.step()`.

        We multiply each batch's mean loss by its size before summing, then divide by the number
        of examples. That handles a smaller final batch correctly. The best checkpoint is chosen
        using **validation loss**, and copied so later updates cannot overwrite it.
        """),
        code("""
        optimizer = torch.optim.SGD(model.parameters(), lr=LR)
        history = []
        best_val_loss = float("inf")
        best_state = deepcopy(model.state_dict())

        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss_sum, train_correct, train_count = 0.0, 0, 0
            for batch_images, batch_labels in train_loader:
                batch_images = batch_images.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                batch_logits = model(batch_images)
                batch_loss = criterion(batch_logits, batch_labels)
                batch_loss.backward()
                optimizer.step()
                count = batch_labels.size(0)
                train_loss_sum += batch_loss.item() * count
                train_correct += (batch_logits.argmax(1) == batch_labels).sum().item()
                train_count += count

            model.eval()
            val_loss_sum, val_correct, val_count = 0.0, 0, 0
            with torch.no_grad():
                for batch_images, batch_labels in val_loader:
                    batch_images = batch_images.to(DEVICE)
                    batch_labels = batch_labels.to(DEVICE)
                    batch_logits = model(batch_images)
                    batch_loss = criterion(batch_logits, batch_labels)
                    count = batch_labels.size(0)
                    val_loss_sum += batch_loss.item() * count
                    val_correct += (batch_logits.argmax(1) == batch_labels).sum().item()
                    val_count += count

            row = {"epoch": epoch, "train_loss": train_loss_sum / train_count,
                   "val_loss": val_loss_sum / val_count,
                   "train_accuracy": train_correct / train_count,
                   "val_accuracy": val_correct / val_count}
            history.append(row)
            if row["val_loss"] < best_val_loss:
                best_val_loss = row["val_loss"]
                best_state = deepcopy(model.state_dict())
            print(f"Epoch {epoch:02d}: train loss={row['train_loss']:.3f}, "
                  f"val loss={row['val_loss']:.3f}, val accuracy={row['val_accuracy']:.1%}")

        model.load_state_dict(best_state)
        print("Restored the checkpoint with lowest validation loss.")
        """),
        md("""
        ## 9. Read the learning curves

        A decreasing training loss shows that optimization is working on training data.
        A growing gap between training and validation performance may indicate overfitting.
        One short run cannot establish a reliable trend.

        Training metrics above were collected while the weights changed across batches;
        validation metrics use one fixed model at the end of the epoch. They are related,
        but are not calculated under identical conditions.
        """),
        code("""
        cu.plot_history(history)
        selected_epoch = min(history, key=lambda row: row["val_loss"])["epoch"]
        print("Selected epoch:", selected_epoch)
        print(f"Validation majority baseline: {baseline_accuracy:.1%}")
        print(f"Selected validation accuracy: {history[selected_epoch - 1]['val_accuracy']:.1%}")
        """),
        md("""
        ## 10. See what the linear model learned

        Each row of the weight matrix can be reshaped into a 28 x 28 image.
        Red and blue indicate weights with different signs under the chosen color scale.
        A weight contributes together with the pixel value, the bias, and competing class scores.
        These pictures are model parameters, not a complete explanation of an individual prediction.
        """),
        code("""
        weight_images = model[1].weight.detach().cpu().view(10, 28, 28)
        scale = weight_images.abs().max().item()
        fig, axes = plt.subplots(2, 5, figsize=(12, 5))
        for class_id, ax in enumerate(axes.flat):
            ax.imshow(weight_images[class_id], cmap="coolwarm", vmin=-scale, vmax=scale)
            ax.set_title(class_names[class_id])
            ax.axis("off")
        fig.suptitle("One learned pixel-weight map per class")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 11. Evaluate the selected model on the held-out test set

        This is the first model assessment using the test set. Once you have seen these results,
        do not repeatedly change settings to improve this same test score. Develop changes using
        validation data, and use a fresh holdout for a new final estimate when needed.
        A subset score describes this sample, not every possible clothing photograph.
        """),
        code("""
        test_logits, test_labels = cu.predict(model, test_loader, DEVICE)
        test_predictions = test_logits.argmax(dim=1)
        test_accuracy = (test_predictions == test_labels).float().mean().item()
        test_loss = criterion(test_logits, test_labels).item()
        print(f"Held-out test accuracy: {test_accuracy:.1%} on {len(test_labels)} images")
        print(f"Held-out test loss: {test_loss:.4f}")

        fig, axes = plt.subplots(2, 4, figsize=(11, 5))
        for index, ax in enumerate(axes.flat):
            image, actual = test_ds[index]
            guess = test_predictions[index].item()
            ax.imshow((image[0] * 0.5 + 0.5).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"True: {class_names[actual]}\\nPred: {class_names[guess]}",
                         color="green" if guess == actual else "crimson", fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 12. Practice and teach it back

        1. Calculate the parameter count for RGB 32 x 32 images and 5 classes without running code.
        2. Repeat the training experiment with learning rates `0.005` and `0.05`, rebuilding the
           model and optimizer from the same seed each time. Compare validation loss only.
        3. Remove `zero_grad()` from a **temporary copy** of the single-step demonstration. Run
           backward twice and inspect the gradient. Why do normal training loops clear it?
        4. Explain why the model receives `[N, 784]` inputs but the loss receives `[N, 10]` logits.

        <details><summary>Solution hints - open after trying</summary>

        The RGB model has `(3 * 32 * 32 + 1) * 5 = 15,365` parameters.
        Rebuild with `cu.seed_everything(SEED)` followed by a new `nn.Sequential(...)` and optimizer
        before each learning-rate run; also recreate the shuffled loader with the same seed.
        Gradients accumulate by default, so missing `zero_grad()` changes the intended update.
        The 784 inputs are image features; the ten outputs are one score for each possible class.

        ```python
        rgb_linear = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 5))
        print(sum(p.numel() for p in rgb_linear.parameters()))
        ```

        </details>

        **Presenter checkpoint:** have a learner narrate the five learning actions and point to
        the exact line that prevents validation data from changing the weights.

        **Limitations:** this model does not explicitly exploit neighboring pixels; these low-resolution
        examples do not establish performance on real product photos. Next, add nonlinear features.

        **References:** [PyTorch optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html),
        [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html),
        [FashionMNIST dataset](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html).
        """),
    ]
    write_notebook("02_linear_classifier.ipynb", cells)


def neural_networks():
    cells = [
        md("""
        # 03 - Neural networks, nonlinear features, and overfitting

        **Goal:** build a multilayer perceptron (MLP), explain why it needs an activation function,
        and observe how training behavior differs from evaluation behavior.

        **Before you start:** lesson 02's logits, cross-entropy, gradients, and train/validation/test
        separation. This notebook creates its own data and model; previous outputs are not required.
        Use **Restart Kernel and Run All Cells** for a clean demonstration.

        **Live demonstration:** 45-60 minutes with discussion. Training time varies by hardware;
        default subsets keep the experiment small. Smoke mode checks execution only.
        We will inspect actual results rather than assume a deeper network always scores higher.
        """),
        code(bootstrap),
        md("""
        ## 1. Set a repeatable experiment

        More parameters give a model more ways to fit the training set. They also make overfitting
        possible. Keep the dataset, seed, and evaluation rule explicit so comparisons have meaning.
        """),
        code("""
        from copy import deepcopy
        from torch import nn
        from torchvision import transforms

        SEED = 42
        cu.seed_everything(SEED)
        TRAIN_N = 96 if SMOKE else (3000 if QUICK else 48000)
        VAL_N = 48 if SMOKE else (800 if QUICK else 12000)
        TEST_N = 48 if SMOKE else (1000 if QUICK else 10000)
        EPOCHS = 1 if SMOKE else (6 if QUICK else 20)
        BATCH_SIZE = 64
        LR = 1e-3
        WEIGHT_DECAY = 1e-4
        print({"device": str(DEVICE), "epochs": EPOCHS, "smoke": SMOKE})
        """),
        md("""
        ## 2. Prepare independent data splits

        Each image starts as `[1, 28, 28]`. The fixed normalization changes `[0, 1]` to `[-1, 1]`.
        Only the training loader shuffles. Validation and test loaders keep a stable order for
        predictions and image galleries. The test set is reserved until the final assessment.
        """),
        code("""
        preprocessing = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
        ])
        train_ds, val_ds, test_ds, class_names = cu.fashion_data(
            train_transform=preprocessing, eval_transform=preprocessing,
            train_limit=TRAIN_N, val_limit=VAL_N, test_limit=TEST_N, seed=SEED,
        )
        train_loader = cu.loader(train_ds, BATCH_SIZE, shuffle=True, seed=SEED)
        val_loader = cu.loader(val_ds, BATCH_SIZE)
        test_loader = cu.loader(test_ds, BATCH_SIZE)
        images, labels = next(iter(train_loader))
        print(images.shape, labels.shape)
        cu.show_images(train_ds, class_names, n=8, mean=(0.5,), std=(0.5,))
        """),
        md(r"""
        ## 3. Why stacking linear layers is not enough

        If `h = x @ W1.T + b1` and `out = h @ W2.T + b2`, substitution gives another single
        linear layer: `W = W2 @ W1`, `b = W2 @ b1 + b2`.
        A second linear layer alone cannot create a nonlinear decision boundary.
        Run this small numerical check before adding an activation.
        """),
        code("""
        first = nn.Linear(2, 3)
        second = nn.Linear(3, 1)
        points = torch.randn(5, 2)
        with torch.no_grad():
            stacked = second(first(points))
            combined_weight = second.weight @ first.weight
            combined_bias = second.weight @ first.bias + second.bias
            combined = points @ combined_weight.T + combined_bias
        print("Largest rounding difference:", (stacked - combined).abs().max().item())
        assert torch.allclose(stacked, combined, atol=1e-6)
        """),
        md("""
        ## 4. Add a nonlinear activation: ReLU

        ReLU means `max(0, x)`: negative values become zero and positive values pass through.
        Putting ReLU between linear layers makes the network able to represent nonlinear functions.
        A hidden unit's value is a learned feature; it is not necessarily a human-readable concept.

        **Pause and predict:** what will ReLU return for `[-2, -1, 0, 1, 2]`?
        """),
        code("""
        values = torch.linspace(-3, 3, 121)
        plt.figure(figsize=(6, 3))
        plt.plot(values.numpy(), torch.relu(values).numpy(), label="ReLU(x)")
        plt.axhline(0, color="gray", linewidth=0.7)
        plt.axvline(0, color="gray", linewidth=0.7)
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.show()
        print(torch.relu(torch.tensor([-2., -1., 0., 1., 2.])))
        """),
        md("""
        ## 5. Define an MLP with explicit dimensions

        The path is `[N,1,28,28] -> [N,784] -> [N,128] -> [N,64] -> [N,10]`.
        Dropout randomly zeros some hidden activations **during training**. At evaluation time it
        becomes an identity operation. Its purpose is regularization, not compression.

        The final layer returns logits. There is no ReLU or softmax after it because cross-entropy
        expects unrestricted raw scores. `nn.Module` registers the layers so their parameters are trained.
        """),
        code("""
        class FashionMLP(nn.Module):
            def __init__(self, hidden=128, dropout=0.2):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(28 * 28, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 10),
                )

            def forward(self, x):
                return self.network(x)

        cu.seed_everything(SEED)
        model = FashionMLP().to(DEVICE)
        print(model)
        print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
        with torch.no_grad():
            print("Output shape:", model(images.to(DEVICE)).shape)
        """),
        md("""
        ## 6. Observe dropout directly

        PyTorch scales the retained activations during training so their expected value is unchanged.
        Two training passes can produce different outputs even when the input and weights are unchanged.
        `eval()` controls dropout; `no_grad()` controls gradient recording. They solve different problems.
        """),
        code("""
        dropout_demo = nn.Dropout(p=0.5)
        ones = torch.ones(12)
        dropout_demo.train()
        print("Training pass A:", dropout_demo(ones).tolist())
        print("Training pass B:", dropout_demo(ones).tolist())
        dropout_demo.eval()
        print("Evaluation pass:", dropout_demo(ones).tolist())

        model.eval()
        with torch.no_grad():
            output_a = model(images.to(DEVICE))
            output_b = model(images.to(DEVICE))
        print("Evaluation outputs match:", torch.allclose(output_a, output_b))
        """),
        md("""
        ## 7. Write reusable, readable train and validation functions

        This is the same learning cycle from lesson 02, organized into functions.
        AdamW adapts updates for different parameters and applies weight decay. Weight decay and
        dropout are separate regularization choices; neither guarantees better validation performance.

        Read `train_one_epoch` from top to bottom. Identify where gradients are cleared, calculated,
        and used. Then find why `validate` cannot update the network.
        """),
        code("""
        criterion = nn.CrossEntropyLoss()

        def train_one_epoch(network, data_loader, optimizer):
            network.train()
            loss_sum, correct, count = 0.0, 0, 0
            for x, y in data_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                logits = network(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                count += y.size(0)
            return {"loss": loss_sum / count, "accuracy": correct / count}

        @torch.no_grad()
        def validate(network, data_loader):
            network.eval()
            loss_sum, correct, count = 0.0, 0, 0
            for x, y in data_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = network(x)
                loss_sum += criterion(logits, y).item() * y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                count += y.size(0)
            return {"loss": loss_sum / count, "accuracy": correct / count}

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        print("Initial validation:", validate(model, val_loader))
        """),
        md("""
        ## 8. Train and restore the best validation checkpoint

        Choose the checkpoint with the lowest validation loss. The last epoch is not automatically
        the best. Copying `state_dict()` preserves the weights at that moment.
        We run a fixed number of epochs for a predictable demonstration; early stopping is a later extension.
        """),
        code("""
        history = []
        best_val_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        for epoch in range(1, EPOCHS + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer)
            val_metrics = validate(model, val_loader)
            row = {"epoch": epoch, "train_loss": train_metrics["loss"],
                   "train_accuracy": train_metrics["accuracy"],
                   "val_loss": val_metrics["loss"], "val_accuracy": val_metrics["accuracy"]}
            history.append(row)
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = deepcopy(model.state_dict())
            print(f"Epoch {epoch:02d}: train={train_metrics['loss']:.3f}, "
                  f"val={val_metrics['loss']:.3f}, val accuracy={val_metrics['accuracy']:.1%}")
        model.load_state_dict(best_state)
        print("Best checkpoint restored:", validate(model, val_loader))
        """),
        md("""
        ## 9. Diagnose fitting behavior

        If both training and validation performance stay poor, possible causes include too few
        updates, a poor learning rate, an input bug, or insufficient model capacity. If training
        improves while validation deteriorates, investigate overfitting and distribution differences.

        Training metrics include active dropout and changing weights. Validation metrics use dropout
        disabled and fixed weights. Validation accuracy can therefore exceed training accuracy,
        especially early on; that observation alone is not evidence of data leakage.
        """),
        code("""
        cu.plot_history(history)
        # Compare both splits using the same fixed model with dropout disabled.
        train_eval_loader = cu.loader(train_ds, BATCH_SIZE, shuffle=False)
        train_eval = validate(model, train_eval_loader)
        val_eval = validate(model, val_loader)
        print("Training set, evaluation mode:", train_eval)
        print("Validation set, evaluation mode:", val_eval)
        print(f"Accuracy gap: {train_eval['accuracy'] - val_eval['accuracy']:+.1%}")
        """),
        md("""
        ## 10. A useful debugging experiment: fit one tiny batch

        A small network should usually be able to fit a tiny collection of consistent examples
        after enough updates. If it cannot, inspect shapes, labels, preprocessing, learning rate,
        gradients, and whether the optimizer contains the model's actual parameters.

        This is a **separate throwaway model**, trained repeatedly on the same training batch with
        dropout off. It demonstrates memorization, not generalization. Smoke mode uses too few
        steps to draw conclusions; even the quick run need not reach perfect accuracy.
        """),
        code("""
        cu.seed_everything(SEED)
        tiny_model = FashionMLP(hidden=64, dropout=0.0).to(DEVICE)
        tiny_optimizer = torch.optim.Adam(tiny_model.parameters(), lr=0.01)
        tiny_x, tiny_y = images[:16].to(DEVICE), labels[:16].to(DEVICE)
        tiny_steps = 3 if SMOKE else (80 if QUICK else 200)
        tiny_losses = []
        tiny_model.train()
        for step in range(tiny_steps):
            tiny_optimizer.zero_grad(set_to_none=True)
            tiny_loss = criterion(tiny_model(tiny_x), tiny_y)
            tiny_loss.backward()
            tiny_optimizer.step()
            tiny_losses.append(tiny_loss.item())
        tiny_model.eval()
        with torch.no_grad():
            tiny_accuracy = (tiny_model(tiny_x).argmax(1) == tiny_y).float().mean().item()
        plt.figure(figsize=(6, 3))
        plt.plot(tiny_losses)
        plt.xlabel("Repeated updates on the same 16 examples")
        plt.ylabel("Cross-entropy")
        plt.show()
        print(f"Tiny training batch accuracy: {tiny_accuracy:.1%}")
        del tiny_model, tiny_optimizer
        """),
        md("""
        ## 11. Assess the main model once on held-out test images

        The tiny-batch demonstration did not modify the main model. Now use its selected validation
        checkpoint for the final test estimate. Inspect a prediction as well as the aggregate score.
        Softmax scores are useful for display, but confidence calibration needs separate evaluation.
        """),
        code("""
        test_logits, test_labels = cu.predict(model, test_loader, DEVICE)
        test_predictions = test_logits.argmax(1)
        test_accuracy = (test_predictions == test_labels).float().mean().item()
        print(f"Final test accuracy: {test_accuracy:.1%} ({len(test_labels)} examples)")
        print(f"Final test loss: {criterion(test_logits, test_labels).item():.4f}")
        first_image, actual = test_ds[0]
        top_scores, top_ids = test_logits[0].softmax(0).topk(3)
        plt.figure(figsize=(3, 3))
        plt.imshow((first_image[0] * 0.5 + 0.5).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
        plt.title(f"Actual: {class_names[actual]}")
        plt.axis("off")
        plt.show()
        for score, class_id in zip(top_scores.tolist(), top_ids.tolist()):
            print(f"{class_names[class_id]:15s} softmax score={score:.3f}")
        """),
        md("""
        ## 12. Save the selected weights with their context

        A weight file alone does not describe preprocessing or the order of class names.
        Save that context with the architecture settings. This notebook writes a new named artifact
        that later lessons do not require; rerunning replaces this lesson's artifact.
        Only load checkpoint files you trust.
        """),
        code("""
        checkpoint_path = cu.ARTIFACTS / "03_fashion_mlp.pt"
        torch.save({
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "architecture": "FashionMLP", "hidden": 128, "dropout": 0.2,
            "class_names": list(class_names), "image_size": [28, 28],
            "mean": [0.5], "std": [0.5], "seed": SEED,
            "smoke": SMOKE, "train_examples": len(train_ds),
        }, checkpoint_path)
        print("Saved:", checkpoint_path)
        """),
        md("""
        ## 13. Exercises and discussion

        1. Predict the parameter count of the first `Linear(784, 128)` layer, then verify it.
        2. Run a controlled comparison with dropout values `0.0` and `0.3`. Use the same initial
           seed, data split, epoch budget, and selection rule; judge with validation data.
        3. Replace both ReLUs with `nn.Tanh()`. Explain why this remains a nonlinear model.
        4. Explain why excellent accuracy on the 16-image experiment does not establish usefulness.

        <details><summary>Solution hints</summary>

        The first layer contains `784 * 128 + 128 = 100,480` trainable numbers.
        Recreate the model, optimizer, and shuffled loader for each dropout condition; do not keep
        training the previous model and call the result a fair comparison. Several seeds provide
        stronger evidence than one run. `Tanh` is nonlinear, though its gradients and scale differ.
        The tiny model has seen the exact evaluation examples repeatedly.

        ```python
        layer = nn.Linear(784, 128)
        print(sum(parameter.numel() for parameter in layer.parameters()))
        # For a fresh controlled run:
        cu.seed_everything(SEED)
        candidate = FashionMLP(hidden=128, dropout=0.3).to(DEVICE)
        candidate_loader = cu.loader(train_ds, BATCH_SIZE, shuffle=True, seed=SEED)
        candidate_optimizer = torch.optim.AdamW(candidate.parameters(), lr=LR,
                                                weight_decay=WEIGHT_DECAY)
        # Repeat steps 8-9 with candidate, candidate_loader, and candidate_optimizer.
        ```

        </details>

        **Presenter checkpoint:** ask a learner to explain the different purposes of `eval()` and
        `no_grad()`, and why raw logits go into cross-entropy.

        **Limitations:** flattening loses an explicit representation of spatial neighborhoods.
        We have not searched architectures systematically, and a quick subset is not a benchmark.
        Next, use convolutional layers to share filters across image locations.

        **References:** [PyTorch model tutorial](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html),
        [Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html),
        [AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
        """),
    ]
    write_notebook("03_neural_networks.ipynb", cells)


def convolutional_networks():
    cells = [
        md("""
        # 04 - Convolutional neural networks: learning spatial features

        **Goal:** apply a filter by hand, track image shapes through a CNN, train a small network,
        and inspect its learned filters and feature maps.

        **Before you start:** lessons 02-03's training loop, logits, activations, and validation.
        This notebook loads its own dataset and trains from scratch. No previous checkpoint is needed.
        Use **Restart Kernel and Run All Cells** to present it from a clean state.

        **Live demonstration:** 45-60 minutes including discussion. A small FashionMNIST subset
        and compact model are suitable for a local NVIDIA GPU; exact runtime varies. Smoke mode
        uses only a few batches and cannot demonstrate model quality.
        """),
        code(bootstrap),
        md("""
        ## 1. Configure the run

        Convolutions reuse a small set of weights across locations. That makes them a useful
        architectural assumption for images: a visual pattern can be relevant in several places.
        It does not make a network perfectly invariant to shifts, rotation, or scale.
        """),
        code("""
        from torch import nn
        import torch.nn.functional as F
        from torchvision import transforms

        SEED = 42
        cu.seed_everything(SEED)
        TRAIN_N = 192 if SMOKE else (6000 if QUICK else 48000)
        VAL_N = 48 if SMOKE else (800 if QUICK else 12000)
        TEST_N = 48 if SMOKE else (1000 if QUICK else 10000)
        EPOCHS = 1 if SMOKE else (5 if QUICK else 15)
        BATCH_SIZE = 64
        print({"device": str(DEVICE), "train_examples": TRAIN_N, "epochs": EPOCHS})
        """),
        md("""
        ## 2. Load and inspect FashionMNIST

        Inputs remain `[N, 1, 28, 28]` through the convolutional layers. We flatten only after
        learning spatial features. Fixed normalization maps the grayscale inputs to `[-1, 1]`.
        Training and validation come from disjoint portions of the original training set.
        """),
        code("""
        preprocessing = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
        ])
        train_ds, val_ds, test_ds, class_names = cu.fashion_data(
            train_transform=preprocessing, eval_transform=preprocessing,
            train_limit=TRAIN_N, val_limit=VAL_N, test_limit=TEST_N, seed=SEED,
        )
        train_loader = cu.loader(train_ds, BATCH_SIZE, shuffle=True, seed=SEED)
        val_loader = cu.loader(val_ds, BATCH_SIZE)
        test_loader = cu.loader(test_ds, BATCH_SIZE)
        cu.show_images(train_ds, class_names, n=8, mean=(0.5,), std=(0.5,))
        """),
        md("""
        ## 3. Apply a hand-designed edge filter

        A 3 x 3 filter slides over the image, multiplying and summing nearby pixels.
        This particular filter responds to horizontal changes in intensity, often highlighting
        vertical edges. Its weights are chosen by us; a CNN learns its own filters from data.

        `F.conv2d` expects a batch and channel dimensions. Strictly, PyTorch implements
        cross-correlation rather than flipping the kernel as mathematical convolution does.
        This naming distinction does not prevent the filters from being learned.
        """),
        code("""
        example, example_label = train_ds[0]
        image_01 = (example * 0.5 + 0.5).clamp(0, 1)
        edge_kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        edge_response = F.conv2d(image_01.unsqueeze(0),
                                 edge_kernel.view(1, 1, 3, 3), padding=1)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        axes[0].imshow(image_01[0], cmap="gray")
        axes[0].set_title(class_names[example_label])
        axes[1].imshow(edge_kernel, cmap="coolwarm")
        axes[1].set_title("Hand-designed 3 x 3 filter")
        axes[2].imshow(edge_response[0, 0], cmap="coolwarm")
        axes[2].set_title("Filter response")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        """),
        md(r"""
        ## 4. Predict convolution shapes and parameter counts

        For kernel size `K`, padding `P`, stride `S`, and dilation 1, one spatial dimension becomes
        `floor((input_size + 2*P - K) / S) + 1`. With size 28, `K=3`, `P=1`, and `S=1`, it stays 28.
        `Conv2d(1, 16, 3, padding=1)` learns 16 filters, so the output has 16 channels.

        Parameters = `out_channels * (in_channels * K * K + 1)` when bias is enabled.
        The extra one is one bias per output channel. Image height and width do not change this count.
        """),
        code("""
        sample_conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        sample_input = torch.zeros(2, 1, 28, 28)
        print("Input:", tuple(sample_input.shape))
        print("Output:", tuple(sample_conv(sample_input).shape))
        print("Weight tensor:", tuple(sample_conv.weight.shape))
        print("Expected parameters:", 16 * (1 * 3 * 3 + 1))
        print("Actual parameters:", sum(p.numel() for p in sample_conv.parameters()))
        """),
        md("""
        ## 5. Pool nearby responses

        Max pooling with a 2 x 2 window and stride 2 keeps the largest value from each region.
        Here 28 x 28 becomes 14 x 14. Pooling has no learned weights. It reduces spatial detail
        and computation, but can also discard information useful for a task.

        **Ask the audience:** calculate the four pooled values from this 4 x 4 example before running it.
        """),
        code("""
        small_grid = torch.tensor([[1., 2., 3., 4.],
                                   [5., 6., 7., 8.],
                                   [9., 10., 11., 12.],
                                   [13., 14., 15., 16.]])
        pooled = F.max_pool2d(small_grid[None, None], kernel_size=2, stride=2)
        print("Before:\\n", small_grid)
        print("After:\\n", pooled[0, 0])
        """),
        md("""
        ## 6. Define a complete, compact CNN

        The first block learns 16 feature maps, then pools to 14 x 14. The second learns 32 maps,
        then pools to 7 x 7. Flattening produces `32 * 7 * 7 = 1,568` features for a small classifier.
        ReLU supplies nonlinear behavior; dropout regularizes the classifier during training.
        The ten final outputs are logits for cross-entropy, so there is no final softmax.
        """),
        code("""
        class FashionCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32 * 7 * 7, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 10),
                )

            def forward(self, x):
                return self.classifier(self.features(x))

        cu.seed_everything(SEED)
        model = FashionCNN().to(DEVICE)
        print(model)
        print("Parameters:", sum(p.numel() for p in model.parameters()))
        """),
        md("""
        ## 7. Trace every shape before training

        Shape errors are easier to fix with one batch than after starting a long run.
        Here we pass a sample through each layer and print the result.
        Switching `28` to `32` without adjusting the classifier would change the flattened width;
        later architectures use adaptive pooling to handle that more flexibly.
        """),
        code("""
        model.eval()
        with torch.no_grad():
            activation = torch.zeros(2, 1, 28, 28, device=DEVICE)
            print("Input:", tuple(activation.shape))
            for layer in list(model.features) + list(model.classifier):
                activation = layer(activation)
                print(f"{layer.__class__.__name__:14s} -> {tuple(activation.shape)}")
        assert activation.shape == (2, 10)
        """),
        md("""
        ## 8. Train the CNN

        The reusable `cu.fit` function performs the loops already written in lessons 02 and 03:
        train mode, clear gradients, forward, cross-entropy, backward, optimizer step, then validation
        in evaluation mode without gradients. It returns the history and restores the lowest
        validation-loss checkpoint. You can inspect its implementation in `course_utils.py`.

        The model architecture changes; the core training process stays the same.
        """),
        code("""
        history = cu.fit(model, train_loader, val_loader, epochs=EPOCHS,
                         device=DEVICE, lr=1e-3, weight_decay=1e-4)
        cu.plot_history(history)
        print("Selected validation metrics:", cu.evaluate(model, val_loader, DEVICE))
        """),
        md("""
        ## 9. Inspect learned first-layer filters

        These filters were optimized for classification rather than set by hand.
        Each is only 3 x 3, so do not expect a detailed picture of a shoe or coat.
        Early filters can respond to local intensity patterns; their role also depends on later layers.
        Color uses one shared scale so filter magnitudes can be compared.
        """),
        code("""
        kernels = model.features[0].weight.detach().cpu()[:, 0]
        limit = max(kernels.abs().max().item(), 1e-8)
        fig, axes = plt.subplots(2, 8, figsize=(13, 4))
        for index, ax in enumerate(axes.flat):
            ax.imshow(kernels[index], cmap="coolwarm", vmin=-limit, vmax=limit)
            ax.set_title(f"Filter {index}", fontsize=9)
            ax.axis("off")
        fig.suptitle("Learned first-layer filters (shared color scale)")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 10. Inspect feature maps for one validation image

        A filter is a small learned weight array; a **feature map** is its response across an input.
        The maps below are after the first ReLU, before pooling. Different channels can respond
        differently to the same clothing item.

        **Discussion:** which maps have strong responses around the outline? Which are mostly quiet?
        A bright location shows an activation, not proof that this pixel caused the final prediction.
        """),
        code("""
        validation_image, validation_label = val_ds[0]
        model.eval()
        with torch.no_grad():
            input_batch = validation_image.unsqueeze(0).to(DEVICE)
            feature_maps = model.features[1](model.features[0](input_batch))[0].cpu()
        fig, axes = plt.subplots(3, 6, figsize=(12, 6))
        axes.flat[0].imshow((validation_image[0] * 0.5 + 0.5).clamp(0, 1), cmap="gray")
        axes.flat[0].set_title(f"Input: {class_names[validation_label]}", fontsize=9)
        maximum = max(feature_maps.max().item(), 1e-8)
        for index in range(16):
            axes.flat[index + 1].imshow(feature_maps[index], cmap="magma", vmin=0, vmax=maximum)
            axes.flat[index + 1].set_title(f"Map {index}", fontsize=9)
        for ax in axes.flat:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 11. Compare model size without confusing it with quality

        A CNN can exploit local structure and shared filters. It can still have many parameters,
        especially in flattened fully connected layers. Parameter count alone cannot establish
        accuracy, inference speed, or memory use; activations and operations also matter.
        This table compares architectures, not trained performance from different experiments.
        """),
        code("""
        linear_count = (28 * 28 + 1) * 10
        first_conv_count = sum(p.numel() for p in model.features[0].parameters())
        feature_count = sum(p.numel() for p in model.features.parameters())
        classifier_count = sum(p.numel() for p in model.classifier.parameters())
        print(f"Linear classifier:  {linear_count:,} parameters")
        print(f"First convolution: {first_conv_count:,} parameters")
        print(f"CNN feature stack: {feature_count:,} parameters")
        print(f"CNN classifier:    {classifier_count:,} parameters")
        print(f"Complete CNN:      {feature_count + classifier_count:,} parameters")
        """),
        md("""
        ## 12. Make the final held-out assessment

        Use the validation-selected checkpoint and deterministic test preprocessing.
        Do not compare this small-run score directly against published full-dataset benchmarks.
        The following lesson will look beyond accuracy and examine class-specific errors.
        """),
        code("""
        test_logits, test_labels = cu.predict(model, test_loader, DEVICE)
        test_predictions = test_logits.argmax(1)
        print(f"Test accuracy: {(test_predictions == test_labels).float().mean().item():.1%}")
        print(f"Test loss: {F.cross_entropy(test_logits, test_labels).item():.4f}")
        print("Test examples:", len(test_labels))

        fig, axes = plt.subplots(2, 4, figsize=(11, 5))
        for index, ax in enumerate(axes.flat):
            image, actual = test_ds[index]
            guess = test_predictions[index].item()
            ax.imshow((image[0] * 0.5 + 0.5).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"True: {class_names[actual]}\\nPred: {class_names[guess]}",
                         color="green" if actual == guess else "crimson", fontsize=9)
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 13. Save a self-described checkpoint

        Save CPU tensors so the checkpoint can later be loaded on a different device.
        We retain architecture and preprocessing information alongside the selected weights.
        Rerunning this cell replaces this lesson's named artifact; other lessons train independently.
        """),
        code("""
        checkpoint_path = cu.ARTIFACTS / "04_fashion_cnn.pt"
        torch.save({
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "architecture": "FashionCNN", "class_names": list(class_names),
            "image_size": [28, 28], "mean": [0.5], "std": [0.5],
            "seed": SEED, "smoke": SMOKE, "train_examples": len(train_ds),
        }, checkpoint_path)
        print(checkpoint_path)
        """),
        md("""
        ## 14. Exercises and demonstration prompts

        1. Predict the shape after a 5 x 5 convolution with no padding on a 28 x 28 image.
        2. Count the parameters of `Conv2d(16, 32, 3, padding=1)` without inspecting the model.
        3. Replace max pooling with average pooling and compare validation performance using
           fresh models, the same seed, and the same training budget.
        4. Change the first layer to accept RGB input. What else must change in the input pipeline?

        <details><summary>Solution hints</summary>

        The convolution produces 24 x 24 spatial dimensions. The second convolution has
        `32 * (16 * 3 * 3 + 1) = 4,640` parameters. Average pooling summarizes the mean rather
        than the maximum; whether it helps is empirical. RGB inputs need three channels,
        appropriate normalization, and data whose resolution matches the classifier.

        ```python
        print(nn.Conv2d(1, 16, 5)(torch.zeros(2, 1, 28, 28)).shape)
        layer = nn.Conv2d(16, 32, 3, padding=1)
        print(sum(p.numel() for p in layer.parameters()))
        # Replace both nn.MaxPool2d(2) layers with nn.AvgPool2d(2), then retrain.
        ```

        </details>

        **Presenter checkpoint:** trace the shape from input to logits without running the model.
        Explain the difference between a filter and a feature map in your own words.

        **Limitations:** these visualizations are exploratory; they are not causal explanations.
        The architecture assumes 28 x 28 grayscale inputs and has not been validated for production.

        **References:** [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html),
        [MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html),
        [PyTorch image-classification tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).
        """),
    ]
    write_notebook("04_convolutional_networks.ipynb", cells)


def augmentation_and_evaluation():
    cells = [
        md("""
        # 05 - Data augmentation and honest model evaluation

        **Goal:** compare two controlled training pipelines, select using validation data, and
        explain final test errors with a confusion matrix, per-class metrics, and an image gallery.

        **Before you start:** lessons 02-04's CNNs, train/eval modes, and validation checkpoints.
        No saved model is needed; both candidates are trained here from the same starting weights.
        Use **Restart Kernel and Run All Cells** for a clean experiment.

        **Live demonstration:** 60-90 minutes including interpretation. Two small training runs
        take longer than one; actual runtime varies. `QUICK=True` limits the dataset and epochs.
        Smoke mode verifies code paths; its tiny samples cannot provide reliable model-quality estimates.

        **Predict first:** will small rotations and shifts help on this dataset? We will measure
        the result rather than assume augmentation must improve accuracy.
        """),
        code(bootstrap),
        md("""
        ## 1. Fix the comparison before training

        Both candidates use the same examples, split, architecture, initialization, batch size,
        optimizer settings, and epoch budget. They differ in training transforms.
        The selection rule is **lowest validation loss**. Decide this before seeing test results.
        """),
        code("""
        from copy import deepcopy
        import pandas as pd
        from IPython.display import display
        from torchvision import transforms
        from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                                     confusion_matrix, ConfusionMatrixDisplay, f1_score)

        SEED = 42
        cu.seed_everything(SEED)
        TRAIN_N = 192 if SMOKE else (6000 if QUICK else 48000)
        VAL_N = 48 if SMOKE else (800 if QUICK else 12000)
        TEST_N = 48 if SMOKE else (1000 if QUICK else 10000)
        EPOCHS = 1 if SMOKE else (5 if QUICK else 15)
        BATCH_SIZE = 64
        MEAN, STD = (0.5,), (0.5,)
        print({"train": TRAIN_N, "validation": VAL_N, "test": TEST_N,
               "epochs_per_candidate": EPOCHS, "smoke": SMOKE})
        """),
        md("""
        ## 2. Define training and evaluation transforms separately

        Small affine transforms rotate, translate, and resize clothing images while usually
        preserving their labels. Label preservation is a task-specific assumption to inspect.
        We avoid vertical flips and aggressive crops that could destroy useful information.

        Only augmented **training** examples receive randomness. Validation and test use the same
        deterministic conversion and normalization. This keeps the evaluation question stable.
        Black padding is added before normalization, so background remains background after scaling.
        """),
        code("""
        eval_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ])
        augmented_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.95, 1.05), fill=0),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        print("Evaluation / baseline training:", eval_transform)
        print("Augmented training:", augmented_transform)
        """),
        md("""
        ## 3. Make identical splits with independent transforms

        A common mistake is to split one dataset and then change its shared `transform`, accidentally
        augmenting validation images too. The course helper creates independent dataset views for
        training and evaluation. Reusing the seed and limits gives both candidates the same indices.
        The assertions below verify split identity instead of merely assuming it.
        """),
        code("""
        split_options = dict(train_limit=TRAIN_N, val_limit=VAL_N, test_limit=TEST_N, seed=SEED)
        train_plain, val_plain, test_plain, class_names = cu.fashion_data(
            train_transform=eval_transform, eval_transform=eval_transform, **split_options,
        )
        train_aug, val_aug, test_aug, aug_class_names = cu.fashion_data(
            train_transform=augmented_transform, eval_transform=eval_transform, **split_options,
        )
        assert list(train_plain.indices) == list(train_aug.indices)
        assert list(val_plain.indices) == list(val_aug.indices)
        assert list(test_plain.indices) == list(test_aug.indices)
        assert list(class_names) == list(aug_class_names)
        assert torch.equal(val_plain[0][0], val_aug[0][0])
        print("Verified identical example indices and deterministic validation preprocessing.")
        val_loader = cu.loader(val_plain, BATCH_SIZE, shuffle=False)
        """),
        md("""
        ## 4. Inspect augmentation before trusting it

        Fetching the same training index repeatedly produces different random views, but its label
        stays the same. Inspect whether a person can still recognize the item in each view.
        Data augmentation adds variation; it does not add genuinely new labeled observations.

        **Audience prompt:** which transforms would be unsuitable for digits, medical scans,
        or traffic signs? Explain the label-preservation assumption for each task.
        """),
        code("""
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        original, actual = train_plain[0]
        axes.flat[0].imshow((original[0] * 0.5 + 0.5).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
        axes.flat[0].set_title(f"Original: {class_names[actual]}", fontsize=9)
        for index in range(1, 8):
            changed, label = train_aug[0]
            assert label == actual
            axes.flat[index].imshow((changed[0] * 0.5 + 0.5).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
            axes.flat[index].set_title(f"Random view {index}", fontsize=9)
        for ax in axes.flat:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 5. Store a common initialization

        `SmallCNN` is the reusable convolutional model defined in `course_utils.py`.
        Print its structure so the comparison is explicit. Copy one initial state and load it into
        each candidate. This avoids attributing an initialization difference to augmentation.

        The loader uses its own seeded random generator, so batch order is controlled independently
        of random augmentation. This model has no dropout. Seeds improve repeatability, but exact
        equality across hardware is not promised. Repeated independent seeds would provide stronger
        evidence than this single-run teaching comparison.
        """),
        code("""
        cu.seed_everything(SEED)
        template_model = cu.SmallCNN(num_classes=10, in_channels=1)
        initial_state = deepcopy(template_model.state_dict())
        print(template_model)
        print("Parameters:", sum(p.numel() for p in template_model.parameters()))
        del template_model
        """),
        md("""
        ## 6. Train both candidates and retain their best validation checkpoints

        Each run resets the seed and builds a new shuffled loader with the same seed.
        `cu.fit` restores its lowest-validation-loss checkpoint. We store CPU copies so there is
        only one actively trained model on the GPU at a time.
        No test results are calculated during candidate selection.
        """),
        code("""
        histories, saved_states, comparisons = {}, {}, []
        for name, training_dataset in [("baseline", train_plain), ("augmented", train_aug)]:
            print(f"\\nTraining candidate: {name}")
            cu.seed_everything(SEED)
            candidate = cu.SmallCNN(num_classes=10, in_channels=1).to(DEVICE)
            candidate.load_state_dict(initial_state)
            candidate_loader = cu.loader(training_dataset, BATCH_SIZE, shuffle=True, seed=SEED)
            history = cu.fit(candidate, candidate_loader, val_loader, epochs=EPOCHS,
                             device=DEVICE, lr=1e-3, weight_decay=1e-4)
            selected_metrics = cu.evaluate(candidate, val_loader, DEVICE)
            histories[name] = history
            saved_states[name] = {key: value.detach().cpu().clone()
                                  for key, value in candidate.state_dict().items()}
            comparisons.append({"candidate": name,
                                "best_epoch": min(history, key=lambda row: row["val_loss"])["epoch"],
                                "val_loss": selected_metrics["loss"],
                                "val_accuracy": selected_metrics["accuracy"]})
            del candidate, candidate_loader
        comparison_table = pd.DataFrame(comparisons).set_index("candidate")
        display(comparison_table)
        """),
        md("""
        ## 7. Compare validation curves and choose the winner

        Augmentation can make training harder and may need more epochs to help. On a short run,
        it may underperform the baseline. Report what happened rather than changing the story.
        Use validation loss for the declared selection rule even if another candidate has slightly
        higher accuracy. Loss accounts for the model's scores as well as its top class.
        """),
        code("""
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name, history in histories.items():
            epochs = [row["epoch"] for row in history]
            axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label=name)
            axes[1].plot(epochs, [row["val_accuracy"] for row in history], marker="o", label=name)
        axes[0].set_title("Validation cross-entropy")
        axes[1].set_title("Validation accuracy")
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()

        winner = min(comparisons, key=lambda row: row["val_loss"])["candidate"]
        selected_model = cu.SmallCNN(num_classes=10, in_channels=1).to(DEVICE)
        selected_model.load_state_dict(saved_states[winner])
        print("Selected using validation loss:", winner)
        """),
        md("""
        ## 8. Freeze the choice, then use the test set

        Now evaluate only the selected candidate on deterministic test inputs.
        We cache logits and labels once and use that same result for all following reports.
        Looking at errors is valuable, but changes inspired by these test errors require a fresh
        final holdout if you want a new unbiased estimate.

        Accuracy is the fraction of correct predictions. Balanced accuracy averages recall over
        the true classes present; macro F1 below gives each of the ten requested classes equal
        weight. These stratified smoke samples include every class, but per-class counts are tiny,
        so the resulting metrics are unstable.
        """),
        code("""
        test_loader = cu.loader(test_plain, BATCH_SIZE, shuffle=False)
        test_logits, test_labels = cu.predict(selected_model, test_loader, DEVICE)
        y_true = test_labels.numpy()
        y_pred = test_logits.argmax(dim=1).numpy()
        all_class_ids = list(range(len(class_names)))
        summary = {
            "selected_candidate": winner,
            "test_examples": len(y_true),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1_all_10_classes": f1_score(y_true, y_pred, labels=all_class_ids,
                                               average="macro", zero_division=0),
            "cross_entropy": torch.nn.functional.cross_entropy(test_logits, test_labels).item(),
        }
        display(pd.Series(summary, name="Held-out test assessment"))
        """),
        md("""
        ## 9. Interpret precision, recall, F1, and support

        **Precision:** of the images predicted as a class, what fraction really belongs to it?
        **Recall:** of the true examples of a class, what fraction did we find?
        **F1:** the harmonic mean of precision and recall. **Support:** the number of true examples.
        For one class, precision is `TP/(TP+FP)` and recall is `TP/(TP+FN)`.

        Macro averages give each class equal weight; weighted averages weight by support.
        We use `zero_division=0` so a class with no predicted examples receives a displayed zero
        precision rather than an undefined numerical result. Treat absent-class metrics cautiously.
        """),
        code("""
        report = classification_report(y_true, y_pred, labels=all_class_ids,
                                       target_names=list(class_names), output_dict=True,
                                       zero_division=0)
        report_table = pd.DataFrame(report).T
        display(report_table.round(3))
        """),
        md("""
        ## 10. Read a confusion matrix in the correct direction

        **Rows are true classes; columns are predicted classes.** The diagonal contains correct
        predictions. A large off-diagonal count identifies a common confusion.
        The second matrix normalizes each true-class row, making recall patterns easier to compare
        when classes have different supports. A zero-support class has an all-zero normalized row.
        """),
        code("""
        counts = confusion_matrix(y_true, y_pred, labels=all_class_ids)
        row_totals = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(counts, row_totals, out=np.zeros_like(counts, dtype=float),
                               where=row_totals != 0)
        fig, axes = plt.subplots(1, 2, figsize=(17, 7))
        ConfusionMatrixDisplay(counts, display_labels=class_names).plot(
            ax=axes[0], xticks_rotation=90, colorbar=False, values_format="d")
        ConfusionMatrixDisplay(normalized, display_labels=class_names).plot(
            ax=axes[1], xticks_rotation=90, colorbar=False, values_format=".2f")
        axes[0].set_title("Counts: true rows, predicted columns")
        axes[1].set_title("Fraction within each true class")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## 11. Find the largest class confusions

        An error table helps turn a large matrix into an investigation plan.
        Check whether the images are visually similar, labels are ambiguous, or the model ignores
        useful features. A confusion count alone does not reveal the cause.
        """),
        code("""
        confusion_pairs = [
            {"true_class": class_names[true_id], "predicted_class": class_names[pred_id],
             "errors": int(counts[true_id, pred_id])}
            for true_id in all_class_ids for pred_id in all_class_ids
            if true_id != pred_id and counts[true_id, pred_id] > 0
        ]
        if confusion_pairs:
            display(pd.DataFrame(confusion_pairs).sort_values("errors", ascending=False).head(8))
        else:
            print("No mistakes in this evaluated sample; inspect a larger independent sample.")
        """),
        md("""
        ## 12. Inspect confident mistakes, not just averages

        Sort incorrect predictions by their maximum softmax score. A wrong answer with a large
        score is a useful reminder that softmax is not a guarantee of correctness.
        The gallery follows the same unshuffled test order used to produce the cached predictions.
        Later lessons examine interpretation and calibration more directly.
        """),
        code("""
        scores = test_logits.softmax(dim=1).max(dim=1).values.numpy()
        wrong_indices = np.flatnonzero(y_pred != y_true)
        ordered_errors = wrong_indices[np.argsort(-scores[wrong_indices])]
        if len(ordered_errors) == 0:
            print("No errors found in this sample.")
        else:
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for ax, index in zip(axes.flat, ordered_errors[:8]):
                image, actual = test_plain[int(index)]
                ax.imshow((image[0] * 0.5 + 0.5).clamp(0, 1), cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"True: {class_names[actual]}\\n"
                             f"Pred: {class_names[y_pred[index]]}\\nScore: {scores[index]:.2f}",
                             fontsize=9, color="crimson")
            for ax in axes.flat:
                ax.axis("off")
            plt.tight_layout()
            plt.show()
        """),
        md("""
        ## 13. Save the experiment evidence

        Record the validation comparison separately from the final test report.
        Keep the selected model's weights, class order, and preprocessing with the result.
        These named files are replaced when you rerun this lesson. Smoke results are labeled in
        the checkpoint and summary so they are not mistaken for real training evidence.
        """),
        code("""
        import json

        comparison_table.to_csv(cu.ARTIFACTS / "05_validation_comparison.csv")
        report_table.to_csv(cu.ARTIFACTS / "05_test_classification_report.csv")
        pd.DataFrame(counts, index=class_names, columns=class_names).to_csv(
            cu.ARTIFACTS / "05_test_confusion_matrix.csv")
        summary["smoke"] = bool(SMOKE)
        summary["quick"] = bool(QUICK)
        (cu.ARTIFACTS / "05_evaluation_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        torch.save({"model_state": saved_states[winner], "architecture": "SmallCNN",
                    "in_channels": 1, "num_classes": 10, "class_names": list(class_names),
                    "mean": list(MEAN), "std": list(STD), "image_size": [28, 28],
                    "selected_candidate": winner, "seed": SEED, "smoke": SMOKE},
                   cu.ARTIFACTS / "05_selected_cnn.pt")
        print("Saved comparison, test metrics, confusion matrix, summary, and selected weights to:")
        print(cu.ARTIFACTS)
        """),
        md("""
        ## 14. Exercises and critical discussion

        1. Recalculate precision and recall manually for one class using the confusion matrix.
        2. Compare rotation limits `0`, `10`, and `25` degrees using validation data and equal budgets.
           Inspect augmented images before training. Why might a stronger transform hurt?
        3. Repeat the baseline/augmented comparison with seeds 7, 42, and 123. Report each validation
           result plus its mean and spread. Do the conclusions survive different seeds?
        4. Imagine 95% of images are shirts. Explain why accuracy alone could be misleading.

        <details><summary>Solution hints</summary>

        For class `k`, true positives are `counts[k,k]`; predicted positives are column `k`'s sum;
        actual positives are row `k`'s sum. Large rotations can introduce images that do not resemble
        realistic examples, or remove distinguishing content. A classifier that always guesses
        shirt could have 95% accuracy in the imagined dataset while finding none of the other classes.

        ```python
        k = 0
        true_positive = counts[k, k]
        precision = true_positive / counts[:, k].sum() if counts[:, k].sum() else 0.0
        recall = true_positive / counts[k, :].sum() if counts[k, :].sum() else 0.0
        print(class_names[k], {"precision": precision, "recall": recall})
        ```

        </details>

        **Presenter checkpoint:** ask a learner to point to the validation-only selection line and
        explain why both candidates were not selected by their test scores. Then ask them to interpret
        one off-diagonal matrix entry and one confident error.

        **Limitations:** one split and one seed are weak evidence of a general improvement. Dataset
        labels may contain ambiguity, the test subset is finite, and real images can differ from
        FashionMNIST. Do not infer deployment readiness from these demonstrations.

        **References:** [Torchvision transforms](https://docs.pytorch.org/vision/stable/transforms.html),
        [scikit-learn classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics),
        [confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html).
        """),
    ]
    write_notebook("05_augmentation_and_evaluation.ipynb", cells)


if __name__ == "__main__":
    linear_classifier()
    neural_networks()
    convolutional_networks()
    augmentation_and_evaluation()
