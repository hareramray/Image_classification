"""Use the model bundle created by notebook 12. Run --help for CLI usage."""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from course_utils import SmallCNN


def load_bundle(bundle, device="cpu"):
    bundle = Path(bundle)
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("architecture") != "SmallCNN":
        raise ValueError("Unsupported bundle schema or architecture.")
    classes = metadata["class_names"]
    spec = metadata["preprocessing"]
    if (metadata["in_channels"] != 1 or spec["color_mode"] != "L"
            or spec["size_hw"] != [28, 28] or spec["interpolation"] != "bilinear"
            or spec["scale"] != "uint8_to_0_1" or spec["mean"] != [0.0] or spec["std"] != [1.0]):
        raise ValueError("This example predictor supports the documented grayscale 28x28 contract only.")
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Class names must be nonempty and unique.")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Try --device cpu.")
    model = SmallCNN(num_classes=len(classes), in_channels=1)
    state = torch.load(bundle / "weights.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    pipeline = transforms.Compose([
        transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize(spec["mean"], spec["std"])])

    def preprocess(image):
        return pipeline(ImageOps.exif_transpose(image).convert("L"))

    return model, metadata, preprocess


def predict_image(bundle, image_path, device="cpu", top_k=3):
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    model, metadata, preprocess = load_bundle(bundle, device)
    with Image.open(image_path) as image:
        tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].cpu()
    values, indices = probabilities.topk(min(top_k, len(metadata["class_names"])))
    return {"image": str(image_path), "smoke_model": metadata.get("smoke_only", False),
            "predictions": [{"class_index": index, "class_name": metadata["class_names"][index],
                             "probability": probability}
                            for probability, index in zip(values.tolist(), indices.tolist())],
            "note": "Uncalibrated model probabilities; supported domain is Fashion-MNIST-like clothing images."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(predict_image(args.bundle, args.image, args.device, args.top_k), indent=2))
