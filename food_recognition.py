import csv
import json
import os
import torch
import argparse
from pathlib import Path
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoFeatureExtractor,
    ViTImageProcessor,
    ViTForImageClassification,
    EfficientNetForImageClassification,
    ResNetForImageClassification,
    ConvNextForImageClassification,
    SwinForImageClassification,
)

# Food-11 class labels (indices 0-10)
FOOD11_LABELS = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles/Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable/Fruit",
]

# Map folder names (with hyphens) to canonical label names
FOLDER_NAME_MAP = {
    "Noodles-Pasta": "Noodles/Pasta",
    "Vegetable-Fruit": "Vegetable/Fruit",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Known architecture classes to try if AutoModel fails
KNOWN_ARCHITECTURES = [
    ViTForImageClassification,
    EfficientNetForImageClassification,
    ResNetForImageClassification,
    ConvNextForImageClassification,
    SwinForImageClassification,
]

# Map architecture name strings (from config.json) to classes
ARCH_NAME_MAP = {
    "ViTForImageClassification": ViTForImageClassification,
    "EfficientNetForImageClassification": EfficientNetForImageClassification,
    "ResNetForImageClassification": ResNetForImageClassification,
    "ConvNextForImageClassification": ConvNextForImageClassification,
    "SwinForImageClassification": SwinForImageClassification,
}


def fetch_raw_config(model_name: str) -> dict:
    """Download and return the model's raw config.json as a dict."""
    try:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(repo_id=model_name, filename="config.json")
        with open(config_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: could not fetch raw config — {e}")
        return {}


def load_processor(model_name: str):
    for loader in [AutoImageProcessor, AutoFeatureExtractor]:
        try:
            return loader.from_pretrained(model_name)
        except Exception:
            continue
    print("  No processor config found — using default ViT processor (224px, ImageNet stats)")
    return ViTImageProcessor(
        size={"height": 224, "width": 224},
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )


def load_model(model_name: str = "BinhQuocNguyen/food-recognition-model"):
    print(f"Loading model: {model_name}")

    # Read the raw config to discover the real backbone architecture
    raw_config = fetch_raw_config(model_name)
    model_type = raw_config.get("model_type", "unknown")
    arch_names = raw_config.get("architectures", [])
    print(f"  Config model_type : {model_type}")
    print(f"  Config architectures: {arch_names}")

    processor = load_processor(model_name)

    model = None

    # 1. If the config lists a known architecture class name, try that first
    for arch_name in arch_names:
        if arch_name in ARCH_NAME_MAP:
            try:
                model = ARCH_NAME_MAP[arch_name].from_pretrained(
                    model_name, ignore_mismatched_sizes=True
                )
                print(f"  Loaded via config architecture: {arch_name}")
                break
            except Exception as e:
                print(f"  {arch_name} failed: {e}")

    # 2. Fall back: try each known architecture until one works
    if model is None:
        print("  Trying known architecture classes...")
        for cls in KNOWN_ARCHITECTURES:
            try:
                model = cls.from_pretrained(model_name, ignore_mismatched_sizes=True)
                print(f"  Loaded as {cls.__name__}")
                break
            except Exception:
                continue

    if model is None:
        raise RuntimeError(
            f"Could not load '{model_name}' with any known architecture.\n"
            "The model may have an incomplete upload on HuggingFace. "
            "Try a different model (e.g. 'nateraw/food')."
        )

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  Running on: {device}\n")
    return model, processor, device


def resolve_label(model, idx: int) -> str:
    """Get label name from model's id2label if available, else our hardcoded list."""
    id2label = getattr(model.config, "id2label", None)
    if id2label:
        # id2label keys can be ints or strings
        label = id2label.get(idx) or id2label.get(str(idx))
        if label:
            return label
    if idx < len(FOOD11_LABELS):
        return FOOD11_LABELS[idx]
    return f"Class {idx}"


def predict_image(image_path: Path, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

    top_prob, top_idx = probs.max(dim=0)
    label = resolve_label(model, top_idx.item())
    return label, top_prob.item()


def resolve_ground_truth(folder_name: str):
    if folder_name in FOLDER_NAME_MAP:
        return FOLDER_NAME_MAP[folder_name]
    if folder_name in FOOD11_LABELS:
        return folder_name
    if folder_name.isdigit() and int(folder_name) < len(FOOD11_LABELS):
        return FOOD11_LABELS[int(folder_name)]
    return None


def collect_images(folder: Path):
    images = []
    for root, _, files in os.walk(folder):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(Path(root) / fname)
    return sorted(images)


def run(folder: str, output_csv: str = "results.csv", max_images: int = None):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    images = collect_images(folder)
    if not images:
        print(f"No images found in: {folder}")
        return

    if max_images:
        images = images[:max_images]

    print(f"Found {len(images)} image(s) in '{folder}'")
    print(f"Results will be saved to: {output_csv}\n")

    model, processor, device = load_model()

    rows = []
    correct = 0
    labelled = 0

    for i, img_path in enumerate(images, 1):
        try:
            predicted_label, confidence = predict_image(img_path, model, processor, device)
            ground_truth = resolve_ground_truth(img_path.parent.name)
            is_correct = predicted_label == ground_truth if ground_truth else None

            rows.append({
                "image": str(img_path.relative_to(folder)),
                "actual": ground_truth if ground_truth else "unknown",
                "predicted": predicted_label,
                "confidence": f"{confidence:.4f}",
                "correct": is_correct,
            })

            status = ""
            if ground_truth:
                labelled += 1
                if is_correct:
                    correct += 1
                    status = " [CORRECT]"
                else:
                    status = " [WRONG]"

            print(f"[{i}/{len(images)}] {img_path.name}")
            print(f"  Actual:    {ground_truth or 'unknown'}")
            print(f"  Predicted: {predicted_label} ({confidence:.1%}){status}\n")

        except Exception as e:
            print(f"  ERROR processing {img_path.name}: {e}\n")
            rows.append({
                "image": str(img_path.relative_to(folder)),
                "actual": resolve_ground_truth(img_path.parent.name) or "unknown",
                "predicted": "ERROR",
                "confidence": "",
                "correct": False,
            })

    output_path = Path(output_csv)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "actual", "predicted", "confidence", "correct"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results saved to: {output_path.resolve()}")
    if labelled > 0:
        print(f"Accuracy: {correct}/{labelled} ({correct/labelled:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Food-11 recognition using HuggingFace model")
    parser.add_argument("folder", help="Path to evaluation folder (with per-class subfolders)")
    parser.add_argument("--output", default="results.csv", help="Output CSV file path (default: results.csv)")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images processed")
    args = parser.parse_args()

    run(args.folder, output_csv=args.output, max_images=args.max_images)
