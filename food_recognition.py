import os
import torch
import argparse
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(model_name: str = "BinhQuocNguyen/food-recognition-model"):
    print(f"Loading model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded. Running on: {device}\n")
    return model, processor, device


def predict_image(image_path: Path, model, processor, device, top_k: int = 3):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        label = FOOD11_LABELS[idx] if idx < len(FOOD11_LABELS) else f"Class {idx}"
        results.append((label, prob))
    return results


def collect_images(folder: Path):
    """Collect all image files recursively from a folder."""
    images = []
    for root, _, files in os.walk(folder):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(Path(root) / fname)
    return sorted(images)


def run(folder: str, top_k: int = 3, max_images: int = None):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    images = collect_images(folder)
    if not images:
        print(f"No images found in: {folder}")
        return

    if max_images:
        images = images[:max_images]

    print(f"Found {len(images)} image(s) in '{folder}'\n")

    model, processor, device = load_model()

    correct = 0
    total = 0

    for img_path in images:
        try:
            predictions = predict_image(img_path, model, processor, device, top_k)
            top_label, top_prob = predictions[0]

            # If image is inside a class subfolder, use the folder name as ground truth
            parent = img_path.parent.name
            ground_truth = None
            if parent.isdigit() and int(parent) < len(FOOD11_LABELS):
                ground_truth = FOOD11_LABELS[int(parent)]
            elif parent in FOOD11_LABELS:
                ground_truth = parent

            print(f"Image : {img_path.relative_to(folder)}")
            print(f"  Top prediction : {top_label} ({top_prob:.1%})")
            for label, prob in predictions[1:]:
                print(f"                   {label} ({prob:.1%})")
            if ground_truth:
                match = "CORRECT" if ground_truth == top_label else "WRONG"
                print(f"  Ground truth   : {ground_truth}  [{match}]")
                correct += ground_truth == top_label
                total += 1
            print()

        except Exception as e:
            print(f"  ERROR processing {img_path.name}: {e}\n")

    if total > 0:
        print(f"Accuracy on labelled images: {correct}/{total} ({correct/total:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Food-11 recognition using HuggingFace model")
    parser.add_argument("folder", help="Path to folder containing food images")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to show (default: 3)")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images processed")
    args = parser.parse_args()

    run(args.folder, top_k=args.top_k, max_images=args.max_images)
