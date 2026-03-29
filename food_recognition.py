import csv
import os
import torch
import argparse
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Food-11 class labels
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

# CLIP performs better with descriptive prompts
CLIP_PROMPTS = [f"a photo of {label.lower()}" for label in FOOD11_LABELS]

# Map folder names (with hyphens) to canonical label names
FOLDER_NAME_MAP = {
    "Noodles-Pasta": "Noodles/Pasta",
    "Vegetable-Fruit": "Vegetable/Fruit",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_NAME = "openai/clip-vit-base-patch32"


def load_model():
    print(f"Loading model: {MODEL_NAME}")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded. Running on: {device}\n")
    return model, processor, device


def predict_image(image_path: Path, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(
        text=CLIP_PROMPTS,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1)[0]

    top_prob, top_idx = probs.max(dim=0)
    return FOOD11_LABELS[top_idx.item()], top_prob.item()


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
    parser = argparse.ArgumentParser(
        description="Food-11 recognition using CLIP zero-shot classification"
    )
    parser.add_argument("folder", help="Path to evaluation folder (with per-class subfolders)")
    parser.add_argument("--output", default="results.csv", help="Output CSV file path (default: results.csv)")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images processed")
    args = parser.parse_args()

    run(args.folder, output_csv=args.output, max_images=args.max_images)
