# HuggingFace Food-11 Recognition

Zero-shot food image classification using OpenAI's CLIP model (`clip-vit-base-patch32`) from HuggingFace. Point it at a folder of images and it classifies each one into one of 11 food categories, then saves the results to a CSV.

## Categories

Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, Vegetable/Fruit

## Setup

**Requirements:** Python 3.8+

1. Clone the repo:
   ```bash
   git clone https://github.com/fa7707/huggingface-food101.git
   cd huggingface-food101
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

The CLIP model (~600 MB) is downloaded automatically from HuggingFace on first run. A GPU is used if available, otherwise it falls back to CPU.

## Usage

```bash
python food_recognition.py <path-to-image-folder> [--output results.csv] [--max-images N]
```

**Arguments:**

| Argument | Description |
|---|---|
| `folder` | Path to folder containing images (required) |
| `--output` | Output CSV file path (default: `results.csv`) |
| `--max-images` | Cap the number of images processed |

**Example:**

```bash
python food_recognition.py ./dataset --output predictions.csv
```

### Folder structure for accuracy measurement

If your images are organized into per-class subfolders, the script will compare predictions against the ground truth and report accuracy:

```
dataset/
  Bread/
    img1.jpg
    img2.jpg
  Meat/
    img3.jpg
  Soup/
    img4.jpg
```

Subfolders can be named with the category name (e.g. `Bread`) or as a zero-based index (e.g. `0` for Bread). The special names `Noodles-Pasta` and `Vegetable-Fruit` are also recognized.

## Output

Results are written to a CSV with these columns:

| Column | Description |
|---|---|
| `image` | Relative path to the image |
| `actual` | Ground truth label (or `unknown` if folder name is unrecognized) |
| `predicted` | CLIP's predicted category |
| `confidence` | Softmax confidence score (0–1) |
| `correct` | `True` / `False` / `None` if ground truth is unknown |

Accuracy is printed to the console at the end when ground truth labels are available.
