# HuggingFace Food-11 Recognition

Zero-shot food image classification using OpenAI's CLIP model (`clip-vit-base-patch32`) from HuggingFace. Point it at a folder of images and it classifies each one into one of 11 food categories, then saves the results to a CSV.

---

## Table of Contents

- [How It Works](#how-it-works)
  - [CLIP Architecture](#clip-architecture)
  - [Zero-Shot Classification](#zero-shot-classification)
  - [Prompt Engineering](#prompt-engineering)
  - [Image Preprocessing](#image-preprocessing)
  - [Confidence Scoring](#confidence-scoring)
- [Why These Choices?](#why-these-choices)
  - [Why CLIP?](#why-clip)
  - [Why ViT-B/32?](#why-vit-b32)
  - [Why No Fine-Tuning?](#why-no-fine-tuning)
- [Categories](#categories)
- [Setup](#setup)
- [Usage](#usage)
- [Output](#output)

---

## How It Works

### CLIP Architecture

The model used is **`openai/clip-vit-base-patch32`**, a Vision-Language model trained by OpenAI using **Contrastive Language–Image Pretraining (CLIP)**. It has two encoder towers:

| Component | Architecture | Role |
|---|---|---|
| **Image Encoder** | Vision Transformer (ViT-B/32) | Embeds images into a shared latent space |
| **Text Encoder** | Transformer (12 layers, 512-dim) | Embeds text prompts into the same space |

**ViT-B/32** means:
- **B** = Base size (~86M parameters in the vision encoder)
- **32** = Each image is divided into 32×32 pixel patches before being fed to the transformer

An input image of 224×224 pixels is split into a 7×7 grid of 32×32 patches (49 patches total). Each patch is linearly projected into a token embedding and processed by the transformer with positional encodings.

The model was originally pretrained on ~400 million image-text pairs scraped from the internet, using a **contrastive loss** that pulled matching image-text pairs closer together in embedding space and pushed non-matching pairs apart. This is what allows it to generalise to unseen categories without any task-specific training.

### Zero-Shot Classification

This script uses CLIP purely for **inference — no training, fine-tuning, or gradient updates occur at runtime.** The classification pipeline is:

1. For each image, the image encoder produces a 512-dimensional embedding vector.
2. For each of the 11 text prompts (one per food class), the text encoder produces a 512-dimensional embedding vector.
3. The **cosine similarity** between the image embedding and each text embedding is computed. CLIP scales these similarities by a learned temperature parameter (`logit_scale`, a scalar trained during pretraining) to produce raw logit scores.
4. A **softmax** is applied over the 11 logits to produce a probability distribution.
5. The class with the highest probability is returned as the prediction.

```
Image → ViT-B/32 → 512-dim vector ─┐
                                     ├─ cosine sim × logit_scale → softmax → predicted class
Text  → Transformer → 512-dim vector ┘
```

### Prompt Engineering

The text prompts are constructed as:

```python
CLIP_PROMPTS = [f"a photo of {label.lower()}" for label in FOOD11_LABELS]
```

This produces prompts like `"a photo of bread"`, `"a photo of meat"`, `"a photo of noodles/pasta"` etc.

**Why this format?** The phrase `"a photo of X"` was established in the original CLIP paper as consistently outperforming bare class-name prompts (e.g. just `"bread"`). During pretraining, image captions rarely consist of just a single word — they are usually natural sentences describing a scene. Using a short sentence reduces the distribution shift between training captions and the inference-time prompts, leading to better alignment between the image and text embeddings.

### Image Preprocessing

The `CLIPProcessor` handles all preprocessing automatically before the image is passed to the vision encoder:

| Step | Detail | Why |
|---|---|---|
| **Convert to RGB** | `Image.open(...).convert("RGB")` | Handles grayscale, RGBA, palette images — CLIP expects 3-channel input |
| **Resize** | Shortest side scaled to 224px | ViT-B/32 requires a fixed 224×224 input |
| **Center crop** | 224×224 crop from the centre | Removes border noise while keeping the subject centred |
| **Normalise** | Mean `[0.48145466, 0.4578275, 0.40821073]`, Std `[0.26862954, 0.26130258, 0.27577711]` | CLIP-specific normalisation values computed over its training set; aligns pixel value distributions with what the model saw during training |
| **Tensor conversion** | HWC numpy → CHW float32 tensor | PyTorch convention |

These steps are **not augmentations** — they are deterministic transforms applied identically to every image. No random flipping, cropping, colour jitter, or other stochastic augmentations are applied because this is inference, not training. Augmentations during inference would introduce variance in predictions without benefiting accuracy.

The text prompts go through CLIP's tokenizer: lowercased, BPE tokenized, padded/truncated to 77 tokens (CLIP's fixed context length), and converted to integer token IDs.

### Confidence Scoring

The confidence score returned is the **softmax probability** of the top predicted class:

```python
probs = outputs.logits_per_image.softmax(dim=-1)[0]
top_prob, top_idx = probs.max(dim=0)
```

This is a value between 0 and 1 representing how much more strongly the model associates the image with the winning class compared to the other 10 classes. A score of 0.90 means the model is highly certain; a score of 0.15 means the top prediction is only marginally preferred over alternatives.

Note: because softmax is applied over exactly 11 classes, this is a **closed-world probability** — it tells you the relative ranking within these 11 categories, not an absolute measure of how food-like the image is.

---

## Why These Choices?

### Why CLIP?

CLIP is ideal for this task because:

- **Zero-shot capability**: No labelled training data or GPU-hours are needed to adapt it to Food-11. The model generalises because it has already seen vast numbers of food-related image-text pairs during pretraining.
- **No hyperparameter tuning**: There are no learning rates, weight decays, batch sizes, or epoch counts to configure for the classification task itself.
- **No augmentation pipeline**: Training augmentation strategies (random crops, flips, colour jitter) are only relevant when training a model. CLIP's weights are frozen.
- **Robust to domain shift**: Because CLIP was trained on internet data rather than a curated benchmark, it handles real-world image variation (different lighting, angles, plating) reasonably well.

### Why ViT-B/32?

The `clip-vit-base-patch32` checkpoint was chosen because:

- **Patch size 32** produces 49 patches from a 224×224 image, making it computationally cheap — inference runs fast even on CPU.
- **Base size** strikes a balance between accuracy and resource usage (~150 MB for the vision encoder).
- It is the most widely tested CLIP variant and the default recommended checkpoint for general-purpose zero-shot tasks.

For higher accuracy at the cost of more compute, `clip-vit-large-patch14` could be substituted by changing `MODEL_NAME` in `food_recognition.py`.

### Why No Fine-Tuning?

Fine-tuning CLIP on Food-11 would require:
- Labelled training data
- Decisions about learning rate, weight decay, scheduler, batch size, number of epochs, and whether to freeze or unfreeze encoder layers
- Risk of catastrophic forgetting of CLIP's general representations

The zero-shot approach avoids all of this and still produces strong results on common food categories. If higher accuracy were required, **linear probing** (training only a classification head on top of frozen CLIP embeddings) or **LoRA fine-tuning** would be the recommended next steps before full fine-tuning.

---

## Categories

The 11 Food-11 categories recognised by this script:

| Index | Label |
|---|---|
| 0 | Bread |
| 1 | Dairy product |
| 2 | Dessert |
| 3 | Egg |
| 4 | Fried food |
| 5 | Meat |
| 6 | Noodles/Pasta |
| 7 | Rice |
| 8 | Seafood |
| 9 | Soup |
| 10 | Vegetable/Fruit |

---

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

   Key packages installed:

   | Package | Version | Purpose |
   |---|---|---|
   | `torch` | ≥2.1.0 | Tensor operations and GPU acceleration |
   | `torchvision` | ≥0.16.0 | Image transforms used by CLIP processor |
   | `transformers` | ≥4.35.0 | Loads `CLIPModel` and `CLIPProcessor` from HuggingFace |
   | `huggingface_hub` | ≥0.19.0 | Downloads model weights (~600 MB) on first run |
   | `Pillow` | ≥10.0.0 | Opens and converts images |
   | `accelerate` | ≥0.24.0 | Optimises model loading and device placement |

3. The CLIP model weights (~600 MB) are downloaded automatically from HuggingFace on first run and cached locally (typically in `~/.cache/huggingface/hub/`). Subsequent runs load from the cache.

4. A **GPU is used automatically** if CUDA is available (`torch.cuda.is_available()`), otherwise the script falls back to CPU. CPU inference is slower but fully functional.

---

## Usage

```bash
python food_recognition.py <path-to-image-folder> [--output results.csv] [--max-images N]
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `folder` | *(required)* | Path to folder containing images |
| `--output` | `results.csv` | Output CSV file path |
| `--max-images` | *(none)* | Cap the number of images processed (useful for testing) |

**Example:**

```bash
# Classify all images and save results
python food_recognition.py ./dataset --output predictions.csv

# Quick test on first 20 images only
python food_recognition.py ./dataset --max-images 20
```

### Folder structure for accuracy measurement

If your images are organised into per-class subfolders, the script compares predictions against the ground truth and reports accuracy:

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

Subfolder names are resolved to class labels using three strategies (in order):
1. **Exact match** — folder name matches a label directly (e.g. `Bread`, `Meat`)
2. **Hyphen aliases** — `Noodles-Pasta` → `Noodles/Pasta`, `Vegetable-Fruit` → `Vegetable/Fruit`
3. **Integer index** — folder named `0` → `Bread`, `6` → `Noodles/Pasta`, etc.

If the folder name does not match any of these, the ground truth is recorded as `unknown` and the image is excluded from accuracy calculation.

---

## Output

Results are written to a CSV with these columns:

| Column | Description |
|---|---|
| `image` | Relative path to the image from the root folder |
| `actual` | Ground truth label (or `unknown` if folder name is unrecognised) |
| `predicted` | CLIP's predicted category |
| `confidence` | Softmax probability of the top class (4 decimal places, 0–1) |
| `correct` | `True` / `False` / `None` if ground truth is unknown |

Accuracy (`correct / labelled`) is printed to the console at the end when ground truth labels are available.
