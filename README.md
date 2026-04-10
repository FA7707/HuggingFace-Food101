# HuggingFace Food-11 Recognition

Zero-shot food image classification using OpenAI's CLIP model (`clip-vit-base-patch32`) from HuggingFace. Point it at a folder of images and it classifies each one into one of 11 food categories, then saves the results to a CSV.

---

## Table of Contents

- [The Dataset](#the-dataset)
- [How It Works](#how-it-works)
  - [CLIP Architecture](#clip-architecture)
  - [Zero-Shot Classification](#zero-shot-classification)
  - [Prompt Engineering](#prompt-engineering)
  - [Image Preprocessing Pipeline](#image-preprocessing-pipeline)
  - [Confidence Scoring](#confidence-scoring)
- [CLIP Pretraining: Augmentations and Optimisation](#clip-pretraining-augmentations-and-optimisation)
  - [Training Data](#training-data)
  - [Augmentation Strategy](#augmentation-strategy)
  - [Optimiser and Hyperparameters](#optimiser-and-hyperparameters)
  - [Loss Function](#loss-function)
  - [Temperature Parameter](#temperature-parameter)
- [Why These Choices?](#why-these-choices)
- [Categories](#categories)
- [Setup](#setup)
- [Usage](#usage)
- [Output](#output)

---

## The Dataset

This script is designed to evaluate against the **Food-11** dataset, created by EPFL's Multimedia Signal Processing Group (MMSPG). It should not be confused with the larger **Food-101** dataset (101 classes, 101,000 images) — Food-11 groups food into 11 broad superclasses, making it better suited for general food-type recognition rather than fine-grained dish identification.

### Origin

Food-11 was assembled by consolidating and re-labelling images from three existing datasets:
- **Food-101** (101 fine-grained classes)
- **UEC-FOOD-100** (100 Japanese-centric food classes)
- **UEC-FOOD-256** (256 food classes)

Images from these sources were mapped into 11 broad categories, giving the dataset cross-cultural diversity (Western, Japanese, and Asian cuisines are all represented).

### Dataset Statistics

| Split | Images |
|---|---|
| Training | 9,866 |
| Validation | 3,430 |
| Test (Evaluation) | 3,347 |
| **Total** | **16,643** |

### Class Distribution

The dataset is **not evenly balanced**. Dessert and Soup are the largest classes; Rice is by far the smallest. This imbalance is worth keeping in mind when interpreting accuracy — a model that is weak on Rice affects the overall score less than one that is weak on Dessert.

| Class | Train | Validation | Test | Total |
|---|---|---|---|---|
| Bread | 994 | 362 | 368 | 1,724 |
| Dairy product | 429 | 144 | 148 | 721 |
| Dessert | 1,500 | 500 | 500 | 2,500 |
| Egg | 986 | 327 | 335 | 1,648 |
| Fried food | 848 | 326 | 287 | 1,461 |
| Meat | 1,325 | 449 | 432 | 2,206 |
| Noodles/Pasta | 440 | 147 | 147 | 734 |
| Rice | 280 | 96 | 96 | 472 |
| Seafood | 855 | 347 | 303 | 1,505 |
| Soup | 1,500 | 500 | 500 | 2,500 |
| Vegetable/Fruit | 709 | 232 | 231 | 1,172 |

### Images and Metadata

- **Format**: JPEG and PNG, variable native resolution (typically several hundred pixels on each side)
- **Colour**: All RGB; some source images were originally greyscale or RGBA — handled by the `convert("RGB")` call in the script
- **Labels**: Class membership is encoded entirely by subfolder name — there are no sidecar metadata files, EXIF annotations, or bounding boxes
- **Total disk size**: ~1.16 GB

### Food-11 vs Food-101 at a glance

| Property | Food-11 | Food-101 |
|---|---|---|
| Classes | 11 broad superclasses | 101 fine-grained dishes |
| Total images | 16,643 | 101,000 |
| Class balance | Moderate imbalance | Perfectly balanced (1,000/class) |
| Use case | General food-type detection | Fine-grained dish recognition |
| Difficulty for zero-shot | Easier (broader categories) | Harder (e.g. "beef tartare" vs "steak") |

---

## How It Works

### CLIP Architecture

The model used is **`openai/clip-vit-base-patch32`**, a Vision-Language model trained by OpenAI using **Contrastive Language–Image Pretraining (CLIP)**. It has two encoder towers:

| Component | Architecture | Role |
|---|---|---|
| **Image Encoder** | Vision Transformer (ViT-B/32) | Embeds images into a shared latent space |
| **Text Encoder** | Transformer (12 layers, 512-dim, 8 heads) | Embeds text prompts into the same space |

**ViT-B/32** means:
- **B** = Base size (~86M parameters in the vision encoder)
- **32** = Each image is divided into 32×32 pixel patches before being fed to the transformer

A 224×224 input is cut into a 7×7 grid of 32×32 patches (49 patches total). Each patch is linearly projected into a 768-dimensional token embedding and processed by 12 transformer layers with multi-head self-attention and positional encodings, producing a single 512-dimensional image embedding.

### Zero-Shot Classification

This script uses CLIP purely for **inference — no training, fine-tuning, or gradient updates happen at runtime.** The pipeline is:

1. The image encoder embeds the input image into a 512-dimensional vector.
2. The text encoder embeds each of the 11 food-class prompts into a 512-dimensional vector.
3. The **cosine similarity** between the image vector and each text vector is computed, then scaled by CLIP's learned `logit_scale` temperature.
4. **Softmax** over the 11 scores produces a probability distribution.
5. The class with the highest probability is the prediction.

```
Image → ViT-B/32 → 512-dim vector ─┐
                                     ├─ cosine sim × logit_scale → softmax → predicted class
Text  → Transformer → 512-dim vector ┘
```

### Prompt Engineering

```python
CLIP_PROMPTS = [f"a photo of {label.lower()}" for label in FOOD11_LABELS]
# → ["a photo of bread", "a photo of dairy product", ...]
```

**Why `"a photo of X"`?** CLIP's pretraining data consisted of natural image captions, not bare class names. Using a descriptive sentence rather than a single word reduces the distribution gap between training captions and inference-time prompts, producing better embedding alignment. This exact template was validated in the original CLIP paper as consistently outperforming bare-word prompts across classification benchmarks.

For even better results, **prompt ensembling** can be used — averaging embeddings across multiple phrasings of the same class (e.g. `"a photo of bread"`, `"a picture of fresh bread"`, `"food: bread"`). This script uses single prompts for simplicity.

### Image Preprocessing Pipeline

The `CLIPProcessor` applies a fixed, deterministic preprocessing pipeline to every image before it reaches the vision encoder. These are **not stochastic augmentations** — they produce the same output every time for the same input image.

| Step | Detail | Why |
|---|---|---|
| **RGB conversion** | `Image.open(...).convert("RGB")` | CLIP's encoder expects exactly 3 channels; this handles greyscale, RGBA, and palette images from the dataset |
| **Resize** | Bicubic resize so the shortest side is 224px | ViT-B/32 requires a fixed 224×224 input; bicubic preserves edge sharpness better than bilinear |
| **Center crop** | 224×224 crop from the centre | Removes peripheral padding/background introduced by the resize while keeping the subject centred |
| **Normalise pixels** | Mean `[0.4815, 0.4578, 0.4082]`, Std `[0.2686, 0.2613, 0.2758]` | CLIP-specific channel statistics computed over its 400M-image training set — aligning inference pixel distributions with what the model was trained on prevents systematic embedding drift |
| **Tensor conversion** | HWC uint8 → CHW float32, values scaled to [0, 1] | PyTorch convention for all vision models |

**Text tokenisation** also goes through the processor: text is lowercased, tokenised with byte-pair encoding (BPE, 49,152-token vocabulary), padded or truncated to exactly **77 tokens** (CLIP's fixed context length), and returned as integer token IDs.

No stochastic augmentations (random flips, colour jitter, random crops, etc.) are applied at inference because they would introduce randomness into predictions with no accuracy benefit. Their role belongs exclusively in a training loop.

### Confidence Scoring

```python
probs = outputs.logits_per_image.softmax(dim=-1)[0]
top_prob, top_idx = probs.max(dim=0)
```

The confidence is the **softmax probability** of the top class — a value in [0, 1] representing how strongly the model favours the winning class over the other ten. It is a **closed-world score**: it reflects relative preference within these 11 categories, not an absolute measure of how food-like or recognisable the image is.

- High score (~0.85+): model is confident and the image is a clear example of one category
- Medium score (~0.30–0.60): image could plausibly belong to multiple categories (e.g. a dish that is both "fried food" and "meat")
- Low score (~0.10–0.20): the model is essentially guessing; the image may be ambiguous, low quality, or not food

---

## CLIP Pretraining: Augmentations and Optimisation

Although no training happens in this script, CLIP's pretrained weights encode everything the model knows about food. Understanding how those weights were produced explains why the model generalises well to Food-11.

### Training Data

CLIP was trained on **WebImageText (WIT)**, a proprietary dataset of approximately **400 million image-text pairs** scraped from the internet. Images were paired with their surrounding alt-text, captions, or titles. The diversity of this data — spanning food photography, restaurant menus, recipe blogs, and cooking videos — is what gives the model strong food-domain priors without any food-specific training.

### Augmentation Strategy

OpenAI applied a deliberately **minimal augmentation strategy** during CLIP pretraining:

| Augmentation | Applied | Rationale |
|---|---|---|
| **Random resized crop** | Yes | Primary spatial augmentation — randomly crops between 75%–100% of the image area and resizes to 224×224, giving the model invariance to scale and position |
| **Random horizontal flip** | No (not in base CLIP) | Avoided in contrastive learning because flipping can break text-image alignment for images with embedded text (e.g. labels, signs) |
| **Colour jitter** | No | Avoided to preserve the colour-text relationships in captions (e.g. "red apple", "golden bread") |
| **Greyscale / Gaussian blur** | No | Same reason — captions often describe colour, so colour information must be preserved |
| **Normalisation** | Yes | Channel-wise mean/std normalisation using the dataset statistics above |

This is much sparser than typical supervised classification augmentation pipelines (which would use all of the above). The constraint is that in contrastive learning the image and its paired caption must still semantically match after augmentation — aggressive visual transforms break that guarantee.

### Optimiser and Hyperparameters

| Hyperparameter | Value | Rationale |
|---|---|---|
| **Optimiser** | Adam | Adaptive learning rates per parameter; well-suited to training transformers with widely varying gradient magnitudes across layers |
| **Learning rate** | 1e-3 peak (5e-4 for ViT-B/32) | High enough for efficient convergence over 400M samples; lower than typical CNNs because transformer training is more sensitive to LR |
| **LR schedule** | Linear warmup → cosine annealing decay | Warmup prevents unstable early gradients; cosine decay smoothly reduces the rate without requiring manual step scheduling |
| **Warmup steps** | ~2,000 | Lets all transformer layers stabilise before the full learning rate is applied |
| **Epochs** | 32 | Enough passes over 400M samples for the contrastive representations to converge |
| **Batch size** | 32,768 | Large batches are critical for contrastive learning — each image is contrasted against all other images in the batch as negatives, so larger batches give a richer and harder negative signal |
| **Weight decay** | 0.2 | Decoupled L2 regularisation (AdamW-style) applied to prevent overfitting on any one domain |
| **Mixed precision** | fp16 | Halves memory usage and speeds up matmul ops on modern GPUs; gradient scaling used to prevent underflow |
| **Gradient clipping** | Applied to temperature | Prevents the logit scale from growing unbounded, which would collapse the softmax into a one-hot and destabilise training |

### Loss Function

CLIP uses a **symmetric cross-entropy contrastive loss**. For a batch of N (image, text) pairs:

1. Compute the N×N matrix of cosine similarities between all image and text embeddings in the batch.
2. Scale by `logit_scale` (a learned temperature).
3. Apply cross-entropy loss **along rows** (each image should match its own text) and **along columns** (each text should match its own image).
4. Average both directions. The model is simultaneously trained to retrieve the right image given a text, and the right text given an image.

The large batch size of 32,768 means the model sees 32,767 hard negatives for each positive pair per step — this forces it to learn genuinely discriminative representations rather than coarse ones.

### Temperature Parameter

`logit_scale` is a **single learnable scalar** (initialised to `log(1/0.07) ≈ 2.659`) that multiplies all cosine similarities before the softmax. A higher temperature makes the probability distribution sharper; a lower temperature makes it flatter.

It is clipped during training so that the effective temperature never exceeds 100 (preventing extreme probability sharpening). At inference the learned value is frozen — this is why CLIP's zero-shot confidence scores are generally well-calibrated without any post-hoc scaling.

---

## Why These Choices?

### Why CLIP?

- **Zero-shot capability**: No labelled training data or GPU time needed to adapt it to Food-11.
- **No task-specific augmentation pipeline**: Augmentation decisions only matter when training. CLIP's weights are frozen; the CLIPProcessor's deterministic preprocessing is all that is needed.
- **No optimiser to configure**: No learning rate, weight decay, scheduler, or batch size decisions at inference time.
- **Handles class imbalance automatically**: Because it never sees the Food-11 label distribution, it cannot overfit to majority classes (unlike a supervised model trained on the imbalanced Food-11 splits).
- **Cross-domain robustness**: Trained on internet images spanning every cuisine, lighting condition, and presentation style.

### Why ViT-B/32?

- **Patch size 32** → 49 patches per image → fast inference, even on CPU
- **Base size** → ~150 MB total model weight, reasonable memory footprint
- Best tested checkpoint for general zero-shot classification

To trade speed for accuracy: change `MODEL_NAME = "openai/clip-vit-base-patch32"` to `"openai/clip-vit-large-patch14"` in `food_recognition.py`.

### Why No Fine-Tuning?

Fine-tuning would require the training split (9,866 images), an augmentation pipeline, an optimiser with tuned hyperparameters, and care to avoid catastrophic forgetting of CLIP's general representations. The zero-shot baseline is a sensible starting point. If accuracy needs improving, the recommended progression is:

1. **Prompt ensembling** (free — no training needed)
2. **Linear probing** — freeze CLIP, train a single linear layer on extracted embeddings
3. **LoRA / adapter fine-tuning** — train small inserted weight matrices while keeping the backbone frozen
4. **Full fine-tuning** — unfreeze all layers with a very low learning rate (~1e-6)

---

## Categories

| Index | Label | Food-11 Training Images |
|---|---|---|
| 0 | Bread | 994 |
| 1 | Dairy product | 429 |
| 2 | Dessert | 1,500 |
| 3 | Egg | 986 |
| 4 | Fried food | 848 |
| 5 | Meat | 1,325 |
| 6 | Noodles/Pasta | 440 |
| 7 | Rice | 280 |
| 8 | Seafood | 855 |
| 9 | Soup | 1,500 |
| 10 | Vegetable/Fruit | 709 |

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

   | Package | Version | Purpose |
   |---|---|---|
   | `torch` | ≥2.1.0 | Tensor ops and GPU acceleration |
   | `torchvision` | ≥0.16.0 | Image transforms used by CLIPProcessor |
   | `transformers` | ≥4.35.0 | Loads `CLIPModel` and `CLIPProcessor` |
   | `huggingface_hub` | ≥0.19.0 | Downloads model weights (~600 MB) on first run |
   | `Pillow` | ≥10.0.0 | Opens and converts images |
   | `accelerate` | ≥0.24.0 | Optimises model loading and device placement |

3. Model weights (~600 MB) are downloaded automatically on first run and cached in `~/.cache/huggingface/hub/`. Subsequent runs load from cache.

4. GPU is used automatically if CUDA is available; falls back to CPU otherwise.

---

## Usage

```bash
python food_recognition.py <path-to-image-folder> [--output results.csv] [--max-images N]
```

| Argument | Default | Description |
|---|---|---|
| `folder` | *(required)* | Path to folder containing images |
| `--output` | `results.csv` | Output CSV file path |
| `--max-images` | *(none)* | Cap the number of images processed |

```bash
# Full evaluation
python food_recognition.py ./dataset --output predictions.csv

# Quick smoke test on 20 images
python food_recognition.py ./dataset --max-images 20
```

### Folder structure for accuracy measurement

```
dataset/
  Bread/
    img1.jpg
  Meat/
    img2.jpg
  Noodles-Pasta/   ← hyphen alias also accepted
    img3.jpg
  6/               ← integer index also accepted (6 = Noodles/Pasta)
    img4.jpg
```

Subfolder names are resolved in this order:
1. **Exact match** — `Bread`, `Meat`, `Soup`, etc.
2. **Hyphen alias** — `Noodles-Pasta` → `Noodles/Pasta`, `Vegetable-Fruit` → `Vegetable/Fruit`
3. **Integer index** — `0` → `Bread` … `10` → `Vegetable/Fruit`

Unrecognised folder names are recorded as `unknown` and excluded from accuracy.

---

## Output

| Column | Description |
|---|---|
| `image` | Relative path to the image |
| `actual` | Ground truth label (or `unknown`) |
| `predicted` | CLIP's predicted category |
| `confidence` | Softmax probability of top class (0–1, 4 d.p.) |
| `correct` | `True` / `False` / `None` if ground truth unknown |

Accuracy is printed to the console when ground truth labels are available.
