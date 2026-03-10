# Problem A: Small Sample Learning for Defect Classification

## Semiconductor Solutions Challenge 2026 — Intel Partnership

---

## 1. Model Approach

### Architecture: ConvNeXt-Base (pretrained ImageNet-21k + ImageNet-1k)
- **Why ConvNeXt-Base**: Modern pure-ConvNet architecture that matches or beats vision transformers on image classification while being efficient at inference. The `fb_in22k_ft_in1k` variant was pretrained on ImageNet-21k (14M images) then fine-tuned on ImageNet-1k, giving extremely strong transfer learning features — critical for our small-sample defect classes.
- **Parameters**: ~88.6M
- **Input resolution**: 384x384 (upscaled from native defect images ~800x1000)
- **Dropout**: 0.4 at classifier head to prevent overfitting on rare classes

### Why Not Other Architectures?
| Model | Params | Accuracy | Inference Speed | Verdict |
|-------|--------|----------|-----------------|---------|
| ResNet-18 | 11M | ~75% | Very fast | Underfits |
| EfficientNet-B2 | 7.7M | ~82% | Fast | Good but limited capacity |
| ConvNeXt-Base | 88.6M | **~90%+** | **Fast (< 200ms)** | **Best accuracy-speed tradeoff** |
| Swin-Base | 87.8M | ~89% | Slower | Transformer overhead |

---

## 2. Handling Class Imbalance

This is the core challenge — the dataset is extremely imbalanced:

| Class | Training Samples | Ratio to Largest |
|-------|-----------------|------------------|
| good | ~5,000 | 1.00x |
| defect8 | ~560 | 0.11x |
| defect10 | ~470 | 0.09x |
| defect5 | ~290 | 0.06x |
| defect9 | ~220 | 0.04x |
| defect1 | ~180 | 0.04x |
| defect2 | ~120 | 0.02x |
| defect4 | ~14 | 0.003x |
| defect3 | ~7 | **0.001x** |

### Multi-Pronged Strategy:

#### a) Weighted Random Sampling
Each training batch is constructed by oversampling rare classes with probability inversely proportional to class frequency. This ensures every batch contains a balanced representation of all defect types.

#### b) Focal Loss (gamma=2.0) with Class-Frequency Alpha
Standard cross-entropy treats all samples equally. Focal Loss down-weights well-classified (easy) samples and focuses on hard, misclassified ones:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

- `gamma=2.0`: Strong focus on hard examples
- `alpha`: Inverse class frequency, normalized to sum to num_classes
- `label_smoothing=0.05`: Prevents overconfident predictions

#### c) Heavy Data Augmentation (albumentations)
- Random flips (horizontal + vertical), 90-degree rotations
- Affine transformations (shift, scale, rotate)
- Gaussian noise and blur
- Random brightness/contrast + CLAHE
- Coarse dropout (simulates partial occlusion)

These augmentations effectively multiply training data 10-50x, especially critical for defect3 (7 samples) and defect4 (14 samples).

#### d) Mixup Regularization (alpha=0.3)
Linearly interpolates between random pairs of images and their labels, creating virtual training examples that smooth decision boundaries.

---

## 3. Training Configuration

```bash
python train_v2.py \
    --model convnext_base.fb_in22k_ft_in1k \
    --img_size 384 \
    --batch_size 64 \
    --epochs 40 \
    --lr 2e-4 \
    --weight_decay 1e-4 \
    --focal_gamma 2.0 \
    --label_smoothing 0.05 \
    --mixup_alpha 0.3 \
    --warmup_epochs 3 \
    --patience 12 \
    --use_ema \
    --amp \
    --num_workers 8
```

### Key Training Features:
- **AdamW optimizer** with weight decay 1e-4
- **Cosine annealing LR** with 3-epoch linear warmup (peak LR=2e-4, min=1e-6)
- **Mixed precision (AMP)**: 2x speedup, lower memory usage
- **Exponential Moving Average (EMA)**: Decay=0.9995, smooths weight updates for better generalization
- **Gradient clipping**: Max norm=1.0 for training stability
- **Early stopping**: Patience=12 epochs
- **Stratified splits**: 70% train / 15% val / 15% test with stratification to preserve class ratios

---

## 4. Evaluation

### Test-Time Augmentation (TTA)
At inference, we average predictions over 5 augmented views of each image:
1. Original
2. Horizontal flip
3. Vertical flip
4. 90-degree rotation
5. Both flips combined

This typically improves accuracy by 1-2% with no additional training.

### Metrics Produced:
- **Overall accuracy** (target: ~85%)
- **Per-class precision, recall, F1-score**
- **Confusion matrix** (visual + CSV)
- **Accuracy vs. class occurrence** scatter plot
- **Learning curve** (accuracy vs. training data size)
- **Per-class metrics** bar chart

---

## 5. Inference Application

```bash
# Single image classification
python inference.py --image path/to/wafer.png --model_path reports/best_model.pt

# Batch directory
python inference.py --image_dir path/to/images/ --model_path reports/best_model.pt

# Benchmark inference speed
python inference.py --benchmark --model_path reports/best_model.pt
```

### Inference Pipeline:
1. Load grayscale image (any resolution up to 1500x2500)
2. Convert to 3-channel (grayscale replicated)
3. Resize to 384x384
4. Normalize with ImageNet statistics
5. Forward pass through ConvNeXt-Base
6. Return: predicted class, confidence, top-3 predictions

### Speed Requirements:
| Resolution | Target | Expected (RTX 5090) |
|-----------|--------|---------------------|
| 384x384 | < 1s | ~15-25 ms |
| 1500x2500 | < 1s | ~100-300 ms |

---

## 6. Hardware Requirements

### Recommended (what we used):
- **GPU**: NVIDIA RTX 5090 (30 GB VRAM)
- **CPU**: Modern multi-core (8+ cores for data loading)
- **RAM**: 32 GB+
- **Storage**: SSD recommended for faster data loading

### Minimum for Training:
- **GPU**: NVIDIA RTX 3060 (12 GB VRAM) — reduce batch_size to 16, img_size to 256
- **GPU**: NVIDIA RTX 4070 (12 GB VRAM) — reduce batch_size to 24, img_size to 320

### Minimum for Inference Only:
- **GPU**: Any NVIDIA GPU with 4+ GB VRAM
- **CPU-only**: Possible but slower (~2-5 seconds per image at 1500x2500)

### Scaling Guide:
| GPU VRAM | Model | Img Size | Batch Size | Expected Accuracy |
|----------|-------|----------|------------|-------------------|
| 4 GB | EfficientNet-B0 | 224 | 32 | ~78-82% |
| 8 GB | EfficientNet-B2 | 260 | 32 | ~82-86% |
| 12 GB | ConvNeXt-Small | 320 | 32 | ~86-89% |
| 16 GB | ConvNeXt-Base | 384 | 32 | ~88-91% |
| 24-30 GB | ConvNeXt-Base | 384 | 64 | **~90-93%** |

---

## 7. Assumptions

1. **Grayscale images**: All images are treated as single-channel grayscale (converted from any input format), then replicated to 3 channels for pretrained model compatibility.
2. **No defect6/defect7**: The provided dataset contains 8 defect classes (1,2,3,4,5,8,9,10) plus "good". Classes 6 and 7 are absent from the training data.
3. **Class imbalance is the primary challenge**: With defect3 having only ~7 training samples vs ~5000 for "good" (714:1 ratio), standard approaches fail. Our multi-pronged strategy (oversampling + focal loss + augmentation + mixup) addresses this.
4. **Transfer learning from natural images works for defect inspection**: ImageNet-pretrained features (edges, textures, patterns) transfer well to semiconductor defect patterns.
5. **Inference hardware**: RTX 5090 or equivalent is available for deployment, enabling real-time classification within the 1-second requirement.

---

## 8. Project Structure

```
threshold/
├── train_v2.py              # Main training pipeline (optimized)
├── inference.py              # Standalone inference application
├── train.py                  # Original baseline training script
├── dataloader.py             # Alternative dataloader
├── SOLUTION.md               # This document
├── TRAINING_GUIDE.md         # Quick-start training guide
├── Data/                     # Dataset
│   ├── good/                 # 7,135 images
│   ├── defect1/              # 253 images
│   ├── defect2/              # 178 images
│   ├── defect3/              # 9 images
│   ├── defect4/              # 14 images
│   ├── defect5/              # 411 images
│   ├── defect8/              # 803 images
│   ├── defect9/              # 319 images
│   └── defect10/             # 674 images
├── reports/                  # Training outputs
│   ├── best_model.pt         # Best model weights
│   ├── best_ema_model.pt     # Best EMA model weights
│   ├── results_summary.json  # Full results summary
│   ├── classification_report.txt
│   ├── confusion_matrix.csv
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   ├── accuracy_vs_occurrence.png
│   ├── per_class_metrics.png
│   ├── learning_curve.png
│   ├── learning_curve.json
│   └── inference_benchmark.json
└── eda_notebook.ipynb        # Exploratory data analysis
```

---

## 9. Key Innovation: Rapid Learning from Few Examples

The problem specifically asks for "small sample learning" — how a model learns from limited data. Our approach demonstrates this through:

1. **Learning curve experiment**: We train on 5%, 10%, 25%, 50%, 75%, and 100% of training data and measure validation accuracy at each point.
2. **Strong pretrained backbone**: ConvNeXt-Base pretrained on ImageNet-21k (14M images) provides rich feature representations that transfer effectively to defect patterns, enabling high accuracy even with very few examples.
3. **Aggressive augmentation**: Multiplies effective training data by 10-50x, critical when defect3 has only ~7 samples.
4. **EMA**: Stabilizes learning and reduces overfitting to rare class noise.

This approach mirrors how a human expert leverages prior experience (pretraining) and learns to recognize new defect patterns from just a handful of examples.
