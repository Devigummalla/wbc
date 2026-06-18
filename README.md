# White Blood Cell Subtype Classification

## A Research-Grade Deep Learning Pipeline Using PyTorch & Transfer Learning

**Author:** [G.J.V.N.Devi]  
**Institution:** [Rajiv Gandhi University of Knowledge Technologies]  
**Date:** June 2026

---

## Abstract

This project implements a complete, research-grade deep learning pipeline for
classifying microscopic white blood cell images into four subtypes: Eosinophil,
Lymphocyte, Monocyte, and Neutrophil. Built exclusively with PyTorch, it
addresses critical methodological flaws found in baseline implementations —
most notably data leakage in train/test splits — and introduces transfer
learning, class-weighted loss, advanced augmentation, rigorous cross-evaluation
metrics, and Grad-CAM++ explainability for clinical trust.

---

## 1. Configuration

All hyperparameters, paths, and random seeds are defined in a single `CONFIG`
dictionary at the top of the notebook. No magic numbers appear anywhere else in
the codebase.

**Why:** Centralizing configuration ensures reproducibility and makes
hyperparameter sweeps straightforward. It also prevents the common anti-pattern
of hardcoding values deep inside training loops, which makes debugging and
comparison across experiments extremely difficult.

**Alternatives rejected:** YAML/JSON config files were considered but rejected
for notebook simplicity — a single dictionary keeps the notebook self-contained.

---

## 2. Environment & Imports

The pipeline uses the following core libraries:

| Library | Purpose |
|---------|---------|
| `torch`, `torchvision` | Core deep learning framework and data utilities |
| `timm` | State-of-the-art pretrained vision models |
| `torchvision.transforms.v2` | Modern, composable augmentation API |
| `numpy`, `pandas` | Numerical computation and data handling |
| `matplotlib`, `seaborn` | Publication-quality visualizations |
| `sklearn` | Metrics, class weights, stratified splitting |
| `pytorch-grad-cam` | Grad-CAM++ explainability |
| `tqdm` | Training progress bars |

**Why PyTorch exclusively:** PyTorch is the dominant framework in academic
research (>80% of NeurIPS/ICML papers as of 2025). Its eager execution model
makes debugging transparent, and the `timm` ecosystem provides access to
hundreds of pretrained architectures.

**Alternatives rejected:** TensorFlow/Keras was rejected due to its declining
adoption in academic research and less transparent computational graph
debugging.

---

## 3. Reproducibility

A `set_seed()` function seeds all sources of randomness:
- Python's `random` module
- NumPy's random generator
- PyTorch CPU and CUDA generators
- CuDNN deterministic and benchmark flags
- Python's `PYTHONHASHSEED` environment variable

**Why:** Reproducibility is a cornerstone of scientific methodology. Without
deterministic seeding, results vary across runs, making it impossible to
attribute performance changes to architectural decisions vs. random
initialization variance.

---

## 4. Dataset Loading

The dataset follows the standard `ImageFolder` layout with four class
subdirectories. Before any splitting, we print a complete class distribution
table to understand the baseline imbalance.

**Why:** Understanding class proportions before training is essential for
designing appropriate loss functions and augmentation strategies. Medical
datasets are frequently imbalanced, and ignoring this leads to models that
trivially predict the majority class.

---

## 5. Data Pipeline — Zero Leakage (Critical Module)

### The Problem

The baseline notebook performed two independent random splits on the full
dataset:

```python
# FLAWED BASELINE:
train_images, test_images = train_test_split(bloodCell_df, test_size=0.3)
train_set, val_set = train_test_split(bloodCell_df, test_size=0.2)  # LEAK!
```

This caused massive overlap between training and evaluation sets. The reported
98.5% accuracy was an artifact of the model being tested on images it had
already memorized.

### Our Solution

We perform a **sequential, stratified, index-level split**:

1. First split: 85% remaining / 15% test
2. Second split (from remaining): ~82.4% train / ~17.6% val

This guarantees:
- **Zero overlap** between any pair of splits (verified programmatically)
- **Preserved class proportions** across all splits via `stratify=labels`

A `TransformSubset` wrapper applies split-specific transforms: full
augmentation for training, deterministic resize+normalize for val/test.

**Why stratified:** In medical imaging, minority classes may comprise <10% of
the dataset. A naive random split could leave certain classes with zero
representation in the test set, producing undefined precision/recall metrics.

**Alternatives rejected:** K-Fold cross-validation was considered but rejected
for the initial implementation to maintain simplicity. It can be layered on top
of this pipeline for future work.

---

## 6. Augmentation Pipeline

Our augmentation strategy is **domain-informed**, designed specifically for
hematology microscopy imaging:

| Transform | Rationale |
|-----------|-----------|
| `RandomHorizontalFlip` | Cells have no canonical orientation |
| `RandomVerticalFlip` | Cells have no canonical orientation |
| `RandomRotation(±30°)` | Microscope slide rotation variance |
| `RandomResizedCrop` | Simulate varying magnification levels |
| `ColorJitter` | **Simulate Giemsa/Wright staining variance** — the primary source of inter-laboratory distribution shift in hematology imaging |
| `GaussianBlur` | Simulate microscope focus plane variance |

All images are normalized using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) to match the pretrained backbone's expected input distribution.

**Critical design choice:** Augmentation is applied **exclusively** to the
training split. Validation and test transforms are deterministic (resize +
normalize only). Augmenting evaluation data would invalidate the metrics.

**Alternatives rejected:** CutMix and MixUp were considered but rejected for
this medical imaging application, as they create biologically implausible
hybrid images that could confuse clinical interpretation of results.

---

## 7. Class Imbalance Handling

We compute inverse-frequency class weights using sklearn's
`compute_class_weight(class_weight='balanced')` and pass them directly into
`nn.CrossEntropyLoss(weight=...)`.

**Why:** Without weighting, the loss function treats all classes equally. If
Eosinophils comprise 15% of the dataset while Neutrophils comprise 40%, the
model learns to bias toward Neutrophils simply because they contribute more to
the total loss. Class weighting ensures equal penalization per class.

**Alternatives rejected:**
- **Oversampling (SMOTE/RandomOverSampler):** Duplicates or synthesizes images
  rather than creating genuine augmentation diversity. Our augmentation pipeline
  already provides variation.
- **Focal Loss:** Effective for extreme imbalance (>1:100 ratios) but
  unnecessarily complex for our moderate imbalance scenario. Class-weighted
  cross-entropy is simpler and sufficient.

---

## 8. Model Architecture

We use **EfficientNetV2-B0** loaded via the `timm` library with ImageNet-1K
pretrained weights.

### Why EfficientNetV2?

| Criterion | EfficientNetV2-B0 | ResNet-50 | VGG-16 |
|-----------|------------------|-----------|--------|
| Parameters | ~7.1M | ~25.6M | ~138M |
| Top-1 (ImageNet) | ~78.7% | ~76.1% | ~71.6% |
| FLOPs | ~0.72B | ~4.1B | ~15.5B |
| Architecture | Compound scaling + Fused-MBConv | Skip connections | Plain sequential |

EfficientNetV2 achieves superior accuracy with significantly fewer parameters
and FLOPs, making it ideal for medical imaging where datasets are often limited
and overfitting is a primary concern.

### Custom Classification Head

The default classifier is replaced with:
```
AdaptiveAvgPool2d(1) → Flatten → Dropout(0.4) → Linear(num_features, 4)
```

**Why Dropout=0.4:** Medical imaging models are particularly prone to
overfitting due to limited dataset sizes. A dropout rate of 0.4 provides
strong regularization without excessively limiting model capacity.

**Alternatives rejected:**
- **Training a CNN from scratch:** Requires orders of magnitude more data to
  learn robust low-level features. Transfer learning leverages billions of
  gradient updates from ImageNet.
- **Vision Transformers (ViT):** Require larger datasets to outperform CNNs
  and are computationally expensive. EfficientNetV2 provides the best
  accuracy/efficiency tradeoff for our dataset scale.

---

## 9. Two-Phase Training Strategy

### Phase 1: Head-Only Training

The entire backbone is frozen. Only the custom classification head (Dropout +
Linear) is trained. This allows the head to adapt to the WBC feature space
without disturbing the backbone's pretrained features.

- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingLR
- **Early Stopping:** Patience=5 on validation loss

### Phase 2: Fine-Tuning

The top 30 layers of the backbone are unfrozen and trained end-to-end with a
**100× lower learning rate** (1e-5) to prevent catastrophic forgetting.

- **Optimizer:** AdamW (lr=1e-5, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingWarmRestarts (T_0=5) for cyclical learning
- **Gradient Clipping:** `max_norm=1.0` — stabilizes training when unfreezing
  deep layers with large gradients
- **Early Stopping:** Patience=7 (more lenient since fine-tuning converges
  more slowly)

**Why two phases?** (Raghu et al., 2019; Kornblith et al., 2019) demonstrated
that training the head first before fine-tuning produces consistently better
results than end-to-end training from the start, especially with limited data.
The head learns a reasonable decision boundary before the backbone features are
adjusted.

**Why CosineAnnealingWarmRestarts in Phase 2?** The periodic warm restarts help
escape sharp local minima that fine-tuning often encounters, producing flatter
minima with better generalization (Loshchilov & Hutter, 2017).

---

## 10. Training History Visualization

A combined 1×2 figure displays:
- **Left:** Training Loss vs. Validation Loss
- **Right:** Training Accuracy vs. Validation Accuracy

A vertical dashed line marks the Phase 1 → Phase 2 boundary, giving a clear
visual narrative of the entire training journey.

---

## 11. Evaluation Metrics

### 11a. Classification Report

Per-class Precision, Recall, and F1-Score, plus **Macro F1-Score** as the
headline metric.

**Why Macro F1 over Accuracy?** Accuracy is misleading with class imbalance. A
model that predicts only the majority class achieves high accuracy but zero
recall on minority classes. Macro F1 weighs all classes equally regardless of
frequency.

### 11b. Confusion Matrix

Seaborn annotated heatmap with class names on both axes. Misclassifications
are immediately visible as off-diagonal hot spots.

### 11c. Multi-Class ROC-AUC

One-vs-Rest (OvR) ROC curves for each class, plotted on a single figure with
AUC scores annotated per curve. A random classifier baseline (diagonal) is
included for reference.

**Why OvR?** It decomposes the 4-class problem into 4 binary classification
problems, making it straightforward to identify which specific classes the
model struggles to distinguish.

---

## 12. Explainable AI — Grad-CAM++

### Method

Grad-CAM++ (Chattopadhay et al., 2018) generates class-discriminative
localization maps by computing a weighted combination of the feature maps in
the last convolutional layer, using the gradients as weights. Grad-CAM++ is an
improvement over vanilla Grad-CAM, providing better localization for
multi-instance and partial object visibility scenarios.

### Clinical Interpretation

| Cell Type | Expected Model Attention | Clinical Significance |
|-----------|-------------------------|----------------------|
| Eosinophil | Bilobed nucleus + eosinophilic granules | Granules contain major basic protein (MBP); bilobed nucleus is the primary diagnostic criterion |
| Lymphocyte | Large, round, darkly-stained nucleus | High nucleus-to-cytoplasm ratio (~90:10) is the morphological hallmark |
| Monocyte | Kidney/horseshoe-shaped nucleus | Largest circulating WBC; indented nucleus is the primary diagnostic feature |
| Neutrophil | Multi-lobed (3-5) segmented nucleus | Lobe count is clinically significant — hypersegmentation (>5 lobes) indicates megaloblastic anemia or B12/folate deficiency |

If the model's attention diverges significantly from these expected regions
(e.g., focusing on background staining artifacts or red blood cells in the
field), this is a strong signal of overfitting to non-diagnostic features and
warrants investigation into data quality and augmentation strategy.

---

## Requirements

```
torch>=2.0
torchvision>=0.15
timm>=0.9
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
tqdm
grad-cam
```

Install with:
```bash
pip install torch torchvision timm numpy pandas matplotlib seaborn scikit-learn Pillow tqdm grad-cam
```

---

## References

1. Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training. *ICML 2021*.
2. Chattopadhay, A., et al. (2018). Grad-CAM++: Generalized Gradient-Based Visual Explanations. *WACV 2018*.
3. Raghu, M., et al. (2019). Transfusion: Understanding Transfer Learning for Medical Imaging. *NeurIPS 2019*.
4. Kornblith, S., et al. (2019). Do Better ImageNet Models Transfer Better? *CVPR 2019*.
5. Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *ICLR 2017*.
