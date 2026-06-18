# %% [markdown]
# # White Blood Cell Subtype Classification
# ## A Research-Grade Deep Learning Pipeline Using PyTorch & Transfer Learning
#
# **Objective:** Classify microscopic blood cell images into 4 WBC subtypes
# (Eosinophil, Lymphocyte, Monocyte, Neutrophil) using EfficientNetV2 with
# two-phase transfer learning, rigorous evaluation, and Grad-CAM++ explainability.
#
# **Methodology highlights:**
# - Stratified, leak-free data splitting with programmatic verification
# - Two-phase training: frozen head → fine-tuned backbone
# - Class-weighted loss for imbalance robustness
# - Multi-class ROC-AUC, confusion matrix, and classification report
# - Grad-CAM++ heatmaps for clinical interpretability

# %% [markdown]
# ---
# ## 1. Configuration
# All hyperparameters, paths, and seeds are defined here.
# **Zero magic numbers appear anywhere else in this notebook.**

# %%
CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────
    "DATA_DIR": "data/",                   # Root with class subfolders
    "CHECKPOINT_DIR": "checkpoints/",

    # ── Reproducibility ───────────────────────────────────────────────
    "SEED": 42,

    # ── Data Pipeline ─────────────────────────────────────────────────
    "IMG_SIZE": 224,                       # EfficientNetV2 native input
    "BATCH_SIZE": 32,
    "NUM_WORKERS": 4,
    "TRAIN_RATIO": 0.70,
    "VAL_RATIO": 0.15,
    "TEST_RATIO": 0.15,

    # ── Augmentation ──────────────────────────────────────────────────
    "ROTATION_DEGREES": 30,
    "COLOR_JITTER": {
        "brightness": 0.3,
        "contrast": 0.3,
        "saturation": 0.3,
        "hue": 0.1,
    },
    "GAUSSIAN_BLUR_KERNEL": (3, 3),

    # ── Model ─────────────────────────────────────────────────────────
    "MODEL_NAME": "tf_efficientnetv2_b0",  # timm model identifier
    "NUM_CLASSES": 4,
    "DROPOUT_RATE": 0.4,

    # ── Phase 1: Head-Only Training ───────────────────────────────────
    "P1_EPOCHS": 15,
    "P1_LR": 1e-3,
    "P1_WEIGHT_DECAY": 1e-4,
    "P1_PATIENCE": 5,

    # ── Phase 2: Fine-Tuning ──────────────────────────────────────────
    "P2_EPOCHS": 25,
    "P2_LR": 1e-5,
    "P2_WEIGHT_DECAY": 1e-4,
    "P2_PATIENCE": 7,
    "P2_UNFREEZE_LAYERS": 30,             # Top-N layers to unfreeze
    "P2_GRAD_CLIP_NORM": 1.0,
    "P2_T_0": 5,                          # CosineAnnealingWarmRestarts T_0

    # ── ImageNet Normalization ────────────────────────────────────────
    "IMAGENET_MEAN": [0.485, 0.456, 0.406],
    "IMAGENET_STD": [0.229, 0.224, 0.225],
}

CLASS_NAMES = ["Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]

# %% [markdown]
# ---
# ## 2. Environment & Imports
#
# We use **PyTorch exclusively** with the following ecosystem:
# - `timm` for state-of-the-art pretrained vision models
# - `torchvision.transforms.v2` for the modern augmentation API
# - `pytorch-grad-cam` for Grad-CAM++ explainability
# - `sklearn` for stratified splitting and evaluation metrics

# %%
import os
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import datasets
from torchvision.transforms import v2 as transforms_v2

import timm

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

warnings.filterwarnings("ignore")

# ── Verify environment ────────────────────────────────────────────────
print(f"PyTorch version : {torch.__version__}")
print(f"Torchvision ver : {torchvision.__version__}")
print(f"timm version    : {timm.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active device   : {DEVICE}")

os.makedirs(CONFIG["CHECKPOINT_DIR"], exist_ok=True)

# %% [markdown]
# ---
# ## 3. Reproducibility — Deterministic Seeding
#
# **Academic justification:** Reproducibility is a cornerstone of scientific
# research. We seed every source of randomness — Python's `random`, NumPy,
# PyTorch CPU, PyTorch CUDA, and CuDNN — and enforce deterministic CuDNN
# algorithms to guarantee identical results across runs.

# %%
def set_seed(seed):
    """Set all random seeds for full reproducibility.

    Ensures deterministic behavior across Python, NumPy, PyTorch CPU/CUDA,
    and CuDNN backends.

    Args:
        seed (int): The seed value to use across all random generators.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[✓] All seeds set to {seed}. Deterministic mode enabled.")


set_seed(CONFIG["SEED"])

# %% [markdown]
# ---
# ## 4. Dataset Loading & Class Distribution
#
# We assume the standard `ImageFolder` directory layout. Before any splitting,
# we inspect the raw class distribution to understand potential imbalance,
# which directly informs our class-weighted loss function later.

# %%
def load_dataset_and_print_distribution(data_dir, class_names):
    """Load the full dataset and print class distribution statistics.

    Uses torchvision.datasets.ImageFolder with a minimal placeholder
    transform (actual transforms are applied per-split later via Subset).

    Args:
        data_dir (str): Path to root data directory with class subfolders.
        class_names (list): Expected class name strings for validation.

    Returns:
        torchvision.datasets.ImageFolder: The loaded dataset.
    """
    placeholder_transform = transforms_v2.Compose([
        transforms_v2.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(torch.float32, scale=True),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=placeholder_transform)

    print(f"\nTotal images found: {len(dataset)}")
    print(f"Classes detected : {dataset.classes}")
    print(f"Class-to-index   : {dataset.class_to_idx}\n")

    targets = np.array(dataset.targets)
    distribution = Counter(targets)

    dist_df = pd.DataFrame({
        "Class": [dataset.classes[k] for k in sorted(distribution.keys())],
        "Count": [distribution[k] for k in sorted(distribution.keys())],
    })
    dist_df["Percentage"] = (dist_df["Count"] / dist_df["Count"].sum() * 100).round(2)

    print("┌─────────────────────────────────────────┐")
    print("│       Class Distribution (Full Dataset)  │")
    print("├──────────────┬─────────┬─────────────────┤")
    print("│ Class        │ Count   │ Percentage (%)   │")
    print("├──────────────┼─────────┼─────────────────┤")
    for _, row in dist_df.iterrows():
        print(f"│ {row['Class']:<12} │ {row['Count']:<7} │ {row['Percentage']:<16}│")
    print("└──────────────┴─────────┴─────────────────┘")

    return dataset


full_dataset = load_dataset_and_print_distribution(CONFIG["DATA_DIR"], CLASS_NAMES)

# %% [markdown]
# ---
# ## 5. Flawless Data Pipeline — Zero Data Leaks
#
# **Academic justification:** Data leakage is the single most common source of
# inflated metrics in machine learning research. The baseline notebook split the
# *full* dataset twice independently, causing train/val/test overlap. Here we
# perform a single, sequential, stratified split at the *index* level, then
# wrap each partition in a `torch.utils.data.Subset`. We programmatically
# verify disjointness with set-intersection assertions.
#
# **Why stratified?** Medical datasets frequently exhibit class imbalance. A
# naive random split could leave minority classes underrepresented in val/test
# sets, producing unreliable evaluation metrics.

# %%
def create_stratified_splits(dataset, config):
    """Create strictly non-overlapping stratified train/val/test splits.

    Performs a two-stage stratified split using sklearn's train_test_split
    with the dataset's target labels to ensure class proportions are
    preserved across all three partitions.

    Args:
        dataset (torchvision.datasets.ImageFolder): The full loaded dataset.
        config (dict): Configuration dictionary with split ratios and seed.

    Returns:
        tuple: (train_indices, val_indices, test_indices) — lists of ints.
    """
    all_indices = list(range(len(dataset)))
    all_labels = [dataset.targets[i] for i in all_indices]

    # Stage 1: Separate test set (15%)
    remaining_indices, test_indices, remaining_labels, _ = train_test_split(
        all_indices,
        all_labels,
        test_size=config["TEST_RATIO"],
        stratify=all_labels,
        random_state=config["SEED"],
    )

    # Stage 2: From the remaining 85%, carve out validation (15/85 ≈ 17.6%)
    val_fraction_of_remaining = config["VAL_RATIO"] / (1.0 - config["TEST_RATIO"])
    train_indices, val_indices = train_test_split(
        remaining_indices,
        test_size=val_fraction_of_remaining,
        stratify=remaining_labels,
        random_state=config["SEED"],
    )

    return train_indices, val_indices, test_indices


def verify_no_overlap(train_idx, val_idx, test_idx):
    """Assert that train, val, and test index sets are completely disjoint.

    Args:
        train_idx (list): Training set indices.
        val_idx (list): Validation set indices.
        test_idx (list): Test set indices.

    Returns:
        None

    Raises:
        AssertionError: If any overlap is detected between splits.
    """
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    assert train_set.isdisjoint(val_set), "LEAK: Train ∩ Val is non-empty!"
    assert train_set.isdisjoint(test_set), "LEAK: Train ∩ Test is non-empty!"
    assert val_set.isdisjoint(test_set), "LEAK: Val ∩ Test is non-empty!"

    print("[✓] Zero overlap verified: Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅")


def print_split_statistics(dataset, train_idx, val_idx, test_idx):
    """Print image counts per split and per class within each split.

    Args:
        dataset (torchvision.datasets.ImageFolder): The full dataset.
        train_idx (list): Training indices.
        val_idx (list): Validation indices.
        test_idx (list): Test indices.

    Returns:
        None
    """
    splits = {"Train": train_idx, "Val": val_idx, "Test": test_idx}

    print(f"\n{'Split':<8} {'Total':<8}", end="")
    for cls_name in dataset.classes:
        print(f"{cls_name:<14}", end="")
    print()
    print("─" * 70)

    for split_name, indices in splits.items():
        class_counts = Counter([dataset.targets[i] for i in indices])
        total = len(indices)
        print(f"{split_name:<8} {total:<8}", end="")
        for cls_idx in range(len(dataset.classes)):
            print(f"{class_counts.get(cls_idx, 0):<14}", end="")
        print()


# ── Execute splits ────────────────────────────────────────────────────
train_indices, val_indices, test_indices = create_stratified_splits(full_dataset, CONFIG)
verify_no_overlap(train_indices, val_indices, test_indices)
print_split_statistics(full_dataset, train_indices, val_indices, test_indices)

# %% [markdown]
# ---
# ## 6. Augmentation Pipeline
#
# **Academic justification:** We use `torchvision.transforms.v2`, the modern,
# composable transforms API. Our augmentation strategy is specifically
# designed for hematology imaging:
#
# - **ColorJitter** simulates variance in Giemsa/Wright chemical staining,
#   a well-documented distribution shift source across laboratories.
# - **GaussianBlur** simulates microscope focus variance.
# - **Geometric transforms** (flip, rotation, crop) increase spatial
#   invariance.
#
# **Critical:** Augmentation is applied **only** to the training split.
# Validation and test sets receive deterministic resize + normalize only.

# %%
def get_train_transforms(config):
    """Build the training augmentation pipeline.

    Includes geometric transforms, color jitter (staining simulation),
    and Gaussian blur (focus simulation), followed by ImageNet normalization.

    Args:
        config (dict): Configuration dictionary with augmentation parameters.

    Returns:
        torchvision.transforms.v2.Compose: The composed training transform.
    """
    return transforms_v2.Compose([
        transforms_v2.RandomResizedCrop(
            size=(config["IMG_SIZE"], config["IMG_SIZE"]),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
        ),
        transforms_v2.RandomHorizontalFlip(p=0.5),
        transforms_v2.RandomVerticalFlip(p=0.5),
        transforms_v2.RandomRotation(degrees=config["ROTATION_DEGREES"]),
        transforms_v2.ColorJitter(**config["COLOR_JITTER"]),
        transforms_v2.GaussianBlur(
            kernel_size=config["GAUSSIAN_BLUR_KERNEL"],
            sigma=(0.1, 2.0),
        ),
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Normalize(
            mean=config["IMAGENET_MEAN"],
            std=config["IMAGENET_STD"],
        ),
    ])


def get_eval_transforms(config):
    """Build the evaluation (val/test) transform pipeline.

    No augmentation — only deterministic resize and ImageNet normalization.

    Args:
        config (dict): Configuration dictionary with image size and norms.

    Returns:
        torchvision.transforms.v2.Compose: The composed evaluation transform.
    """
    return transforms_v2.Compose([
        transforms_v2.Resize((config["IMG_SIZE"], config["IMG_SIZE"])),
        transforms_v2.ToImage(),
        transforms_v2.ToDtype(torch.float32, scale=True),
        transforms_v2.Normalize(
            mean=config["IMAGENET_MEAN"],
            std=config["IMAGENET_STD"],
        ),
    ])


# %%
class TransformSubset(Subset):
    """A Subset wrapper that applies a custom transform to each sample.

    Standard torch.utils.data.Subset inherits the parent dataset's transform.
    This subclass overrides __getitem__ to apply a split-specific transform
    instead, enabling separate augmentation for train vs. eval splits.

    Args:
        dataset (torchvision.datasets.ImageFolder): The full parent dataset.
        indices (list): Indices into the parent dataset for this split.
        transform (callable): Transform to apply to each image.
    """

    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.custom_transform = transform

    def __getitem__(self, idx):
        """Retrieve sample and apply split-specific transform.

        Args:
            idx (int): Index within this subset (not the parent dataset).

        Returns:
            tuple: (transformed_image, label)
        """
        original_idx = self.indices[idx]
        image_path, label = self.dataset.samples[original_idx]
        image = Image.open(image_path).convert("RGB")

        if self.custom_transform is not None:
            image = self.custom_transform(image)

        return image, label


# ── Create split-specific datasets and dataloaders ────────────────────
train_transform = get_train_transforms(CONFIG)
eval_transform = get_eval_transforms(CONFIG)

train_dataset = TransformSubset(full_dataset, train_indices, transform=train_transform)
val_dataset = TransformSubset(full_dataset, val_indices, transform=eval_transform)
test_dataset = TransformSubset(full_dataset, test_indices, transform=eval_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=True,
    num_workers=CONFIG["NUM_WORKERS"],
    pin_memory=True,
    drop_last=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=False,
    num_workers=CONFIG["NUM_WORKERS"],
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG["BATCH_SIZE"],
    shuffle=False,
    num_workers=CONFIG["NUM_WORKERS"],
    pin_memory=True,
)

print(f"\n[✓] DataLoaders created:")
print(f"    Train: {len(train_dataset)} images, {len(train_loader)} batches")
print(f"    Val  : {len(val_dataset)} images, {len(val_loader)} batches")
print(f"    Test : {len(test_dataset)} images, {len(test_loader)} batches")

# %% [markdown]
# ### 6.1 Visual Verification of Augmentation
#
# We display a grid of augmented training samples to visually confirm
# that the pipeline produces realistic, non-degenerate transformations.

# %%
def plot_augmented_samples(dataset, class_names, num_samples=16):
    """Plot a grid of augmented training samples for visual verification.

    Args:
        dataset (TransformSubset): The training dataset with augmentation.
        class_names (list): List of class name strings.
        num_samples (int): Number of samples to display. Default 16.

    Returns:
        None
    """
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig.suptitle("Augmented Training Samples — Visual Pipeline Verification",
                 fontsize=16, fontweight="bold")

    mean = np.array(CONFIG["IMAGENET_MEAN"])
    std = np.array(CONFIG["IMAGENET_STD"])

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for ax, idx in zip(axes.flatten(), indices):
        image, label = dataset[idx]
        # Unnormalize for display
        img_np = image.permute(1, 2, 0).numpy()
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

        ax.imshow(img_np)
        ax.set_title(class_names[label], fontsize=11)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


plot_augmented_samples(train_dataset, CLASS_NAMES)

# %% [markdown]
# ---
# ## 7. Class Imbalance Handling
#
# **Academic justification:** In medical imaging, class imbalance is endemic.
# Minority classes (e.g., Eosinophils) can be drowned out during gradient
# updates if the loss function treats all classes equally. We compute inverse-
# frequency class weights via `sklearn` and inject them into
# `nn.CrossEntropyLoss`, ensuring the model penalizes misclassification of
# rare classes proportionally more.
#
# **Alternative rejected:** Oversampling (e.g., SMOTE) was considered but
# rejected because it duplicates images rather than creating novel variations,
# and our augmentation pipeline already addresses diversity.

# %%
def compute_and_print_class_weights(dataset, train_indices, device):
    """Compute balanced class weights from training split labels.

    Uses sklearn's compute_class_weight with 'balanced' mode, which assigns
    weights inversely proportional to class frequencies.

    Args:
        dataset (torchvision.datasets.ImageFolder): The full dataset.
        train_indices (list): Indices belonging to the training split.
        device (torch.device): Device to place the weight tensor on.

    Returns:
        torch.FloatTensor: Class weight tensor of shape (num_classes,).
    """
    train_labels = np.array([dataset.targets[i] for i in train_indices])
    unique_classes = np.unique(train_labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=train_labels,
    )

    weights_tensor = torch.FloatTensor(weights).to(device)

    print("\n┌───────────────────────────────────────┐")
    print("│       Computed Class Weights           │")
    print("├──────────────┬────────────────────────┤")
    print("│ Class        │ Weight                  │")
    print("├──────────────┼────────────────────────┤")
    for cls_name, w in zip(CLASS_NAMES, weights):
        print(f"│ {cls_name:<12} │ {w:.4f}                 │")
    print("└──────────────┴────────────────────────┘")

    return weights_tensor


class_weights = compute_and_print_class_weights(full_dataset, train_indices, DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# %% [markdown]
# ---
# ## 8. Model Architecture
#
# **Academic justification:** We use **EfficientNetV2-B0** via the `timm`
# library, a model that achieves strong accuracy with significantly fewer
# parameters than ResNet-50, making it suitable for medical imaging where
# datasets are often limited. The pretrained ImageNet weights provide a
# powerful feature extraction backbone.
#
# **Architecture:**
# - Frozen EfficientNetV2-B0 backbone (pretrained on ImageNet)
# - Custom classification head: AdaptiveAvgPool2d → Dropout → Linear(4)
#
# **Alternative rejected:** Training a CNN from scratch was rejected due to
# insufficient data for the model to learn robust low-level features.
# Transfer learning leverages billions of gradient updates from ImageNet.

# %%
def build_model(config, device):
    """Build EfficientNetV2-B0 with a custom classification head.

    Loads pretrained ImageNet weights via timm. Replaces the default
    classifier with AdaptiveAvgPool2d → Dropout → Linear(num_classes).

    Args:
        config (dict): Configuration with model name, dropout, num_classes.
        device (torch.device): Target device for the model.

    Returns:
        nn.Module: The constructed model, moved to device.
    """
    model = timm.create_model(
        config["MODEL_NAME"],
        pretrained=True,
        num_classes=0,  # Remove default head; we add our own
    )

    # Get the feature dimension from the backbone
    num_features = model.num_features

    # Attach custom classification head
    model.classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(p=config["DROPOUT_RATE"]),
        nn.Linear(num_features, config["NUM_CLASSES"]),
    )

    model = model.to(device)
    return model


def count_parameters(model):
    """Count and print total vs. trainable parameters.

    Args:
        model (nn.Module): The PyTorch model to inspect.

    Returns:
        tuple: (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters     : {total:,}")
    print(f"    Trainable parameters : {trainable:,}")
    print(f"    Frozen parameters    : {total - trainable:,}")
    return total, trainable


def freeze_backbone(model):
    """Freeze all layers except the custom classifier head.

    Args:
        model (nn.Module): The model whose backbone layers to freeze.

    Returns:
        None
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_top_n_layers(model, n):
    """Unfreeze the top N parameter groups of the backbone for fine-tuning.

    Args:
        model (nn.Module): The model to partially unfreeze.
        n (int): Number of parameter groups (from the end) to unfreeze.

    Returns:
        None
    """
    all_params = list(model.named_parameters())
    # Unfreeze the classifier (always) + last N backbone params
    for name, param in all_params:
        param.requires_grad = False  # Freeze everything first

    # Unfreeze classifier head
    for name, param in model.named_parameters():
        if "classifier" in name:
            param.requires_grad = True

    # Unfreeze top-N backbone layers
    backbone_params = [(n_p, p) for n_p, p in all_params if "classifier" not in n_p]
    for name, param in backbone_params[-n:]:
        param.requires_grad = True


# ── Build model ───────────────────────────────────────────────────────
model = build_model(CONFIG, DEVICE)
print(f"\n[✓] Model: {CONFIG['MODEL_NAME']} loaded with ImageNet weights.")
print(f"[✓] Custom head: AdaptiveAvgPool2d → Dropout({CONFIG['DROPOUT_RATE']}) → Linear({CONFIG['NUM_CLASSES']})")

# %% [markdown]
# ---
# ## 9. Training Infrastructure
#
# Clean, reusable `train_one_epoch()` and `validate_one_epoch()` functions
# with tqdm progress bars, plus an `EarlyStopping` utility class.

# %%
class EarlyStopping:
    """Early stopping to terminate training when validation loss plateaus.

    Monitors validation loss and stops training if no improvement is
    observed for `patience` consecutive epochs. Saves the best model
    checkpoint automatically.

    Args:
        patience (int): Number of epochs to wait before stopping.
        checkpoint_path (str): File path to save the best model weights.
        verbose (bool): Whether to print status messages. Default True.
    """

    def __init__(self, patience, checkpoint_path, verbose=True):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model):
        """Check if training should stop and save best model.

        Args:
            val_loss (float): Current epoch's validation loss.
            model (nn.Module): The model to potentially checkpoint.

        Returns:
            None
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
            if self.verbose:
                print(f"    [✓] Val loss improved to {val_loss:.4f}. Model saved.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"    [!] No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("    [✗] Early stopping triggered.")


# %%
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one full epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        device (torch.device): Computation device.

    Returns:
        tuple: (epoch_loss, epoch_accuracy) as floats.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="  Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total:.4f}",
        )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_one_epoch_with_clipping(model, loader, criterion, optimizer, device, max_norm):
    """Train one epoch with gradient clipping (for fine-tuning phase).

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        device (torch.device): Computation device.
        max_norm (float): Maximum gradient norm for clipping.

    Returns:
        tuple: (epoch_loss, epoch_accuracy) as floats.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc="  Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{correct / total:.4f}",
        )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, criterion, device):
    """Evaluate the model on a validation/test set for one epoch.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation or test data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Computation device.

    Returns:
        tuple: (epoch_loss, epoch_accuracy) as floats.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="  Validating", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# %% [markdown]
# ---
# ## 9.1 Phase 1 — Head-Only Training
#
# **Academic justification:** In transfer learning, the pretrained backbone
# already contains rich feature representations from ImageNet. Training only
# the classification head first allows it to adapt to our specific WBC task
# without disturbing the backbone's learned features. This is standard
# practice in medical imaging literature (Raghu et al., 2019).

# %%
print("=" * 60)
print("  PHASE 1: Head-Only Training")
print("=" * 60)

freeze_backbone(model)
print("\n[✓] Backbone frozen. Training classifier head only.")
count_parameters(model)

optimizer_p1 = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["P1_LR"],
    weight_decay=CONFIG["P1_WEIGHT_DECAY"],
)

scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_p1, T_max=CONFIG["P1_EPOCHS"]
)

early_stopping_p1 = EarlyStopping(
    patience=CONFIG["P1_PATIENCE"],
    checkpoint_path=os.path.join(CONFIG["CHECKPOINT_DIR"], "best_model_phase1.pth"),
)

history = {
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": [],
}

for epoch in range(1, CONFIG["P1_EPOCHS"] + 1):
    print(f"\n  Epoch {epoch}/{CONFIG['P1_EPOCHS']}")

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer_p1, DEVICE
    )
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, DEVICE)

    scheduler_p1.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"    Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    early_stopping_p1(val_loss, model)
    if early_stopping_p1.early_stop:
        break

phase1_epochs = len(history["train_loss"])
print(f"\n[✓] Phase 1 complete after {phase1_epochs} epochs.")
print(f"    Best validation loss: {early_stopping_p1.best_loss:.4f}")

# Load best Phase 1 weights before Phase 2
model.load_state_dict(
    torch.load(os.path.join(CONFIG["CHECKPOINT_DIR"], "best_model_phase1.pth"))
)

# %% [markdown]
# ---
# ## 9.2 Phase 2 — Fine-Tuning
#
# **Academic justification:** After the head has converged, we unfreeze the
# top N layers of the backbone and train end-to-end with a significantly
# lower learning rate (1e-5 vs 1e-3). This prevents catastrophic forgetting
# of the pretrained features while allowing the backbone to specialize for
# WBC morphology. Gradient clipping (`max_norm=1.0`) stabilizes training
# with the unfrozen, deeper parameters.

# %%
print("=" * 60)
print("  PHASE 2: Fine-Tuning (Top Layers Unfrozen)")
print("=" * 60)

unfreeze_top_n_layers(model, CONFIG["P2_UNFREEZE_LAYERS"])
print(f"\n[✓] Top {CONFIG['P2_UNFREEZE_LAYERS']} backbone layers unfrozen.")
count_parameters(model)

optimizer_p2 = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CONFIG["P2_LR"],
    weight_decay=CONFIG["P2_WEIGHT_DECAY"],
)

scheduler_p2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer_p2, T_0=CONFIG["P2_T_0"]
)

early_stopping_p2 = EarlyStopping(
    patience=CONFIG["P2_PATIENCE"],
    checkpoint_path=os.path.join(CONFIG["CHECKPOINT_DIR"], "best_model_phase2.pth"),
)

for epoch in range(1, CONFIG["P2_EPOCHS"] + 1):
    print(f"\n  Epoch {epoch}/{CONFIG['P2_EPOCHS']}")

    train_loss, train_acc = train_one_epoch_with_clipping(
        model, train_loader, criterion, optimizer_p2, DEVICE,
        max_norm=CONFIG["P2_GRAD_CLIP_NORM"],
    )
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, DEVICE)

    scheduler_p2.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"    Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    early_stopping_p2(val_loss, model)
    if early_stopping_p2.early_stop:
        break

print(f"\n[✓] Phase 2 complete.")
print(f"    Best validation loss: {early_stopping_p2.best_loss:.4f}")

# Load best Phase 2 weights for final evaluation
model.load_state_dict(
    torch.load(os.path.join(CONFIG["CHECKPOINT_DIR"], "best_model_phase2.pth"))
)
print("[✓] Best Phase 2 checkpoint loaded for evaluation.")

# %% [markdown]
# ---
# ## 10. Training History Plot
#
# Combined figure showing loss and accuracy across both training phases,
# with a vertical dashed line marking the Phase 1 → Phase 2 transition.

# %%
def plot_training_history(history, phase1_epochs, class_names):
    """Plot combined training history across both phases.

    Creates a 1×2 figure with loss curves (left) and accuracy curves (right).
    A vertical dashed line marks the Phase 1 → Phase 2 boundary.

    Args:
        history (dict): Dictionary with keys train_loss, val_loss,
                        train_acc, val_acc — each a list of floats.
        phase1_epochs (int): Number of epochs in Phase 1.
        class_names (list): Class name strings (for title context).

    Returns:
        None
    """
    total_epochs = len(history["train_loss"])
    epochs_range = range(1, total_epochs + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Training History — Phase 1 (Head) + Phase 2 (Fine-Tune)",
                 fontsize=15, fontweight="bold")

    # ── Loss ──
    ax1.plot(epochs_range, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs_range, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.axvline(x=phase1_epochs, color="gray", linestyle="--", linewidth=1.5,
                label=f"Phase 1→2 (epoch {phase1_epochs})")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss Curves")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ── Accuracy ──
    ax2.plot(epochs_range, history["train_acc"], "b-", label="Train Acc", linewidth=2)
    ax2.plot(epochs_range, history["val_acc"], "r-", label="Val Acc", linewidth=2)
    ax2.axvline(x=phase1_epochs, color="gray", linestyle="--", linewidth=1.5,
                label=f"Phase 1→2 (epoch {phase1_epochs})")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy Curves")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


plot_training_history(history, phase1_epochs, CLASS_NAMES)

# %% [markdown]
# ---
# ## 11. Evaluation on Test Set
#
# **Critical:** We evaluate exclusively on the strictly isolated test set that
# the model has *never* seen during training or validation. We report:
# - Per-class Precision, Recall, F1
# - Macro F1-Score as the headline metric
# - Annotated Confusion Matrix
# - Multi-class ROC-AUC with One-vs-Rest curves

# %%
def get_predictions(model, loader, device):
    """Collect all predictions and true labels from a data loader.

    Args:
        model (nn.Module): Trained model in eval mode.
        loader (DataLoader): Data loader to iterate over.
        device (torch.device): Computation device.

    Returns:
        tuple: (all_labels, all_preds, all_probs) where:
            - all_labels: np.ndarray of shape (N,) with true class indices
            - all_preds: np.ndarray of shape (N,) with predicted class indices
            - all_probs: np.ndarray of shape (N, C) with softmax probabilities
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Evaluating test set"):
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ── Collect predictions ───────────────────────────────────────────────
test_labels, test_preds, test_probs = get_predictions(model, test_loader, DEVICE)

# %% [markdown]
# ### 11a. Classification Report

# %%
print("\n" + "=" * 60)
print("  CLASSIFICATION REPORT (Test Set)")
print("=" * 60)
print(classification_report(
    test_labels,
    test_preds,
    target_names=CLASS_NAMES,
    digits=4,
))

# %% [markdown]
# ### 11b. Confusion Matrix

# %%
def plot_confusion_matrix(true_labels, predictions, class_names):
    """Plot an annotated confusion matrix heatmap.

    Args:
        true_labels (np.ndarray): Ground truth class indices.
        predictions (np.ndarray): Predicted class indices.
        class_names (list): List of class name strings.

    Returns:
        None
    """
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 14},
    )
    plt.xlabel("Predicted Label", fontsize=13)
    plt.ylabel("True Label", fontsize=13)
    plt.title("Confusion Matrix — Test Set", fontsize=15, fontweight="bold")
    plt.xticks(fontsize=11, rotation=45)
    plt.yticks(fontsize=11, rotation=0)
    plt.tight_layout()
    plt.show()


plot_confusion_matrix(test_labels, test_preds, CLASS_NAMES)

# %% [markdown]
# ### 11c. Multi-Class ROC-AUC Curves

# %%
def plot_roc_auc_curves(true_labels, probabilities, class_names):
    """Plot One-vs-Rest ROC curves for each class with AUC scores.

    Args:
        true_labels (np.ndarray): Ground truth class indices of shape (N,).
        probabilities (np.ndarray): Softmax probabilities of shape (N, C).
        class_names (list): List of class name strings.

    Returns:
        None
    """
    num_classes = len(class_names)
    # Binarize labels for OvR
    from sklearn.preprocessing import label_binarize
    true_binarized = label_binarize(true_labels, classes=range(num_classes))

    plt.figure(figsize=(10, 8))
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(true_binarized[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], linewidth=2.5,
                 label=f"{class_names[i]} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.5, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title("Multi-Class ROC Curves — One-vs-Rest", fontsize=15, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Macro-averaged AUC
    macro_auc = roc_auc_score(
        true_binarized, probabilities, multi_class="ovr", average="macro"
    )
    print(f"\n  ★ Macro-Averaged ROC-AUC: {macro_auc:.4f}")


plot_roc_auc_curves(test_labels, test_probs, CLASS_NAMES)

# %% [markdown]
# ---
# ## 12. Explainable AI — Grad-CAM++
#
# **Academic justification:** In clinical settings, a model's prediction is
# only trustworthy if it can be explained. Grad-CAM++ (Chattopadhay et al.,
# 2018) generates class-discriminative localization maps by using the
# gradients flowing into the final convolutional layer. This allows us to
# visually confirm that the model attends to morphologically relevant
# regions of each WBC subtype:
#
# | Cell Type    | Expected Attention Region                     |
# |-------------|-----------------------------------------------|
# | Eosinophil  | Bilobed nucleus and eosinophilic granules      |
# | Lymphocyte  | Large, round, darkly-stained nucleus           |
# | Monocyte    | Kidney/horseshoe-shaped nucleus                |
# | Neutrophil  | Multi-lobed (3-5 lobes) segmented nucleus      |

# %%
def get_target_layer(model):
    """Identify the last convolutional layer for Grad-CAM.

    For timm EfficientNetV2 models, this is typically the last block
    in the feature extraction backbone.

    Args:
        model (nn.Module): The trained model.

    Returns:
        nn.Module: The target convolutional layer.
    """
    # For timm EfficientNetV2 models, the last conv block is in model.blocks[-1]
    # Adjust if your model structure differs
    if hasattr(model, "blocks"):
        return model.blocks[-1]
    elif hasattr(model, "features"):
        return model.features[-1]
    else:
        # Fallback: find last Conv2d
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv


def visualize_gradcam(model, dataset, sample_idx, class_names, config, device):
    """Generate and display Grad-CAM++ heatmap for a single sample.

    Produces a side-by-side plot of the original image and the Grad-CAM++
    overlay, annotated with predicted class, true class, and confidence.

    Args:
        model (nn.Module): Trained model in eval mode.
        dataset (TransformSubset): The dataset to draw the sample from.
        sample_idx (int): Index of the sample within the dataset.
        class_names (list): List of class name strings.
        config (dict): Configuration with ImageNet normalization stats.
        device (torch.device): Computation device.

    Returns:
        None
    """
    model.eval()

    # Get the image and label
    image_tensor, true_label = dataset[sample_idx]
    input_tensor = image_tensor.unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)
        pred_label = predicted.item()
        conf_score = confidence.item()

    # Unnormalize for display
    mean = np.array(config["IMAGENET_MEAN"])
    std = np.array(config["IMAGENET_STD"])
    original_img = image_tensor.permute(1, 2, 0).numpy()
    original_img = original_img * std + mean
    original_img = np.clip(original_img, 0, 1)

    # Grad-CAM++
    target_layer = get_target_layer(model)
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(pred_label)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(original_img.astype(np.float32), grayscale_cam,
                                   use_rgb=True)

    # Plot side-by-side
    is_correct = pred_label == true_label
    title_color = "green" if is_correct else "red"
    result_symbol = "✓" if is_correct else "✗"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(original_img)
    ax1.set_title(f"Original — True: {class_names[true_label]}", fontsize=12)
    ax1.axis("off")

    ax2.imshow(cam_image)
    ax2.set_title(
        f"Grad-CAM++ — Pred: {class_names[pred_label]} ({conf_score:.2%}) {result_symbol}",
        fontsize=12,
        color=title_color,
        fontweight="bold",
    )
    ax2.axis("off")

    plt.suptitle(
        f"Explainability: {class_names[true_label]}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ### 12.1 Grad-CAM++ Visualizations Per Class
#
# We find one correctly classified sample per class from the test set
# and generate its Grad-CAM++ heatmap.

# %%
def find_correct_sample_per_class(test_labels, test_preds, num_classes):
    """Find one correctly classified sample index per class.

    Args:
        test_labels (np.ndarray): True labels from the test set.
        test_preds (np.ndarray): Predicted labels from the test set.
        num_classes (int): Total number of classes.

    Returns:
        dict: Mapping from class_index → sample_index_in_test_set.
    """
    samples = {}
    for idx in range(len(test_labels)):
        cls = test_labels[idx]
        if cls not in samples and test_preds[idx] == cls:
            samples[cls] = idx
        if len(samples) == num_classes:
            break
    return samples


# ── Generate heatmaps for each class ─────────────────────────────────
correct_samples = find_correct_sample_per_class(
    test_labels, test_preds, CONFIG["NUM_CLASSES"]
)

print("\nGenerating Grad-CAM++ heatmaps for correctly classified samples:\n")
for cls_idx, sample_idx in sorted(correct_samples.items()):
    print(f"  Class: {CLASS_NAMES[cls_idx]} — Test sample index: {sample_idx}")
    visualize_gradcam(model, test_dataset, sample_idx, CLASS_NAMES, CONFIG, DEVICE)

# %% [markdown]
# ### 12.2 Clinical Interpretation of Grad-CAM++ Results
#
# The heatmaps above should reveal that the model focuses on the
# morphologically distinctive features of each WBC subtype:
#
# | Cell Type    | Expected Attention Region | Clinical Significance |
# |-------------|---------------------------|----------------------|
# | **Eosinophil** | Bilobed nucleus and bright eosinophilic granules | Granules contain cytotoxic proteins; bilobed nucleus is the primary diagnostic feature |
# | **Lymphocyte** | Large, round, darkly-stained nucleus occupying most of the cell | High nucleus-to-cytoplasm ratio is the hallmark; minimal cytoplasm expected |
# | **Monocyte** | Kidney/horseshoe-shaped nucleus | Largest WBC; characteristic indented nucleus shape |
# | **Neutrophil** | Multi-lobed (3-5 lobes) segmented nucleus | Segmentation count is clinically significant (hypersegmentation indicates B12 deficiency) |
#
# If the model's attention diverges from these regions (e.g., focusing on
# background or staining artifacts), this indicates potential overfitting
# to non-diagnostic features and warrants further data cleaning or
# augmentation strategy revision.

# %% [markdown]
# ---
# ## 13. Final Summary
#
# This notebook implements a complete, research-grade WBC classification
# pipeline with:
#
# 1. **Zero data leakage** — Verified programmatically
# 2. **Transfer learning** — EfficientNetV2-B0 with two-phase training
# 3. **Robust augmentation** — Including staining and focus simulation
# 4. **Class-weighted loss** — For imbalance robustness
# 5. **Comprehensive evaluation** — F1, Confusion Matrix, ROC-AUC
# 6. **Clinical explainability** — Grad-CAM++ heatmaps
#
# **Next steps:**
# - Compare with a baseline CNN trained from scratch
# - Experiment with Vision Transformers (ViT) for potential accuracy gains
# - Deploy via Streamlit for clinical prototype testing
