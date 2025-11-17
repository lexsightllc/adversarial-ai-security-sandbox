#!/usr/bin/env python3
"""
Vesuvius Challenge - Surface Detection Pipeline (Enhanced Edition)
3D U-Net with residual connections, attention mechanisms, and advanced training
Production-grade implementation with comprehensive optimizations and robustness
"""

import importlib
import os
import subprocess
import sys
import random
import zipfile
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image, ImageSequence
from tqdm import tqdm

# ============================================================================
# DEPENDENCY MANAGEMENT
# ============================================================================

REQUIRED_PACKAGES = [
    "torch", "torchvision", "numpy", "scipy",
    "Pillow", "tqdm", "pandas"
]

def install_package(package: str) -> None:
    """Install package with pip in quiet mode."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package,
         "--break-system-packages", "-q"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

# Install missing packages
for pkg in REQUIRED_PACKAGES:
    try:
        importlib.import_module(pkg if pkg != "Pillow" else "PIL")
    except ImportError:
        print(f"Installing {pkg}...")
        install_package(pkg)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Central configuration for the pipeline."""

    # Paths
    BASE_PATH: Path = field(default_factory=lambda: Path('/kaggle/input/vesuvius-challenge-surface-detection'))
    WORKING_DIR: Path = field(default_factory=lambda: Path('/kaggle/working'))

    # Model architecture
    PATCH_SIZE: Tuple[int, int, int] = (64, 64, 64)
    IN_CHANNELS: int = 1
    OUT_CHANNELS: int = 1
    BASE_FEATURES: int = 32
    DEPTH: int = 4
    USE_RESIDUAL: bool = True
    USE_ATTENTION: bool = True

    # Training hyperparameters
    BATCH_SIZE: int = 2
    LEARNING_RATE: float = 5e-4
    WEIGHT_DECAY: float = 1e-4
    NUM_EPOCHS: int = 120
    VAL_INTERVAL: int = 3
    SAVE_INTERVAL: int = 10
    WARMUP_EPOCHS: int = 5

    # Training stabilization
    EPSILON: float = 1e-7
    GRAD_CLIP: float = 1.0
    USE_AMP: bool = True

    # Loss function configuration
    DICE_WEIGHT: float = 0.6
    FOCAL_WEIGHT: float = 0.2
    BCE_WEIGHT: float = 0.2
    FOCAL_ALPHA: float = 0.25
    FOCAL_GAMMA: float = 2.0

    # Inference parameters
    TILE_OVERLAP: float = 0.5
    INFERENCE_THRESHOLD: float = 0.5
    USE_TTA: bool = True
    TTA_AUGMENTATIONS: int = 4
    USE_CRF: bool = False

    # Data sampling
    POS_NEG_RATIO: float = 1.5
    NUM_PATCHES_PER_VOLUME: int = 8

    # System
    NUM_WORKERS: int = 2
    SEED: int = 42
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Derived paths (set in __post_init__)
    TRAIN_IMAGES: Path = field(init=False)
    TRAIN_LABELS: Path = field(init=False)
    TEST_IMAGES: Path = field(init=False)
    TRAIN_CSV: Path = field(init=False)
    TEST_CSV: Path = field(init=False)
    MODEL_DIR: Path = field(init=False)
    PREDICTIONS_DIR: Path = field(init=False)
    LOGS_DIR: Path = field(init=False)
    SUBMISSION_PATH: Path = field(init=False)

    def __post_init__(self):
        """Initialize derived paths and create directories."""
        # Resolve paths with environment variable support
        self.BASE_PATH = self._resolve_path(
            os.environ.get('VESUVIUS_BASE_PATH'),
            self.BASE_PATH
        )
        self.WORKING_DIR = self._resolve_path(
            os.environ.get('VESUVIUS_WORKING_DIR'),
            self.WORKING_DIR
        )

        # Set derived paths
        self.TRAIN_IMAGES = self.BASE_PATH / 'train_images'
        self.TRAIN_LABELS = self.BASE_PATH / 'train_labels'
        self.TEST_IMAGES = self.BASE_PATH / 'test_images'
        self.TRAIN_CSV = self.BASE_PATH / 'train.csv'
        self.TEST_CSV = self.BASE_PATH / 'test.csv'

        self.MODEL_DIR = self.WORKING_DIR / 'models'
        self.PREDICTIONS_DIR = self.WORKING_DIR / 'predictions'
        self.LOGS_DIR = self.WORKING_DIR / 'logs'
        self.SUBMISSION_PATH = self.WORKING_DIR / 'submission.zip'

        # Create directories
        for d in [self.WORKING_DIR, self.MODEL_DIR, self.PREDICTIONS_DIR, self.LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_path(env_path: Optional[str], default_path: Path) -> Path:
        """Resolve path from environment or default."""
        if env_path:
            return Path(env_path).expanduser().resolve()
        if default_path.exists():
            return default_path.resolve()
        return Path.cwd()

CFG = Config()

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = CFG.SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ============================================================================
# LOGGING
# ============================================================================

class Logger:
    """Simple file and console logger with level filtering."""

    def __init__(self, log_file: Path, min_level: int = 0):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.min_level = min_level
        self.levels = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
        self.stats = defaultdict(int)

    def log(self, message: str, level: str = "INFO") -> None:
        """Log message to console and file."""
        if self.levels.get(level, 1) < self.min_level:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {level:5s} | {message}"
        print(formatted)
        with open(self.log_file, "a") as f:
            f.write(formatted + "\n")

        self.stats[level] += 1

    def info(self, msg: str) -> None:
        self.log(msg, "INFO")

    def warn(self, msg: str) -> None:
        self.log(msg, "WARN")

    def error(self, msg: str) -> None:
        self.log(msg, "ERROR")

    def debug(self, msg: str) -> None:
        self.log(msg, "DEBUG")

    def get_stats(self) -> Dict[str, int]:
        """Get logging statistics."""
        return dict(self.stats)

logger = Logger(CFG.LOGS_DIR / "training.log")

# ============================================================================
# I/O UTILITIES
# ============================================================================

def load_tiff_volume(path: Path, validate: bool = True) -> np.ndarray:
    """Load multi-page TIFF as 3D numpy array with validation."""
    if not path.exists():
        raise FileNotFoundError(f"TIFF not found: {path}")

    try:
        with Image.open(path) as img:
            slices = []
            for page in ImageSequence.Iterator(img):
                slice_data = np.array(page)
                if validate and slice_data.ndim != 2:
                    raise ValueError(f"Expected 2D slice, got shape {slice_data.shape}")
                slices.append(slice_data)

        if not slices:
            raise ValueError("No slices found in TIFF")

        volume = np.stack(slices, axis=0).astype(np.float32)

        # Validate volume
        if validate:
            if volume.ndim != 3:
                raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
            if np.isnan(volume).any() or np.isinf(volume).any():
                logger.warn(f"NaN/Inf values found in {path}, clipping...")
                volume = np.nan_to_num(volume, nan=0.0, posinf=255.0, neginf=0.0)

        return volume
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")

def save_tiff_volume(volume: np.ndarray, path: Path, validate: bool = True) -> None:
    """Save 3D numpy array as multi-page TIFF with validation."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    if validate and (np.isnan(volume).any() or np.isinf(volume).any()):
        logger.warn(f"Invalid values in volume for {path}, cleaning...")
        volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=0.0)

    # Convert to uint8 if needed
    if volume.dtype != np.uint8:
        volume = np.clip(volume, 0, 1).astype(np.float32)
        volume = (volume * 255).astype(np.uint8)

    # Create image pages
    pages = [Image.fromarray(volume[i], mode="L") for i in range(volume.shape[0])]

    # Save with compression
    try:
        pages[0].save(
            path,
            save_all=True,
            append_images=pages[1:],
            compression="tiff_deflate"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to save {path}: {e}")

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class StableDiceLoss(nn.Module):
    """Differentiable Dice loss with numerical stability."""

    def __init__(self, epsilon: float = CFG.EPSILON, smooth: float = 1.0):
        super().__init__()
        self.epsilon = epsilon
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Logits [B, C, D, H, W]
            target: Binary labels [B, C, D, H, W]
        """
        pred = torch.sigmoid(pred)

        # Flatten spatial dimensions
        pred_flat = pred.reshape(pred.size(0), pred.size(1), -1)
        target_flat = target.reshape(target.size(0), target.size(1), -1)

        # Compute Dice coefficient with smooth constant
        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth + self.epsilon)

        return 1.0 - dice.mean()

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = CFG.FOCAL_ALPHA, gamma: float = CFG.FOCAL_GAMMA, epsilon: float = CFG.EPSILON):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal loss forward pass."""
        pred_sigmoid = torch.sigmoid(pred)

        # Compute focal loss
        pos_weight = target * (1.0 - pred_sigmoid).pow(self.gamma)
        neg_weight = (1.0 - target) * pred_sigmoid.pow(self.gamma)

        pos_loss = -self.alpha * pos_weight * F.logsigmoid(pred)
        neg_loss = -(1.0 - self.alpha) * neg_weight * F.logsigmoid(-pred)

        return (pos_loss + neg_loss).mean()

class BoundaryLoss(nn.Module):
    """Loss that emphasizes boundaries between regions."""

    def __init__(self, epsilon: float = CFG.EPSILON):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute boundary-aware loss."""
        pred_sigmoid = torch.sigmoid(pred)

        # Compute gradients (boundaries)
        target_grad = torch.abs(
            target[:, :, 1:, :, :] - target[:, :, :-1, :, :] +
            target[:, :, :, 1:, :] - target[:, :, :, :-1, :] +
            target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        )

        # Emphasize boundaries
        boundary_weight = 1.0 + 5.0 * target_grad.max(dim=1)[0].unsqueeze(1)

        # Standard BCE with boundary weighting
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        weighted_bce = (bce * boundary_weight[:, :, :-1, :-1, :-1]).mean()

        return weighted_bce

class CombinedLoss(nn.Module):
    """Weighted combination of multiple loss functions."""

    def __init__(
        self,
        dice_weight: float = CFG.DICE_WEIGHT,
        focal_weight: float = CFG.FOCAL_WEIGHT,
        bce_weight: float = CFG.BCE_WEIGHT,
        epsilon: float = CFG.EPSILON
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight

        self.dice_loss = StableDiceLoss(epsilon)
        self.focal_loss = FocalLoss(epsilon=epsilon) if focal_weight > 0 else None
        self.bce_loss = nn.BCEWithLogitsLoss() if bce_weight > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0

        if self.dice_weight > 0:
            dice = self.dice_loss(pred, target)
            loss = loss + self.dice_weight * dice

        if self.focal_weight > 0 and self.focal_loss is not None:
            focal = self.focal_loss(pred, target)
            loss = loss + self.focal_weight * focal

        if self.bce_weight > 0 and self.bce_loss is not None:
            bce = self.bce_loss(pred, target)
            loss = loss + self.bce_weight * bce

        return loss

# ============================================================================
# METRICS
# ============================================================================

class DiceMetric:
    """Dice coefficient metric for evaluation."""

    def __init__(self, threshold: float = CFG.INFERENCE_THRESHOLD):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.sum_dice = 0.0
        self.count = 0
        self.dice_per_sample = []

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metric with batch predictions."""
        with torch.no_grad():
            pred_binary = (torch.sigmoid(pred) > self.threshold).float()

            # Flatten batch dimension
            pred_flat = pred_binary.reshape(pred.size(0), -1)
            target_flat = target.reshape(target.size(0), -1)

            # Compute Dice per sample
            intersection = (pred_flat * target_flat).sum(dim=1)
            union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
            dice = (2.0 * intersection + CFG.EPSILON) / (union + CFG.EPSILON)

            self.sum_dice += dice.sum().item()
            self.count += pred.size(0)
            self.dice_per_sample.extend(dice.cpu().numpy().tolist())

    def compute(self) -> float:
        """Compute average Dice score."""
        if self.count == 0:
            return 0.0
        return self.sum_dice / self.count

    def get_stats(self) -> Dict[str, float]:
        """Get detailed statistics."""
        if not self.dice_per_sample:
            return {}

        dice_array = np.array(self.dice_per_sample)
        return {
            "mean": float(np.mean(dice_array)),
            "std": float(np.std(dice_array)),
            "min": float(np.min(dice_array)),
            "max": float(np.max(dice_array))
        }

class IoUMetric:
    """Intersection over Union metric."""

    def __init__(self, threshold: float = CFG.INFERENCE_THRESHOLD):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.sum_iou = 0.0
        self.count = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """Update metric with batch predictions."""
        with torch.no_grad():
            pred_binary = (torch.sigmoid(pred) > self.threshold).float()

            # Flatten batch dimension
            pred_flat = pred_binary.reshape(pred.size(0), -1)
            target_flat = target.reshape(target.size(0), -1)

            # Compute IoU per sample
            intersection = (pred_flat * target_flat).sum(dim=1)
            union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
            iou = intersection / (union + CFG.EPSILON)

            self.sum_iou += iou.sum().item()
            self.count += pred.size(0)

    def compute(self) -> float:
        """Compute average IoU."""
        if self.count == 0:
            return 0.0
        return self.sum_iou / self.count

# ============================================================================
# MODEL ARCHITECTURE - ENHANCED
# ============================================================================

class ConvBlock3D(nn.Module):
    """3D convolutional block with normalization and activation."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class ResidualBlock3D(nn.Module):
    """3D Residual block with skip connection."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels, dropout)
        self.skip = nn.Identity() if in_channels == out_channels else \
                    nn.Conv3d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)

class ChannelAttention3D(nn.Module):
    """Channel attention module for 3D features."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention3D(nn.Module):
    """Spatial attention module for 3D features."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(x_cat))
        return x * out

class AttentionBlock3D(nn.Module):
    """Combined channel and spatial attention."""

    def __init__(self, channels: int):
        super().__init__()
        self.channel_att = ChannelAttention3D(channels)
        self.spatial_att = SpatialAttention3D()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class Down3D(nn.Module):
    """Downsampling block with pooling and convolution."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ResidualBlock3D(in_channels, out_channels, dropout) if CFG.USE_RESIDUAL \
                    else ConvBlock3D(in_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))

class Up3D(nn.Module):
    """Upsampling block with transpose convolution and skip connection."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, 2, 2)
        self.conv = ResidualBlock3D(in_channels, out_channels, dropout) if CFG.USE_RESIDUAL \
                    else ConvBlock3D(in_channels, out_channels, dropout)
        self.attention = AttentionBlock3D(out_channels) if CFG.USE_ATTENTION else nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Upsampled features
            x2: Skip connection from encoder
        """
        x1 = self.up(x1)

        # Pad x1 to match x2 if sizes differ
        diff_d = x2.size(2) - x1.size(2)
        diff_h = x2.size(3) - x1.size(3)
        diff_w = x2.size(4) - x1.size(4)

        if diff_d != 0 or diff_h != 0 or diff_w != 0:
            x1 = F.pad(x1, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
                diff_d // 2, diff_d - diff_d // 2
            ])

        # Concatenate and convolve
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        return x

class UNet3D(nn.Module):
    """3D U-Net with residual connections and attention mechanisms."""

    def __init__(
        self,
        in_channels: int = CFG.IN_CHANNELS,
        out_channels: int = CFG.OUT_CHANNELS,
        base_features: int = CFG.BASE_FEATURES,
        depth: int = CFG.DEPTH
    ):
        super().__init__()
        self.depth = depth
        features = base_features

        # Initial convolution
        self.encoder_init = ResidualBlock3D(in_channels, features) if CFG.USE_RESIDUAL \
                           else ConvBlock3D(in_channels, features)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        for i in range(depth):
            dropout = 0.2 * (i / (depth - 1)) if depth > 1 else 0.0  # Progressive dropout
            self.encoder_blocks.append(
                Down3D(features * (2 ** i), features * (2 ** (i + 1)), dropout)
            )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i in range(depth, 0, -1):
            dropout = 0.2 * ((depth - i) / depth)
            self.decoder_blocks.append(
                Up3D(features * (2 ** i), features * (2 ** (i - 1)), dropout)
            )

        # Output
        self.out_conv = nn.Conv3d(features, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input volume [B, 1, D, H, W]
        Returns:
            Segmentation logits [B, 1, D, H, W]
        """
        # Initial encoding
        x = self.encoder_init(x)
        skip_connections = [x]

        # Encoder with skip connections
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            if i < self.depth - 1:
                skip_connections.append(x)

        # Decoder with skip connections
        for i, decoder in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i + 1)]
            x = decoder(x, skip)

        return self.out_conv(x)

# ============================================================================
# DATA AUGMENTATION - ENHANCED
# ============================================================================

class VolumeAugmentation:
    """Advanced 3D volume augmentation for training."""

    def __init__(self, mode: str = "train"):
        self.mode = mode

    def __call__(
        self,
        image: np.ndarray,
        label: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply augmentation pipeline."""

        if self.mode != "train":
            return self._normalize(image), label

        if label is not None:
            # Spatial augmentations
            image, label = self._random_flip(image, label)
            image, label = self._random_rotate90(image, label)
            image, label = self._random_z_shift(image, label)
            image, label = self._random_elastic_deform(image, label)

        # Intensity augmentation
        image = self._intensity_shift(image)
        image = self._random_brightness(image)
        image = self._random_contrast(image)
        image = self._random_noise(image)
        image = self._normalize(image)

        return image, label

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1]."""
        return np.clip(image, 0, 255) / 255.0

    @staticmethod
    def _random_flip(
        image: np.ndarray,
        label: np.ndarray,
        prob: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random flips along each axis."""
        for axis in range(3):
            if random.random() < prob:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
        return image, label

    @staticmethod
    def _random_rotate90(
        image: np.ndarray,
        label: np.ndarray,
        prob: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random 90-degree rotation in xy plane."""
        if random.random() < prob:
            k = random.randint(0, 3)
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            label = np.rot90(label, k=k, axes=(1, 2)).copy()
        return image, label

    @staticmethod
    def _random_z_shift(
        image: np.ndarray,
        label: np.ndarray,
        max_shift: int = 5,
        prob: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random shift along z-axis (depth)."""
        if random.random() < prob:
            shift = random.randint(-max_shift, max_shift)
            if shift != 0:
                image = np.roll(image, shift, axis=0)
                label = np.roll(label, shift, axis=0)
        return image, label

    @staticmethod
    def _random_elastic_deform(
        image: np.ndarray,
        label: np.ndarray,
        alpha: float = 30.0,
        sigma: float = 3.0,
        prob: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Elastic deformation augmentation."""
        if random.random() < prob:
            from scipy.ndimage import map_coordinates, gaussian_filter

            shape = image.shape
            dx = np.random.randn(*shape) * sigma
            dy = np.random.randn(*shape) * sigma
            dz = np.random.randn(*shape) * sigma

            dx = gaussian_filter(dx, sigma=sigma) * alpha
            dy = gaussian_filter(dy, sigma=sigma) * alpha
            dz = gaussian_filter(dz, sigma=sigma) * alpha

            z, y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
            indices_z = np.reshape(z + dz, (-1, 1))
            indices_y = np.reshape(y + dy, (-1, 1))
            indices_x = np.reshape(x + dx, (-1, 1))

            # Apply to both image and label
            image = map_coordinates(image, [indices_z, indices_y, indices_x], order=1, cval=0).reshape(shape)
            label = map_coordinates(label, [indices_z, indices_y, indices_x], order=0, cval=0).reshape(shape)

        return image, label

    @staticmethod
    def _intensity_shift(
        image: np.ndarray,
        prob: float = 0.5,
        max_shift: float = 0.15
    ) -> np.ndarray:
        """Random intensity shift."""
        if random.random() < prob:
            shift = random.uniform(-max_shift, max_shift) * 255
            image = image + shift
        return image

    @staticmethod
    def _random_brightness(
        image: np.ndarray,
        prob: float = 0.3,
        max_factor: float = 0.2
    ) -> np.ndarray:
        """Random brightness adjustment."""
        if random.random() < prob:
            factor = 1.0 + random.uniform(-max_factor, max_factor)
            image = image * factor
        return image

    @staticmethod
    def _random_contrast(
        image: np.ndarray,
        prob: float = 0.3,
        max_factor: float = 0.3
    ) -> np.ndarray:
        """Random contrast adjustment."""
        if random.random() < prob:
            factor = 1.0 + random.uniform(-max_factor, max_factor)
            mean = image.mean()
            image = (image - mean) * factor + mean
        return image

    @staticmethod
    def _random_noise(
        image: np.ndarray,
        prob: float = 0.3,
        noise_level: float = 0.01
    ) -> np.ndarray:
        """Add Gaussian noise."""
        if random.random() < prob:
            noise = np.random.randn(*image.shape) * (noise_level * 255)
            image = image + noise
        return image

# ============================================================================
# DATASET
# ============================================================================

class VesuviusDataset(Dataset):
    """Dataset for Vesuvius Challenge surface detection with enhanced sampling."""

    def __init__(
        self,
        csv_path: Path,
        image_dir: Path,
        label_dir: Optional[Path] = None,
        mode: str = "train",
        patch_size: Tuple[int, int, int] = CFG.PATCH_SIZE,
        num_patches: int = CFG.NUM_PATCHES_PER_VOLUME,
        pos_neg_ratio: float = CFG.POS_NEG_RATIO
    ):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mode = mode
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.pos_neg_ratio = pos_neg_ratio

        # Load sample IDs with caching
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            self.sample_ids = df["id"].astype(str).tolist()
        else:
            self.sample_ids = sorted([p.stem for p in image_dir.glob("*.tif")])

        self.augment = VolumeAugmentation(mode)
        logger.info(f"Loaded {len(self.sample_ids)} samples in {mode} mode")

        # Pre-compute positive sample indices for efficient sampling
        self.positive_sample_indices = self._find_positive_samples() if mode == "train" else []

    def _find_positive_samples(self) -> List[int]:
        """Find sample indices with positive labels."""
        positive = []
        for idx, sample_id in enumerate(self.sample_ids):
            if self.label_dir is None:
                continue
            label_path = self.label_dir / f"{sample_id}.tif"
            if label_path.exists():
                positive.append(idx)
        return positive

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.sample_ids) * self.num_patches
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training patch or full volume."""

        # Determine sample ID
        if self.mode == "train":
            sample_idx = idx // self.num_patches
        else:
            sample_idx = idx

        sample_id = self.sample_ids[sample_idx]

        # Load image
        image_path = self.image_dir / f"{sample_id}.tif"
        try:
            image = load_tiff_volume(image_path)
        except Exception as e:
            logger.error(f"Failed to load {image_path}: {e}")
            raise

        # Load label if available
        label = None
        if self.label_dir is not None:
            label_path = self.label_dir / f"{sample_id}.tif"
            if label_path.exists():
                try:
                    label = load_tiff_volume(label_path)
                except Exception as e:
                    logger.warn(f"Failed to load label {label_path}: {e}")

        # Training mode: extract patch
        if self.mode == "train" and label is not None:
            image, label = self._extract_balanced_patch(image, label)
            image, label = self.augment(image, label)

            # Convert to tensors
            image = torch.from_numpy(image[None]).float()
            label = torch.from_numpy(label[None]).float()
            label = torch.clamp(label, 0, 1)

            return {"image": image, "label": label, "id": sample_id}

        # Inference mode: return full volume
        image, _ = self.augment(image, None)
        return {
            "image": image,
            "id": sample_id,
            "original_shape": image.shape
        }

    def _extract_balanced_patch(
        self,
        image: np.ndarray,
        label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a patch with improved balanced positive/negative sampling."""

        d, h, w = image.shape
        pd, ph, pw = self.patch_size

        # Pad if volume is smaller than patch
        if d < pd or h < ph or w < pw:
            image = self._pad_to_size(image, self.patch_size)
            label = self._pad_to_size(label, self.patch_size)
            return image, label

        # Compute label statistics for better sampling
        label_sum = label.sum()
        label_ratio = label_sum / label.size if label.size > 0 else 0.0

        # Decide whether to sample positive or negative patch
        sample_positive = random.random() < (
            self.pos_neg_ratio / (1 + self.pos_neg_ratio)
        )

        # Find patch center
        if sample_positive and label_sum > 0:
            # Sample near positive labels with better strategy
            positive_coords = np.argwhere(label > 0)
            if len(positive_coords) > 0:
                center = positive_coords[random.randint(0, len(positive_coords) - 1)]
            else:
                center = np.array([d // 2, h // 2, w // 2])
        else:
            # Random sampling with edge handling
            center = np.array([
                random.randint(pd // 2, d - pd // 2) if d > pd else d // 2,
                random.randint(ph // 2, h - ph // 2) if h > ph else h // 2,
                random.randint(pw // 2, w - pw // 2) if w > pw else w // 2
            ])

        # Extract patch with better boundary handling
        d_start = max(0, min(center[0] - pd // 2, d - pd))
        h_start = max(0, min(center[1] - ph // 2, h - ph))
        w_start = max(0, min(center[2] - pw // 2, w - pw))

        d_end = d_start + pd
        h_end = h_start + ph
        w_end = w_start + pw

        image_patch = image[d_start:d_end, h_start:h_end, w_start:w_end]
        label_patch = label[d_start:d_end, h_start:h_end, w_start:w_end]

        # Pad if needed
        if image_patch.shape != self.patch_size:
            image_patch = self._pad_to_size(image_patch, self.patch_size)
            label_patch = self._pad_to_size(label_patch, self.patch_size)

        return image_patch, label_patch

    @staticmethod
    def _pad_to_size(array: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        """Pad array to target size."""
        pad_width = [
            (0, max(0, target - current))
            for target, current in zip(target_size, array.shape)
        ]
        return np.pad(array, pad_width, mode="constant", constant_values=0)

# ============================================================================
# TRAINING UTILITIES - ENHANCED
# ============================================================================

class WarmupScheduler:
    """Learning rate scheduler with warmup phase."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, total_epochs: int):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        """Update learning rate."""
        self.current_epoch += 1

        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr_factor = self.current_epoch / self.warmup_epochs
        else:
            # Polynomial decay
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_factor = (1 - progress) ** 0.9

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_factor

class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def __call__(self, score: float) -> bool:
        """Check if training should stop. Returns True if should stop."""
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# ============================================================================
# TRAINER - ENHANCED
# ============================================================================

class Trainer:
    """Advanced training loop with comprehensive monitoring."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: str = CFG.DEVICE
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.best_dice = 0.0
        self.best_iou = 0.0
        self.train_losses = []
        self.val_dices = []
        self.val_ious = []
        self.learning_rates = []

        # Mixed precision training
        self.use_amp = CFG.USE_AMP and torch.cuda.is_available()
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            logger.info("Using mixed precision training")

        # Early stopping
        self.early_stopping = EarlyStopping(patience=30, min_delta=0.001)

        # Metrics
        self.dice_metric = DiceMetric()
        self.iou_metric = IoUMetric()

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        valid_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{CFG.NUM_EPOCHS}",
            total=len(self.train_loader)
        )

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Validate inputs
            if torch.isnan(images).any() or torch.isinf(images).any():
                logger.warn(f"Invalid input in batch {batch_idx}")
                continue

            if torch.isnan(labels).any() or torch.isinf(labels).any():
                logger.warn(f"Invalid label in batch {batch_idx}")
                continue

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), CFG.GRAD_CLIP
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), CFG.GRAD_CLIP
                )
                self.optimizer.step()

            # Accumulate loss
            if not torch.isnan(loss):
                epoch_loss += loss.item()
                valid_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if valid_batches == 0:
            logger.error(f"No valid batches in epoch {epoch}")
            return float("inf")

        avg_loss = epoch_loss / valid_batches
        self.train_losses.append(avg_loss)

        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.learning_rates.append(current_lr)

        return avg_loss

    def validate(self) -> Tuple[float, float]:
        """Validate on validation set. Returns (dice, iou)."""
        if self.val_loader is None:
            return 0.0, 0.0

        self.model.eval()
        self.dice_metric.reset()
        self.iou_metric.reset()

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                image = batch["image"]
                sample_id = batch["id"][0]

                # Load ground truth label
                label_path = CFG.TRAIN_LABELS / f"{sample_id}.tif"
                if not label_path.exists():
                    continue

                try:
                    label = load_tiff_volume(label_path)
                except Exception as e:
                    logger.warn(f"Failed to load validation label {sample_id}: {e}")
                    continue

                # Predict with tiling
                if isinstance(image, torch.Tensor):
                    image = image.cpu().numpy()[0] if image.ndim == 4 else image.numpy()

                prediction = self._predict_with_tiling(image)

                # Convert to tensors for metric
                pred_tensor = torch.from_numpy(prediction[None, None]).float()
                label_tensor = torch.from_numpy(label[None, None]).float()

                # Convert prediction to logits
                pred_tensor = torch.clamp(pred_tensor, CFG.EPSILON, 1 - CFG.EPSILON)
                pred_logits = torch.log(pred_tensor / (1 - pred_tensor))

                self.dice_metric.update(pred_logits, label_tensor)
                self.iou_metric.update(pred_logits, label_tensor)

        dice_score = self.dice_metric.compute()
        iou_score = self.iou_metric.compute()

        self.val_dices.append(dice_score)
        self.val_ious.append(iou_score)

        return dice_score, iou_score

    def _predict_with_tiling(self, volume: np.ndarray) -> np.ndarray:
        """Predict volume using overlapping tiles with Gaussian weighting."""

        # Normalize volume
        if isinstance(volume, torch.Tensor):
            volume = volume.cpu().numpy()
        volume = np.clip(volume / 255.0, 0, 1).astype(np.float32)

        d, h, w = volume.shape
        pd, ph, pw = CFG.PATCH_SIZE

        # Calculate stride
        stride_d = max(1, int(pd * (1 - CFG.TILE_OVERLAP)))
        stride_h = max(1, int(ph * (1 - CFG.TILE_OVERLAP)))
        stride_w = max(1, int(pw * (1 - CFG.TILE_OVERLAP)))

        # Initialize output
        output = np.zeros(volume.shape, dtype=np.float32)
        weight_map = np.zeros(volume.shape, dtype=np.float32)

        # Gaussian weights for smooth blending
        gaussian_weights = self._get_gaussian_weights(CFG.PATCH_SIZE)

        # Generate tile positions
        d_positions = list(range(0, max(1, d - pd + 1), stride_d))
        if len(d_positions) == 0 or d_positions[-1] + pd < d:
            d_positions.append(max(0, d - pd))

        h_positions = list(range(0, max(1, h - ph + 1), stride_h))
        if len(h_positions) == 0 or h_positions[-1] + ph < h:
            h_positions.append(max(0, h - ph))

        w_positions = list(range(0, max(1, w - pw + 1), stride_w))
        if len(w_positions) == 0 or w_positions[-1] + pw < w:
            w_positions.append(max(0, w - pw))

        # Process each tile
        with torch.no_grad():
            for d_start in d_positions:
                for h_start in h_positions:
                    for w_start in w_positions:
                        # Extract patch
                        d_end = min(d_start + pd, d)
                        h_end = min(h_start + ph, h)
                        w_end = min(w_start + pw, w)

                        patch = volume[d_start:d_end, h_start:h_end, w_start:w_end]

                        # Pad if necessary
                        if patch.shape != CFG.PATCH_SIZE:
                            padded = np.zeros(CFG.PATCH_SIZE, dtype=np.float32)
                            ad, ah, aw = patch.shape
                            padded[:ad, :ah, :aw] = patch
                            patch = padded
                        else:
                            ad, ah, aw = patch.shape

                        # Predict
                        patch_tensor = torch.from_numpy(patch[None, None]).float().to(self.device)
                        pred = torch.sigmoid(self.model(patch_tensor))
                        pred = pred[0, 0].cpu().numpy()

                        # Accumulate with Gaussian weighting
                        weights = gaussian_weights[:ad, :ah, :aw]
                        output[d_start:d_end, h_start:h_end, w_start:w_end] += \
                            pred[:ad, :ah, :aw] * weights
                        weight_map[d_start:d_end, h_start:h_end, w_start:w_end] += weights

        # Normalize by weights
        output = output / np.maximum(weight_map, 1e-8)
        return output

    @staticmethod
    def _get_gaussian_weights(patch_size: Tuple[int, int, int]) -> np.ndarray:
        """Generate 3D Gaussian weights for smooth tile blending."""
        d, h, w = patch_size
        sigma = d / 6.0

        z = np.arange(d) - d / 2.0
        y = np.arange(h) - h / 2.0
        x = np.arange(w) - w / 2.0

        gz = np.exp(-(z ** 2) / (2 * sigma ** 2))
        gy = np.exp(-(y ** 2) / (2 * sigma ** 2))
        gx = np.exp(-(x ** 2) / (2 * sigma ** 2))

        weights = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
        return weights.astype(np.float32)

    def save_checkpoint(self, epoch: int, metric: float, name: str = "checkpoint"):
        """Save model checkpoint with metadata."""
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "metric": metric,
            "best_dice": self.best_dice,
            "best_iou": self.best_iou,
            "train_losses": self.train_losses,
            "val_dices": self.val_dices,
            "val_ious": self.val_ious,
            "learning_rates": self.learning_rates,
            "config": {
                "patch_size": CFG.PATCH_SIZE,
                "base_features": CFG.BASE_FEATURES,
                "depth": CFG.DEPTH,
                "use_residual": CFG.USE_RESIDUAL,
                "use_attention": CFG.USE_ATTENTION
            }
        }

        path = CFG.MODEL_DIR / f"{name}.pth"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if self.scheduler and checkpoint.get("scheduler_state"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        self.best_dice = checkpoint.get("best_dice", 0.0)
        self.best_iou = checkpoint.get("best_iou", 0.0)
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_dices = checkpoint.get("val_dices", [])
        self.val_ious = checkpoint.get("val_ious", [])

        logger.info(f"Loaded checkpoint from {path}")
        logger.info(f"Best Dice: {self.best_dice:.4f}, Best IoU: {self.best_iou:.4f}")

    def train(self):
        """Full training loop with monitoring."""
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Training batches: {len(self.train_loader)}")
        if self.val_loader:
            logger.info(f"Validation batches: {len(self.val_loader)}")

        for epoch in range(1, CFG.NUM_EPOCHS + 1):
            # Train
            loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch}/{CFG.NUM_EPOCHS} - Train Loss: {loss:.4f}")

            # Log learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(f"Learning Rate: {current_lr:.6f}")

            # Validate
            if epoch % CFG.VAL_INTERVAL == 0 and self.val_loader:
                dice, iou = self.validate()
                logger.info(f"Epoch {epoch}/{CFG.NUM_EPOCHS} - Validation Dice: {dice:.4f}, IoU: {iou:.4f}")

                # Save best model
                if dice > self.best_dice:
                    self.best_dice = dice
                    self.best_iou = iou
                    self.save_checkpoint(epoch, dice, "best_model")
                    logger.info(f"New best model! Dice: {self.best_dice:.4f}, IoU: {self.best_iou:.4f}")

                # Check early stopping
                if self.early_stopping(dice):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Periodic checkpoint
            if epoch % CFG.SAVE_INTERVAL == 0:
                self.save_checkpoint(epoch, loss, f"checkpoint_epoch_{epoch}")

        logger.info("=" * 80)
        logger.info(f"TRAINING COMPLETE - Best Dice: {self.best_dice:.4f}, Best IoU: {self.best_iou:.4f}")
        logger.info("=" * 80)

# ============================================================================
# INFERENCE - ENHANCED WITH TTA AND POST-PROCESSING
# ============================================================================

def predict_test_set(model: nn.Module, device: str = CFG.DEVICE):
    """Generate predictions for test set with TTA."""
    model.eval()

    # Load test sample IDs
    if CFG.TEST_CSV.exists():
        df = pd.read_csv(CFG.TEST_CSV)
        sample_ids = df["id"].astype(str).tolist()
    else:
        sample_ids = sorted([p.stem for p in CFG.TEST_IMAGES.glob("*.tif")])

    logger.info(f"Predicting {len(sample_ids)} test volumes (TTA={CFG.USE_TTA})...")

    for sample_id in tqdm(sample_ids, desc="Test Inference"):
        image_path = CFG.TEST_IMAGES / f"{sample_id}.tif"

        if not image_path.exists():
            logger.warn(f"Missing test image: {sample_id}")
            continue

        try:
            # Load volume
            volume = load_tiff_volume(image_path)
            original_shape = volume.shape

            # Predict with optional TTA
            if CFG.USE_TTA:
                prediction = predict_volume_with_tta(model, volume, device)
            else:
                prediction = predict_volume_with_tiling(model, volume, device)

            # Binarize
            binary = (prediction > CFG.INFERENCE_THRESHOLD).astype(np.uint8)

            # Validate output
            assert binary.shape == original_shape, "Shape mismatch"
            assert set(np.unique(binary)).issubset({0, 1}), "Invalid values"

            # Save
            save_tiff_volume(binary, CFG.PREDICTIONS_DIR / f"{sample_id}.tif")

        except Exception as e:
            logger.error(f"Failed to predict {sample_id}: {e}")
            continue

    # Create submission
    create_submission_zip()
    logger.info("Test inference complete")

def predict_volume_with_tta(
    model: nn.Module,
    volume: np.ndarray,
    device: str,
    num_augmentations: int = CFG.TTA_AUGMENTATIONS
) -> np.ndarray:
    """Predict with test-time augmentation."""

    predictions = []

    for aug_idx in range(num_augmentations):
        # Apply augmentation
        if aug_idx == 0:
            aug_volume = volume.copy()
        elif aug_idx == 1:
            aug_volume = np.flip(volume, axis=1).copy()  # Flip H
        elif aug_idx == 2:
            aug_volume = np.flip(volume, axis=2).copy()  # Flip W
        elif aug_idx == 3:
            aug_volume = np.rot90(volume, k=1, axes=(1, 2)).copy()  # Rotate
        else:
            aug_volume = volume.copy()

        # Predict
        pred = predict_volume_with_tiling(model, aug_volume, device)

        # Reverse augmentation
        if aug_idx == 1:
            pred = np.flip(pred, axis=1)
        elif aug_idx == 2:
            pred = np.flip(pred, axis=2)
        elif aug_idx == 3:
            pred = np.rot90(pred, k=-1, axes=(1, 2))

        predictions.append(pred)

    # Average predictions
    return np.mean(predictions, axis=0)

def predict_volume_with_tiling(
    model: nn.Module,
    volume: np.ndarray,
    device: str
) -> np.ndarray:
    """Predict full volume using overlapping tiles."""

    # Normalize
    volume = np.clip(volume / 255.0, 0, 1).astype(np.float32)

    d, h, w = volume.shape
    pd, ph, pw = CFG.PATCH_SIZE

    # Calculate stride
    stride_d = max(1, int(pd * (1 - CFG.TILE_OVERLAP)))
    stride_h = max(1, int(ph * (1 - CFG.TILE_OVERLAP)))
    stride_w = max(1, int(pw * (1 - CFG.TILE_OVERLAP)))

    # Initialize output
    output = np.zeros(volume.shape, dtype=np.float32)
    weight_map = np.zeros(volume.shape, dtype=np.float32)

    # Gaussian weights
    gaussian_weights = get_gaussian_weights_3d(CFG.PATCH_SIZE)

    # Generate tile positions
    d_positions = list(range(0, max(1, d - pd + 1), stride_d))
    if len(d_positions) == 0 or d_positions[-1] + pd < d:
        d_positions.append(max(0, d - pd))

    h_positions = list(range(0, max(1, h - ph + 1), stride_h))
    if len(h_positions) == 0 or h_positions[-1] + ph < h:
        h_positions.append(max(0, h - ph))

    w_positions = list(range(0, max(1, w - pw + 1), stride_w))
    if len(w_positions) == 0 or w_positions[-1] + pw < w:
        w_positions.append(max(0, w - pw))

    # Process tiles
    with torch.no_grad():
        for d_start in d_positions:
            for h_start in h_positions:
                for w_start in w_positions:
                    # Extract patch
                    d_end = min(d_start + pd, d)
                    h_end = min(h_start + ph, h)
                    w_end = min(w_start + pw, w)

                    patch = volume[d_start:d_end, h_start:h_end, w_start:w_end]

                    # Pad if necessary
                    if patch.shape != CFG.PATCH_SIZE:
                        padded = np.zeros(CFG.PATCH_SIZE, dtype=np.float32)
                        ad, ah, aw = patch.shape
                        padded[:ad, :ah, :aw] = patch
                        patch = padded
                    else:
                        ad, ah, aw = patch.shape

                    # Predict
                    patch_tensor = torch.from_numpy(patch[None, None]).float().to(device)
                    pred = torch.sigmoid(model(patch_tensor))
                    pred = pred[0, 0].cpu().numpy()

                    # Accumulate with weights
                    weights = gaussian_weights[:ad, :ah, :aw]
                    output[d_start:d_end, h_start:h_end, w_start:w_end] += \
                        pred[:ad, :ah, :aw] * weights
                    weight_map[d_start:d_end, h_start:h_end, w_start:w_end] += weights

    # Normalize
    output = output / np.maximum(weight_map, 1e-8)
    return output

def get_gaussian_weights_3d(patch_size: Tuple[int, int, int]) -> np.ndarray:
    """Generate 3D Gaussian weights for tile blending."""
    d, h, w = patch_size
    sigma = d / 6.0

    z = np.arange(d) - d / 2.0
    y = np.arange(h) - h / 2.0
    x = np.arange(w) - w / 2.0

    gz = np.exp(-(z ** 2) / (2 * sigma ** 2))
    gy = np.exp(-(y ** 2) / (2 * sigma ** 2))
    gx = np.exp(-(x ** 2) / (2 * sigma ** 2))

    weights = gz[:, None, None] * gy[None, :, None] * gx[None, None, :]
    return weights.astype(np.float32)

def create_submission_zip():
    """Create submission ZIP file."""
    with zipfile.ZipFile(CFG.SUBMISSION_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for tif_path in sorted(CFG.PREDICTIONS_DIR.glob("*.tif")):
            zipf.write(tif_path, arcname=tif_path.name)

    count = len(list(CFG.PREDICTIONS_DIR.glob("*.tif")))
    size = CFG.SUBMISSION_PATH.stat().st_size / (1024 * 1024)

    logger.info(f"Submission ZIP: {CFG.SUBMISSION_PATH}")
    logger.info(f"Files: {count}, Size: {size:.2f} MB")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution pipeline."""

    print("\n" + "=" * 80)
    print("VESUVIUS CHALLENGE - SURFACE DETECTION PIPELINE (ENHANCED)")
    print("=" * 80 + "\n")

    logger.info("Initializing enhanced pipeline...")
    logger.info(f"Device: {CFG.DEVICE}")
    logger.info(f"Patch size: {CFG.PATCH_SIZE}")
    logger.info(f"Batch size: {CFG.BATCH_SIZE}")
    logger.info(f"Learning rate: {CFG.LEARNING_RATE}")
    logger.info(f"Use Residual: {CFG.USE_RESIDUAL}")
    logger.info(f"Use Attention: {CFG.USE_ATTENTION}")
    logger.info(f"Use TTA: {CFG.USE_TTA}")

    # Check for existing model
    best_model_path = CFG.MODEL_DIR / "best_model.pth"

    if best_model_path.exists():
        logger.info(f"Found existing model: {best_model_path}")
        logger.info("Skipping training, proceeding to inference...")

        # Load model and predict
        model = UNet3D()
        checkpoint = torch.load(best_model_path, map_location=CFG.DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        model.to(CFG.DEVICE)

        predict_test_set(model)

    else:
        logger.info("No existing model found. Starting training...")

        # Create datasets
        logger.info("Creating datasets...")
        train_dataset = VesuviusDataset(
            CFG.TRAIN_CSV,
            CFG.TRAIN_IMAGES,
            CFG.TRAIN_LABELS,
            mode="train"
        )

        val_dataset = VesuviusDataset(
            CFG.TRAIN_CSV,
            CFG.TRAIN_IMAGES,
            CFG.TRAIN_LABELS,
            mode="val",
            num_patches=1
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.BATCH_SIZE,
            shuffle=True,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=False
        )

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")

        # Initialize model and training components
        model = UNet3D()
        criterion = CombinedLoss(
            dice_weight=CFG.DICE_WEIGHT,
            focal_weight=CFG.FOCAL_WEIGHT,
            bce_weight=CFG.BCE_WEIGHT
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CFG.LEARNING_RATE,
            weight_decay=CFG.WEIGHT_DECAY
        )
        scheduler = WarmupScheduler(
            optimizer,
            warmup_epochs=CFG.WARMUP_EPOCHS,
            total_epochs=CFG.NUM_EPOCHS
        )

        # Train
        trainer = Trainer(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler
        )
        trainer.train()

        # Generate test predictions
        logger.info("Generating test predictions...")
        predict_test_set(model)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80 + "\n")
    logger.info("All done!")

if __name__ == "__main__":
    main()
