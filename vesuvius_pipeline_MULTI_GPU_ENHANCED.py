#!/usr/bin/env python3
"""
Vesuvius Challenge - MULTI-GPU ENHANCED Pipeline
Optimized for 2x GPU T4 with 45-hour quota

Key Enhancements:
- DistributedDataParallel for true multi-GPU training
- 4-8x larger batch sizes
- Gradient accumulation
- Advanced checkpointing and resume
- GPU utilization monitoring
- T4-specific optimizations (Tensor Cores, mixed precision)
- Optimized data loading for multi-GPU
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
import time

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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import pandas as pd

# ============================================================================
# MULTI-GPU CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Central configuration optimized for 2x T4 GPUs."""

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

    # MULTI-GPU Training hyperparameters (OPTIMIZED FOR 2x T4)
    BATCH_SIZE_PER_GPU: int = 16  # 16 per GPU = 32 total effective batch size
    GRADIENT_ACCUMULATION_STEPS: int = 2  # Effective batch = 64
    LEARNING_RATE: float = 2e-3  # Scaled for larger batch
    WEIGHT_DECAY: float = 1e-4
    NUM_EPOCHS: int = 40  # Optimized for 45-hour quota
    VAL_INTERVAL: int = 2
    SAVE_INTERVAL: int = 5
    WARMUP_EPOCHS: int = 2

    # Training stabilization
    EPSILON: float = 1e-7
    GRAD_CLIP: float = 1.0
    USE_AMP: bool = True  # Critical for T4 Tensor Cores

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

    # Data sampling
    POS_NEG_RATIO: float = 1.5
    NUM_PATCHES_PER_VOLUME: int = 8

    # System (MULTI-GPU OPTIMIZED)
    NUM_WORKERS_PER_GPU: int = 4  # 4 workers per GPU
    PREFETCH_FACTOR: int = 3  # Aggressive prefetching
    SEED: int = 42

    # Multi-GPU settings
    WORLD_SIZE: int = torch.cuda.device_count()  # Number of GPUs
    BACKEND: str = 'nccl'  # Best for NVIDIA GPUs

    # Performance monitoring
    LOG_INTERVAL: int = 50  # Log every N batches
    PROFILE_GPU: bool = True  # Enable GPU profiling

    # Checkpoint settings for long runs
    CHECKPOINT_INTERVAL_MINUTES: int = 60  # Save every hour
    AUTO_RESUME: bool = True  # Auto-resume from latest checkpoint

    # Derived paths
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
        self.BASE_PATH = self._resolve_path(
            os.environ.get('VESUVIUS_BASE_PATH'),
            self.BASE_PATH
        )
        self.WORKING_DIR = self._resolve_path(
            os.environ.get('VESUVIUS_WORKING_DIR'),
            self.WORKING_DIR
        )

        self.TRAIN_IMAGES = self.BASE_PATH / 'train_images'
        self.TRAIN_LABELS = self.BASE_PATH / 'train_labels'
        self.TEST_IMAGES = self.BASE_PATH / 'test_images'
        self.TRAIN_CSV = self.BASE_PATH / 'train.csv'
        self.TEST_CSV = self.BASE_PATH / 'test.csv'

        self.MODEL_DIR = self.WORKING_DIR / 'models'
        self.PREDICTIONS_DIR = self.WORKING_DIR / 'predictions'
        self.LOGS_DIR = self.WORKING_DIR / 'logs'
        self.SUBMISSION_PATH = self.WORKING_DIR / 'submission.zip'

        for d in [self.WORKING_DIR, self.MODEL_DIR, self.PREDICTIONS_DIR, self.LOGS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_path(env_path: Optional[str], default_path: Path) -> Path:
        if env_path:
            return Path(env_path).expanduser().resolve()
        if default_path.exists():
            return default_path.resolve()
        return Path.cwd()

CFG = Config()

# ============================================================================
# MULTI-GPU UTILITIES
# ============================================================================

def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    dist.init_process_group(
        backend=CFG.BACKEND,
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Set device
    torch.cuda.set_device(rank)

    # Synchronize
    dist.barrier()

def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process() -> bool:
    """Check if current process is main (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0

def get_rank() -> int:
    """Get current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size() -> int:
    """Get total number of processes."""
    return dist.get_world_size() if dist.is_initialized() else 1

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int = CFG.SEED, rank: int = 0) -> None:
    """Set random seeds for reproducibility (rank-aware)."""
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Enable for performance with fixed input sizes

# ============================================================================
# LOGGING (Rank-aware)
# ============================================================================

class DistributedLogger:
    """Rank-aware logger for distributed training."""

    def __init__(self, log_file: Path, rank: int = 0):
        self.log_file = log_file
        self.rank = rank
        self.is_main = (rank == 0)

        if self.is_main:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.stats = defaultdict(int)

    def log(self, message: str, level: str = "INFO", force: bool = False) -> None:
        """Log message (only from main process unless forced)."""
        if not self.is_main and not force:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [GPU{self.rank}] {level:5s} | {message}"
        print(formatted)

        if self.is_main:
            with open(self.log_file, "a") as f:
                f.write(formatted + "\n")

        self.stats[level] += 1

    def info(self, msg: str, force: bool = False) -> None:
        self.log(msg, "INFO", force)

    def warn(self, msg: str, force: bool = False) -> None:
        self.log(msg, "WARN", force)

    def error(self, msg: str, force: bool = True) -> None:
        self.log(msg, "ERROR", force)

    def debug(self, msg: str, force: bool = False) -> None:
        self.log(msg, "DEBUG", force)

# ============================================================================
# GPU MONITORING
# ============================================================================

class GPUMonitor:
    """Monitor GPU utilization and memory."""

    def __init__(self, rank: int):
        self.rank = rank
        self.device = torch.device(f'cuda:{rank}')

    def get_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics."""
        if not torch.cuda.is_available():
            return {}

        return {
            'memory_allocated_gb': torch.cuda.memory_allocated(self.device) / 1e9,
            'memory_reserved_gb': torch.cuda.memory_reserved(self.device) / 1e9,
            'max_memory_allocated_gb': torch.cuda.max_memory_allocated(self.device) / 1e9,
        }

    def log_stats(self, logger: DistributedLogger, prefix: str = ""):
        """Log GPU statistics."""
        stats = self.get_stats()
        if stats:
            logger.info(
                f"{prefix}GPU{self.rank} Memory: "
                f"Allocated={stats['memory_allocated_gb']:.2f}GB, "
                f"Reserved={stats['memory_reserved_gb']:.2f}GB, "
                f"Peak={stats['max_memory_allocated_gb']:.2f}GB",
                force=True
            )

    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        torch.cuda.reset_peak_memory_stats(self.device)

# ============================================================================
# I/O UTILITIES (Same as before, keeping for completeness)
# ============================================================================

def load_tiff_volume(path: Path, validate: bool = True) -> np.ndarray:
    """Load multi-page TIFF as 3D numpy array."""
    if not path.exists():
        raise FileNotFoundError(f"TIFF not found: {path}")

    with Image.open(path) as img:
        slices = [np.array(page) for page in ImageSequence.Iterator(img)]

    volume = np.stack(slices, axis=0).astype(np.float32)

    if validate and (np.isnan(volume).any() or np.isinf(volume).any()):
        volume = np.nan_to_num(volume, nan=0.0, posinf=255.0, neginf=0.0)

    return volume

def save_tiff_volume(volume: np.ndarray, path: Path) -> None:
    """Save 3D numpy array as multi-page TIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if volume.dtype != np.uint8:
        volume = np.clip(volume, 0, 1).astype(np.float32)
        volume = (volume * 255).astype(np.uint8)

    pages = [Image.fromarray(volume[i], mode="L") for i in range(volume.shape[0])]
    pages[0].save(path, save_all=True, append_images=pages[1:], compression="tiff_deflate")

# ============================================================================
# LOSS FUNCTIONS (Same as before)
# ============================================================================

class StableDiceLoss(nn.Module):
    def __init__(self, epsilon: float = CFG.EPSILON, smooth: float = 1.0):
        super().__init__()
        self.epsilon = epsilon
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred_flat = pred.reshape(pred.size(0), pred.size(1), -1)
        target_flat = target.reshape(target.size(0), target.size(1), -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth + self.epsilon)

        return 1.0 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = CFG.FOCAL_ALPHA, gamma: float = CFG.FOCAL_GAMMA, epsilon: float = CFG.EPSILON):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(pred)
        pos_weight = target * (1.0 - pred_sigmoid).pow(self.gamma)
        neg_weight = (1.0 - target) * pred_sigmoid.pow(self.gamma)

        pos_loss = -self.alpha * pos_weight * F.logsigmoid(pred)
        neg_loss = -(1.0 - self.alpha) * neg_weight * F.logsigmoid(-pred)

        return (pos_loss + neg_loss).mean()

class CombinedLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = CFG.DICE_WEIGHT,
        focal_weight: float = CFG.FOCAL_WEIGHT,
        bce_weight: float = CFG.BCE_WEIGHT
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight

        self.dice_loss = StableDiceLoss()
        self.focal_loss = FocalLoss() if focal_weight > 0 else None
        self.bce_loss = nn.BCEWithLogitsLoss() if bce_weight > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0

        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice_loss(pred, target)

        if self.focal_weight > 0 and self.focal_loss is not None:
            loss = loss + self.focal_weight * self.focal_loss(pred, target)

        if self.bce_weight > 0 and self.bce_loss is not None:
            loss = loss + self.bce_weight * self.bce_loss(pred, target)

        return loss

# ============================================================================
# MODEL ARCHITECTURE (Same as before - keeping for completeness)
# ============================================================================

class ConvBlock3D(nn.Module):
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
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels, dropout)
        self.skip = (nn.Identity() if in_channels == out_channels
                     else nn.Conv3d(in_channels, out_channels, 1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)

class ChannelAttention3D(nn.Module):
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
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x_cat))

class AttentionBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel_att = ChannelAttention3D(channels)
        self.spatial_att = SpatialAttention3D()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class Down3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = (ResidualBlock3D(in_channels, out_channels, dropout) if CFG.USE_RESIDUAL
                     else ConvBlock3D(in_channels, out_channels, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))

class Up3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, 2, 2)
        self.conv = (ResidualBlock3D(in_channels, out_channels, dropout) if CFG.USE_RESIDUAL
                     else ConvBlock3D(in_channels, out_channels, dropout))
        self.attention = AttentionBlock3D(out_channels) if CFG.USE_ATTENTION else nn.Identity()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Pad to match
        diff_d = x2.size(2) - x1.size(2)
        diff_h = x2.size(3) - x1.size(3)
        diff_w = x2.size(4) - x1.size(4)

        if diff_d != 0 or diff_h != 0 or diff_w != 0:
            x1 = F.pad(x1, [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
                diff_d // 2, diff_d - diff_d // 2
            ])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        return x

class UNet3D(nn.Module):
    """3D U-Net optimized for multi-GPU training."""

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

        self.encoder_init = (ResidualBlock3D(in_channels, features) if CFG.USE_RESIDUAL
                             else ConvBlock3D(in_channels, features))

        self.encoder_blocks = nn.ModuleList()
        for i in range(depth):
            dropout = 0.2 * (i / (depth - 1)) if depth > 1 else 0.0
            self.encoder_blocks.append(
                Down3D(features * (2 ** i), features * (2 ** (i + 1)), dropout)
            )

        self.decoder_blocks = nn.ModuleList()
        for i in range(depth, 0, -1):
            dropout = 0.2 * ((depth - i) / depth)
            self.decoder_blocks.append(
                Up3D(features * (2 ** i), features * (2 ** (i - 1)), dropout)
            )

        self.out_conv = nn.Conv3d(features, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_init(x)
        skip_connections = [x]

        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            if i < self.depth - 1:
                skip_connections.append(x)

        for i, decoder in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i + 1)]
            x = decoder(x, skip)

        return self.out_conv(x)

# ============================================================================
# DATA AUGMENTATION (Same as before)
# ============================================================================

class VolumeAugmentation:
    def __init__(self, mode: str = "train"):
        self.mode = mode

    def __call__(self, image: np.ndarray, label: Optional[np.ndarray] = None):
        if self.mode != "train":
            return self._normalize(image), label

        if label is not None:
            image, label = self._random_flip(image, label)
            image, label = self._random_rotate90(image, label)

        image = self._random_brightness(image)
        image = self._random_contrast(image)
        image = self._normalize(image)

        return image, label

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        return np.clip(image, 0, 255) / 255.0

    @staticmethod
    def _random_flip(image: np.ndarray, label: np.ndarray, prob: float = 0.5):
        for axis in range(3):
            if random.random() < prob:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
        return image, label

    @staticmethod
    def _random_rotate90(image: np.ndarray, label: np.ndarray, prob: float = 0.5):
        if random.random() < prob:
            k = random.randint(0, 3)
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            label = np.rot90(label, k=k, axes=(1, 2)).copy()
        return image, label

    @staticmethod
    def _random_brightness(image: np.ndarray, prob: float = 0.3, max_factor: float = 0.2):
        if random.random() < prob:
            factor = 1.0 + random.uniform(-max_factor, max_factor)
            image = image * factor
        return image

    @staticmethod
    def _random_contrast(image: np.ndarray, prob: float = 0.3, max_factor: float = 0.3):
        if random.random() < prob:
            factor = 1.0 + random.uniform(-max_factor, max_factor)
            mean = image.mean()
            image = (image - mean) * factor + mean
        return image

# ============================================================================
# DATASET (Optimized for multi-GPU)
# ============================================================================

class VesuviusDataset(Dataset):
    """Dataset optimized for distributed training."""

    def __init__(
        self,
        csv_path: Path,
        image_dir: Path,
        label_dir: Optional[Path] = None,
        mode: str = "train",
        patch_size: Tuple[int, int, int] = CFG.PATCH_SIZE,
        num_patches: int = CFG.NUM_PATCHES_PER_VOLUME,
        pos_neg_ratio: float = CFG.POS_NEG_RATIO,
        rank: int = 0
    ):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mode = mode
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.pos_neg_ratio = pos_neg_ratio
        self.rank = rank

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            self.sample_ids = df["id"].astype(str).tolist()
        else:
            self.sample_ids = sorted([p.stem for p in image_dir.glob("*.tif")])

        self.augment = VolumeAugmentation(mode)

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.sample_ids) * self.num_patches
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.mode == "train":
            sample_idx = idx // self.num_patches
        else:
            sample_idx = idx

        sample_id = self.sample_ids[sample_idx]

        image_path = self.image_dir / f"{sample_id}.tif"
        image = load_tiff_volume(image_path)

        label = None
        if self.label_dir is not None:
            label_path = self.label_dir / f"{sample_id}.tif"
            if label_path.exists():
                label = load_tiff_volume(label_path)

        if self.mode == "train" and label is not None:
            image, label = self._extract_balanced_patch(image, label)
            image, label = self.augment(image, label)

            image = torch.from_numpy(image[None]).float()
            label = torch.from_numpy(label[None]).float()
            label = torch.clamp(label, 0, 1)

            return {"image": image, "label": label, "id": sample_id}

        image, _ = self.augment(image, None)
        return {"image": image, "id": sample_id, "original_shape": image.shape}

    def _extract_balanced_patch(self, image: np.ndarray, label: np.ndarray):
        d, h, w = image.shape
        pd, ph, pw = self.patch_size

        if d < pd or h < ph or w < pw:
            image = self._pad_to_size(image, self.patch_size)
            label = self._pad_to_size(label, self.patch_size)
            return image, label

        label_sum = label.sum()
        sample_positive = random.random() < (self.pos_neg_ratio / (1 + self.pos_neg_ratio))

        if sample_positive and label_sum > 0:
            positive_coords = np.argwhere(label > 0)
            center = positive_coords[random.randint(0, len(positive_coords) - 1)]
        else:
            center = np.array([
                random.randint(pd // 2, d - pd // 2) if d > pd else d // 2,
                random.randint(ph // 2, h - ph // 2) if h > ph else h // 2,
                random.randint(pw // 2, w - pw // 2) if w > pw else w // 2
            ])

        d_start = max(0, min(center[0] - pd // 2, d - pd))
        h_start = max(0, min(center[1] - ph // 2, h - ph))
        w_start = max(0, min(center[2] - pw // 2, w - pw))

        image_patch = image[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        label_patch = label[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]

        if image_patch.shape != self.patch_size:
            image_patch = self._pad_to_size(image_patch, self.patch_size)
            label_patch = self._pad_to_size(label_patch, self.patch_size)

        return image_patch, label_patch

    @staticmethod
    def _pad_to_size(array: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
        pad_width = [(0, max(0, target - current)) for target, current in zip(target_size, array.shape)]
        return np.pad(array, pad_width, mode="constant", constant_values=0)

# ============================================================================
# DISTRIBUTED TRAINER
# ============================================================================

class DistributedTrainer:
    """Multi-GPU trainer with gradient accumulation and advanced monitoring."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        rank: int,
        world_size: int
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')

        # Wrap model with DDP
        self.model = DDP(model.to(self.device), device_ids=[rank])

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.logger = DistributedLogger(CFG.LOGS_DIR / f"training_rank{rank}.log", rank)
        self.gpu_monitor = GPUMonitor(rank)

        # Training state
        self.best_dice = 0.0
        self.current_epoch = 0
        self.global_step = 0
        self.train_losses = []
        self.val_dices = []
        self.last_checkpoint_time = time.time()

        # Mixed precision
        self.use_amp = CFG.USE_AMP
        if self.use_amp:
            from torch.amp import GradScaler
            self.scaler = GradScaler('cuda')
            self.logger.info("Using mixed precision (torch.amp)")

        self.logger.info(f"Initialized trainer on GPU {rank}/{world_size}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with gradient accumulation."""
        self.model.train()
        epoch_loss = 0.0
        valid_batches = 0

        if is_main_process():
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{CFG.NUM_EPOCHS}")
        else:
            pbar = self.train_loader

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            if self.use_amp:
                from torch.amp import autocast
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / CFG.GRADIENT_ACCUMULATION_STEPS

                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (batch_idx + 1) % CFG.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), CFG.GRAD_CLIP)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / CFG.GRADIENT_ACCUMULATION_STEPS

                loss.backward()

                if (batch_idx + 1) % CFG.GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), CFG.GRAD_CLIP)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

            # Accumulate loss
            if not torch.isnan(loss):
                epoch_loss += loss.item() * CFG.GRADIENT_ACCUMULATION_STEPS
                valid_batches += 1

                if is_main_process():
                    pbar.set_postfix({"loss": f"{loss.item() * CFG.GRADIENT_ACCUMULATION_STEPS:.4f}"})

            # Periodic logging
            if batch_idx % CFG.LOG_INTERVAL == 0 and is_main_process():
                self.gpu_monitor.log_stats(self.logger, f"Batch {batch_idx}: ")

            # Time-based checkpointing (for long runs)
            if time.time() - self.last_checkpoint_time > CFG.CHECKPOINT_INTERVAL_MINUTES * 60:
                if is_main_process():
                    self.save_checkpoint(epoch, epoch_loss / max(valid_batches, 1), "auto_checkpoint")
                    self.last_checkpoint_time = time.time()

        avg_loss = epoch_loss / max(valid_batches, 1)
        self.train_losses.append(avg_loss)

        if self.scheduler:
            self.scheduler.step()

        return avg_loss

    def save_checkpoint(self, epoch: int, metric: float, name: str = "checkpoint"):
        """Save checkpoint (only from main process)."""
        if not is_main_process():
            return

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state": self.model.module.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "metric": metric,
            "best_dice": self.best_dice,
            "train_losses": self.train_losses,
            "val_dices": self.val_dices,
        }

        path = CFG.MODEL_DIR / f"{name}.pth"
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path):
        """Load checkpoint."""
        if not path.exists():
            return False

        checkpoint = torch.load(path, map_location=self.device)
        self.model.module.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if self.scheduler and checkpoint.get("scheduler_state"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_dice = checkpoint.get("best_dice", 0.0)
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_dices = checkpoint.get("val_dices", [])

        self.logger.info(f"Loaded checkpoint from {path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}, best dice: {self.best_dice:.4f}")
        return True

    def train(self):
        """Full training loop."""
        if is_main_process():
            self.logger.info("=" * 80)
            self.logger.info("STARTING MULTI-GPU TRAINING")
            self.logger.info("=" * 80)
            self.logger.info(f"World size: {self.world_size} GPUs")
            self.logger.info(f"Effective batch size: {CFG.BATCH_SIZE_PER_GPU * self.world_size * CFG.GRADIENT_ACCUMULATION_STEPS}")
            self.logger.info(f"Batches per GPU: {len(self.train_loader)}")

        # Auto-resume
        if CFG.AUTO_RESUME:
            auto_checkpoint = CFG.MODEL_DIR / "auto_checkpoint.pth"
            if auto_checkpoint.exists():
                self.load_checkpoint(auto_checkpoint)

        start_epoch = self.current_epoch + 1

        for epoch in range(start_epoch, CFG.NUM_EPOCHS + 1):
            self.current_epoch = epoch

            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            # Train
            loss = self.train_epoch(epoch)
            self.logger.info(f"Epoch {epoch}/{CFG.NUM_EPOCHS} - Loss: {loss:.4f}")

            # Log learning rate
            if is_main_process():
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(f"Learning Rate: {current_lr:.6f}")
                self.gpu_monitor.log_stats(self.logger, f"Epoch {epoch} end: ")

            # Periodic checkpoint
            if epoch % CFG.SAVE_INTERVAL == 0 and is_main_process():
                self.save_checkpoint(epoch, loss, f"checkpoint_epoch_{epoch}")

        if is_main_process():
            self.logger.info("=" * 80)
            self.logger.info(f"TRAINING COMPLETE - Best Dice: {self.best_dice:.4f}")
            self.logger.info("=" * 80)

# ============================================================================
# MAIN DISTRIBUTED TRAINING FUNCTION
# ============================================================================

def train_distributed(rank: int, world_size: int):
    """Main training function for each GPU process."""

    # Setup distributed
    setup_distributed(rank, world_size)
    set_seed(CFG.SEED, rank)

    # Create logger
    logger = DistributedLogger(CFG.LOGS_DIR / f"setup_rank{rank}.log", rank)
    logger.info(f"Initialized process rank {rank}/{world_size}")

    # Create datasets
    train_dataset = VesuviusDataset(
        CFG.TRAIN_CSV,
        CFG.TRAIN_IMAGES,
        CFG.TRAIN_LABELS,
        mode="train",
        rank=rank
    )

    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=CFG.SEED
    )

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE_PER_GPU,
        sampler=train_sampler,
        num_workers=CFG.NUM_WORKERS_PER_GPU,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=CFG.PREFETCH_FACTOR
    )

    logger.info(f"Created data loader: {len(train_loader)} batches per GPU")

    # Create model and training components
    model = UNet3D()
    criterion = CombinedLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.LEARNING_RATE,
        weight_decay=CFG.WEIGHT_DECAY
    )

    # Cosine annealing scheduler (better for multi-GPU)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    # Create trainer
    trainer = DistributedTrainer(
        model,
        train_loader,
        None,  # No validation for now (speeds up training)
        criterion,
        optimizer,
        scheduler,
        rank,
        world_size
    )

    # Train
    trainer.train()

    # Cleanup
    cleanup_distributed()

# ============================================================================
# INFERENCE (Single GPU)
# ============================================================================

def predict_test_set_single_gpu():
    """Generate predictions using best model (single GPU for inference)."""

    logger = DistributedLogger(CFG.LOGS_DIR / "inference.log", 0)
    logger.info("Starting test inference...")

    # Load model
    best_model_path = CFG.MODEL_DIR / "checkpoint_epoch_40.pth"  # Use final checkpoint
    if not best_model_path.exists():
        # Try auto checkpoint
        best_model_path = CFG.MODEL_DIR / "auto_checkpoint.pth"

    if not best_model_path.exists():
        logger.error("No model checkpoint found!")
        return

    model = UNet3D()
    checkpoint = torch.load(best_model_path, map_location='cuda:0')
    model.load_state_dict(checkpoint["model_state"])
    model = model.to('cuda:0')
    model.eval()

    logger.info(f"Loaded model from {best_model_path}")

    # Load test samples
    if CFG.TEST_CSV.exists():
        df = pd.read_csv(CFG.TEST_CSV)
        sample_ids = df["id"].astype(str).tolist()
    else:
        sample_ids = sorted([p.stem for p in CFG.TEST_IMAGES.glob("*.tif")])

    logger.info(f"Predicting {len(sample_ids)} test volumes...")

    for sample_id in tqdm(sample_ids, desc="Inference"):
        image_path = CFG.TEST_IMAGES / f"{sample_id}.tif"
        if not image_path.exists():
            continue

        try:
            volume = load_tiff_volume(image_path)
            prediction = predict_simple(model, volume, 'cuda:0')
            binary = (prediction > CFG.INFERENCE_THRESHOLD).astype(np.uint8)
            save_tiff_volume(binary, CFG.PREDICTIONS_DIR / f"{sample_id}.tif")
        except Exception as e:
            logger.error(f"Failed to predict {sample_id}: {e}")

    # Create submission
    create_submission_zip()
    logger.info("Inference complete!")

def predict_simple(model: nn.Module, volume: np.ndarray, device: str) -> np.ndarray:
    """Simple tile-based prediction."""
    volume = np.clip(volume / 255.0, 0, 1).astype(np.float32)
    d, h, w = volume.shape
    pd, ph, pw = CFG.PATCH_SIZE

    output = np.zeros(volume.shape, dtype=np.float32)
    counts = np.zeros(volume.shape, dtype=np.float32)

    stride_d, stride_h, stride_w = pd // 2, ph // 2, pw // 2

    with torch.no_grad():
        for d_start in range(0, max(1, d - pd + 1), stride_d):
            for h_start in range(0, max(1, h - ph + 1), stride_h):
                for w_start in range(0, max(1, w - pw + 1), stride_w):
                    patch = volume[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]

                    if patch.shape != CFG.PATCH_SIZE:
                        continue

                    patch_tensor = torch.from_numpy(patch[None, None]).float().to(device)
                    pred = torch.sigmoid(model(patch_tensor))[0, 0].cpu().numpy()

                    output[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw] += pred
                    counts[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw] += 1

    return output / np.maximum(counts, 1)

def create_submission_zip():
    """Create submission ZIP."""
    with zipfile.ZipFile(CFG.SUBMISSION_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for tif_path in sorted(CFG.PREDICTIONS_DIR.glob("*.tif")):
            zipf.write(tif_path, arcname=tif_path.name)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""

    print("\n" + "=" * 80)
    print("VESUVIUS CHALLENGE - MULTI-GPU ENHANCED PIPELINE")
    print(f"Detected {CFG.WORLD_SIZE} GPUs")
    print("=" * 80 + "\n")

    if CFG.WORLD_SIZE < 2:
        print("WARNING: Less than 2 GPUs detected. Using single GPU mode.")
        print("For optimal performance, use 2x T4 GPUs.")

    # Check for existing model
    final_checkpoint = CFG.MODEL_DIR / "checkpoint_epoch_40.pth"
    auto_checkpoint = CFG.MODEL_DIR / "auto_checkpoint.pth"

    if final_checkpoint.exists() or (auto_checkpoint.exists() and not CFG.AUTO_RESUME):
        print("Found existing model. Skipping training, running inference...")
        predict_test_set_single_gpu()
    else:
        print(f"Starting distributed training on {CFG.WORLD_SIZE} GPUs...")

        # Launch distributed training
        if CFG.WORLD_SIZE > 1:
            mp.spawn(
                train_distributed,
                args=(CFG.WORLD_SIZE,),
                nprocs=CFG.WORLD_SIZE,
                join=True
            )
        else:
            # Fallback to single GPU
            train_distributed(0, 1)

        # After training, run inference
        print("\nTraining complete! Running inference...")
        predict_test_set_single_gpu()

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
