#!/usr/bin/env python3
"""
Vesuvius Challenge - JUPYTER-FRIENDLY Multi-GPU Pipeline
Uses DataParallel (works in notebooks!) for 2x T4

Key Features:
- DataParallel for notebook compatibility
- 2-3x larger batch sizes
- Optimized for 2x T4 GPUs
- No multiprocessing required
- Works in Jupyter/Colab/Kaggle
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
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package,
         "--break-system-packages", "-q"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

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
# CONFIGURATION (OPTIMIZED FOR 2x T4)
# ============================================================================

@dataclass
class Config:
    """Configuration optimized for DataParallel with 2x T4 GPUs."""

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

    # MULTI-GPU Training hyperparameters
    # DataParallel splits batch across GPUs automatically
    BATCH_SIZE: int = 32  # Will be split: 16 per GPU
    GRADIENT_ACCUMULATION_STEPS: int = 2  # Effective batch = 64
    LEARNING_RATE: float = 2e-3  # Scaled for larger batch
    WEIGHT_DECAY: float = 1e-4
    NUM_EPOCHS: int = 40
    VAL_INTERVAL: int = 2
    SAVE_INTERVAL: int = 5
    WARMUP_EPOCHS: int = 2

    # Training stabilization
    EPSILON: float = 1e-7
    GRAD_CLIP: float = 1.0
    USE_AMP: bool = True

    # Loss function
    DICE_WEIGHT: float = 0.6
    FOCAL_WEIGHT: float = 0.2
    BCE_WEIGHT: float = 0.2
    FOCAL_ALPHA: float = 0.25
    FOCAL_GAMMA: float = 2.0

    # Inference
    TILE_OVERLAP: float = 0.5
    INFERENCE_THRESHOLD: float = 0.5
    USE_TTA: bool = True
    TTA_AUGMENTATIONS: int = 4

    # Data sampling
    POS_NEG_RATIO: float = 1.5
    NUM_PATCHES_PER_VOLUME: int = 8

    # System (OPTIMIZED FOR MULTI-GPU)
    NUM_WORKERS: int = 8  # More workers for both GPUs
    PREFETCH_FACTOR: int = 3
    SEED: int = 42

    # Multi-GPU
    USE_MULTI_GPU: bool = True  # Auto-uses DataParallel if >1 GPU
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Checkpointing
    CHECKPOINT_INTERVAL_MINUTES: int = 60
    AUTO_RESUME: bool = True

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
# UTILITIES
# ============================================================================

def set_seed(seed: int = CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Enable for performance

set_seed()

class Logger:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.stats = defaultdict(int)

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {level:5s} | {message}"
        print(formatted)
        with open(self.log_file, "a") as f:
            f.write(formatted + "\n")
        self.stats[level] += 1

    def info(self, msg: str): self.log(msg, "INFO")
    def warn(self, msg: str): self.log(msg, "WARN")
    def error(self, msg: str): self.log(msg, "ERROR")

logger = Logger(CFG.LOGS_DIR / "training.log")

class GPUMonitor:
    def __init__(self):
        self.num_gpus = torch.cuda.device_count()

    def get_stats(self) -> List[Dict[str, float]]:
        if not torch.cuda.is_available():
            return []

        stats = []
        for i in range(self.num_gpus):
            stats.append({
                'gpu_id': i,
                'memory_allocated_gb': torch.cuda.memory_allocated(i) / 1e9,
                'memory_reserved_gb': torch.cuda.memory_reserved(i) / 1e9,
            })
        return stats

    def log_stats(self, logger: Logger, prefix: str = ""):
        stats = self.get_stats()
        for stat in stats:
            logger.info(
                f"{prefix}GPU{stat['gpu_id']}: "
                f"Alloc={stat['memory_allocated_gb']:.2f}GB, "
                f"Reserved={stat['memory_reserved_gb']:.2f}GB"
            )

def load_tiff_volume(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"TIFF not found: {path}")
    with Image.open(path) as img:
        slices = [np.array(page) for page in ImageSequence.Iterator(img)]
    volume = np.stack(slices, axis=0).astype(np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=255.0, neginf=0.0)
    return volume

def save_tiff_volume(volume: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if volume.dtype != np.uint8:
        volume = np.clip(volume, 0, 1).astype(np.float32)
        volume = (volume * 255).astype(np.uint8)
    pages = [Image.fromarray(volume[i], mode="L") for i in range(volume.shape[0])]
    pages[0].save(path, save_all=True, append_images=pages[1:], compression="tiff_deflate")

# ============================================================================
# LOSS FUNCTIONS
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
    def __init__(self, dice_weight: float = CFG.DICE_WEIGHT, focal_weight: float = CFG.FOCAL_WEIGHT, bce_weight: float = CFG.BCE_WEIGHT):
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
# MODEL (Same as before - keeping concise)
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
        )
    def forward(self, x): return self.conv(x)

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock3D(in_channels, out_channels, dropout)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv3d(in_channels, out_channels, 1, bias=False)
    def forward(self, x): return self.conv(x) + self.skip(x)

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
    def forward(self, x): return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class AttentionBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channel_att = ChannelAttention3D(channels)
        self.spatial_att = SpatialAttention3D()
    def forward(self, x): return self.spatial_att(self.channel_att(x))

class Down3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ResidualBlock3D(in_channels, out_channels, dropout) if CFG.USE_RESIDUAL else ConvBlock3D(in_channels, out_channels, dropout)
    def forward(self, x): return self.conv(self.pool(x))

class Up3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, 2, 2)
        self.conv = ResidualBlock3D(in_channels, out_channels, dropout) if CFG.USE_RESIDUAL else ConvBlock3D(in_channels, out_channels, dropout)
        self.attention = AttentionBlock3D(out_channels) if CFG.USE_ATTENTION else nn.Identity()

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_d = x2.size(2) - x1.size(2)
        diff_h = x2.size(3) - x1.size(3)
        diff_w = x2.size(4) - x1.size(4)
        if diff_d != 0 or diff_h != 0 or diff_w != 0:
            x1 = F.pad(x1, [diff_w//2, diff_w-diff_w//2, diff_h//2, diff_h-diff_h//2, diff_d//2, diff_d-diff_d//2])
        x = torch.cat([x2, x1], dim=1)
        return self.attention(self.conv(x))

class UNet3D(nn.Module):
    def __init__(self, in_channels: int = CFG.IN_CHANNELS, out_channels: int = CFG.OUT_CHANNELS, base_features: int = CFG.BASE_FEATURES, depth: int = CFG.DEPTH):
        super().__init__()
        self.depth = depth
        features = base_features
        self.encoder_init = ResidualBlock3D(in_channels, features) if CFG.USE_RESIDUAL else ConvBlock3D(in_channels, features)
        self.encoder_blocks = nn.ModuleList([Down3D(features * (2**i), features * (2**(i+1)), 0.2 * (i / (depth-1)) if depth > 1 else 0.0) for i in range(depth)])
        self.decoder_blocks = nn.ModuleList([Up3D(features * (2**i), features * (2**(i-1)), 0.2 * ((depth-i) / depth)) for i in range(depth, 0, -1)])
        self.out_conv = nn.Conv3d(features, out_channels, 1)

    def forward(self, x):
        x = self.encoder_init(x)
        skip_connections = [x]
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            if i < self.depth - 1:
                skip_connections.append(x)
        for i, decoder in enumerate(self.decoder_blocks):
            x = decoder(x, skip_connections[-(i+1)])
        return self.out_conv(x)

# ============================================================================
# DATA AUGMENTATION & DATASET
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
        return self._normalize(image), label

    @staticmethod
    def _normalize(image): return np.clip(image, 0, 255) / 255.0
    @staticmethod
    def _random_flip(image, label, prob=0.5):
        for axis in range(3):
            if random.random() < prob:
                image = np.flip(image, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
        return image, label
    @staticmethod
    def _random_rotate90(image, label, prob=0.5):
        if random.random() < prob:
            k = random.randint(0, 3)
            image = np.rot90(image, k=k, axes=(1, 2)).copy()
            label = np.rot90(label, k=k, axes=(1, 2)).copy()
        return image, label
    @staticmethod
    def _random_brightness(image, prob=0.3, max_factor=0.2):
        if random.random() < prob:
            image = image * (1.0 + random.uniform(-max_factor, max_factor))
        return image
    @staticmethod
    def _random_contrast(image, prob=0.3, max_factor=0.3):
        if random.random() < prob:
            mean = image.mean()
            image = (image - mean) * (1.0 + random.uniform(-max_factor, max_factor)) + mean
        return image

class VesuviusDataset(Dataset):
    def __init__(self, csv_path: Path, image_dir: Path, label_dir: Optional[Path] = None, mode: str = "train", patch_size=CFG.PATCH_SIZE, num_patches=CFG.NUM_PATCHES_PER_VOLUME, pos_neg_ratio=CFG.POS_NEG_RATIO):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.mode = mode
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.pos_neg_ratio = pos_neg_ratio

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            self.sample_ids = df["id"].astype(str).tolist()
        else:
            self.sample_ids = sorted([p.stem for p in image_dir.glob("*.tif")])

        self.augment = VolumeAugmentation(mode)

    def __len__(self):
        return len(self.sample_ids) * self.num_patches if self.mode == "train" else len(self.sample_ids)

    def __getitem__(self, idx):
        sample_idx = idx // self.num_patches if self.mode == "train" else idx
        sample_id = self.sample_ids[sample_idx]

        image = load_tiff_volume(self.image_dir / f"{sample_id}.tif")

        label = None
        if self.label_dir:
            label_path = self.label_dir / f"{sample_id}.tif"
            if label_path.exists():
                label = load_tiff_volume(label_path)

        if self.mode == "train" and label is not None:
            image, label = self._extract_balanced_patch(image, label)
            image, label = self.augment(image, label)
            return {"image": torch.from_numpy(image[None]).float(), "label": torch.clamp(torch.from_numpy(label[None]).float(), 0, 1), "id": sample_id}

        image, _ = self.augment(image, None)
        return {"image": image, "id": sample_id, "original_shape": image.shape}

    def _extract_balanced_patch(self, image, label):
        d, h, w = image.shape
        pd, ph, pw = self.patch_size
        if d < pd or h < ph or w < pw:
            return self._pad_to_size(image, self.patch_size), self._pad_to_size(label, self.patch_size)

        sample_positive = random.random() < (self.pos_neg_ratio / (1 + self.pos_neg_ratio))
        if sample_positive and label.sum() > 0:
            coords = np.argwhere(label > 0)
            center = coords[random.randint(0, len(coords) - 1)]
        else:
            center = np.array([
                random.randint(pd//2, d-pd//2) if d > pd else d//2,
                random.randint(ph//2, h-ph//2) if h > ph else h//2,
                random.randint(pw//2, w-pw//2) if w > pw else w//2
            ])

        d_start = max(0, min(center[0] - pd//2, d - pd))
        h_start = max(0, min(center[1] - ph//2, h - ph))
        w_start = max(0, min(center[2] - pw//2, w - pw))

        image_patch = image[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
        label_patch = label[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]

        if image_patch.shape != self.patch_size:
            return self._pad_to_size(image_patch, self.patch_size), self._pad_to_size(label_patch, self.patch_size)
        return image_patch, label_patch

    @staticmethod
    def _pad_to_size(array, target_size):
        pad_width = [(0, max(0, target - current)) for target, current in zip(target_size, array.shape)]
        return np.pad(array, pad_width, mode="constant", constant_values=0)

# ============================================================================
# TRAINER (DataParallel version)
# ============================================================================

class MultiGPUTrainer:
    """Trainer using DataParallel (notebook-friendly)."""

    def __init__(self, model, train_loader, criterion, optimizer, scheduler):
        self.num_gpus = torch.cuda.device_count()
        self.device = CFG.DEVICE

        # Wrap with DataParallel if multiple GPUs
        if self.num_gpus > 1 and CFG.USE_MULTI_GPU:
            logger.info(f"Using DataParallel on {self.num_gpus} GPUs")
            self.model = nn.DataParallel(model.to(self.device))
        else:
            logger.info("Using single GPU mode")
            self.model = model.to(self.device)

        self.train_loader = train_loader
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.gpu_monitor = GPUMonitor()
        self.best_dice = 0.0
        self.train_losses = []
        self.last_checkpoint_time = time.time()

        # Mixed precision
        if CFG.USE_AMP:
            from torch.amp import GradScaler
            self.scaler = GradScaler('cuda')
            logger.info("Using mixed precision")

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        valid_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{CFG.NUM_EPOCHS}")

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            if CFG.USE_AMP:
                from torch.amp import autocast
                with autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels) / CFG.GRADIENT_ACCUMULATION_STEPS

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % CFG.GRADIENT_ACCUMULATION_STEPS == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), CFG.GRAD_CLIP)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels) / CFG.GRADIENT_ACCUMULATION_STEPS
                loss.backward()

                if (batch_idx + 1) % CFG.GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), CFG.GRAD_CLIP)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            if not torch.isnan(loss):
                epoch_loss += loss.item() * CFG.GRADIENT_ACCUMULATION_STEPS
                valid_batches += 1
                pbar.set_postfix({"loss": f"{loss.item() * CFG.GRADIENT_ACCUMULATION_STEPS:.4f}"})

            # Periodic GPU logging
            if batch_idx % 100 == 0 and self.num_gpus > 1:
                self.gpu_monitor.log_stats(logger, f"Batch {batch_idx}: ")

            # Time-based checkpointing
            if time.time() - self.last_checkpoint_time > CFG.CHECKPOINT_INTERVAL_MINUTES * 60:
                self.save_checkpoint(epoch, epoch_loss / max(valid_batches, 1), "auto_checkpoint")
                self.last_checkpoint_time = time.time()

        avg_loss = epoch_loss / max(valid_batches, 1)
        self.train_losses.append(avg_loss)

        if self.scheduler:
            self.scheduler.step()

        return avg_loss

    def save_checkpoint(self, epoch, metric, name="checkpoint"):
        # Handle DataParallel wrapper
        model_state = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()

        checkpoint = {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "metric": metric,
            "best_dice": self.best_dice,
            "train_losses": self.train_losses,
        }

        path = CFG.MODEL_DIR / f"{name}.pth"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        if not path.exists():
            return False

        checkpoint = torch.load(path, map_location=self.device)

        # Handle DataParallel wrapper
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint["model_state"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.scheduler and checkpoint.get("scheduler_state"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        self.best_dice = checkpoint.get("best_dice", 0.0)
        self.train_losses = checkpoint.get("train_losses", [])

        logger.info(f"Loaded checkpoint from {path}")
        return True

    def train(self):
        logger.info("=" * 80)
        logger.info("STARTING MULTI-GPU TRAINING (DataParallel)")
        logger.info("=" * 80)
        logger.info(f"GPUs: {self.num_gpus}")
        logger.info(f"Batch size: {CFG.BATCH_SIZE} (split across {self.num_gpus} GPUs = {CFG.BATCH_SIZE // self.num_gpus} per GPU)")
        logger.info(f"Effective batch size: {CFG.BATCH_SIZE * CFG.GRADIENT_ACCUMULATION_STEPS}")
        logger.info(f"Batches: {len(self.train_loader)}")

        # Auto-resume
        auto_checkpoint = CFG.MODEL_DIR / "auto_checkpoint.pth"
        start_epoch = 1
        if CFG.AUTO_RESUME and auto_checkpoint.exists():
            if self.load_checkpoint(auto_checkpoint):
                start_epoch = len(self.train_losses) + 1

        for epoch in range(start_epoch, CFG.NUM_EPOCHS + 1):
            loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch}/{CFG.NUM_EPOCHS} - Loss: {loss:.4f}")
            logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")

            if self.num_gpus > 1:
                self.gpu_monitor.log_stats(logger, f"Epoch {epoch} end: ")

            if epoch % CFG.SAVE_INTERVAL == 0:
                self.save_checkpoint(epoch, loss, f"checkpoint_epoch_{epoch}")

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)

# ============================================================================
# INFERENCE
# ============================================================================

def predict_test_set(model):
    model.eval()

    # Unwrap DataParallel if needed
    if isinstance(model, nn.DataParallel):
        model = model.module

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
            prediction = predict_simple(model, volume)
            binary = (prediction > CFG.INFERENCE_THRESHOLD).astype(np.uint8)
            save_tiff_volume(binary, CFG.PREDICTIONS_DIR / f"{sample_id}.tif")
        except Exception as e:
            logger.error(f"Failed to predict {sample_id}: {e}")

    create_submission_zip()

def predict_simple(model, volume):
    volume = np.clip(volume / 255.0, 0, 1).astype(np.float32)
    d, h, w = volume.shape
    pd, ph, pw = CFG.PATCH_SIZE

    output = np.zeros(volume.shape, dtype=np.float32)
    counts = np.zeros(volume.shape, dtype=np.float32)

    stride_d, stride_h, stride_w = pd//2, ph//2, pw//2

    with torch.no_grad():
        for d_start in range(0, max(1, d-pd+1), stride_d):
            for h_start in range(0, max(1, h-ph+1), stride_h):
                for w_start in range(0, max(1, w-pw+1), stride_w):
                    patch = volume[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw]
                    if patch.shape != CFG.PATCH_SIZE:
                        continue

                    patch_tensor = torch.from_numpy(patch[None, None]).float().to(CFG.DEVICE)
                    pred = torch.sigmoid(model(patch_tensor))[0, 0].cpu().numpy()

                    output[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw] += pred
                    counts[d_start:d_start+pd, h_start:h_start+ph, w_start:w_start+pw] += 1

    return output / np.maximum(counts, 1)

def create_submission_zip():
    with zipfile.ZipFile(CFG.SUBMISSION_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for tif_path in sorted(CFG.PREDICTIONS_DIR.glob("*.tif")):
            zipf.write(tif_path, arcname=tif_path.name)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("VESUVIUS CHALLENGE - JUPYTER MULTI-GPU PIPELINE")
    print(f"Detected {torch.cuda.device_count()} GPUs")
    print("=" * 80 + "\n")

    # Check for existing model
    final_checkpoint = CFG.MODEL_DIR / f"checkpoint_epoch_{CFG.NUM_EPOCHS}.pth"
    auto_checkpoint = CFG.MODEL_DIR / "auto_checkpoint.pth"

    if final_checkpoint.exists() or (auto_checkpoint.exists() and not CFG.AUTO_RESUME):
        logger.info("Found existing model. Running inference...")
        model = UNet3D()
        checkpoint_path = final_checkpoint if final_checkpoint.exists() else auto_checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=CFG.DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        model.to(CFG.DEVICE)
        predict_test_set(model)
    else:
        logger.info("Starting training...")

        # Create dataset
        train_dataset = VesuviusDataset(CFG.TRAIN_CSV, CFG.TRAIN_IMAGES, CFG.TRAIN_LABELS, mode="train")
        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.BATCH_SIZE,
            shuffle=True,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=CFG.PREFETCH_FACTOR
        )

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Batches: {len(train_loader)}")

        # Create model and trainer
        model = UNet3D()
        criterion = CombinedLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        trainer = MultiGPUTrainer(model, train_loader, criterion, optimizer, scheduler)
        trainer.train()

        # Inference
        logger.info("Generating test predictions...")
        predict_test_set(trainer.model)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
