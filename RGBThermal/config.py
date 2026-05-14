"""Configuration for Multispectral (RGB + Thermal) Object Detection on M3FD."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

_HERE = Path(__file__).parent


@dataclass
class DataConfig:
    """Dataset configuration for M3FD."""
    root: str = str(_HERE / "data" / "M3FD")
    rgb_dir: str = "Vis"         # visible (RGB) images
    thermal_dir: str = "Ir"      # infrared (thermal) images
    ann_dir: str = "Annotation"  # VOC XML annotations

    # Train/val split ratio
    val_ratio: float = 0.2
    split_seed: int = 42

    # Target classes (M3FD classes)
    target_classes: List[str] = field(
        default_factory=lambda: ["People", "Car", "Bus", "Motorcycle", "Lamp", "Truck"]
    )
    num_classes: int = 6  # excluding background

    # Image size (H, W) — resize for training
    img_size: Tuple[int, int] = (640, 640)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    backbone: str = "resnet50"  # resnet50, resnet101
    pretrained: bool = True

    # Cross-Modal Attention Fusion
    fusion_num_heads: int = 8
    fusion_dropout: float = 0.1

    # FPN channels
    fpn_out_channels: int = 256

    # Faster R-CNN
    # 4 levels: P3, P4, P5, pool (from FPN with LastLevelMaxPool)
    rpn_anchor_sizes: Tuple = ((32,), (64,), (128,), (256,))
    rpn_aspect_ratios: Tuple = ((0.5, 1.0, 2.0),) * 4
    roi_score_thresh: float = 0.05
    roi_nms_thresh: float = 0.5
    roi_detections_per_img: int = 100

    # Backbone feature channels (ResNet-50/101)
    backbone_channels: Tuple[int, ...] = (256, 512, 1024, 2048)


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 30
    batch_size: int = 8
    num_workers: int = 4

    # Optimizer
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005

    # LR scheduler
    lr_step_size: int = 10
    lr_gamma: float = 0.1
    warmup_epochs: int = 1
    warmup_factor: float = 0.001

    # Augmentation
    use_augmentation: bool = True
    hflip_prob: float = 0.5

    # Checkpointing
    save_dir: str = str(_HERE / "runs")
    save_every: int = 5
    resume: str = ""  # path to checkpoint to resume from

    # Logging
    print_freq: int = 50

    # Mixed precision
    amp: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
