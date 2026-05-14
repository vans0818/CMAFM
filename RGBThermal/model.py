"""
Dual-Backbone Multispectral Detector with Cross-Modal Attention Fusion.

Architecture:
    RGB  ──► ResNet-50 ──► {C2,C3,C4,C5} ──┐
                                             ├──► Cross-Modal Attention ──► FPN ──► Faster R-CNN Head
    Thermal ──► ResNet-50 ──► {C2,C3,C4,C5} ─┘
"""

import math
from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101, ResNet50_Weights, ResNet101_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from config import ModelConfig


# ---------------------------------------------------------------------------
# Cross-Modal Attention Fusion Module
# ---------------------------------------------------------------------------

class CrossModalAttentionFusion(nn.Module):
    """
    Efficient bi-directional cross-modal attention fusion.

    Combines two complementary attention mechanisms:
      1. Channel cross-attention: global context exchange (GAP → cross-attend → excite)
      2. Spatial cross-attention: local detail exchange (depthwise conv + cross-gating)
      3. Gated fusion with residual

    Memory-efficient: avoids N×N spatial attention matrices.
    """

    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        # --- Channel cross-attention (lightweight: C×C attention, not H*W × H*W) ---
        self.rgb_channel_q = nn.Linear(channels, channels)
        self.thermal_channel_kv = nn.Linear(channels, channels * 2)
        self.thermal_channel_q = nn.Linear(channels, channels)
        self.rgb_channel_kv = nn.Linear(channels, channels * 2)
        self.channel_scale = (channels // num_heads) ** -0.5

        # --- Spatial cross-attention (conv-based, no N×N matrix) ---
        self.rgb_spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.thermal_spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

        # Cross spatial gates
        self.rgb_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.thermal_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # Norms
        self.norm_rgb = nn.GroupNorm(32, channels)
        self.norm_thermal = nn.GroupNorm(32, channels)

        # Gated fusion
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(32, channels),
            nn.Sigmoid(),
        )

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
        )

        self.dropout = nn.Dropout(dropout)

    def _channel_cross_attn(self, q_proj, kv_proj, feat_q, feat_kv):
        """Channel cross-attention: GAP → project → dot-product scaling."""
        B, C, _, _ = feat_q.shape

        q_vec = feat_q.mean(dim=[2, 3])   # B, C
        kv_vec = feat_kv.mean(dim=[2, 3])  # B, C

        q = q_proj(q_vec)                 # B, C
        kv = kv_proj(kv_vec)              # B, C*2
        k, v = kv.chunk(2, dim=-1)        # B, C each

        # Per-channel dot-product scaling
        scale = (q * k).sigmoid()         # B, C
        out = v * scale                   # B, C
        return out.view(B, C, 1, 1)

    def forward(self, rgb_feat: torch.Tensor, thermal_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = rgb_feat.shape

        # 1. Channel cross-attention: exchange global context
        rgb_ch_attn = self._channel_cross_attn(
            self.rgb_channel_q, self.thermal_channel_kv, rgb_feat, thermal_feat)
        thermal_ch_attn = self._channel_cross_attn(
            self.thermal_channel_q, self.rgb_channel_kv, thermal_feat, rgb_feat)

        rgb_enhanced = rgb_feat * rgb_ch_attn
        thermal_enhanced = thermal_feat * thermal_ch_attn

        # 2. Spatial cross-attention: exchange local details
        rgb_sp = self.rgb_spatial(rgb_enhanced)
        thermal_sp = self.thermal_spatial(thermal_enhanced)

        # Cross-gating: RGB spatial features gated by thermal context and vice versa
        rgb_cross = self.rgb_gate(torch.cat([rgb_sp, thermal_sp], dim=1)) * rgb_enhanced
        thermal_cross = self.thermal_gate(torch.cat([thermal_sp, rgb_sp], dim=1)) * thermal_enhanced

        rgb_out = self.norm_rgb(rgb_feat + rgb_cross)
        thermal_out = self.norm_thermal(thermal_feat + thermal_cross)

        # 3. Gated fusion
        gate = self.fusion_gate(torch.cat([rgb_out, thermal_out], dim=1))
        fused = gate * rgb_out + (1 - gate) * thermal_out

        # Output
        fused = self.out_proj(fused) + fused
        return fused


# ---------------------------------------------------------------------------
# Channel Alignment (for lower-dim backbone stages → uniform channels)
# ---------------------------------------------------------------------------

class ChannelAlign(nn.Module):
    """1x1 conv to align channel dimensions before fusion."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# Dual Backbone with FPN
# ---------------------------------------------------------------------------

class DualBackboneWithFPN(nn.Module):
    """
    Two ResNet backbones (RGB + Thermal) with Cross-Modal Attention Fusion
    at C3, C4, C5 levels, followed by FPN.

    Output: OrderedDict of feature maps at P3, P4, P5 (+ P6, P7 from MaxPool).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.out_channels = cfg.fpn_out_channels

        # Create backbones
        if cfg.backbone == "resnet50":
            weights = ResNet50_Weights.DEFAULT if cfg.pretrained else None
            rgb_resnet = resnet50(weights=weights)
            thermal_resnet = resnet50(weights=weights)
        elif cfg.backbone == "resnet101":
            weights = ResNet101_Weights.DEFAULT if cfg.pretrained else None
            rgb_resnet = resnet101(weights=weights)
            thermal_resnet = resnet101(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {cfg.backbone}")

        # Extract feature stages (C1→C5)
        def _make_stages(resnet):
            return nn.ModuleDict({
                "stem": nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
                "layer1": resnet.layer1,  # C2: stride 4,  256ch
                "layer2": resnet.layer2,  # C3: stride 8,  512ch
                "layer3": resnet.layer3,  # C4: stride 16, 1024ch
                "layer4": resnet.layer4,  # C5: stride 32, 2048ch
            })

        self.rgb_stages = _make_stages(rgb_resnet)
        self.thermal_stages = _make_stages(thermal_resnet)

        # Cross-Modal Attention Fusion at C3, C4, C5
        c3_ch, c4_ch, c5_ch = cfg.backbone_channels[1], cfg.backbone_channels[2], cfg.backbone_channels[3]

        self.fusion_c3 = CrossModalAttentionFusion(c3_ch, cfg.fusion_num_heads, cfg.fusion_dropout)
        self.fusion_c4 = CrossModalAttentionFusion(c4_ch, cfg.fusion_num_heads, cfg.fusion_dropout)
        self.fusion_c5 = CrossModalAttentionFusion(c5_ch, cfg.fusion_num_heads, cfg.fusion_dropout)

        # FPN on fused features
        in_channels_list = [c3_ch, c4_ch, c5_ch]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=cfg.fpn_out_channels,
            extra_blocks=LastLevelMaxPool(),
        )

    def _forward_backbone(self, stages, x):
        """Extract multi-scale features from one backbone."""
        x = stages["stem"](x)
        c2 = stages["layer1"](x)
        c3 = stages["layer2"](c2)
        c4 = stages["layer3"](c3)
        c5 = stages["layer4"](c4)
        return c2, c3, c4, c5

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor):
        # Extract features
        _, rgb_c3, rgb_c4, rgb_c5 = self._forward_backbone(self.rgb_stages, rgb)
        _, thr_c3, thr_c4, thr_c5 = self._forward_backbone(self.thermal_stages, thermal)

        # Cross-modal fusion
        fused_c3 = self.fusion_c3(rgb_c3, thr_c3)
        fused_c4 = self.fusion_c4(rgb_c4, thr_c4)
        fused_c5 = self.fusion_c5(rgb_c5, thr_c5)

        # FPN — use string-integer keys to match FasterRCNN's expectations
        fpn_input = OrderedDict([
            ("0", fused_c3),
            ("1", fused_c4),
            ("2", fused_c5),
        ])
        fpn_output = self.fpn(fpn_input)
        return fpn_output


# ---------------------------------------------------------------------------
# Full Multispectral Detector
# ---------------------------------------------------------------------------

class MultispectralDetector(nn.Module):
    """
    Complete multispectral object detector:
        Dual ResNet-50 + Cross-Modal Attention Fusion + FPN + Faster R-CNN

    Training: returns dict of losses
    Inference: returns list of {boxes, labels, scores} per image
    """

    def __init__(self, cfg: ModelConfig, num_classes: int):
        super().__init__()
        self.backbone = DualBackboneWithFPN(cfg)

        # Anchor generator
        anchor_sizes = cfg.rpn_anchor_sizes
        aspect_ratios = cfg.rpn_aspect_ratios
        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

        # RPN head
        rpn_head = RPNHead(
            in_channels=cfg.fpn_out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )

        # Build Faster R-CNN with a dummy backbone wrapper
        # We wrap our dual backbone to match FasterRCNN's expected interface
        backbone_wrapper = _BackboneWrapper(self.backbone, cfg.fpn_out_channels)

        # 6-channel input (RGB+Thermal) — extend normalization params
        image_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]

        self.detector = FasterRCNN(
            backbone=backbone_wrapper,
            num_classes=num_classes + 1,  # +1 for background
            rpn_anchor_generator=anchor_generator,
            rpn_head=rpn_head,
            box_score_thresh=cfg.roi_score_thresh,
            box_nms_thresh=cfg.roi_nms_thresh,
            box_detections_per_img=cfg.roi_detections_per_img,
            image_mean=image_mean,
            image_std=image_std,
        )

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor,
                targets: List[Dict] = None):
        """
        Args:
            rgb: (B, 3, H, W) normalized RGB images
            thermal: (B, 3, H, W) normalized Thermal images (3ch repeated)
            targets: list of dicts with 'boxes' and 'labels' (training only)

        Returns:
            Training: dict of losses
            Inference: list of dicts with 'boxes', 'labels', 'scores'
        """
        # Pack rgb+thermal into a single tensor for the wrapper
        # Channel dim: [rgb(3) + thermal(3)] = 6 channels
        combined = torch.cat([rgb, thermal], dim=1)

        # Convert to list of images (FasterRCNN expects list)
        images = [combined[i] for i in range(combined.shape[0])]

        if targets is not None:
            return self.detector(images, targets)
        else:
            return self.detector(images)


class _BackboneWrapper(nn.Module):
    """
    Wraps DualBackboneWithFPN to match FasterRCNN's backbone interface.
    Input is 6-channel (RGB+Thermal concatenated), splits internally.
    """

    def __init__(self, dual_backbone: DualBackboneWithFPN, out_channels: int):
        super().__init__()
        self.dual_backbone = dual_backbone
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Split 6-channel input back to RGB(3ch) + Thermal(3ch)
        rgb = x[:, :3]
        thermal = x[:, 3:]
        return self.dual_backbone(rgb, thermal)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig, num_classes: int) -> MultispectralDetector:
    """Build the multispectral detector."""
    model = MultispectralDetector(cfg, num_classes)

    # Initialize fusion modules
    for module in model.modules():
        if isinstance(module, CrossModalAttentionFusion):
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model built: {total_params / 1e6:.1f}M params ({trainable_params / 1e6:.1f}M trainable)")

    return model
