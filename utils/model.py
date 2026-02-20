"""
3D CNN Stress Predictor Model
==============================
A dual-head 3D Convolutional Neural Network that takes a binary voxel grid
as input and predicts:
  1. Classification: pass (0) / fail (1) — whether the part has a critical
     stress concentration near an edge.
  2. Bounding box regression: (x, y, z, w, h, d) in voxel-space coordinates
     of the danger zone (only meaningful when classification = fail).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Conv3d → BatchNorm → ReLU → MaxPool3d"""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StressPredictor3DCNN(nn.Module):
    """
    Dual-head 3D CNN for stress-concentration prediction.

    Input shape : (B, 1, 64, 64, 64)   — binary voxel grid
    Output dict :
        "classification" : (B, 1)       — sigmoid probability of failure
        "bbox"           : (B, 6)       — predicted danger-zone bounding box
                                           [x, y, z, w, h, d] in voxel space
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        # ---- Encoder (shared backbone) ----
        self.enc1 = ConvBlock3D(in_channels, 32)   # 64 -> 32
        self.enc2 = ConvBlock3D(32, 64)             # 32 -> 16
        self.enc3 = ConvBlock3D(64, 128)            # 16 -> 8
        self.enc4 = ConvBlock3D(128, 256)            # 8  -> 4

        self.global_pool = nn.AdaptiveAvgPool3d(1)  # -> (B, 256, 1, 1, 1)

        # ---- Classification head ----
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            # Sigmoid applied in forward() for numerical stability with BCEWithLogitsLoss
        )

        # ---- Bounding-box regression head ----
        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 6),
            nn.ReLU(inplace=True),   # bbox coords are non-negative
        )

    def forward(self, x: torch.Tensor) -> dict:
        # Shared feature extraction
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        features = self.global_pool(x)         # (B, 256, 1, 1, 1)

        # Heads
        cls_logit = self.cls_head(features)     # (B, 1)
        bbox = self.bbox_head(features)         # (B, 6)

        return {
            "classification": cls_logit,
            "bbox": bbox,
        }


def load_model(weights_path: str, device: str = "cpu") -> StressPredictor3DCNN:
    """
    Convenience function to instantiate and load trained weights.

    Parameters
    ----------
    weights_path : str
        Path to the saved `.pth` state dict.
    device : str
        Device to map weights to ('cpu' or 'cuda').

    Returns
    -------
    StressPredictor3DCNN
        Model in eval mode with loaded weights.
    """
    model = StressPredictor3DCNN()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
