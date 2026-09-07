"""Model factory: ResNet-50 (the reported architecture), plus two alternatives
kept available for the future work noted in the dissertation (EfficientNetV2,
ViT) but not benchmarked in this run.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

try:
    import timm  # only required for the efficientnet_v2 / vit options
except ImportError:
    timm = None


def get_model(
    model_name: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Build a classifier.

    dropout: applied right before the final linear layer. The original run
        used a bare `nn.Linear` head straight after global average pooling —
        with only ~4.5k training images, that head overfits fast. A dropout
        layer costs nothing at inference and meaningfully slows memorization.
    freeze_backbone: freezes every ImageNet-pretrained conv layer and trains
        only the new head. This is the single biggest lever for both
        overfitting (far fewer trainable parameters relative to dataset
        size) and training speed/memory (no gradients computed or stored for
        ~23M frozen parameters). Good default for a first run or a quick
        iteration; unfreeze (the default) for the best final accuracy once
        you have epochs to spare.
    pretrained: set False in tests/CI to build the architecture without
        downloading ImageNet weights.
    """
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )

    elif model_name == "efficientnet_v2":
        if timm is None:
            raise ImportError("pip install timm to use efficientnet_v2")
        model = timm.create_model(
            "efficientnetv2_s", pretrained=pretrained, num_classes=num_classes, drop_rate=dropout,
        )
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

    elif model_name == "vit":
        if timm is None:
            raise ImportError("pip install timm to use vit")
        model = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes, drop_rate=dropout,
        )
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "head" not in name:
                    param.requires_grad = False

    else:
        raise ValueError("Unknown model! Use 'resnet50', 'efficientnet_v2', or 'vit'")

    return model


def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    """Returns (trainable, total) parameter counts — useful to confirm
    freeze_backbone actually reduced what the optimizer has to update."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
