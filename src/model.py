"""Model factory: ResNet-50 (the reported architecture), plus two alternatives
kept available for the future work noted in the dissertation (EfficientNetV2,
ViT) but not benchmarked in this run.
"""

import torch
import torch.nn as nn
from torchvision import models

try:
    import timm  # only required for the efficientnet_v2 / vit options
except ImportError:
    timm = None


def get_model(model_name: str = "resnet50", num_classes: int = 2) -> nn.Module:
    if model_name == "resnet50":
        # ImageNet-pretrained backbone; transfer learning keeps the data
        # requirement manageable for a ~5.7k-image medical dataset.
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_v2":
        if timm is None:
            raise ImportError("pip install timm to use efficientnet_v2")
        model = timm.create_model("efficientnetv2_s", pretrained=True, num_classes=num_classes)

    elif model_name == "vit":
        if timm is None:
            raise ImportError("pip install timm to use vit")
        model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=num_classes)

    else:
        raise ValueError("Unknown model! Use 'resnet50', 'efficientnet_v2', or 'vit'")

    return model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
