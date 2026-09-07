"""Loss functions.

The reported run (see README) trains with standard Cross-Entropy Loss.
FocalLoss is included as a ready-to-use alternative for the class-imbalance
mitigation noted as future work in the dissertation — down-weighting
easy/majority examples so the model spends more gradient signal on the
harder, minority-class cases.
"""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()
