"""Tests for the overfitting/efficiency improvements — all run on CPU with
pretrained=False, so no GPU and no network access (ImageNet weight download)
is needed."""

import torch.nn as nn

from src.dataset import TRAIN_TRANSFORM, VAL_TRANSFORM
from src.model import count_trainable_params, get_model
from src.train import EarlyStopping


def test_train_transform_includes_augmentation_val_does_not():
    train_ops = [type(t).__name__ for t in TRAIN_TRANSFORM.transforms]
    val_ops = [type(t).__name__ for t in VAL_TRANSFORM.transforms]

    assert "RandomHorizontalFlip" in train_ops
    assert "RandomAffine" in train_ops
    assert "RandomHorizontalFlip" not in val_ops
    assert val_ops == ["Resize", "ToTensor", "Normalize"]


def test_model_head_has_dropout():
    model = get_model("resnet50", pretrained=False, dropout=0.3)
    assert isinstance(model.fc, nn.Sequential)
    assert isinstance(model.fc[0], nn.Dropout)
    assert model.fc[0].p == 0.3


def test_freeze_backbone_only_trains_head():
    model = get_model("resnet50", pretrained=False, freeze_backbone=True)
    trainable, total = count_trainable_params(model)

    # Only the new fc head (Dropout has no params, Linear has weight+bias)
    # should require grad; everything else should be frozen.
    assert trainable < total
    head_params = sum(p.numel() for p in model.fc.parameters())
    assert trainable == head_params


def test_unfrozen_model_trains_everything():
    model = get_model("resnet50", pretrained=False, freeze_backbone=False)
    trainable, total = count_trainable_params(model)
    assert trainable == total


def test_early_stopping_triggers_after_patience_epochs_without_improvement():
    stopper = EarlyStopping(patience=2)

    assert stopper.step(0.60) is False  # first score, always "improves"
    assert stopper.step(0.65) is False  # improved
    assert stopper.step(0.64) is False  # no improvement (1/2)
    assert stopper.step(0.63) is True   # no improvement (2/2) -> stop


def test_early_stopping_resets_on_improvement():
    stopper = EarlyStopping(patience=2)

    stopper.step(0.60)
    stopper.step(0.59)  # 1/2 without improvement
    stopper.step(0.62)  # improved -> resets counter
    assert stopper.epochs_without_improvement == 0
    assert stopper.step(0.61) is False  # 1/2 again, not stopped yet
