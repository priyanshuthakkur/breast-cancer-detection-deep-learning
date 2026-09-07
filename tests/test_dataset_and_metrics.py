"""Lightweight tests that don't need the (5GB, gitignored) CBIS-DDSM dataset —
they use temp images and small arrays instead."""

import pandas as pd
import torch
from PIL import Image

from src.dataset import BreastCancerDataset
from src.losses import FocalLoss
from src.train import calc_metrics


def test_breast_cancer_dataset_returns_tensor_and_label(tmp_path):
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (300, 300), color=(128, 64, 64)).save(img_path)

    df = pd.DataFrame({"image_path": [str(img_path)], "label": [1]})
    dataset = BreastCancerDataset(df)

    image_tensor, label = dataset[0]
    assert image_tensor.shape == (3, 224, 224)
    assert label == 1


def test_breast_cancer_dataset_falls_back_on_unreadable_file(tmp_path):
    bad_path = tmp_path / "corrupted.jpg"
    bad_path.write_bytes(b"not a real image")

    df = pd.DataFrame({"image_path": [str(bad_path)], "label": [0]})
    dataset = BreastCancerDataset(df)

    image_tensor, label = dataset[0]  # should not raise
    assert image_tensor.shape == (3, 224, 224)
    assert label == 0


def test_calc_metrics_all_correct():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 1, 0]
    acc, prec, rec, f1 = calc_metrics(y_true, y_pred)
    assert acc == prec == rec == f1 == 1.0


def test_focal_loss_returns_scalar():
    loss_fn = FocalLoss()
    logits = torch.randn(4, 2)
    targets = torch.tensor([0, 1, 1, 0])
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss.item() >= 0
