"""End-to-end training entry point.

    python run_train.py --epochs 10

Expects the CBIS-DDSM dataset already downloaded and unzipped into
data/raw/ (see README > Getting started for the Kaggle download step —
it's a ~5GB dataset, so it's not fetched automatically).
"""

import argparse

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data_prep import build_dataset
from src.dataset import BreastCancerDataset
from src.evaluate import evaluate_model, visualize_predictions
from src.model import get_device, get_model
from src.train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model", default="resnet50", choices=["resnet50", "efficientnet_v2", "vit"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    args = parser.parse_args()

    train_df, val_df = build_dataset(args.data_dir)

    train_loader = DataLoader(BreastCancerDataset(train_df), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(BreastCancerDataset(val_df), batch_size=args.batch_size, shuffle=False)

    device = get_device()
    print("Using device:", device)
    model = get_model(args.model, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=args.epochs, device=device, checkpoint_path=args.checkpoint,
    )

    evaluate_model(model, val_loader, device=device, save_dir="assets")
    visualize_predictions(model, val_loader, device=device, save_dir="assets")


if __name__ == "__main__":
    main()
