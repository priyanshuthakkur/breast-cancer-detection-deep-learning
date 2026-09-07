"""End-to-end training entry point.

    python run_train.py --epochs 15

Expects the CBIS-DDSM dataset already downloaded and unzipped into
data/raw/ (see README > Getting started for the Kaggle download step —
it's a ~5GB dataset, so it's not fetched automatically).

Defaults here differ from the original dissertation run in a few
deliberate ways — augmentation on, dropout on, early stopping on, mixed
precision on — see README > Improvements over the original run for why.
"""

import argparse

import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data_prep import build_dataset
from src.dataset import TRAIN_TRANSFORM, VAL_TRANSFORM, BreastCancerDataset
from src.evaluate import evaluate_model, visualize_predictions
from src.model import count_trainable_params, get_device, get_model
from src.train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--model", default="resnet50", choices=["resnet50", "efficientnet_v2", "vit"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", default="models/best_model.pth")

    # Regularization / overfitting controls
    parser.add_argument("--no-augment", action="store_true", help="disable training-time data augmentation")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--freeze-backbone", action="store_true",
                         help="train only the classifier head (faster, fewer params, good first run)")
    parser.add_argument("--patience", type=int, default=3, help="early-stopping patience, in epochs")

    # Efficiency controls
    parser.add_argument("--no-amp", action="store_true", help="disable mixed-precision training")
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    train_df, val_df = build_dataset(args.data_dir)

    train_transform = VAL_TRANSFORM if args.no_augment else TRAIN_TRANSFORM
    train_loader = DataLoader(
        BreastCancerDataset(train_df, transform=train_transform),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        BreastCancerDataset(val_df, transform=VAL_TRANSFORM),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = get_device()
    print("Using device:", device)

    model = get_model(
        args.model, num_classes=2, dropout=args.dropout, freeze_backbone=args.freeze_backbone,
    ).to(device)
    trainable, total = count_trainable_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable / total:.1%})")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=args.epochs, device=device, checkpoint_path=args.checkpoint,
        patience=args.patience, use_amp=not args.no_amp,
    )

    evaluate_model(model, val_loader, device=device, save_dir="assets")
    visualize_predictions(model, val_loader, device=device, save_dir="assets")


if __name__ == "__main__":
    main()
