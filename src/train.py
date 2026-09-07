"""Training loop with per-epoch metrics, early stopping, mixed precision, and
best-checkpoint saving.
"""

from typing import Optional

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1


class EarlyStopping:
    """Stops training once validation F1 hasn't improved for `patience` epochs.

    The original dissertation run trained a fixed 10 epochs regardless of
    what validation loss/F1 was doing — by its own account, validation loss
    was still rising at the end while training loss kept falling (textbook
    overfitting). Stopping at the actual best epoch instead of a fixed count
    is free accuracy and free compute.
    """

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = 0.0
        self.epochs_without_improvement = 0

    def step(self, val_f1: float) -> bool:
        """Call once per epoch with the epoch's validation F1. Returns True
        when training should stop."""
        if val_f1 > self.best_score + self.min_delta:
            self.best_score = val_f1
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        return self.epochs_without_improvement >= self.patience


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs: int = 10,
    device="cpu",
    checkpoint_path: str = "models/best_model.pth",
    patience: int = 3,
    use_amp: bool = True,
    grad_clip_norm: Optional[float] = 1.0,
):
    """Trains for up to `epochs`, tracking loss/accuracy/precision/recall/F1
    on both splits, and stopping early once validation F1 plateaus.

    use_amp: mixed-precision training via torch.cuda.amp — roughly 1.5-2x
        faster on a Colab/consumer GPU with lower memory use, at effectively
        no accuracy cost. Automatically a no-op on CPU.
    grad_clip_norm: clips gradient norm to stabilize training, particularly
        useful once the backbone is unfrozen and gradients flow through the
        whole network. Set to None to disable.
    """
    history = {k: [] for k in (
        "train_loss", "val_loss", "train_acc", "val_acc",
        "train_prec", "val_prec", "train_rec", "val_rec",
        "train_f1", "val_f1",
    )}

    amp_enabled = use_amp and torch.device(device).type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    early_stopping = EarlyStopping(patience=patience)

    best_val_f1 = 0.0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_loss, all_preds, all_labels = 0.0, [], []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc, train_prec, train_rec, train_f1 = calc_metrics(all_labels, all_preds)

        # ---- Validation ----
        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc, val_prec, val_rec, val_f1 = calc_metrics(all_labels, all_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_prec"].append(train_prec)
        history["val_prec"].append(val_prec)
        history["train_rec"].append(train_rec)
        history["val_rec"].append(val_rec)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        scheduler.step(val_loss)

        # Model selection is on F1, not accuracy: with a near-balanced but
        # imperfect class split, F1 is the more trustworthy signal (see README).
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            print("Best model saved (F1 improved).")

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} "
            f"Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f} "
            f"Val F1: {val_f1:.4f}"
        )

        if early_stopping.step(val_f1):
            print(
                f"Early stopping: val F1 hasn't improved in {patience} epochs "
                f"(best: {early_stopping.best_score:.4f}). Stopping at epoch {epoch + 1}/{epochs}."
            )
            break

    return model, history
