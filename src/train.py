"""Training loop with per-epoch metrics and best-checkpoint saving."""

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calc_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, prec, rec, f1


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs: int = 10,
    device: str = "cpu",
    checkpoint_path: str = "models/best_model.pth",
):
    """Trains for `epochs`, tracking loss/accuracy/precision/recall/F1 on both
    splits. The checkpoint with the best validation F1 (not accuracy — see
    README for why) is written to `checkpoint_path`.
    """
    history = {k: [] for k in (
        "train_loss", "val_loss", "train_acc", "val_acc",
        "train_prec", "val_prec", "train_rec", "val_rec",
        "train_f1", "val_f1",
    )}

    best_val_f1 = 0.0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        train_loss, all_preds, all_labels = 0.0, [], []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_acc, train_prec, train_rec, train_f1 = calc_metrics(all_labels, all_preds)

        # ---- Validation ----
        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
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

    return model, history
