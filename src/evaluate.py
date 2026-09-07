"""Final evaluation: metrics, classification report, confusion matrix, and a
grid of sample predictions — the same plots used in the README.
"""

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from src.train import calc_metrics


def evaluate_model(model, val_loader, device="cpu", save_dir=None):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc, prec, rec, f1 = calc_metrics(all_labels, all_preds)
    print("Final Evaluation on Validation Set")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Benign", "Malignant"]))

    disp = ConfusionMatrixDisplay.from_predictions(
        all_labels, all_preds,
        display_labels=["Benign", "Malignant"],
        cmap="Blues", normalize=None,
    )
    disp.ax_.set_title("Confusion Matrix")
    if save_dir:
        plt.savefig(f"{save_dir}/confusion_matrix.png", bbox_inches="tight")
    plt.show()

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def visualize_predictions(model, val_loader, device="cpu", num_images=6, save_dir=None):
    model.eval()
    images_shown = 0
    plt.figure(figsize=(12, 8))

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)

            for i in range(images.size(0)):
                if images_shown >= num_images:
                    if save_dir:
                        plt.savefig(f"{save_dir}/sample_predictions.png", bbox_inches="tight")
                    return

                img = images[i].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())

                true_label = "Malignant" if labels[i].item() == 1 else "Benign"
                pred_label = "Malignant" if preds[i].item() == 1 else "Benign"

                plt.subplot(2, num_images // 2, images_shown + 1)
                plt.imshow(img)
                plt.axis("off")
                plt.title(f"T:{true_label}\nP:{pred_label}")
                images_shown += 1
