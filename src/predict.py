"""Single-image inference helper, shared by the CLI and the Streamlit demo."""

from pathlib import Path

import torch
from PIL import Image

from src.dataset import TRANSFORM
from src.model import get_device, get_model

CLASS_NAMES = ["Benign", "Malignant"]


def load_trained_model(checkpoint_path: str = "models/best_model.pth", model_name: str = "resnet50"):
    """Loads the architecture and, if present, trained weights.

    Returns (model, device, weights_loaded: bool) — weights_loaded is False
    when no checkpoint is on disk yet, so callers (e.g. the demo app) can
    show an honest "untrained model" notice instead of silently guessing.
    """
    device = get_device()
    model = get_model(model_name, num_classes=2).to(device)

    weights_loaded = False
    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        weights_loaded = True

    model.eval()
    return model, device, weights_loaded


def predict_image(model, device, image: Image.Image) -> dict:
    """Runs one PIL image through the model and returns class + confidence."""
    x = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    pred_idx = int(torch.argmax(logits, dim=1).item())
    return {
        "label": CLASS_NAMES[pred_idx],
        "confidence": probs[pred_idx],
        "probabilities": dict(zip(CLASS_NAMES, probs)),
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.predict <path_to_mammogram_image>")
        sys.exit(1)

    model, device, weights_loaded = load_trained_model()
    if not weights_loaded:
        print("Warning: no trained checkpoint found at models/best_model.pth — "
              "predictions are from an untrained (ImageNet-only) backbone. "
              "Run run_train.py first.")

    image = Image.open(sys.argv[1])
    result = predict_image(model, device, image)
    print(f"Prediction: {result['label']} ({result['confidence']:.1%} confidence)")
    print(f"Probabilities: {result['probabilities']}")
