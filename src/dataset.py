"""Dataset class and image transforms for the CBIS-DDSM mammogram classifier."""

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Validation/inference: deterministic resize + normalize only — no
# augmentation, so evaluation numbers are stable and reproducible.
VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Kept as an alias so existing imports (predict.py, older scripts) keep working.
TRANSFORM = VAL_TRANSFORM

# Training: light, label-preserving augmentation. The original dissertation
# run used no augmentation at all, which is the single biggest driver of the
# overfitting it reports (97.9% train acc vs 65% val acc by epoch 10) — the
# model had nothing to do but memorize ~4.5k fixed crops. Flips and small
# rotations/translations are safe for mammogram crops (a lesion is still the
# same lesion mirrored or nudged a few pixels); we avoid anything that could
# change apparent tissue density (e.g. no aggressive color/contrast jitter),
# since density is diagnostically meaningful.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class BreastCancerDataset(Dataset):
    """Wraps a dataframe of (image_path, label) rows for CBIS-DDSM mammograms.

    label: 0 = benign, 1 = malignant.

    A handful of files in the public CBIS-DDSM export are corrupted or
    truncated; rather than let one bad file crash a training epoch, unreadable
    images fall back to a blank placeholder so the run keeps going.
    """

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        # Defaults to the non-augmented transform if the caller doesn't pass
        # one — always pass TRAIN_TRANSFORM explicitly for a training split.
        self.transform = transform or VAL_TRANSFORM

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "image_path"]
        label = int(self.df.loc[idx, "label"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: could not read {img_path}: {e}")
            image = Image.new("RGB", (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label
