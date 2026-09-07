"""Dataset class and image transforms for the CBIS-DDSM mammogram classifier."""

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Resize to 224x224 and normalize with ImageNet statistics — required because
# the model starts from ImageNet-pretrained weights (see src/model.py).
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
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
        self.transform = transform or TRANSFORM

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
