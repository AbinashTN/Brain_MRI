import hashlib
import logging
from pathlib import Path

import config
from config import CLASSES
from PIL import Image
from torch.utils.data import Dataset


def load_train_test(data_dir=None):
    """Return (path, label) lists, keeping each RGB image only once."""
    data_dir = config.DATA_DIR if data_dir is None else Path(data_dir)
    seen = {}  # Shared across Training and Testing to avoid overlap

    def read_set(split):
        samples = []
        duplicates = 0
        for label, name in enumerate(CLASSES):
            for path in sorted((data_dir / split / name).glob("*.jpg")):
                try:
                    with Image.open(path) as image:
                        image = image.convert("RGB")
                        # Compare pixels, not filenames or JPEG metadata
                        hash_image = hashlib.sha256(str(image.size).encode() + image.tobytes()).digest()
                except OSError as error:
                    logging.error("Error opening image %s: %s", path, error)
                    continue
                if hash_image in seen:
                    if seen[hash_image] != label:
                        raise ValueError(f"Identical images have different labels: {path}")
                    duplicates += 1
                    continue
                seen[hash_image] = label
                samples.append((path, label))
        if not samples:
            raise ValueError(f"No usable JPG images found in {data_dir / split}")
        print(f"{split}: {len(samples)} images, {duplicates} duplicates excluded")
        return samples

    train_set = read_set("Training")
    test_set = read_set("Testing")  # Also excludes images already present in Training
    return train_set, test_set


class MRIDataset(Dataset):
    """Open and prepare each image when the DataLoader requests it."""

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label
