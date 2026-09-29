from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights

WEIGHTS = MobileNet_V3_Small_Weights.IMAGENET1K_V1


def build_transform(training=False):
    # Resize, crop to 224, convert to a tensor and normalize for MobileNet.
    # Small rotations add variety only during training.
    if training:
        return transforms.Compose([
            transforms.RandomRotation(5),
            WEIGHTS.transforms(),
        ])
    return WEIGHTS.transforms()
