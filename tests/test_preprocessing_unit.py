from PIL import Image
import torch

from src.preprocessing import WEIGHTS, build_transform


def test_shapes_and_determinism():
    image = Image.new("RGB", (200, 150), (100, 120, 130))
    transform = build_transform()
    output = transform(image)
    assert torch.equal(output, transform(image))
    augmented = build_transform(training=True)(image)
    assert output.shape == augmented.shape == (3, 224, 224)
    assert torch.isfinite(augmented).all()


def test_transform_matches_pretrained_weights():
    image = Image.new("RGB", (280, 250), (100, 120, 130))
    assert torch.equal(build_transform()(image), WEIGHTS.transforms()(image))
