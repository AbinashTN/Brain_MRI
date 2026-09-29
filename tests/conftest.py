import pytest
from PIL import Image

from src.dataset import CLASSES
from src.utils import seed_everything


@pytest.fixture(autouse=True)
def seed_tests():
    seed_everything(42)


@pytest.fixture
def sample_data(tmp_path):
    root = tmp_path / "data"
    # Distinct colors keep these synthetic images from being deduplicated
    for split, blue in (("Training", 0), ("Testing", 100)):
        for label, name in enumerate(CLASSES):
            folder = root / split / name
            folder.mkdir(parents=True)
            for i in range(10):
                Image.new("RGB", (32, 32), (label * 50, i * 20, blue)).save(folder / f"{i}.jpg")
    return root
