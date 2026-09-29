import pytest
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

from src.dataset import MRIDataset, load_train_test
from src.preprocessing import build_transform


def test_load_train_test_fixed_layout(sample_data):
    train_set, test_set = load_train_test(sample_data)
    assert len(train_set) == len(test_set) == 40
    assert train_set[0][1] == 0


def test_duplicates_are_excluded_within_and_between_sets(sample_data):
    source = sample_data / "Training/glioma/0.jpg"
    (source.parent / "copy.jpg").write_bytes(source.read_bytes())
    (sample_data / "Testing/glioma/copy.jpg").write_bytes(source.read_bytes())
    train_set, test_set = load_train_test(sample_data)
    assert len(train_set) == len(test_set) == 40
    assert source in [path for path, _ in train_set]
    assert not any(path.name == "copy.jpg" for path, _ in train_set + test_set)
    assert (source.parent / "copy.jpg").exists()  # Original files are preserved.


def test_conflicting_duplicate_labels_stop_loading(sample_data):
    source = sample_data / "Training/glioma/0.jpg"
    (sample_data / "Training/meningioma/copy.jpg").write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="different labels"):
        load_train_test(sample_data)


def test_unreadable_image_is_logged_and_skipped(sample_data, caplog):
    (sample_data / "Training/glioma/broken.jpg").write_bytes(b"not an image")
    train_set, _ = load_train_test(sample_data)
    assert len(train_set) == 40
    assert "Error opening image" in caplog.text


def test_split_preserves_classes_and_has_no_overlap(sample_data):
    train_set, test_set = load_train_test(sample_data)
    train_set, val_set = train_test_split(
        train_set, test_size=0.2, random_state=42, stratify=[y for _, y in train_set])
    assert len(train_set) == 32
    assert len(val_set) == 8
    assert {y for _, y in val_set} == {0, 1, 2, 3}
    assert not set(train_set) & set(val_set)
    assert not set(train_set + val_set) & set(test_set)


def test_dataset_converts_grayscale_to_rgb(tmp_path):
    path = tmp_path / "gray.jpg"
    Image.new("L", (40, 40), 100).save(path)
    dataset = MRIDataset([(path, 0)], build_transform())
    image, label = dataset[0]
    assert image.shape == (3, 224, 224)
    assert torch.isfinite(image).all()
    assert label == 0


def test_empty_dataset_has_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="No usable JPG images found"):
        load_train_test(tmp_path)


def test_testing_empty_after_deduplication_has_a_clear_error(sample_data):
    # Replace each test image with its training copy, leaving no unique test data.
    for path in (sample_data / "Testing").glob("*/*.jpg"):
        source = sample_data / "Training" / path.parent.name / path.name
        path.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="No usable JPG images found.*Testing"):
        load_train_test(sample_data)
