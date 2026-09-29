import math

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import CLASSES, MRIDataset, load_train_test
from src.inference import InferenceModel
from src.model import build_model
from src.preprocessing import build_transform
from src.train import run_epoch


def test_train_save_load_predict(sample_data, tmp_path):
    train_set, test_set = load_train_test(sample_data)
    train_set, val_set = train_test_split(
        train_set, test_size=0.2, random_state=42, stratify=[y for _, y in train_set])
    train_loader = DataLoader(MRIDataset(train_set, build_transform(training=True)), batch_size=8, shuffle=True)
    val_loader = DataLoader(MRIDataset(val_set, build_transform()), batch_size=8)
    model = build_model(pretrained=False)
    # Compare weights and BatchNorm statistics before and after training.
    features_before = {k: v.clone() for k, v in model.features.state_dict().items()}
    head_before = model.classifier[-1].weight.detach().clone()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

    train_metrics = run_epoch(model, train_loader, "cpu", optimizer)
    val_metrics = run_epoch(model, val_loader, "cpu")
    for metrics in (train_metrics, val_metrics):
        assert math.isfinite(metrics["loss"])
        assert 0 <= metrics["accuracy"] <= 1
    assert not torch.equal(head_before, model.classifier[-1].weight)
    assert all(torch.equal(features_before[k], v) for k, v in model.features.state_dict().items())

    # Reload from disk to check the same path used by the application.
    path = tmp_path / "model.pth"
    torch.save({"model_state_dict": model.state_dict(), "class_names": CLASSES}, path)
    service = InferenceModel(path)
    with Image.open(test_set[0][0]) as image:
        result = service.predict(image)
        assert result == service.predict(image)
    assert result.predicted_class in CLASSES
    assert math.isclose(sum(result.probabilities.values()), 1, abs_tol=1e-6)


def test_main_scripts(sample_data, tmp_path, monkeypatch):
    import config
    from src import train, evaluate
    from scripts import explore_data

    # Keep generated files temporary and avoid downloading pretrained weights.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "build_model", lambda: build_model(pretrained=False))
    monkeypatch.setattr(config, "DATA_DIR", sample_data)
    monkeypatch.setattr(config, "EPOCHS", 1)
    monkeypatch.setattr(config, "BATCH_SIZE", 8)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "saved/model.pth")
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "figures")
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path / "metrics")
    train.main()
    assert config.MODEL_PATH.exists()
    assert (config.FIGURES_DIR / "training_history.png").exists()
    assert not (config.METRICS_DIR / "split_manifest.json").exists()

    evaluate.main()
    assert (config.METRICS_DIR / "test_metrics.json").exists()
    assert (config.FIGURES_DIR / "confusion_matrix.png").exists()
    explore_data.main()
    assert (config.FIGURES_DIR / "sample_images.png").exists()
