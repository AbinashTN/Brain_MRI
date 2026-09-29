from io import BytesIO

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from api import main
from src.dataset import CLASSES
from src.inference import InferenceModel
from src.model import build_model


def test_real_checkpoint_to_http_prediction(tmp_path, monkeypatch):
    path = tmp_path / "model.pth"
    model = build_model(pretrained=False)
    torch.save({"model_state_dict": model.state_dict(), "class_names": CLASSES}, path)
    monkeypatch.setattr(main, "InferenceModel", lambda: InferenceModel(path))
    buffer = BytesIO()
    Image.new("RGB", (64, 64), (80, 80, 80)).save(buffer, format="JPEG")
    with TestClient(main.app) as client:
        response = client.post("/predict", files={"file": ("scan.jpg", buffer.getvalue(), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in CLASSES
    assert 0 <= body["confidence"] <= 1
    assert list(body["probabilities"]) == CLASSES
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-6)


def test_checkpoint_with_different_class_order_is_rejected(tmp_path):
    path = tmp_path / "model.pth"
    model = build_model(pretrained=False)
    torch.save({"model_state_dict": model.state_dict(), "class_names": CLASSES[::-1]}, path)
    with pytest.raises(ValueError, match="class order"):
        InferenceModel(path)
