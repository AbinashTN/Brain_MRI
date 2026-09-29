from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import main
from src.dataset import CLASSES
from src.inference import Prediction


class FakeModel:
    classes = CLASSES

    def predict(self, image):
        assert image.mode == "RGB"
        return Prediction(predicted_class="glioma", confidence=0.7,
                          probabilities=dict(zip(CLASSES, [0.7, 0.1, 0.1, 0.1])))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "InferenceModel", FakeModel)
    with TestClient(main.app) as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "classes": CLASSES}


def test_prediction(client):
    buffer = BytesIO()
    Image.new("L", (48, 48), 80).save(buffer, format="JPEG")
    response = client.post("/predict", files={"file": ("scan.jpg", buffer.getvalue(), "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["predicted_class"] == "glioma"
    assert sum(response.json()["probabilities"].values()) == pytest.approx(1)


@pytest.mark.parametrize("data", [b"", b"not an image"])
def test_unreadable_upload(client, data):
    response = client.post("/predict", files={"file": ("scan.jpg", data, "image/jpeg")})
    assert response.status_code == 400


def test_large_upload(client):
    response = client.post("/predict", files={"file": ("scan.jpg", b"x" * (10 * 1024 * 1024 + 1), "image/jpeg")})
    assert response.status_code == 413


def test_missing_file(client):
    assert client.post("/predict").status_code == 422
