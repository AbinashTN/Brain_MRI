from pathlib import Path

import httpx
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from streamlit.testing.v1 import AppTest

import config
from api import main
from src.dataset import CLASSES
from src.inference import InferenceModel
from src.model import build_model


@pytest.fixture
def interface(tmp_path, monkeypatch):
    folder = tmp_path / "Testing" / CLASSES[0]
    folder.mkdir(parents=True)
    Image.new("RGB", (40, 40), (80, 80, 80)).save(folder / "example.jpg")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "API_URL", "http://testserver")
    path = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"
    app = AppTest.from_file(str(path)).run()
    app.radio[0].set_value("Choose example").run()
    assert not app.exception
    return app


def test_interface_sends_image_to_api(interface, tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pth"
    model = build_model(pretrained=False)
    torch.save({"model_state_dict": model.state_dict(), "class_names": CLASSES}, checkpoint)
    monkeypatch.setattr(main, "InferenceModel", lambda: InferenceModel(checkpoint))
    requests = []
    with TestClient(main.app) as client:
        def post(url, **kwargs):
            requests.append(url)
            assert kwargs.pop("timeout") == 30
            return client.post(url, **kwargs)

        monkeypatch.setattr(httpx, "post", post)
        interface.button[0].click().run(timeout=30)

    assert requests == ["http://testserver/predict"]
    assert not interface.exception
    assert not interface.error
    assert interface.success
    assert interface.metric[0].label == "Confidence"


@pytest.mark.parametrize("failure", ["connection", "timeout", "http"])
def test_interface_displays_api_errors(interface, monkeypatch, failure):
    def post(url, **kwargs):
        if failure == "connection":
            raise httpx.ConnectError("Connection refused")
        if failure == "timeout":
            raise httpx.ReadTimeout("Request timed out")
        return httpx.Response(413, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    interface.button[0].click().run()
    assert not interface.exception
    assert not interface.success
    message = interface.error[0].value
    assert ("HTTP 413" if failure == "http" else "Cannot reach the API") in message
