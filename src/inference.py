import config
import torch
from pydantic import BaseModel

from src.dataset import CLASSES
from src.model import build_model
from src.preprocessing import build_transform


class Prediction(BaseModel):
    """Prediction format shared by the API and Streamlit."""
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


class InferenceModel:
    """For inference with the saved model."""
    def __init__(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = config.MODEL_PATH
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.classes = checkpoint["class_names"]
        self.transform = build_transform()
        self.model = build_model(pretrained=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image):
        tensor = self.transform(image.convert("RGB")).unsqueeze(0)
        probabilities = self.model(tensor).softmax(dim=1)[0]
        confidence, index = probabilities.max(dim=0)
        return Prediction(
            predicted_class=self.classes[index.item()],
            confidence=confidence.item(),
            probabilities=dict(zip(self.classes, probabilities.tolist())),
        )
