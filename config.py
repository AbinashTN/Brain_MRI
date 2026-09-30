from pathlib import Path
import os

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Paths are relative to the project root
DATA_DIR = Path("data")
EXAMPLES_DIR = Path(__file__).resolve().parent / "app" / "examples"
MODEL_PATH = Path("models/best_model.pth")
FIGURES_DIR = Path("outputs/figures")
METRICS_DIR = Path("outputs/metrics")

# Training settings
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SIZE = 0.2
SEED = 42

# Prediction server used by Streamlit.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")




