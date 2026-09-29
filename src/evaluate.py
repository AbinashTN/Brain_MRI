import config

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader

from src.dataset import MRIDataset, load_train_test
from src.inference import InferenceModel
from src.utils import save_json, seed_everything


def plot_predictions(test_set, predicted, confidence, classes):
    # Use a fixed random sample so the displayed examples stay comparable.
    indices = np.random.default_rng(config.SEED).choice(len(test_set), min(12, len(test_set)), replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(12, 10))
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, indices):
        path, label = test_set[index]
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"True: {classes[label]}\nPred: {classes[predicted[index]]} ({confidence[index]:.1%})")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "predictions.png")
    plt.close(fig)


def main():
    seed_everything()

    # Load Test set, model and DataLoader
    _, test_set = load_train_test()
    inference_model = InferenceModel()
    loader = DataLoader(MRIDataset(test_set, inference_model.transform), batch_size=config.BATCH_SIZE)

    # Collect predictions in dataset order for the metrics and example plots
    truth, predicted, confidence = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            probabilities = inference_model.model(images).softmax(dim=1)
            values, indices = probabilities.max(dim=1)
            truth.extend(labels.tolist())
            predicted.extend(indices.tolist())
            confidence.extend(values.tolist())

    # Save and plots
    classes = inference_model.classes
    labels = list(range(len(classes)))
    report = classification_report(truth, predicted, labels=labels, target_names=classes, zero_division=0)
    print(report)
    save_json(config.METRICS_DIR / "test_metrics.json", classification_report(truth, predicted, labels=labels, target_names=classes, zero_division=0, output_dict=True))
    (config.METRICS_DIR / "classification_report.txt").write_text(report)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_predictions(truth, predicted, labels=labels, display_labels=classes, cmap="Blues", colorbar=False)
    display.figure_.tight_layout()
    display.figure_.savefig(config.FIGURES_DIR / "confusion_matrix.png")
    plt.close(display.figure_)
    plot_predictions(test_set, predicted, confidence, classes)


if __name__ == "__main__":
    main()
