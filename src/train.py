import config

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import CLASSES, MRIDataset, load_train_test
from src.model import build_model
from src.preprocessing import build_transform
from src.utils import seed_everything


def run_epoch(model, loader, device, optimizer=None):
    """Run one epoch."""
    # Optimizer only for training
    training = optimizer is not None
    model.train(training)
    model.features.eval()  # Keep frozen BatchNorm statistics unchanged
    total_loss, correct = 0.0, 0
    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
    return {"loss": total_loss / len(loader.dataset), "accuracy": correct / len(loader.dataset)}


def plot_history(history):
    """Compare training and validation loss and accuracy over the epochs."""
    #Create figures dir
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    #Plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history) + 1)
    for ax, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "validation"):
            ax.plot(epochs, [row[split][metric] for row in history], label=split)
        ax.set(xlabel="Epoch", ylabel=metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "training_history.png")
    plt.close(fig)


def main():
    seed_everything()

    # Load and split data + DataLoader
    train_set, _ = load_train_test()
    train_set, val_set = train_test_split(
        train_set, test_size=config.VALIDATION_SIZE, random_state=config.SEED,
        stratify=[label for _, label in train_set],
    )
    train_loader = DataLoader(
        MRIDataset(train_set, build_transform(training=True)),
        batch_size=config.BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(MRIDataset(val_set, build_transform()), batch_size=config.BATCH_SIZE)

    # Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=config.LEARNING_RATE)
    history, best_loss = [], float("inf")

    #Create the dir where the checkpoints will be saved
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device)
        history.append({"train": train_metrics, "validation": val_metrics})
        print(f"Epoch {epoch}/{config.EPOCHS}: train={train_metrics}, validation={val_metrics}")

        # Keep the model that performs best on unseen validation images.
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save({"model_state_dict": model.state_dict(), "class_names": CLASSES}, config.MODEL_PATH)
            
    plot_history(history)
    print(f"Best model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
