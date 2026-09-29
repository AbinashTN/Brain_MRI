from collections import Counter
import config
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.dataset import CLASSES, load_train_test


def main():
    train_set, test_set = load_train_test()
    output_dir = config.FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compare the number of usable images in each class and split
    num_classes = len(CLASSES)
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, name, samples in ((-0.2, "Training", train_set), (0.2, "Testing", test_set)):
        counts = Counter(label for _, label in samples)
        bars = ax.bar(np.arange(num_classes) + offset, [counts[i] for i in range(num_classes)], 0.4, label=name)
        ax.bar_label(bars)
    ax.set(xticks=range(num_classes), xticklabels=CLASSES, ylabel="Images")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "class_distribution.png")
    plt.close(fig)

    # Display up to four training examples per class
    rng = np.random.default_rng(config.SEED)
    fig, axes = plt.subplots(num_classes, 4, figsize=(10, 10), squeeze=False)
    for label, row in enumerate(axes):
        paths = [path for path, target in train_set if target == label]
        for ax in row:
            ax.axis("off")
        for ax, index in zip(row, rng.choice(len(paths), min(4, len(paths)), replace=False)):
            with Image.open(paths[index]) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(CLASSES[label])
    fig.tight_layout()
    fig.savefig(output_dir / "sample_images.png")
    plt.close(fig)

    # Image dimensions plot
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, samples in (("Training", train_set), ("Testing", test_set)):
        sizes = []
        for path, _ in samples:
            with Image.open(path) as image:
                sizes.append(image.size)
        widths, heights = zip(*sizes)
        ax.scatter(widths, heights, s=8, alpha=0.3, label=name)
    ax.set(xlabel="Width", ylabel="Height", title="Image dimensions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "image_dimensions.png")
    plt.close(fig)


    print(f"Saved exploration figures to {output_dir}")


if __name__ == "__main__":
    main()
