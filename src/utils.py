import json
import random
from pathlib import Path

import config
import numpy as np
import torch


def seed_everything(seed=None):
    """Use the same random seeds to make runs easier to compare."""
    if seed is None:
        seed = config.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)  # Limit CPU usage for this small project


def save_json(path, value):
    """Save results as readable JSON, creating the output folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
