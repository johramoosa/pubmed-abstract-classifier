import random
from pathlib import Path

import numpy as np
import torch


def capture_random_states(train_loader):
    """Capture random states for the current CPU training workflow."""
    numpy_state = np.random.get_state()

    return {
        "python": random.getstate(),
        # Store the NumPy array as a list for weights_only=True loading.
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "train_loader": train_loader.generator.get_state(),
    }


def restore_random_states(states, train_loader):
    """Restore random states after constructing/loading model and optimizer."""
    random.setstate(states["python"])

    numpy_state = states["numpy"]
    np.random.set_state((
        numpy_state["algorithm"],
        np.array(numpy_state["keys"], dtype=np.uint32),
        numpy_state["position"],
        numpy_state["has_gauss"],
        numpy_state["cached_gaussian"],
    ))

    torch.set_rng_state(states["torch"])
    train_loader.generator.set_state(states["train_loader"])


def save_checkpoint(checkpoint, path):
    """Write a temporary file before replacing the previous checkpoint."""
    path = Path(path)
    temporary_path = path.with_suffix(".tmp")

    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)
