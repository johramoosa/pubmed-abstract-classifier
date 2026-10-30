import random

import numpy as np
import torch


def set_random_seed(seed):
    """Set random seeds before creating a model and training loader."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make cuDNN convolution behaviour more reproducible.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
