import random
import numpy as np


def set_seed(seed: int = 42):
    """Set random seed across Python, NumPy, and PyTorch (CPU and CUDA).

    Args:
        seed: Random seed value to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass
