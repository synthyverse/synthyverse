import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, full_determinism: bool = False):
    """Set random seed across Python, NumPy, and PyTorch (CPU and CUDA).

    Args:
        seed: Random seed value to use for all random number generators.
        full_determinism: If True, enable deterministic framework behavior where
            available. This may reduce performance or raise errors for
            nondeterministic operations.
    """
    if full_determinism:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if full_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(False)
