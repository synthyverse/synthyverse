import os
import random
import numpy as np


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
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if full_determinism:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # torch.backends.cuda.matmul.allow_tf32 = False
            # torch.backends.cudnn.allow_tf32 = False
            torch.use_deterministic_algorithms(True)
    except:
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        if full_determinism:
            try:
                tf.config.experimental.enable_op_determinism()
            except:
                pass
    except:
        pass
    try:
        import tensorflow.compat.v1 as tf

        tf.set_random_seed(seed)
    except:
        pass
    try:
        import jax

        jax.random.PRNGKey(seed)
    except:
        pass
