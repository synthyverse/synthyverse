import gc, math, sys


def get_total_trainable_params(model):
    """Calculate the total number of trainable parameters in a model.

    Args:
        model: PyTorch model with parameters.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def resolve_epochs_from_training_steps(
    epochs: int,
    training_steps: int,
    sample_size: int,
    batch_size: int,
) -> int:
    """Resolve epochs, optionally overriding them with a fixed training step count."""
    if training_steps is None:
        return epochs

    if training_steps <= 0:
        raise ValueError("training_steps must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    steps_per_epoch = max(sample_size // batch_size, 1)
    return math.ceil(training_steps / steps_per_epoch)


def free_up_memory():
    """Aggressively release Python-level CPU and GPU memory.

    This function performs comprehensive memory cleanup across multiple frameworks:
    - Python garbage collection
    - PyTorch CUDA cache clearing
    - TensorFlow/Keras session clearing
    - JAX engine clearing
    - Matplotlib figure closing
    """

    # --- 1. Drop Python references ----------------------------------------
    gc.collect()  # clear cyclic refs first pass
    for name, mod in list(sys.modules.items()):
        # Unload large, one-off modules you know you won't reuse (optional).
        # Example heuristic: anything imported from inside a loop.
        if mod is None or "your_temp_pkg" in name:
            sys.modules.pop(name, None)
    gc.collect()  # second pass after pruning modules

    # --- 2. PyTorch (if present) ------------------------------------------
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # release unused cached blocks
            torch.cuda.ipc_collect()  # flush inter-process cached blocks
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    # --- 3. TensorFlow / Keras (if present) -------------------------------
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()  # frees graph + variables
    except ImportError:
        pass

    # --- 4. JAX (if present) ----------------------------------------------
    try:
        import jax
        from jax._src import api

        api._clear_engine()  # clears XLA backend cache
    except Exception:
        pass  # JAX API is private; ignore if not available

    # --- 5. Forcefully close matplotlib figures (notorious leak) ----------
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except ImportError:
        pass

    # Final sweep
    gc.collect()
