import gc, sys, os, functools


def get_generator(generator: str):

    from ..generators import all_generators

    available_generators = [g for g in all_generators if g is not None]

    generator_map = {g.name: g for g in available_generators}
    if generator not in generator_map.keys():
        raise ValueError(f"Generator {generator} not found")

    return generator_map[generator]


def free_up_memory():
    """
    Aggressively release Python-level CPU and GPU memory.
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
