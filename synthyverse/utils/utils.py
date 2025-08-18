import gc, sys, os, functools
import pandas as pd


def calculate_column_precision(col_values: pd.Series) -> int:
    """
    Calculate the maximum precision needed for a numerical column.

    Args:
        col_values: Pandas Series containing numerical values

    Returns:
        int: Maximum precision (number of decimal places) needed
    """

    # Convert to string and split by decimal point
    str_values = col_values.astype(str)

    # Vectorized operation to find decimal parts
    decimal_parts = str_values.str.split(".").str[-1]

    # Handle cases where there's no decimal point (integer values)
    has_decimal = str_values.str.contains(".")

    # Calculate precision for each value
    def get_precision(decimal_part, has_dec):
        if not has_dec or pd.isna(decimal_part):
            return 0
        # Find last non-zero digit from the right
        for i in range(len(decimal_part) - 1, -1, -1):
            if decimal_part[i] != "0":
                return i + 1
        return 0

    # Apply the precision calculation vectorized
    precisions = [
        get_precision(part, has_dec)
        for part, has_dec in zip(decimal_parts, has_decimal)
    ]

    return max(precisions) if precisions else 0


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


def suppress_print(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

    return wrapper
