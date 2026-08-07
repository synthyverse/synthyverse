import gc, math
from typing import Any, Iterable
import torch


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
    drop_last: bool = False,
) -> int:
    """Resolve epochs, optionally overriding them with a fixed training step count."""
    if training_steps is None:
        return epochs

    if training_steps <= 0:
        raise ValueError("training_steps must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    steps_per_epoch = (
        sample_size // batch_size if drop_last else math.ceil(sample_size / batch_size)
    )
    steps_per_epoch = max(steps_per_epoch, 1)
    return math.ceil(training_steps / steps_per_epoch)


def memory_guarded(items: Iterable[Any]) -> Iterable[Any]:
    """memory guarded iterator to release memory in costly loop-based pipelines."""
    for item in items:
        free_up_memory()
        try:
            yield item
        finally:
            free_up_memory()


def free_up_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()

    gc.collect()
