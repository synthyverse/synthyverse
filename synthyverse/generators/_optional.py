from importlib.util import find_spec


CTGAN_EXTRA_MESSAGE = (
    "CTGANGenerator and TVAEGenerator require the optional ctgan dependency. "
    "Install it with `pip install synthyverse[ctgan]` and review the ctgan "
    "Business Source License before using these generators."
)


def has_ctgan() -> bool:
    return find_spec("ctgan") is not None


def require_ctgan() -> None:
    if not has_ctgan():
        raise ImportError(CTGAN_EXTRA_MESSAGE)
