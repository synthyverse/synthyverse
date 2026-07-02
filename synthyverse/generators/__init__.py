from importlib import import_module

from .base import (
    BaseGenerator,
    ConstraintEnforcer,
    DataProcessor,
    SynthyverseGenerator,
    TabularImputer,
    TabularSchema,
)
from ._optional import has_ctgan, require_ctgan

_BASE_GENERATORS = {
    "ARFGenerator": (".arf_generator", "arf"),
    "TabSynGenerator": (".tabsyn_generator", "tabsyn"),
    "CDTDGenerator": (".cdtd_generator", "cdtd"),
    "TabDDPMGenerator": (".tabddpm_generator", "tabddpm"),
    "TabDiffGenerator": (".tabdiff_generator", "tabdiff"),
    "UnivariateGenerator": (".univariate_generator", "univariate"),
    "SMOTEGenerator": (".smote_generator", "smote"),
    "SynthpopGenerator": (".synthpop_generator", "synthpop"),
}

_CTGAN_GENERATORS = {
    "CTGANGenerator": (".ctgan_generator", "ctgan"),
    "TVAEGenerator": (".tvae_generator", "tvae"),
}

_GENERATORS = {**_BASE_GENERATORS, **_CTGAN_GENERATORS}
_GENERATOR_BY_NAME = {name: cls for cls, (_, name) in _GENERATORS.items()}


def _available_generators():
    if has_ctgan():
        return _GENERATORS
    return _BASE_GENERATORS


def __getattr__(name: str):
    if name == "all_generators":
        all_generators = [__getattr__(cls) for cls in _available_generators()]
        globals()[name] = all_generators
        return all_generators
    if name in _CTGAN_GENERATORS and not has_ctgan():
        require_ctgan()
    if name in _GENERATORS:
        module_name, _ = _GENERATORS[name]
        generator = getattr(import_module(module_name, __name__), name)
        globals()[name] = generator
        return generator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_generator(generator_name: str):
    """Get a generator class by name."""
    class_name = _GENERATOR_BY_NAME.get(generator_name)
    if class_name is None:
        raise ValueError(f"Generator {generator_name} not found")
    if class_name in _CTGAN_GENERATORS and not has_ctgan():
        require_ctgan()
    return __getattr__(class_name)


__all__ = [
    "BaseGenerator",
    "ConstraintEnforcer",
    "DataProcessor",
    "SynthyverseGenerator",
    "TabularImputer",
    "TabularSchema",
    *list(_available_generators()),
    "all_generators",
    "get_generator",
]
