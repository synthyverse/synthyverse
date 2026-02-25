def _make_unavailable_generator(
    class_name: str,
    generator_name: str,
    extra_name: str,
    import_error: Exception,
):
    class _UnavailableGenerator:
        name = generator_name

        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{class_name} is unavailable because its dependencies could not be imported. "
                f"Install the required extras with `pip install synthyverse[{extra_name}]` "
                f"and verify the environment is healthy. Original import error: {import_error!r}"
            )

    _UnavailableGenerator.__name__ = class_name
    return _UnavailableGenerator


try:
    from .arf_generator import ARFGenerator
except Exception as exc:
    ARFGenerator = _make_unavailable_generator("ARFGenerator", "arf", "arf", exc)

try:
    from .bn_generator import BNGenerator
except Exception as exc:
    BNGenerator = _make_unavailable_generator("BNGenerator", "bn", "bn", exc)

try:
    from .ctgan_generator import CTGANGenerator
except Exception as exc:
    CTGANGenerator = _make_unavailable_generator("CTGANGenerator", "ctgan", "ctgan", exc)

try:
    from .tvae_generator import TVAEGenerator
except Exception as exc:
    TVAEGenerator = _make_unavailable_generator("TVAEGenerator", "tvae", "tvae", exc)

try:
    from .tabsyn_generator import TabSynGenerator
except Exception as exc:
    TabSynGenerator = _make_unavailable_generator("TabSynGenerator", "tabsyn", "tabsyn", exc)

try:
    from .cdtd_generator import CDTDGenerator
except Exception as exc:
    CDTDGenerator = _make_unavailable_generator("CDTDGenerator", "cdtd", "cdtd", exc)

try:
    from .tabargn_generator import TabARGNGenerator
except Exception as exc:
    TabARGNGenerator = _make_unavailable_generator(
        "TabARGNGenerator", "tabargn", "tabargn", exc
    )

try:
    from .tabddpm_generator import TabDDPMGenerator
except Exception as exc:
    TabDDPMGenerator = _make_unavailable_generator(
        "TabDDPMGenerator", "tabddpm", "tabddpm", exc
    )

try:
    from .realtabformer_generator import RealTabFormerGenerator
except Exception as exc:
    RealTabFormerGenerator = _make_unavailable_generator(
        "RealTabFormerGenerator", "realtabformer", "realtabformer", exc
    )

try:
    from .ctabgan_generator import CTABGANGenerator
except Exception as exc:
    CTABGANGenerator = _make_unavailable_generator(
        "CTABGANGenerator", "ctabgan", "ctabgan", exc
    )

try:
    from .permutation_generator import PermutationGenerator
except Exception as exc:
    PermutationGenerator = _make_unavailable_generator(
        "PermutationGenerator", "permutation", "permutation", exc
    )

try:
    from .forestdiffusion_generator import ForestDiffusionGenerator
except Exception as exc:
    ForestDiffusionGenerator = _make_unavailable_generator(
        "ForestDiffusionGenerator", "forestdiffusion", "forestdiffusion", exc
    )

try:
    from .unmaskingtrees_generator import UnmaskingTreesGenerator
except Exception as exc:
    UnmaskingTreesGenerator = _make_unavailable_generator(
        "UnmaskingTreesGenerator", "unmaskingtrees", "unmaskingtrees", exc
    )

try:
    from .synthpop_generator import SynthpopGenerator
except Exception as exc:
    SynthpopGenerator = _make_unavailable_generator(
        "SynthpopGenerator", "synthpop", "synthpop", exc
    )

try:
    from .nrgboost_generator import NRGBoostGenerator
except Exception as exc:
    NRGBoostGenerator = _make_unavailable_generator(
        "NRGBoostGenerator", "nrgboost", "nrgboost", exc
    )

try:
    from .smote_generator import SMOTEGenerator
except Exception as exc:
    SMOTEGenerator = _make_unavailable_generator("SMOTEGenerator", "smote", "smote", exc)


def get_generator(generator_name: str):
    """Get a generator class by name.

    Args:
        generator_name: Name of the generator to retrieve.

    Returns:
        class: Generator class corresponding to the name.

    Raises:
        ValueError: If generator name is not found.
    """
    generator_map = {g.name: g for g in all_generators}
    if generator_name not in generator_map.keys():
        raise ValueError(f"Generator {generator_name} not found")

    return generator_map[generator_name]


all_generators = [
    ARFGenerator,
    BNGenerator,
    CTGANGenerator,
    TVAEGenerator,
    TabSynGenerator,
    CDTDGenerator,
    TabARGNGenerator,
    TabDDPMGenerator,
    RealTabFormerGenerator,
    CTABGANGenerator,
    PermutationGenerator,
    ForestDiffusionGenerator,
    UnmaskingTreesGenerator,
    NRGBoostGenerator,
    SynthpopGenerator,
    SMOTEGenerator,
]
