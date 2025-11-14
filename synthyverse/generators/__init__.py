try:
    from .arf_generator import ARFGenerator
except:
    ARFGenerator = None

try:
    from .bn_generator import BNGenerator
except:
    BNGenerator = None

try:
    from .ctgan_generator import CTGANGenerator
except:
    CTGANGenerator = None

try:
    from .tvae_generator import TVAEGenerator
except:
    TVAEGenerator = None

try:
    from .tabsyn_generator import TabSynGenerator
except:
    TabSynGenerator = None

try:
    from .cdtd_generator import CDTDGenerator
except:
    CDTDGenerator = None

try:
    from .tabargn_generator import TabARGNGenerator
except:
    TabARGNGenerator = None

try:
    from .tabddpm_generator import TabDDPMGenerator
except:
    TabDDPMGenerator = None

try:
    from .realtabformer_generator import RealTabFormerGenerator
except:
    RealTabFormerGenerator = None

try:
    from .ctabgan_generator import CTABGANGenerator
except:
    CTABGANGenerator = None

try:
    from .permutation_generator import PermutationGenerator
except:
    PermutationGenerator = None

try:
    from .forestdiffusion_generator import ForestDiffusionGenerator
except:
    ForestDiffusionGenerator = None

try:
    from .unmaskingtrees_generator import UnmaskingTreesGenerator
except:
    UnmaskingTreesGenerator = None

try:
    from .synthpop_generator import SynthpopGenerator
except:
    SynthpopGenerator = None

try:
    from .nrgboost_generator import NRGBoostGenerator
except:
    NRGBoostGenerator = None


def get_generator(generator_name: str):
    """Get a generator class by name.

    Args:
        generator_name: Name of the generator to retrieve.

    Returns:
        class: Generator class corresponding to the name.

    Raises:
        ValueError: If generator name is not found.
    """
    available_generators = [g for g in all_generators if g is not None]

    generator_map = {g.name: g for g in available_generators}
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
]
