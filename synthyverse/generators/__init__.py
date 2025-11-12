try:
    from .arf_generator import ARFGenerator
except ImportError:
    ARFGenerator = None

try:
    from .bn_generator import BNGenerator
except ImportError:
    BNGenerator = None

try:
    from .ctgan_generator import CTGANGenerator
except ImportError:
    CTGANGenerator = None

try:
    from .tvae_generator import TVAEGenerator
except ImportError:
    TVAEGenerator = None

try:
    from .tabsyn_generator import TabSynGenerator
except ImportError:
    TabSynGenerator = None

try:
    from .cdtd_generator import CDTDGenerator
except ImportError:
    CDTDGenerator = None

try:
    from .tabargn_generator import TabARGNGenerator
except ImportError:
    TabARGNGenerator = None

try:
    from .tabddpm_generator import TabDDPMGenerator
except ImportError:
    TabDDPMGenerator = None

try:
    from .realtabformer_generator import RealTabFormerGenerator
except ImportError:
    RealTabFormerGenerator = None

try:
    from .ctabgan_generator import CTABGANGenerator
except ImportError:
    CTABGANGenerator = None

try:
    from .permutation_generator import PermutationGenerator
except ImportError:
    PermutationGenerator = None

try:
    from .forestdiffusion_generator import ForestDiffusionGenerator
except ImportError:
    ForestDiffusionGenerator = None

try:
    from .unmaskingtrees_generator import UnmaskingTreesGenerator
except ImportError:
    UnmaskingTreesGenerator = None

try:
    from .synthpop_generator import SynthpopGenerator
except ImportError:
    SynthpopGenerator = None

try:
    from .nrgboost_generator import NRGBoostGenerator
except ImportError:
    NRGBoostGenerator = None


def get_generator(generator_name: str):

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
