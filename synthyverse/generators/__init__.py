from .base import BaseGenerator

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

all_generators = [
    ARFGenerator,
    BNGenerator,
    CTGANGenerator,
    TVAEGenerator,
    TabSynGenerator,
]
