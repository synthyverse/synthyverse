try:
    from .synthesis import TabularSynthesisBenchmark
except ImportError:
    TabularSynthesisBenchmark = None

try:
    from .imputation import TabularImputationBenchmark
except ImportError:
    TabularImputationBenchmark = None
