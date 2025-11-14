try:
    from .synthesis import TabularSynthesisBenchmark
except:
    TabularSynthesisBenchmark = None

try:
    from .imputation import TabularImputationBenchmark
except:
    TabularImputationBenchmark = None
