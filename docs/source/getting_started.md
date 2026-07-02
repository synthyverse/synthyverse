# Installation
Install synthyverse from PyPI:

```bash
pip install synthyverse
```

The base install is MIT licensed and excludes CTGAN-backed functionality.
Install the optional CTGAN extra only when you need `CTGANGenerator` or
`TVAEGenerator`:

```bash
pip install "synthyverse[ctgan]"
```

The `ctgan` dependency is distributed under the Business Source License. Review
that license before using the CTGAN or TVAE generators.

For local development from a clone:

```bash
pip install -e .
```

## First Synthetic Dataset

The high-level `SynthyverseGenerator` wrapper combines shared preprocessing with a low-level generator.

```python
from synthyverse.generators import SynthyverseGenerator

generator = SynthyverseGenerator(
    "univariate",
    missing_imputation_method="median",
    random_state=42,
)

generator.fit(X, discrete_features=["category", "target"])
X_syn = generator.generate(1000)
```

For manual preprocessing, direct generator usage, evaluation metrics, and benchmarking examples, continue to the in-depth usage guide.
