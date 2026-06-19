# Installation
Install synthyverse from PyPI:

```bash
pip install synthyverse
```

For local development from a clone:

```bash
pip install -e .
```

## First Synthetic Dataset

The high-level `SynthyverseGenerator` wrapper combines shared preprocessing with a low-level generator.

```python
from synthyverse.generators import SynthyverseGenerator

generator = SynthyverseGenerator(
    "ctgan",
    generator_params={"epochs": 300},
    missing_imputation_method="median",
    random_state=42,
)

generator.fit(X, discrete_features=["category", "target"])
X_syn = generator.generate(1000)
```

For manual preprocessing, direct generator usage, evaluation metrics, and benchmarking examples, continue to the in-depth usage guide.
