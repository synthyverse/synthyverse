<table align="center" border="0">
<tr>
<td align="center">

<img src="https://raw.githubusercontent.com/synthyverse/synthyverse/main/logo/logo.png" alt="Synthyverse logo" width="250" height="auto">

<br/>
<br/>

Welcome to the synthyverse!

An extensive ecosystem for synthetic data generation and evaluation in Python.

[Read the docs](https://synthyverse.readthedocs.io) for in-depth usage.

_The synthyverse is a work in progress. Please provide any suggestions through a GitHub Issue._

</td>
</tr>
</table>

<div style="clear: both;"></div>

# Features
- **Tabular synthetic data generators.** Use low-level generators directly, or wrap them with shared preprocessing through `SynthyverseGenerator`.
- **Evaluation metrics.** Compare synthetic data with fidelity, utility, and privacy metrics through individual metric classes or `TabularMetricEvaluator`.
- **Benchmarking workflows.** Train, sample, evaluate, and save benchmark artifacts with `TabularSynthesisBenchmark`.
- **Shared preprocessing.** Reuse `DataProcessor` for missing-value handling, schema restoration, and column constraints.

# Installation
Install synthyverse from PyPI:

```bash
pip install synthyverse
```

For local development from a clone:

```bash
pip install -e .
```

# Usage

Use a high-level wrapper when you want preprocessing and schema restoration handled for you:

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

Use the lower-level APIs when you want explicit control over preprocessing, generator fitting, metrics, or benchmarking. See the [docs](https://synthyverse.readthedocs.io) for complete examples.


