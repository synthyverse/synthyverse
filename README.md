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
- **Modular installation.** Install only the generator and evaluation extras you need for a given environment.
- **Tabular synthetic data generators.** Use low-level generators directly, or wrap them with shared preprocessing through `SynthyverseGenerator`.
- **Evaluation metrics.** Compare synthetic data with fidelity, utility, and privacy metrics through individual metric classes or `TabularMetricEvaluator`.
- **Benchmarking workflows.** Train, sample, evaluate, and save benchmark artifacts with `TabularSynthesisBenchmark`.
- **Shared preprocessing.** Reuse `DataProcessor` for missing-value handling, schema restoration, and column constraints.

# Installation
The core package installs the shared tabular data dependencies used by the base generator and preprocessing APIs. Optional extras install dependencies for specific generators or for the evaluation module.

Install only the extras you need for a run. Some generator dependencies are heavy or may conflict with each other, so separate virtual environments are recommended when combining many extras.

## Available Installation Templates

The following installation templates are available:

| Template Name | Category | Installation Command |
|---------------|----------|----------------------|
| `arf` | Generator | `pip install synthyverse[arf]` |
| `bn` | Generator | `pip install synthyverse[bn]` |
| `cdtd` | Generator | `pip install synthyverse[cdtd]` |
| `ctgan` | Generator | `pip install synthyverse[ctgan]` |
| `permutation` | Generator | `pip install synthyverse[permutation]` |
| `smote` | Generator | `pip install synthyverse[smote]` |
| `tabargn` | Generator | `pip install synthyverse[tabargn]` |
| `tabddpm` | Generator | `pip install synthyverse[tabddpm]` |
| `tabsyn` | Generator | `pip install synthyverse[tabsyn]` |
| `tvae` | Generator | `pip install synthyverse[tvae]` |
| `base` | Generator | `pip install synthyverse[base]` |
| `eval` | Evaluation | `pip install synthyverse[eval]` |
| `full` | All | `pip install synthyverse[full]` |

**Note:** You can install multiple templates by separating them with commas, e.g., `pip install synthyverse[ctgan,eval]`
### General Installation Template

```bash
pip install synthyverse[template]
```

### Installation Examples
```bash
pip install synthyverse[ctgan]
```


```bash
pip install synthyverse[arf,bn,ctgan,tvae]
```

```bash
pip install synthyverse[ctgan,eval]
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


