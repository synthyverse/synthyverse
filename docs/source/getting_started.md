# Installation
The synthyverse uses optional installation extras for generator-specific and evaluation-specific dependencies. The core package installs the shared tabular dependencies used by the base generator and preprocessing APIs.

Install multiple extras to use multiple modules in one environment, for example a generator plus the evaluation module.

**We strongly advise installing only the extras required for a specific run. Some generator dependencies are heavy or may conflict with each other, so separate virtual environments are recommended when combining many extras.**

## Available Installation Templates

The following installation templates are available:

| Template Name | Category | Installation Command |
|---------------|----------|----------------------|
| `arf` | Generator | `pip install synthyverse[arf]` |
| `bn` | Generator | `pip install synthyverse[bn]` |
| `cdtd` | Generator | `pip install synthyverse[cdtd]` |
| `ctgan` | Generator | `pip install synthyverse[ctgan]` |
| `smote` | Generator | `pip install synthyverse[smote]` |
| `tabargn` | Generator | `pip install synthyverse[tabargn]` |
| `tabddpm` | Generator | `pip install synthyverse[tabddpm]` |
| `tabsyn` | Generator | `pip install synthyverse[tabsyn]` |
| `tvae` | Generator | `pip install synthyverse[tvae]` |
| `univariate` | Generator | `pip install synthyverse[univariate]` |
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
