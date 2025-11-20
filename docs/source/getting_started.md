# Installation
The synthyverse is unique in its modular installation set-up. To avoid conflicting dependencies, we provide various installation templates. Each template installs only those dependencies which are required to access certain modules. 

Templates provide installation for specific generators, the evaluation module, and more. Install multiple templates to get access to multiple modules of the synthyverse, e.g., multiple generators and evaluation. 

**We strongly advise to only install templates which you require during a specific run. Installing multiple templates gives rise to potential dependency conflicts. Use separate virtual environments across installations.**

**Note that the core installation without any template doesn't install any modules.**

See the [overview of templates](https://github.com/synthyverse/synthyverse/blob/main/synthyverse/TEMPLATES.md).

## Available Installation Templates

The following installation templates are available:

| Template Name | Category | Installation Command |
|---------------|----------|----------------------|
| `arf` | Generator | `pip install synthyverse[arf]` |
| `bn` | Generator | `pip install synthyverse[bn]` |
| `cdtd` | Generator | `pip install synthyverse[cdtd]` |
| `ctabgan` | Generator | `pip install synthyverse[ctabgan]` |
| `ctgan` | Generator | `pip install synthyverse[ctgan]` |
| `forestdiffusion` | Generator | `pip install synthyverse[forestdiffusion]` |
| `forestlm` | Generator | `pip install synthyverse[forestlm]` |
| `nrgboost` | Generator | `pip install synthyverse[nrgboost]` |
| `permutation` | Generator | `pip install synthyverse[permutation]` |
| `realtabformer` | Generator | `pip install synthyverse[realtabformer]` |
| `synthpop` | Generator | `pip install synthyverse[synthpop]` |
| `tabargn` | Generator | `pip install synthyverse[tabargn]` |
| `tabddpm` | Generator | `pip install synthyverse[tabddpm]` |
| `tabsyn` | Generator | `pip install synthyverse[tabsyn]` |
| `tvae` | Generator | `pip install synthyverse[tvae]` |
| `unmaskingtrees` | Generator | `pip install synthyverse[unmaskingtrees]` |
| `base` | Generator | `pip install synthyverse[base]` |
| `ice` | Imputer | `pip install synthyverse[ice]` |
| `missforest` | Imputer | `pip install synthyverse[missforest]` |
| `ot` | Imputer | `pip install synthyverse[ot]` |
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