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
- 🔧 **Highly modular installation.** Install only those modules which you require to keep your installation lightweight.
- 📚 **Extensive library for synthetic data.** Any generator or metric can be quickly added without dependency conflicts due to synthyverse's modular installation. This allows the synthyverse to host a great amount of generators and evaluation metrics. It also allows the synthyverse to wrap around any existing synthetic data library.
- ⚙️ **Benchmarking module for simplified synthetic data pipelines.** The benchmarking module executes a modular pipeline of synthetic data generation and evaluation. Choose a generator, set of evaluation metrics, and pipeline parameters, and obtain results on synthetic data quality.
- 👷 **Minimal preprocessing required.** All preprocessing is handled by the synthyverse, so no need for scaling, one-hot encoding, or handling missing values. Different preprocessing schemes can be used by setting simple parameters.
- 👍 **Set constraints for your synthetic data.** You can specify inter-column constraints which you want your synthetic data to follow. Constraints are modelled explicitly by the synthyverse, not through oversampling. This ensures efficient and reliable constraint setting.

# Installation
The synthyverse is unique in its modular installation set-up. To avoid conflicting dependencies, we provide various installation templates. Each template installs only those dependencies which are required to access certain modules. 

Templates provide installation for specific generators, the evaluation module, and more. Install multiple templates to get access to multiple modules of the synthyverse, e.g., multiple generators and evaluation. 

**We strongly advise to only install templates which you require during a specific run. Installing multiple templates gives rise to potential dependency conflicts. Use separate virtual environments across installations.**

**Note that the core installation without any template doesn't install any modules.**

See the [overview of templates](https://github.com/synthyverse/synthyverse/blob/main/synthyverse/TEMPLATES.md).

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

We refer to the [docs](https://synthyverse.readthedocs.io) to learn how to use the synthyverse!


# Tutorials
- [Tabular Synthetic Data with the synthyverse: Introduction](https://github.com/synthyverse/synthyverse/blob/main/tutorial.ipynb)
