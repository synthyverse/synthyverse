<table align="center" border="0">
<tr>
<td align="center">

<img src="logo/logo.png" alt="Synthyverse logo" width="250" height="auto">

<br/>
<br/>

Welcome to the synthyverse!

An extensive ecosystem for synthetic data generation and evaluation in Python.

_The synthyverse is a work in progress. Please provide any suggestions through a GitHub Issue._

</td>
</tr>
</table>

<div style="clear: both;"></div>

# Features
- 🔧 **Highly modular installation.** Install only those modules which you require to keep your installation lightweight.
- 📚 **Extensive library for synthetic data.** Any generator or metric can be quickly added without dependency conflicts due to synthyverse's modular installation. This allows the synthyverse to host a great amount of generators and evaluation metrics. It also allows the synthyverse to wrap around any existing synthetic data library.
- ⚙️ **Benchmarking module for simplified synthetic data pipelines.** The benchmarking module executes a modular pipeline of synthetic data generation and evaluation. Choose a generator, set of evaluation metrics, and pipeline parameters, and obtain results on synthetic data quality.
- 👷 **Minimal preprocessing required.** All preprocessing is handled under the hood in the synthyverse, so no need for scaling, one-hot encoding, or handling missing values.

# Installation
The synthyverse is unique in its modular installation set-up. To avoid conflicting dependencies, we provide various installation templates. Each template installs only those dependencies which are required to access certain modules. 

Templates provide installation for specific generators, the evaluation module, and more. Install multiple templates to get access to multiple modules of the synthyverse, e.g., multiple generators and evaluation. 

**We strongly advise to only install templates which you require during a specific run. Installing multiple templates gives rise to potential dependency conflicts. Use separate virtual environments across installations.**

**Note that the core installation without any template doesn't install any modules.**

See the [overview of templates](synthyverse/TEMPLATES.md).

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

### Synthetic Data Generation
Import desired generator. Note that you can only import generators according to your installed synthyverse template.

See [all available generators](synthyverse/generators/GENERATORS.md).
```python
from synthyverse.generators import ARFGenerator
generator = ARFGenerator(num_trees=20, random_state=0)
```

Fit the generator. For tabular data, also pass which columns are discrete, as these often need to be handled differently than numerical features. If the target column is discrete, it should also be included in the discrete features list. 
```python
from sklearn.datasets import load_breast_cancer
X = load_breast_cancer(as_frame=True).frame
generator.fit(X, discrete_features=["target"])
```

Sample a synthetic dataset.
```python
syn = generator.generate(len(X))
```

### Synthetic Data Evaluation
Choose a set of metrics. Either choose default metrics as a list, or provide them as a dictionary with carefully selected hyperparameters. Add a dash to the metric name to compute various configurations of the same evaluation metric.

See [all available metrics](synthyverse/evaluation/METRICS.md).
```python
metrics = ["mle", "dcr", "similarity"]
metrics={
        "mle-trts": {"train_set": "real"},
        "mle-tstr": {"train_set": "synthetic"},
        "dcr": {"estimates": ["mean", 0.01, 0.05]},
        "similarity":{}
    }
```

Set-up a MetricEvaluator object.

```python
from synthyverse.evaluation import MetricEvaluator

evaluator = MetricEvaluator(
    metrics=metrics,
    discrete_features=["target"],
    target_column="target",
    random_state=seed
)
```

Evaluate the metrics with respect to the synthetic data, the training data used to fit the generator, and an independent holdout/test set of real data.

```python
results = evaluator.evaluate(X_train, X_test, syn)
```

### Benchmarking
The benchmarking module performs synthetic data generation and evaluation in a single pipeline. 

Set-up a benchmarking object. Supply the [generator name and its parameters](synthyverse/generators/GENERATORS.md), [evaluation metrics](synthyverse/evaluation/METRICS.md), the number of random train-test splits to fit the generator to, number of random initializations to fit the generator to, the number of synthetic sets to sample for each fitted generator, and the size of the test set.

```python
from synthyverse.benchmark import TabularBenchmark

benchmark = TabularBenchmark(
    generator_name="arf",
    generator_params={"num_trees": 20},
    n_random_splits=3,
    n_inits=3,
    n_generated_datasets=20,
    metrics=["classifier_test", "mle", "dcr"],
    test_size=0.3,
)
```

Run the benchmarking pipeline on a dataset. 
```python
results = benchmark.run(X, target_column="target", discrete_columns=["target"])
```

# Tutorials
- [Tabular Synthetic Data with the synthyverse: Introduction](tutorial.ipynb)
