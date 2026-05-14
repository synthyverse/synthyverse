# In-Depth Usage

Synthyverse has two layers for tabular synthesis:

- **Low-level components** give explicit control over preprocessing, generator fitting, metrics, and persistence.
- **High-level wrappers** combine those pieces into shorter workflows for common use cases.

Most users should start with `SynthyverseGenerator` for a single generator workflow or `TabularSynthesisBenchmark` for repeatable train/evaluate runs. Use the low-level classes when you want to inspect intermediate data, reuse one processor across generators, or call metric classes directly.

## Low-Level Preprocessing

Low-level generators expect data that is already suitable for the model. Shared tabular preprocessing is handled by `DataProcessor`.

`DataProcessor` can:

- drop, keep, or impute missing numerical values;
- apply equality and inequality constraints before training;
- restore the original column order, dtypes, and numerical precision after generation.

```python
import pandas as pd
from synthyverse.generators import DataProcessor

X = pd.read_csv("data.csv")
discrete_features = ["category", "target"]

processor = DataProcessor(
    constraints=["total=part_a+part_b", "age>=18"],
    missing_imputation_method="median",
    random_state=42,
)

X_model = processor.preprocess(X, discrete_features=discrete_features)
```

If you provide validation data, both datasets are transformed with the same fitted processor.

```python
X_train_model, X_val_model = processor.preprocess(
    X_train,
    discrete_features=discrete_features,
    X_val=X_val,
)
```

After a generator produces model-space data, call `postprocess()` to return to the original schema.

```python
X_syn_model = generator.generate(1000)
X_syn = processor.postprocess(X_syn_model)
```

### Missing Values

The current tabular imputation methods are:

- `drop`: remove rows with missing numerical values.
- `keep`: leave missing values unchanged for generators that can handle them.
- `mean`: fill numerical missing values with the mean.
- `median`: fill numerical missing values with the median.
- `most_frequent`: fill numerical missing values with the most frequent value.
- `missforest`: use iterative imputation with a random forest regressor.

Missing categorical values are left in categorical columns and can be handled by generator-specific categorical processing.

### Constraints

Constraints are strings evaluated against the tabular columns.

Equality constraints remove one side of the equation during preprocessing and recompute it during postprocessing.

```python
processor = DataProcessor(constraints=["total=part_a+part_b"])
```

Inequality constraints store a difference during preprocessing and reconstruct the constrained column during postprocessing.

```python
processor = DataProcessor(constraints=["age>=18", "income>expenses"])
```

Constraints should already hold in the training data. Imputation can change constrained columns before the constraint transform runs, so constraints are most reliable when the constrained columns are complete or when the chosen imputation preserves the intended relationship.

## Low-Level Generators

Every public generator inherits from `BaseGenerator` and follows the same public interface:

```python
from synthyverse.generators import CTGANGenerator

generator = CTGANGenerator(epochs=300, batch_size=500, random_state=42)
generator.fit(X_model, discrete_features=discrete_features)
X_syn_model = generator.generate(1000)
```

Constructor arguments are generator-specific model hyperparameters. Shared preprocessing arguments such as `constraints` and `missing_imputation_method` belong on `DataProcessor` or `SynthyverseGenerator`, not on the low-level generator classes.

You can resolve generators by registry name when building configurable workflows.

```python
from synthyverse.generators import get_generator

Generator = get_generator("ctgan")
generator = Generator(epochs=100, random_state=42)
```

Common registry names include `arf`, `bn`, `ctgan`, `tvae`, `tabsyn`, `cdtd`, `tabargn`, `tabddpm`, `permutation`, and `smote`.

## Custom Modular Workflows

The modular setup is useful when you want to control the full workflow yourself instead of letting `SynthyverseGenerator` do every step. The core pattern is:

1. Fit a `DataProcessor` on real data.
2. Train one or more low-level generators on the processed model-space data.
3. Generate model-space synthetic data.
4. Use the same fitted processor to restore the original schema.

```python
import pandas as pd
from synthyverse.generators import CTGANGenerator, DataProcessor

X_train = pd.read_csv("train.csv")
X_val = pd.read_csv("validation.csv")
discrete_features = ["category", "target"]

processor = DataProcessor(
    constraints=["total=part_a+part_b", "age>=18"],
    missing_imputation_method="median",
    random_state=42,
)

X_train_model, X_val_model = processor.preprocess(
    X_train,
    discrete_features=discrete_features,
    X_val=X_val,
)

generator = CTGANGenerator(epochs=300, batch_size=500, random_state=42)
generator.fit(
    X_train_model,
    discrete_features=discrete_features,
    X_val=X_val_model,
)

X_syn_model = generator.generate(1000)
X_syn = processor.postprocess(X_syn_model)
```

In this workflow, the generator only sees the processed model-space columns. The fitted processor owns the dataset-level contract: missing-value handling, constraint transforms, column order, dtypes, and numeric precision. That is why the same processor must be used for both preprocessing related real datasets and postprocessing generated data.

Because `DataProcessor.preprocess()` fits only on the first call, you can reuse a fitted processor to transform later datasets with the same original schema. This is useful when you want synthetic train and test samples in the same schema, or when you want to score a held-out real dataset in model space.

```python
X_test_model = processor.preprocess(X_test)
X_syn_test = processor.postprocess(generator.generate(len(X_test)))
```

You can also reuse one processed dataset across multiple generators. This keeps preprocessing fixed while you compare model behavior.

```python
from synthyverse.generators import CTGANGenerator, TVAEGenerator

generators = {
    "ctgan": CTGANGenerator(epochs=300, random_state=42),
    "tvae": TVAEGenerator(epochs=300, random_state=42),
}

synthetic_sets = {}
for name, generator in generators.items():
    generator.fit(
        X_train_model,
        discrete_features=discrete_features,
        X_val=X_val_model,
    )
    synthetic_sets[name] = processor.postprocess(generator.generate(1000))
```

For configurable experiments, combine the registry with the same modular pattern.

```python
from synthyverse.generators import get_generator

generator_configs = {
    "ctgan": {"epochs": 300, "batch_size": 500},
    "tvae": {"epochs": 300},
}

synthetic_sets = {}
for generator_name, params in generator_configs.items():
    Generator = get_generator(generator_name)
    generator = Generator(**params, random_state=42)
    generator.fit(X_train_model, discrete_features=discrete_features)
    X_syn_model = generator.generate(1000)
    synthetic_sets[generator_name] = processor.postprocess(X_syn_model)
```

When you need persistence, save the processor and each low-level generator separately. Load both pieces later and keep the same order: generate with the low-level generator, then postprocess with the loaded processor.

```python
processor.save("saved_models/shared_processor")
generator.save("saved_models/ctgan_low_level")

loaded_processor = DataProcessor.load("saved_models/shared_processor")
loaded_generator = CTGANGenerator.load("saved_models/ctgan_low_level")

X_syn_model = loaded_generator.generate(1000)
X_syn = loaded_processor.postprocess(X_syn_model)
```

## High-Level Generator Wrapper

`SynthyverseGenerator` combines a `DataProcessor` with any low-level generator. It accepts a generator registry name, class, or instance.

```python
from synthyverse.generators import SynthyverseGenerator

generator = SynthyverseGenerator(
    "ctgan",
    generator_params={"epochs": 300, "batch_size": 500},
    constraints=["total=part_a+part_b"],
    missing_imputation_method="median",
    random_state=42,
)

generator.fit(X, discrete_features=discrete_features)
X_syn = generator.generate(1000)
```

The wrapper preprocesses `X`, fits the low-level generator, samples model-space rows, and postprocesses those rows back into the original schema. The wrapped pieces remain available as `generator.generator` and `generator.processor` when you need lower-level control.

You can also pass a preconfigured processor.

```python
from synthyverse.generators import DataProcessor, SynthyverseGenerator, TVAEGenerator

processor = DataProcessor(missing_imputation_method="most_frequent", random_state=42)
wrapper = SynthyverseGenerator(
    TVAEGenerator(epochs=100, random_state=42),
    processor=processor,
)

wrapper.fit(X, discrete_features=discrete_features)
X_syn = wrapper.sample(500)
```

## Metrics and Evaluation

Metric classes can be used directly when you want full control.

```python
from synthyverse.evaluation import Wasserstein, DCR, MLE

wasserstein = Wasserstein(discrete_features=discrete_features)
dcr = DCR(discrete_features=discrete_features)
mle = MLE(
    target_column="target",
    discrete_features=discrete_features,
    train_set="synthetic",
    random_state=42,
)

fidelity_results = wasserstein.evaluate(X_train=X_train, X_syn=X_syn)
privacy_results = dcr.evaluate(X_train=X_train, X_syn=X_syn)
utility_results = mle.evaluate(X_train=X_train, X_test=X_test, X_syn=X_syn, X_val=X_val)
```

Use `TabularMetricEvaluator` to run a group of metrics with consistent metadata. The evaluator is imported from the `eval` submodule.

```python
from synthyverse.evaluation.eval import TabularMetricEvaluator

evaluator = TabularMetricEvaluator(
    metrics={
        "wasserstein": {},
        "dcr": {},
        "mle-tstr": {"train_set": "synthetic"},
        "mle-trts": {"train_set": "real"},
    },
    discrete_features=discrete_features,
    target_column="target",
    random_state=42,
)

results = evaluator.evaluate(
    X_train=X_train,
    X_test=X_test,
    X_syn=X_syn,
    X_syn_test=X_syn_test,
    X_val=X_val,
)
```

Metric registry names are resolved with `get_metric()`. The suffix after a dash is ignored for lookup, which lets you run several configurations of the same metric in one evaluator.

```python
from synthyverse.evaluation import get_metric

Metric = get_metric("wasserstein")
metric = Metric(discrete_features=discrete_features)
```

## Benchmarking

`TabularSynthesisBenchmark` is the highest-level workflow. It creates train/validation/test splits, fits or loads processors, trains or loads generators, samples synthetic datasets, evaluates metrics, and writes long-format result rows.

```python
from synthyverse.benchmark.synthesis import TabularSynthesisBenchmark

benchmark = TabularSynthesisBenchmark(
    X=X,
    save_dir="runs/ctgan",
    generator="ctgan",
    generator_params={"epochs": 300, "batch_size": 500},
    categorical_features=discrete_features,
    target_column="target",
    constraints=["total=part_a+part_b"],
    missing_imputation_method="median",
    random_state=42,
)

results = benchmark.run(
    metrics=["wasserstein", "dcr"],
    n_train_seeds=3,
    n_sets=2,
)
```

Use `train()` and `eval()` separately when you want to reuse saved models or run new metrics later.

```python
benchmark.train(n_train_seeds=3)
results = benchmark.eval(
    metrics={"mle-tstr": {"train_set": "synthetic"}},
    n_train_seeds=3,
    n_sets=1,
)
```

## Saving and Loading

Low-level generators save their fitted state in a directory containing `generator.pkl`.

```python
from synthyverse.generators import CTGANGenerator

generator = CTGANGenerator(epochs=300, random_state=42)
generator.fit(X_model, discrete_features)
generator.save("saved_models/ctgan_low_level")

loaded = CTGANGenerator.load("saved_models/ctgan_low_level")
X_syn_model = loaded.generate(1000)
```

`DataProcessor` saves to `processor.pkl` when given a directory.

```python
from synthyverse.generators import DataProcessor

processor.save("saved_models/processor")
loaded_processor = DataProcessor.load("saved_models/processor")
```

`SynthyverseGenerator` saves the wrapper state, processor, and wrapped generator together. This is the easiest persistence option when you want to generate data in the original schema after loading.

```python
generator = SynthyverseGenerator("ctgan", generator_params={"epochs": 300})
generator.fit(X, discrete_features=discrete_features)
generator.save("saved_models/ctgan_wrapper")

loaded = SynthyverseGenerator.load("saved_models/ctgan_wrapper")
X_syn = loaded.generate(1000)
```

## Practical Guidance

Start with `SynthyverseGenerator` when you need one synthetic dataset and want preprocessing handled for you. Use `DataProcessor` plus a low-level generator when you need to inspect or reuse the model-space data. Use `TabularMetricEvaluator` for small metric suites, and `TabularSynthesisBenchmark` when you need reproducible splits, saved artifacts, multiple seeds, or repeated synthetic sets.
