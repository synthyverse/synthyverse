# In-Depth Usage: Base Generator Parameters

All generators in synthyverse inherit from a base generator class (currently `TabularBaseGenerator` for tabular data), which provides preprocessing and other data handling capabilities. These parameters can be passed to any generator to customize how your data is preprocessed and how synthetic data is generated.

## Overview

The Base Generator provides four key capabilities:

1. **Constraints**: Enforce inter-column constraints in synthetic data
2. **Missing Value Imputation**: Handle missing values in your training data
3. **Retain Missingness**: Preserve missing value patterns in synthetic data
4. **Encode Mixed Numerical-Discrete Features**: Handle features with mixed characteristics (e.g., zero-inflated, discrete spikes)

All of these parameters can be passed to any generator when instantiating it:

```python
from synthyverse.generators import ARFGenerator

generator = ARFGenerator(
    num_trees=50,
    constraints=["col1=col2+col3"],
    missing_imputation_method="mean",
    retain_missingness=True,
    encode_mixed_numerical_features=True,
    random_state=42
)
```

## Constraints

Constraints allow you to enforce relationships that must hold in your synthetic data. This is particularly useful when your data has mathematical relationships (e.g., totals, differences) or logical constraints.

### Equality Constraints

Equality constraints enforce that one feature equals an expression involving other features. For tabular data, this means one column equals an expression involving other columns:

```python
# Example: col1 must equal col2 + col3
generator = ARFGenerator(
    constraints=["col1=col2+col3"],
    random_state=42
)

# Multiple constraints
generator = CTGANGenerator(
    constraints=["total=price+tax", "net=total-discount"],
    random_state=42
)
```

**How it works**: During preprocessing, the left-hand side feature is removed from the data (since it can be computed exactly from the right-hand side). After generation, the left-hand side is recomputed from the generated right-hand side values.

**Important**: The constraints must already hold in your training data. The generator will enforce them in synthetic data, but cannot learn relationships that don't exist in the training set.

### Inequality Constraints

Inequality constraints enforce relationships like `feature1 < feature2` or `feature1 > feature2`. For tabular data, this means relationships between columns:

```python
# Example: col1 must be less than col2
generator = TVAEGenerator(
    constraints=["col1<col2+col3"],
    random_state=42
)

# Multiple constraints
generator = TabSynGenerator(
    target_column="target",
    constraints=["age<retirement_age", "yearly_income>=monthly_income*12"],
    random_state=42
)
```

**How it works**: The left-hand side is replaced with the difference to the right-hand side during preprocessing. After generation, the constraint is enforced by adding the right-hand side to the difference.

**Note**: Inequality constraints will only strictly hold if the generator outputs values within the range of the training data. The exact behavior may vary depending on the data type and generator used.


## Missing Value Imputation

Most generative models cannot natively handle missing values. The Base Generator provides several imputation strategies to handle missing data before training. Note that this only concerns missing numerical data, as missing categorical data can be encoded as a separate category. The available methods and their behavior may vary depending on the data type.

### Available Methods

The following methods are available for tabular data (current implementation):

- **`"drop"`** (default): Remove rows/instances with missing values
- **`"random"`**: Randomly sample from existing values
- **`"mean"`**: Fill with the mean value
- **`"median"`**: Fill with the median value
- **`"mode"`**: Fill with the most frequent value

### Examples

```python
# Drop rows with missing values (default)
generator = ARFGenerator(
    missing_imputation_method="drop",
    random_state=42
)

# Impute with mean for numerical columns
generator = CTGANGenerator(
    missing_imputation_method="mean",
    random_state=42
)

# Impute with median (more robust to outliers)
generator = TVAEGenerator(
    missing_imputation_method="median",
    random_state=42
)

# Random imputation (preserves distribution)
generator = TabSynGenerator(
    target_column="target",
    missing_imputation_method="random",
    random_state=42
)
```

### Best Practices

- Use `"random"` to preserve the shape of the original distribution
- Use `"drop"` if missing values are rare and you have sufficient data
- Use `"mean"`, `"median"`, or `"mode"` if you wish to use a simple imputation

## Retain Missingness

By default, missing values are imputed and synthetic data will not contain missing values. However, you can preserve missing value patterns in synthetic data by setting `retain_missingness=True`.

### How It Works

When `retain_missingness=True`:

1. Missing value indicators are added (for tabular data, these are binary columns like `col1_MISSING`)
2. The generator learns the pattern of missingness
3. Missing values are reinstated in synthetic data based on the learned patterns

The exact implementation may vary depending on the data type.

### Example

```python
# Preserve missing value patterns in synthetic data
generator = ARFGenerator(
    missing_imputation_method="mean",  # Still need to impute for training
    retain_missingness=True,             # But preserve pattern in output
    random_state=42
)

generator.fit(X, discrete_features)
X_syn = generator.generate(1000)  # May contain NaN values
```

### Use Cases

- When missingness is informative (e.g., "did not answer" vs "answered zero")
- When you want synthetic data to match real-world missing data patterns
- For data quality assessment and imputation method evaluation

**Note**: You must still specify a `missing_imputation_method` (the generator needs complete data for training), but the missingness pattern will be preserved in the output. Other libraries often choose random imputation when retaining missingness to ensure similarly shaped marginal distributions. 

## Encode Mixed Features

Some features have mixed characteristics (e.g., discrete spikes in otherwise continuous features) that are not well-handled by some generative models. The `encode_mixed_numerical_features` parameter handles these automatically.

### What Are Mixed Features?

Mixed features (currently implemented for numerical tabular features) are features that contain:
- Continuous values (e.g., 0.5, 1.2, 3.7)
- Discrete spikes (e.g., many zeros, or specific values like 100)

Common examples:
- Zero-inflated features (many zeros, some positive values)
- Features with default values (many 0s or -1s, some actual values)
- Features with special codes (e.g., -999 for "not applicable")

### How It Works

When `encode_mixed_numerical_features=True` (for tabular data):

1. The preprocessor detects discrete spikes (values that appear in ≥30% of instances)
2. Discrete spikes are encoded as separate indicator features (one-hot encoded for tabular data)
3. The original feature values at spike positions are replaced with random samples
4. After generation, spikes are reinstated based on the learned patterns

The exact encoding method may vary for different data types.

### Example

```python
# Handle zero-inflated and mixed-type features
generator = CTGANGenerator(
    encode_mixed_numerical_features=True,
    random_state=42
)

generator.fit(X, discrete_features)
X_syn = generator.generate(1000)
```

### Detection Criteria

For tabular data, a feature is considered "mixed" if:
- It has at least one value that appears in ≥30% of instances (discrete spike)
- It still has at least 20 distinct values outside the spikes (continuous component)
- Up to 3 discrete spikes per feature are detected

Detection criteria may vary for other data types.


## Complete Example

Here's a comprehensive example using all Base Generator parameters for tabular data:

```python
import pandas as pd
from synthyverse.generators import ARFGenerator

# Load your data
X = pd.read_csv("data.csv")
discrete_features = ["category_col", "status_col"]

# Create generator with all Base Generator parameters
generator = ARFGenerator(
    # Generator-specific parameters
    num_trees=50,
    max_iters=10,
    
    # Base Generator parameters (available for all generators)
    constraints=["total=item1+item2+item3", "age>=18"],
    missing_imputation_method="mean",
    retain_missingness=True,
    encode_mixed_numerical_features=True,
    random_state=42
)

# Fit and generate
generator.fit(X, discrete_features)
X_syn = generator.generate(1000)

# The synthetic data will:
# - Enforce the constraints
# - Have missing values if retain_missingness=True
# - Properly handle mixed numerical features
```

## Tips and Best Practices

1. **Start Simple**: Begin with default parameters, then add complexity as needed
2. **Check Your Data**: Ensure constraints hold in training data before using them
3. **Missing Data**: Use `retain_missingness=True` only if missingness is informative
4. **Mixed Features**: Enable `encode_mixed_numerical_features` if you have features with discrete spikes or mixed characteristics
5. **Performance**: Some preprocessing steps (especially encoding mixed features) can increase training time
6. **Generator-Specific Considerations**: Some generators already handle some of the issues which these preprocessing schemes aim to address, so be sure to check your generator's documentation beforehand


