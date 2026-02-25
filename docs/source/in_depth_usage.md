# In-Depth Usage: Base Generator Parameters

All generators in synthyverse inherit from a base generator class (currently `TabularBaseGenerator` for tabular data), which provides preprocessing and other data handling capabilities. These parameters can be passed to any generator to customize how your data is preprocessed and how synthetic data is generated.

## Overview

The Base Generator provides three key capabilities:

1. **Constraints**: Enforce inter-column constraints in synthetic data
2. **Missing Value Imputation**: Handle missing values in your training data
3. **Retain Missingness**: Preserve missing value patterns in synthetic data

All of these parameters can be passed to any generator when instantiating it:

```python
from synthyverse.generators import ARFGenerator

generator = ARFGenerator(
    num_trees=50,
    constraints=["col1=col2+col3"],
    missing_imputation_method="mean",
    retain_missingness=True,
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

**Important (missing values)**: Constraint setting may not work as expected when you choose to impute missing values in the training data (`missing_imputation_method` other than `"drop"`). Imputed values can violate the intended relationships, so constraints learned from imputed training data may be unreliable.

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

By default, missing values are dropped or imputed and synthetic data will not contain missing values. However, you can preserve missing value patterns in synthetic data by setting `retain_missingness=True`.

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
    random_state=42
)

# Fit and generate
generator.fit(X, discrete_features)
X_syn = generator.generate(1000)

# The synthetic data will:
# - Enforce the constraints
# - Have missing values if retain_missingness=True
```

## Tips and Best Practices

1. **Start Simple**: Begin with default parameters, then add complexity as needed
3. **Missing Data**: Use `retain_missingness=True` only if missingness is informative
4. **Using Constraints**: Expect constraint setting to be most reliable when i) constraints already hold before training, and ii) no imputation is needed. If missing values are imputed, the imputed training values may violate constraints
5. **Performance**: Some preprocessing steps can increase training time, for example, by increasing dimensionality. 
6. **Generator-Specific Considerations**: Some generators already handle some of the issues which these preprocessing schemes aim to address, so be sure to check your generator's documentation beforehand


