import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer

def replace_empty_with_nan(df):
    """Replace empty strings or whitespace-only strings with NaN for object columns."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].apply(
                lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
            )
    return df_copy

def replace_infinite_with_nan(df):
    """Replace infinite numeric values with NaN."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].replace([np.inf, -np.inf], np.nan)
    return df_copy

def impute_missing_values(df, num_strategy='mean', cat_strategy='mode', fill_constant='Unknown'):
    """Handle missing values for numeric and categorical columns."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            if num_strategy == 'mean':
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            elif num_strategy == 'median':
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            elif num_strategy == 'zero':
                df_copy[col] = df_copy[col].fillna(0)
        else:
            if cat_strategy == 'mode':
                if df_copy[col].dropna().empty:
                    df_copy[col] = df_copy[col].fillna(fill_constant)
                else:
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
            elif cat_strategy == 'constant':
                df_copy[col] = df_copy[col].fillna(fill_constant)
            elif cat_strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
    return df_copy

def correct_skewness(df, skew_threshold=1.0, skew_method='log'):
    """
    Correct skewness for numeric columns if above threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    skew_threshold : float, default=1.0
        Threshold above which skewness is considered high.
    skew_method : str, {'log', 'boxcox', 'yeo-johnson'}, default='log'
        Transformation method to reduce skewness.

    Returns
    -------
    df_copy : pd.DataFrame
        Transformed dataframe with reduced skewness.
    transformations : dict
        Dictionary recording transformations applied to each column.
    """
    df_copy = df.copy()
    transformations = {}
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        # 1. Skip true ID columns based on name AND high cardinality
        is_id_like_name = any(keyword in col.lower() for keyword in ['id', 'num', 'code', 'index'])
        is_mostly_unique = df_copy[col].nunique() > 0.9 * len(df_copy)
        
        # A column is a true ID if it has an ID-like name AND is mostly unique
        if is_id_like_name and is_mostly_unique:
            print(f"Skipping {col} (ID-like column)")
            continue
            
        # 2. Skip constant columns or columns with no variance
        if df_copy[col].nunique() <= 1:
            print(f"Skipping {col} (constant column)")
            continue
            
        # 3. NEW: Check if the column has any variability left for transformation
        if df_copy[col].std() < 1e-8:  # Check for near-zero standard deviation
            print(f"Skipping {col} (near-zero variance)")
            continue

        skewness_value = df_copy[col].skew()
        if abs(skewness_value) > skew_threshold:
            print(f"Correcting skewness for {col}: {skewness_value:.2f}")
            try:
                if skew_method == 'log':
                    if (df_copy[col] <= 0).any():
                        shift = abs(df_copy[col].min()) + 1
                        df_copy[col] = np.log1p(df_copy[col] + shift)
                        transformations[col] = ('log', shift)
                    else:
                        df_copy[col] = np.log1p(df_copy[col])
                        transformations[col] = ('log', 0)

                elif skew_method == 'boxcox':
                    # boxcox requires positive data
                    if (df_copy[col] <= 0).any():
                        shift = abs(df_copy[col].min()) + 1
                        transformed_data, lam = boxcox(df_copy[col] + shift)
                    else:
                        transformed_data, lam = boxcox(df_copy[col])
                    df_copy[col] = transformed_data
                    transformations[col] = ('boxcox', lam)

                elif skew_method == 'yeo-johnson':
                    pt = PowerTransformer(method='yeo-johnson', standardize=False)
                    # Reshape the data for sklearn and assign back correctly
                    transformed_data = pt.fit_transform(df_copy[[col]])
                    df_copy[col] = transformed_data[:, 0] # Extract the first (and only) column
                    transformations[col] = ('yeo-johnson', pt.lambdas_[0])

                # Ensure numeric dtype and handle any new NaNs (e.g., from log(0))
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                if df_copy[col].isnull().any():
                    median_val = df_copy[col].median()
                    df_copy[col] = df_copy[col].fillna(median_val)
                    print(f"Generated NaNs in {col} after transform. Filled with median.")

                # Check post-transformation skewness
                new_skew = df_copy[col].skew()
                print(f"Skewness reduced: {skewness_value:.2f} → {new_skew:.2f}")
                if abs(new_skew) > skew_threshold:
                    print(f"Warning: {col} is still skewed.")

            # CATCH SPECIFIC ERRORS, NOT EVERYTHING
            except ValueError as e:
                print(f"ValueError for {col} during {skew_method}: {e}. Trying Yeo-Johnson fallback.")
                # Fallback to Yeo-Johnson
                try:
                    pt = PowerTransformer(method='yeo-johnson', standardize=False)
                    transformed_data = pt.fit_transform(df_copy[[col]])
                    df_copy[col] = transformed_data[:, 0]
                    transformations[col] = ('yeo-johnson-fallback', pt.lambdas_[0])
                    print(f"Fallback Yeo-Johnson applied to {col}")
                except Exception as e2:
                    print(f"Fallback also failed for {col}: {e2}. Column skipped.")
            except Exception as e:
                print(f"Unexpected error for {col}: {e}. Column skipped.")
        else:
            print(f"Skipping {col} (skewness {skewness_value:.2f} below threshold)")

    return df_copy, transformations


def drop_duplicates(df):
    """Drop duplicate rows."""
    return df.drop_duplicates()

# Utility to make JSON serializable
def make_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(i) for i in obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    return obj

def remove_empty_and_constant_columns(df, remove_empty=True, remove_constant=True):
    """
    Remove empty (all NaN) and constant (single unique value) columns.

    Parameters:
    - remove_empty: If True, drops columns with all NaN values
    - remove_constant: If True, drops columns with only a single unique value
    """
    df_copy = df.copy()

    if remove_empty:
        empty_cols = [col for col in df_copy.columns if df_copy[col].isna().all()]
        df_copy = df_copy.drop(columns=empty_cols)
        if empty_cols:
            print(f"Removed empty columns: {empty_cols}")

    if remove_constant:
        constant_cols = [col for col in df_copy.columns if df_copy[col].nunique(dropna=True) == 1]
        df_copy = df_copy.drop(columns=constant_cols)
        if constant_cols:
            print(f"Removed constant columns: {constant_cols}")

    return df_copy