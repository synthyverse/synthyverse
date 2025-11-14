import pandas as pd


class BaseImputer:
    """Base class for imputation methods.

    This class provides a common interface for all imputation methods.
    It handles preprocessing, precision preservation, and data type restoration.
    It also provides the common API for training imputers and filling missing values.

    Args:
        random_state (int): Random seed for reproducibility. Default: 0.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.imputers import ICEImputer
        >>>
        >>> # Load data with missing values
        >>> X = pd.read_csv("data_with_missing.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create and fit imputer
        >>> imputer = ICEImputer(random_state=42)
        >>> imputer.fit(X, discrete_features)
        >>>
        >>> # Transform data
        >>> X_imputed = imputer.transform(X)
    """

    def __init__(self, random_state: int = 0):
        self.__dict__.update(locals())

    def fit(self, X: pd.DataFrame, discrete_features: list):
        """Fit the imputer on training data.

        Args:
            X: Training data as a pandas DataFrame.
            discrete_features: List of column names that are discrete/categorical.
        """

        self.discrete_features = discrete_features
        self.ori_cols = X.columns.tolist()
        self.numerical_features = [
            col for col in X.columns if col not in discrete_features
        ]
        self.ori_dtypes = X.dropna().dtypes
        self.ori_precision = {}
        for col in self.numerical_features:
            self.ori_precision[col] = calculate_column_precision(X[col])

        self._fit(X)

    def transform(self, X: pd.DataFrame):
        """Transform data by imputing missing values.

        Args:
            X: Data with missing values to impute.

        Returns:
            pd.DataFrame: Data with imputed values, preserving original precision
                and data types.
        """
        imputed = self._transform(X)
        if not isinstance(imputed, pd.DataFrame):
            imputed = pd.DataFrame(imputed, columns=self.ori_cols)
        # round to original precision
        for col in self.numerical_features:
            imputed[col] = imputed[col].astype(float).round(self.ori_precision[col])

        # cast to original dtypes
        imputed = imputed.astype(self.ori_dtypes)

        return imputed

    def _fit(self, X: pd.DataFrame):
        """Fit the underlying imputation model.

        This method must be implemented by subclasses.

        Args:
            X: Training data with missing values.
        """
        raise NotImplementedError("Subclasses must implement _fit")

    def _transform(self, X: pd.DataFrame):
        """Perform the actual imputation transformation.

        This method must be implemented by subclasses.

        Args:
            X: Data with missing values to impute.

        Returns:
            pd.DataFrame or np.ndarray: Imputed data.
        """
        raise NotImplementedError("Subclasses must implement _transform")


def calculate_column_precision(col_values: pd.Series) -> int:
    """Calculate the maximum precision within a numerical column.

    Args:
        col_values: Pandas Series containing numerical values.

    Returns:
        int: Maximum precision (number of decimal places) needed.
    """

    # Convert to string and split by decimal point
    str_values = col_values.dropna().astype(str)

    # Vectorized operation to find decimal parts
    decimal_parts = str_values.str.split(".").str[-1]

    # Handle cases where there's no decimal point (integer values)
    has_decimal = str_values.str.contains(".")

    # Calculate precision for each value
    def get_precision(decimal_part, has_dec):
        if not has_dec or pd.isna(decimal_part):
            return 0
        # Find last non-zero digit from the right
        for i in range(len(decimal_part) - 1, -1, -1):
            if decimal_part[i] != "0":
                return i + 1
        return 0

    # Apply the precision calculation vectorized
    precisions = [
        get_precision(part, has_dec)
        for part, has_dec in zip(decimal_parts, has_decimal)
    ]

    return max(precisions) if precisions else 0
