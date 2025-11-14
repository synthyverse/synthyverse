import pandas as pd
from typing import Union
from ..preprocessing.tabular import TabularPreprocessor


class TabularBaseGenerator:
    """Base class for tabular data generators.

    This class provides a common interface for all tabular synthetic data generators.
    It handles preprocessing, missing value imputation, and constraint enforcement.
    It also provides the common API for training and generation across all generators.

    Args:
        constraints (Union[list, str]): List of constraint strings or single constraint string.
            Constraints can be equality (e.g., "col1=col2+col3") or inequality
            (e.g., "col1<col2"). Default: [].
        missing_imputation_method (str): Method for handling missing values.
            Options: "drop", "random", "mean", "median", "mode". Default: "drop".
        retain_missingness (bool): Whether to propogate/generate missing values in the synthetic data. Default: False.
        encode_mixed_numerical_features (bool): Whether to encode mixed numerical-categorical features.
            If True, discrete spikes in numerical features are one-hot encoded and replaced by randomly imputed numerical values. Default: False.
        quantile_transform_numericals (bool): Whether to apply quantile transformation (normalization) to
            numerical features for better distribution matching. Can be useful for generative models which expect normally distributed data,
            but which do not perform this preprocessing itself. Default: False.
        random_state (int): Random seed for reproducibility. Default: 0.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import ARFGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator with constraints
        >>> generator = ARFGenerator(
        ...     constraints=["col1=col2+col3"],
        ...     missing_imputation_method="mean",
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    def __init__(
        self,
        constraints: Union[list, str] = [],
        missing_imputation_method: str = "drop",
        retain_missingness: bool = False,
        encode_mixed_numerical_features: bool = False,
        quantile_transform_numericals: bool = False,
        random_state: int = 0,
    ):
        self.random_state = random_state
        if isinstance(constraints, str):
            self.constraints = [constraints]
        else:
            self.constraints = constraints
        self.missing_imputation_method = missing_imputation_method
        self.retain_missingness = retain_missingness
        self.quantile_transform_numericals = quantile_transform_numericals
        self.encode_mixed_numerical_features = encode_mixed_numerical_features

    def fit(self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None):
        """Fit the generator on training data.

        Args:
            X: Training data as a pandas DataFrame.
            discrete_features: List of column names that are discrete/categorical.
            X_val: Optional validation data as a pandas DataFrame.

        Returns:
            self: Returns self for method chaining.
        """
        self.base_discrete_features = discrete_features.copy()
        X_prep = X.copy()

        # execute a preprocessing pipeline
        self.preprocessor = TabularPreprocessor(
            discrete_features=self.base_discrete_features,
            random_state=self.random_state,
        )
        # fit validation preprocessor first to ensure we end up with training data params
        if X_val is not None:
            X_val_prep = self.preprocessor.pipeline(
                X=X_val,
                missing_imputation_method=self.missing_imputation_method,
                retain_missingness=self.retain_missingness,
                encode_mixed_numerical_features=self.encode_mixed_numerical_features,
                quantile_transform_numericals=self.quantile_transform_numericals,
                constraints=self.constraints,
            )
        else:
            X_val_prep = None
        X_prep = self.preprocessor.pipeline(
            X=X_prep,
            missing_imputation_method=self.missing_imputation_method,
            retain_missingness=self.retain_missingness,
            encode_mixed_numerical_features=self.encode_mixed_numerical_features,
            quantile_transform_numericals=self.quantile_transform_numericals,
            constraints=self.constraints,
        )

        # update which features are discrete
        self.base_discrete_features = self.preprocessor.discrete_features.copy()

        self._fit_model(
            X=X_prep, discrete_features=self.base_discrete_features, X_val=X_val_prep
        )

        return self

    def generate(self, n: int):
        """Generate synthetic data.

        Args:
            n: Number of synthetic samples to generate.

        Returns:
            pd.DataFrame: Generated synthetic data with same format as training data.
        """
        syn_X = self._generate_data(n)

        # inverse the preprocessing pipeline
        syn_X = self.preprocessor.inverse_pipeline(syn_X)

        return syn_X

    def _fit_model(self, X: pd.DataFrame, discrete_features: list = []):
        """Fit the underlying generative model.

        This method must be implemented by subclasses.

        Args:
            X: Preprocessed training data.
            discrete_features: List of discrete feature names after preprocessing.
        """
        raise NotImplementedError("Subclasses must implement _fit_model")

    def _generate_data(self, n: int) -> pd.DataFrame:
        """Generate raw synthetic data from the fitted model.

        This method must be implemented by subclasses. The output should be
        preprocessed data that can be inverse transformed.

        Args:
            n: Number of samples to generate.

        Returns:
            pd.DataFrame: Preprocessed synthetic data.
        """
        raise NotImplementedError("Subclasses must implement _generate_data")
