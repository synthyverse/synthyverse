import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from synthpop.method.helpers import proper, smooth
from synthpop.constants import NUM_COLS_DTYPES, CAT_COLS_DTYPES


from tqdm import tqdm


class CARTMethod:
    """


    Attributes:
        metadata (dict): Mapping of column names to abstract data types
                         (e.g., "numerical", "categorical", "boolean", "datetime", "timedelta").
        smoothing (bool): Whether to apply smoothing to numerical predictions.
        proper (bool): Whether to apply a resampling (proper) step during fitting.
        minibucket (int): Minimum samples per leaf in the decision tree.
        random_state (int or None): Random seed.
        tree_params (dict): Additional parameters to pass to the decision tree constructors.
    """

    def __init__(
        self,
        metadata,
        smoothing=False,
        proper=False,
        minibucket=5,
        random_state=None,
        tree_params=None,
    ):
        self.metadata = metadata
        self.smoothing = smoothing
        self.proper = proper
        self.minibucket = minibucket
        self.random_state = random_state
        self.tree_params = tree_params or {}
        self.models = {}  # Dict: column -> fitted decision tree model
        self.leaf_values = (
            {}
        )  # Dict: column -> dict mapping leaf id -> array of training y values
        self.y_bounds = (
            {}
        )  # Dict: column -> (y_real_min, y_real_max) for numerical columns
        self.fitted = False
        self._train_data = None  # Copy of preprocessed training data

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit a CART model for each column using the remaining columns as predictors.
        For numerical (and related) columns, stores the min and max of y for smoothing.
        Uses the 'proper' function to optionally resample the data.

        Args:
            data (pd.DataFrame): Preprocessed data.
        """
        self._train_data = data.copy()
        pbar = tqdm(data.columns)
        for col in pbar:
            pbar.set_description(f"Fitting column: {col}")
            # Prepare predictors (X) and target (y)
            X = data.drop(columns=[col])
            y = data[col]
            if self.proper:
                X, y = proper(X_df=X, y_df=y, random_state=self.random_state)
            dtype = self.metadata.get(col, "numerical")
            # Choose the appropriate decision tree
            if dtype in ["numerical", "datetime", "timedelta"]:
                model = DecisionTreeRegressor(
                    min_samples_leaf=self.minibucket,
                    random_state=self.random_state,
                    **self.tree_params,
                )
                # Store bounds for smoothing
                self.y_bounds[col] = (np.min(y.to_numpy()), np.max(y.to_numpy()))
            elif dtype in ["categorical", "boolean"]:
                model = DecisionTreeClassifier(
                    min_samples_leaf=self.minibucket,
                    random_state=self.random_state,
                    **self.tree_params,
                )
            else:
                print(f"Unknown data type for column '{col}', defaulting to regressor.")
                model = DecisionTreeRegressor(
                    min_samples_leaf=self.minibucket,
                    random_state=self.random_state,
                    **self.tree_params,
                )
            try:
                X_np = X.to_numpy()
                y_np = y.to_numpy()
                model.fit(X_np, y_np)
                self.models[col] = model
                # Compute leaf indices for training data and group target values by leaf.
                leaves = model.apply(X_np)
                df_leaves = pd.DataFrame({"leaf": leaves, "y": y_np})
                leaf_dict = (
                    df_leaves.groupby("leaf")["y"]
                    .apply(lambda arr: arr.values)
                    .to_dict()
                )
                self.leaf_values[col] = leaf_dict
            except Exception as e:
                print(f"Error fitting model for column '{col}': {e}")
        self.fitted = True

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic predictions using leaf-based sampling.
        For each column, the method predicts the leaf for each test row and then samples
        randomly from the training values associated with that leaf.
        Optionally applies smoothing to numerical columns.

        Args:
            X_test (pd.DataFrame): Preprocessed predictors (should contain same columns as training data).

        Returns:
            pd.DataFrame: A DataFrame with synthetic predictions for each column.
        """
        if not self.fitted:
            raise ValueError("The model must be fitted before prediction.")

        predictions = {}
        pbar = tqdm(self.models.items())
        for col, model in pbar:
            pbar.set_description(f"Synthesizing column: {col}")
            dtype = self.metadata.get(col, "numerical")
            # Prepare predictors for this column (drop the target if present)
            X = X_test.drop(columns=[col], errors="ignore")
            X_np = X.to_numpy()
            # Get leaf indices for test data
            leaves_pred = model.apply(X_np)
            y_pred = np.empty(len(leaves_pred), dtype=object)
            # Group indices by leaf
            leaf_indices = (
                pd.DataFrame({"leaf": leaves_pred, "index": range(len(leaves_pred))})
                .groupby("leaf")["index"]
                .apply(list)
                .to_dict()
            )
            for leaf, indices in leaf_indices.items():
                if leaf in self.leaf_values[col]:
                    samples = np.random.choice(
                        self.leaf_values[col][leaf], size=len(indices), replace=True
                    )
                else:
                    # Fallback: if unseen leaf, use direct prediction.
                    samples = model.predict(X_np[indices])
                for i, idx in enumerate(indices):
                    y_pred[idx] = samples[i]
            y_pred = np.array(y_pred)
            # Apply smoothing if enabled and if numeric/datetime/timedelta
            if self.smoothing and dtype in ["numerical", "datetime", "timedelta"]:
                y_real_min, y_real_max = self.y_bounds[col]
                y_pred = smooth(dtype, y_pred, y_real_min, y_real_max)
            predictions[col] = y_pred
        return pd.DataFrame(predictions)

    def sample(self, num_rows: int) -> pd.DataFrame:
        """
        Generate synthetic data with a specified number of rows.

        The predictor sampling uses the maximum of the requested number of rows
        and the size of the original training data (to ensure the trees see as much data
        as possible). However, the returned DataFrame has the user-specified number of rows.

        Args:
            num_rows (int): The number of synthetic samples to generate.

        Returns:
            pd.DataFrame: A DataFrame containing synthetic data with num_rows rows.
        """
        if not self.fitted:
            raise ValueError(
                "The model must be fitted before generating synthetic data."
            )

        # Use the maximum between num_rows and the original data size for predictor sampling
        sample_size = max(num_rows, len(self._train_data))
        synthetic_input = self._train_data.sample(
            n=sample_size, replace=True, random_state=self.random_state
        )

        # Generate synthetic data using the predict method
        synthetic_full = self.predict(synthetic_input)

        # Return only the first num_rows synthetic observations
        return synthetic_full.iloc[:num_rows].reset_index(drop=True)
