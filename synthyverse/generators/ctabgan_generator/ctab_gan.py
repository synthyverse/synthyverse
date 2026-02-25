from ..base import TabularBaseGenerator
import pandas as pd
from scipy.stats import normaltest

from .ctabgan_dir.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer
from .ctabgan_dir.pipeline.data_preparation import DataPrep


class CTABGANGenerator(TabularBaseGenerator):
    """Conditional Tabular GAN (CTABGAN).

    This is the CTABGAN+ implementation from the original paper.
    Improves on previous conditional GANs through convolutional layers and elaborate preprocessing schemes.
    Unlike the original implementation, we automatically detect feature-type categories (e.g., gaussian-like columns) as part of preprocessing.

    Paper: "Ctab-gan+: Enhancing tabular data synthesis" by Zhao et al. (2024).

    Args:
        target_column (str): Name of the target column.
        class_dim (tuple): Tuple of dimensions for class-specific layers. Default: (256, 256, 256, 256).
        random_dim (int): Dimension of random noise vector. Default: 100.
        num_channels (int): Number of channels in generator. Default: 64.
        l2scale (float): L2 regularization scale. Default: 1e-5.
        batch_size (int): Batch size for training. Default: 500.
        epochs (int): Number of training epochs. Default: 150.
        sides (list): List of side dimensions for generator. Default: [4, 8, 16, 24, 32, 64].
        random_state (int): Random seed for reproducibility. Default: 0.
        **kwargs: Additional arguments passed to TabularBaseGenerator.

    Example:
        >>> import pandas as pd
        >>> from synthyverse.generators import CTABGANGenerator
        >>>
        >>> # Load data
        >>> X = pd.read_csv("data.csv")
        >>> discrete_features = ["category_col"]
        >>>
        >>> # Create generator (requires target column)
        >>> generator = CTABGANGenerator(
        ...     target_column="target",
        ...     epochs=150,
        ...     batch_size=500,
        ...     random_state=42
        ... )
        >>>
        >>> # Fit and generate
        >>> generator.fit(X, discrete_features)
        >>> X_syn = generator.generate(1000)
    """

    name = "ctabgan"
    needs_target_column = True

    # TBD: add detection of long-tailed features

    def __init__(
        self,
        target_column: str,
        class_dim: tuple = (256, 256, 256, 256),
        random_dim: int = 100,
        num_channels: int = 64,
        l2scale: float = 1e-5,
        batch_size: int = 500,
        epochs: int = 150,
        sides: list = [4, 8, 16, 24, 32, 64],
        random_state: int = 0,
        **kwargs,
    ):
        super().__init__(random_state=random_state, **kwargs)
        self.target_column = target_column
        self.model = CTABGANSynthesizer(
            class_dim=class_dim,
            random_dim=random_dim,
            num_channels=num_channels,
            l2scale=l2scale,
            batch_size=batch_size,
            epochs=epochs,
            sides=sides,
        )

    def _fit_model(
        self, X: pd.DataFrame, discrete_features: list, X_val: pd.DataFrame = None
    ):
        if self.target_column in discrete_features:
            problem_type = {"Classification": self.target_column}
        else:
            problem_type = {"Regression": self.target_column}

        numerical_features = [x for x in X.columns if x not in discrete_features]

        try:
            mixed_features = self._detect_mixed_features(X[numerical_features])
        except Exception as e:
            # try-catch block to avoid errors when there are are no numerical features
            print(e)
            mixed_features = {}
        numerical_features = [
            x for x in numerical_features if x not in mixed_features.keys()
        ]

        simple_gaussians = self._detect_simple_gaussians(X[numerical_features])
        numerical_features = [
            x for x in numerical_features if x not in simple_gaussians
        ]

        print(f"simple gaussians: {simple_gaussians}")
        print(f"mixed features: {mixed_features}")
        print(f"numerical features: {numerical_features}")

        self.data_prep = DataPrep(
            X.copy(),
            discrete_features.copy(),
            [],  # TBD: add logic to check long-tailed features
            mixed_features,
            simple_gaussians,
            numerical_features,
            [],  # integer rounding already handled in basegenerator
            problem_type,
            # 0.0,  # no test set
        )
        self.model.fit(
            train_data=self.data_prep.df,
            categorical=self.data_prep.column_types["categorical"],
            mixed=self.data_prep.column_types["mixed"],
            general=self.data_prep.column_types["general"],
            non_categorical=self.data_prep.column_types["non_categorical"],
            type=problem_type,
        )

    def _generate_data(self, n: int):
        syn = self.model.sample(n)
        syn = self.data_prep.inverse_prep(syn)
        return syn

    def _detect_mixed_features(
        self,
        df: pd.DataFrame,
        min_spike_prop: float = 0.2,
        rounding: int = 6,
        min_cont_unique: int = 20,
        max_discrete_values: int = 3,
    ):
        """Detect numeric features that are a mix of continuous values + discrete spikes.

        Args:
            df: Input data.
            min_spike_prop: "How discrete" a spike must be: a value is considered a
                discrete spike in a column if it accounts for at least this fraction
                of the non-missing rows in that column. Raise this if you want fewer
                columns to qualify (stricter), lower it to be looser.
            rounding: Number of decimal places to round before counting unique values
                (helps merge near-identical floats like 0.30000000004).
            min_cont_unique: Require at least this many distinct (rounded) values
                outside the detected spikes for the column to be considered "mixed"
                rather than purely discrete.
            max_discrete_values: Upper bound on how many spike values to return per
                column (safety against pathological cases).

        Returns:
            dict: Mapping of column name -> sorted list of detected discrete spike
                values. Only columns that meet the "mixed" criterion are included.

        Note:
            Typical zero-inflated columns will be captured by setting min_spike_prop
            somewhere around 0.05–0.20 depending on your dataset size. If a column
            is fully discrete (e.g., only a handful of unique values total), it will
            be excluded unless there are at least `min_cont_unique` unique non-spike
            values remaining.
        """

        result = {}

        for col in df.columns:
            s = df[col].copy()

            # Round to reduce float noise before counting unique values
            sr = s.round(rounding)

            vc = sr.value_counts(dropna=False)
            n = int(vc.sum())
            props = vc / n

            # Candidate spikes: values with large mass
            spikes = props[props >= min_spike_prop].index.tolist()

            if not spikes:
                continue

            # Check that there's still a meaningful continuous "tail" outside spikes
            mask_non_spike = ~sr.isin(spikes)
            cont_unique = sr[mask_non_spike].nunique()

            if cont_unique >= min_cont_unique:
                # Sort spikes by value and cap the length
                spikes_sorted = sorted(spikes)[:max_discrete_values]
                # Cast to builtins (float) for clean JSON/serialization
                result[col] = [float(v) for v in spikes_sorted]

        return result

    def _detect_simple_gaussians(
        self,
        df: pd.DataFrame,
        alpha: float = 0.05,
    ):
        """Detect numeric columns that are plausibly Gaussian.

        Uses D'Agostino & Pearson's normality test to identify columns that
        appear to follow a Gaussian distribution.

        Args:
            df: Input data.
            alpha: Significance level. A column is considered Gaussian-like if
                test p-value > alpha (fail to reject normality).

        Returns:
            list: List of column names that appear Gaussian-like.
        """

        result = []
        for col in df.columns:
            _, p = normaltest(df[col])
            if p > alpha:
                result.append(col)

        return result
