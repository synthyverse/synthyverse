from ..base import BaseGenerator
import pandas as pd
from scipy.stats import normaltest

from .ctabgan_dir.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer
from .ctabgan_dir.pipeline.data_preparation import DataPrep


class CTABGANGenerator(BaseGenerator):
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
    ):
        super().__init__(random_state=random_state)
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

    def _fit_model(self, X: pd.DataFrame, discrete_features: list):

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
            discrete_features,
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
        """
        Detect numeric features that are a mix of continuous values + discrete spikes.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        min_spike_prop : float, default 0.05
            "How discrete" a spike must be: a value is considered a discrete spike in a column
            if it accounts for at least this fraction of the non-missing rows in that column.
            Raise this if you want *fewer* columns to qualify (stricter), lower it to be looser.
        rounding : int, default 6
            Number of decimal places to round before counting unique values (helps merge
            near-identical floats like 0.30000000004).
        min_cont_unique : int, default 10
            Require at least this many distinct (rounded) values *outside* the detected spikes
            for the column to be considered “mixed” rather than purely discrete.
        max_discrete_values : int, default 20
            Upper bound on how many spike values to return per column (safety against
            pathological cases).

        Returns
        -------
        dict[str, list[float]]
            Mapping of column name -> sorted list of detected discrete spike values.
            Only columns that meet the “mixed” criterion are included.

        Notes
        -----
        - Typical zero-inflated columns will be captured by setting min_spike_prop
        somewhere around 0.05–0.20 depending on your dataset size.
        - If a column is *fully* discrete (e.g., only a handful of unique values total),
        it will be *excluded* unless there are at least `min_cont_unique` unique
        non-spike values remaining.
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
        """
        Detect numeric columns that are plausibly Gaussian
        using D’Agostino & Pearson’s normality test.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        columns : iterable of str, optional
            Subset of columns to check. Defaults to all numeric columns.
        alpha : float, default 0.05
            Significance level. A column is considered Gaussian-like if
            test p-value > alpha (fail to reject normality).

        Returns
        -------
        list[str]
            Columns that appear Gaussian-like.
        """

        result = []
        for col in df.columns:
            _, p = normaltest(df[col])
            if p > alpha:
                result.append(col)

        return result
