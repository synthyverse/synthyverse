import os
import pandas as pd

from ..base import BaseGenerator
from tabpfn import TabPFNClassifier, TabPFNRegressor
from .tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel


class TabPFNGenerator(BaseGenerator):

    name = "tabpfn"
    supports_purely_numerical = True
    supports_purely_categorical = True

    def __init__(
        self,
        api_key: str | None = None,
        t: float = 1.0,
        n_permutations: int = 3,
        dag: dict[int, list[int]] | None = None,
        tabpfn_clf_kwargs: dict | None = None,  # TabPFNClassifier-only kwargs
        tabpfn_reg_kwargs: dict | None = None,  # TabPFNRegressor-only kwargs
        random_state=0,
        full_determinism=False,
    ):
        super().__init__(random_state, full_determinism)
        # Optionally supply a token for model access.
        if api_key is not None:
            os.environ["TABPFN_TOKEN"] = api_key
        self.t = t
        self.n_permutations = n_permutations
        self.dag = dag
        self.tabpfn_clf_kwargs = tabpfn_clf_kwargs
        self.tabpfn_reg_kwargs = tabpfn_reg_kwargs

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        self.x_columns = X.columns
        tabpfn_clf = tabpfn_reg = None
        if len(discrete_features) > 0:
            tabpfn_clf = TabPFNClassifier(**(self.tabpfn_clf_kwargs or {}))
        if (len(X.columns) - len(discrete_features)) > 0:
            tabpfn_reg = TabPFNRegressor(**(self.tabpfn_reg_kwargs or {}))
        self.model = TabPFNUnsupervisedModel(
            tabpfn_clf=tabpfn_clf, tabpfn_reg=tabpfn_reg
        )
        discrete_feature_indices = X.columns.get_indexer(discrete_features).tolist()
        self.model.set_categorical_features(discrete_feature_indices)
        self.model.fit(X)
        return self

    def _generate(self, n: int):
        syn_X = self.model.generate_synthetic_data(
            n, self.t, self.n_permutations, self.dag
        )
        syn_X = pd.DataFrame(syn_X, columns=self.x_columns)
        self.version_ = (
            self.model.tabpfn_clf.configs_[0].name
            if self.model.tabpfn_clf is not None
            else self.model.tabpfn_reg.configs_[0].name
        )
        return syn_X
