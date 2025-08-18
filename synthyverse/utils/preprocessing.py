from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    FunctionTransformer,
)
import pandas as pd


ENCODERS = {
    "one-hot": OneHotEncoder(sparse_output=False),
    "ordinal": OrdinalEncoder(),
    "standard": StandardScaler(),
    "minmax": MinMaxScaler(feature_range=(-1, 1)),
    "none": FunctionTransformer(func=lambda x: x),
}


def scale(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_syn: pd.DataFrame,
    discrete_features: list = [],
    discrete_method: str = "one-hot",
    scaling_method: str = "standard",
):

    numerical_features = [
        col for col in X_train.columns if col not in discrete_features
    ]
    # one-hot or labelencode discretes
    all_df = pd.concat([X_train, X_test, X_syn])
    discrete_encoder = ENCODERS[discrete_method]
    discrete_encoder.fit(all_df[discrete_features])
    X_train = pd.concat(
        [
            X_train[numerical_features],
            pd.DataFrame(
                discrete_encoder.transform(X_train[discrete_features]),
                columns=discrete_encoder.get_feature_names_out(),
            ),
        ],
        axis=1,
    )
    X_test = pd.concat(
        [
            X_test[numerical_features],
            pd.DataFrame(
                discrete_encoder.transform(X_test[discrete_features]),
                columns=discrete_encoder.get_feature_names_out(),
            ),
        ],
        axis=1,
    )
    X_syn = pd.concat(
        [
            X_syn[numerical_features],
            pd.DataFrame(
                discrete_encoder.transform(X_syn[discrete_features]),
                columns=discrete_encoder.get_feature_names_out(),
            ),
        ],
        axis=1,
    )

    # standard or minmax scale
    scaler = ENCODERS[scaling_method]
    X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.fit_transform(X_test[numerical_features])
    X_syn[numerical_features] = scaler.fit_transform(X_syn[numerical_features])

    return X_train, X_test, X_syn
