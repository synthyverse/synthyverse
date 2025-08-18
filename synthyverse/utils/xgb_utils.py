from xgboost import XGBClassifier
from xgboost.core import XGBoostError


def get_xgb_tree_method():

    try:
        # Attempt dummy model on GPU
        X_dummy = [[0.0]]
        y_dummy = [0]
        model = XGBClassifier(
            tree_method="gpu_hist",
            gpu_id=0,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        model.fit(X_dummy, y_dummy)
        return "gpu_hist"
    except XGBoostError:
        return "hist"
