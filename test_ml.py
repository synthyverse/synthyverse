from sklearn.datasets import load_diabetes
from synthyverse.evaluation.ml import ml_task
from synthyverse.evaluation.utility import MLE



df = load_diabetes(as_frame=True).frame
target_col = "target"

X_train = df[:300].drop("target", axis=1)
y_train = df[:300]["target"]
X_test = df[300:].drop("target", axis=1)
y_test = df[300:]["target"]


results = ml_task(X_train, X_test, y_train, y_test, discrete_features=[], task="regression", model_name="tfm", score_fns=["rmse", "r2"])

res = []
for _ in range(5):
    res.append(ml_task(X_train, X_test, y_train, y_test, discrete_features=[], task="regression", model_name="tfm", score_fns=["rmse", "r2"]))


#### MLE metric
df = load_diabetes(as_frame=True).frame
target_col = "target"
X_train = df[:200]
X_test = df[200:400]

metric = MLE("target", model_name="tfm")
metric.evaluate(X_train, X_train, X_test, discrete_features=[])





