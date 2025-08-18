# Available Evaluation Metrics
## Fidelity Metrics

|Name|Description|Parameters|Reference|
|---|---|---|---|
|similarity|Calculates Shape and Trend scores. Shape indicates similarity in marginal distributions, Trend indicates similarity in bivariate correlations.|No parameters.|[SDMetrics](https://docs.sdv.dev/sdmetrics)|
|classifier_test|AUROC of a classifier (XGBoost) aiming to distinguish synthetic from real data.|No parameters.||
|prauth|Calculates alpha-precision, Beta-recall, and authenticity scores.|No parameters.|[Alaa et al. (2022)](https://proceedings.mlr.press/v162/alaa22a.html).|

## Utility Metrics
|Name|Description|Parameters|Reference|
|---|---|---|---|
|mle|Calculates Machine Learning Efficacy of synthetic data using an XGBoost regressor or classifier. Uses R2 score for regression targets and AUC for classification targets. Either takes synthetic data as training data or test data. Also outputs machine learning efficacy of real data. Also called Train Synthetic Test Real and Train Real Test Synthetic.|**train_set: ("synthetic","real"), default="synthetic"** <br/> Whether to take synthetic or real data as training data. The opposite is used as test data.|[Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs](https://arxiv.org/abs/1706.02633)|


## Privacy Metrics
|Name|Description|Parameters|Reference|
|---|---|---|---|
|dcr|Calculates Distance to Closest Record for the training data versus synthetic data and independent test set.|**estimates: list of floats or str: "mean", default=["mean",0.01,0.05,0.1,0.25,0.5]** <br/> Which estimates to calculate for DCR. Can calculate the mean or any percentile. <br/> <br/> **batch_size: int or None, default=16000** <br/> Allows calculating DCR in batches for very large datasets.||
