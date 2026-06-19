from importlib import import_module

_METRICS = {
    "ClassifierTest": (".fidelity", "classifier_test"),
    "AlphaPrecisionBetaRecall": (".fidelity", "alphaprecisionbetarecall"),
    "PRDC": (".fidelity", "prdc"),
    # "Wasserstein": (".fidelity", "wasserstein"),
    "ShapeTrend": (".fidelity", "shapetrend"),
    "Marginals": (".fidelity", "marginals"),
    "Correlations": (".fidelity", "correlations"),
    "ARM": (".fidelity", "arm"),
    "NMI": (".fidelity", "nmi"),
    "FeatureWisePlots": (".fidelity", "featurewiseplots"),
    "MLE": (".utility", "mle"),
    "AIA": (".privacy", "aia"),
    "DCR": (".privacy", "dcr"),
    "DOMIAS": (".privacy", "mia.domias"),
    "DPI": (".privacy", "mia.dpi"),
    "ClassifierMIA": (".privacy", "mia.classifier"),
    "EnsembleMIA": (".privacy", "mia.ensemble"),
    "DomainConstraint": (".fidelity", "domainconstraint"),
}
_METRIC_BY_NAME = {name: cls for cls, (_, name) in _METRICS.items()}


def __getattr__(name: str):
    if name == "TabularMetricEvaluator":
        evaluator = import_module(".eval", __name__).TabularMetricEvaluator
        globals()[name] = evaluator
        return evaluator
    if name == "all_metrics":
        all_metrics = [__getattr__(cls) for cls in _METRICS]
        globals()[name] = all_metrics
        return all_metrics
    if name in _METRICS:
        module_name, _ = _METRICS[name]
        metric = getattr(import_module(module_name, __name__), name)
        globals()[name] = metric
        return metric
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_metric(metric_name: str):
    """Get a metric class by name."""
    class_name = _METRIC_BY_NAME.get(metric_name)
    if class_name is None:
        raise ValueError(f"Metric {metric_name} not found")
    return __getattr__(class_name)


__all__ = [
    *list(_METRICS),
    "TabularMetricEvaluator",
    "all_metrics",
    "get_metric",
]
