def _make_unavailable_metric(
    class_name: str,
    metric_name: str,
    import_error: Exception,
):
    class _UnavailableMetric:
        name = metric_name

        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{class_name} is unavailable because evaluation dependencies could not be imported. "
                "Install the evaluation extras with `pip install synthyverse[eval]` "
                f"and verify the environment is healthy. Original import error: {import_error!r}"
            )

    _UnavailableMetric.__name__ = class_name
    return _UnavailableMetric


try:
    from .eval import TabularMetricEvaluator
except Exception as exc:
    TabularMetricEvaluator = _make_unavailable_metric(
        "TabularMetricEvaluator", "tabular_metric_evaluator", exc
    )


try:
    from .fidelity import (
        ClassifierTest,
        AlphaPrecisionBetaRecallAuthenticity,
        ShapeTrend,
        Marginals,
        Correlations,
    )
except Exception as exc:
    ClassifierTest = _make_unavailable_metric("ClassifierTest", "classifier_test", exc)
    AlphaPrecisionBetaRecallAuthenticity = _make_unavailable_metric(
        "AlphaPrecisionBetaRecallAuthenticity", "prauth", exc
    )
    ShapeTrend = _make_unavailable_metric("ShapeTrend", "shapetrend", exc)
    Marginals = _make_unavailable_metric("Marginals", "marginals", exc)
    Correlations = _make_unavailable_metric("Correlations", "correlations", exc)

try:
    from .utility import MLE
except Exception as exc:
    MLE = _make_unavailable_metric("MLE", "mle", exc)

try:
    from .privacy import DCR, DOMIAS
except Exception as exc:
    DCR = _make_unavailable_metric("DCR", "dcr", exc)
    DOMIAS = _make_unavailable_metric("DOMIAS", "domias", exc)


all_metrics = [
    ClassifierTest,
    AlphaPrecisionBetaRecallAuthenticity,
    ShapeTrend,
    Marginals,
    Correlations,
    MLE,
    DCR,
    DOMIAS,
]


def get_metric(metric_name: str):
    """Get a metric class by name.

    Args:
        metric_name: Name of the metric to retrieve.

    Returns:
        class: Metric class corresponding to the name.

    Raises:
        ValueError: If metric name is not found.
    """
    metric_map = {m.name: m for m in all_metrics}
    if metric_name not in metric_map.keys():
        raise ValueError(f"Metric {metric_name} not found")
    return metric_map[metric_name]
