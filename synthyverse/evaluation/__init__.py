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
    from .fidelity import (
        ClassifierTest,
        AlphaPrecisionBetaRecall,
        PRDC,
        Wasserstein,
        ShapeTrend,
        Marginals,
        Correlations,
        ARM,
        NMI,
        FeatureWisePlots,
        DomainConstraint,
    )
except Exception as exc:
    ClassifierTest = _make_unavailable_metric("ClassifierTest", "classifier_test", exc)
    AlphaPrecisionBetaRecall = _make_unavailable_metric(
        "AlphaPrecisionBetaRecall", "alphaprecisionbetarecall", exc
    )
    PRDC = _make_unavailable_metric("PRDC", "prdc", exc)
    Wasserstein = _make_unavailable_metric("Wasserstein", "wasserstein", exc)
    ShapeTrend = _make_unavailable_metric("ShapeTrend", "shapetrend", exc)
    Marginals = _make_unavailable_metric("Marginals", "marginals", exc)
    Correlations = _make_unavailable_metric("Correlations", "correlations", exc)
    ARM = _make_unavailable_metric("ARM", "arm", exc)
    NMI = _make_unavailable_metric("NMI", "nmi", exc)
    FeatureWisePlots = _make_unavailable_metric(
        "FeatureWisePlots", "featurewiseplots", exc
    )
    DomainConstraint = _make_unavailable_metric(
        "DomainConstraint", "domainconstraint", exc
    )

try:
    from .utility import MLE
except Exception as exc:
    MLE = _make_unavailable_metric("MLE", "mle", exc)

try:
    from .privacy import AIA, DCR, DOMIAS, DPI, ClassifierMIA, EnsembleMIA
except Exception as exc:
    AIA = _make_unavailable_metric("AIA", "aia", exc)
    DCR = _make_unavailable_metric("DCR", "dcr", exc)
    DOMIAS = _make_unavailable_metric("DOMIAS", "mia.domias", exc)
    DPI = _make_unavailable_metric("DPI", "mia.dpi", exc)
    ClassifierMIA = _make_unavailable_metric("ClassifierMIA", "mia.classifier", exc)
    EnsembleMIA = _make_unavailable_metric("EnsembleMIA", "mia.ensemble", exc)


all_metrics = [
    ClassifierTest,
    AlphaPrecisionBetaRecall,
    PRDC,
    Wasserstein,
    ShapeTrend,
    Marginals,
    Correlations,
    ARM,
    NMI,
    FeatureWisePlots,
    MLE,
    AIA,
    DCR,
    DOMIAS,
    DPI,
    ClassifierMIA,
    EnsembleMIA,
    DomainConstraint,
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


def __getattr__(name: str):
    if name == "TabularMetricEvaluator":
        from .eval import TabularMetricEvaluator

        return TabularMetricEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ClassifierTest",
    "AlphaPrecisionBetaRecall",
    "PRDC",
    "Wasserstein",
    "ShapeTrend",
    "Marginals",
    "Correlations",
    "ARM",
    "NMI",
    "FeatureWisePlots",
    "MLE",
    "AIA",
    "DCR",
    "DOMIAS",
    "DPI",
    "ClassifierMIA",
    "EnsembleMIA",
    "DomainConstraint",
    "TabularMetricEvaluator",
    "all_metrics",
    "get_metric",
]
