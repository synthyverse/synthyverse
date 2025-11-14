try:
    from .eval import TabularMetricEvaluator
except:
    TabularMetricEvaluator = None


try:
    from .fidelity import (
        ClassifierTest,
        AlphaPrecisionBetaRecallAuthenticity,
        Similarity,
        ImputationMAE_MAD,
    )
except:
    ClassifierTest = None
    AlphaPrecisionBetaRecallAuthenticity = None
    Similarity = None

try:
    from .utility import MLE
except:
    MLE = None

try:
    from .privacy import DCR, DOMIAS
except:
    DCR = None
    DOMIAS = None


all_metrics = [
    ImputationMAE_MAD,
    ClassifierTest,
    AlphaPrecisionBetaRecallAuthenticity,
    Similarity,
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
    available_metrics = [m for m in all_metrics if m is not None]
    metric_map = {m.name: m for m in available_metrics}
    if metric_name not in metric_map.keys():
        raise ValueError(f"Metric {metric_name} not found")
    return metric_map[metric_name]
