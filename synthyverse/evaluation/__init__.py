try:
    from .eval import TabularMetricEvaluator
except ImportError:
    TabularMetricEvaluator = None


try:
    from .fidelity import (
        ClassifierTest,
        AlphaPrecisionBetaRecallAuthenticity,
        Similarity,
        ImputationMAE_MAD,
    )
except ImportError:
    ClassifierTest = None
    AlphaPrecisionBetaRecallAuthenticity = None
    Similarity = None

try:
    from .utility import MLE
except ImportError:
    MLE = None

try:
    from .privacy import DCR, DOMIAS
except ImportError:
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
    available_metrics = [m for m in all_metrics if m is not None]
    metric_map = {m.name: m for m in available_metrics}
    if metric_name not in metric_map.keys():
        raise ValueError(f"Metric {metric_name} not found")
    return metric_map[metric_name]
