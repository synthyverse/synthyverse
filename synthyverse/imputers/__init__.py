try:
    from .ice_imputer import ICEImputer
except ImportError:
    ICEImputer = None

try:
    from .missforest_imputer import MissForestImputer
except ImportError:
    MissForestImputer = None

try:
    from .ot_imputer import OTImputer
except ImportError:
    OTImputer = None


def get_imputer(imputer_name: str):
    imputer_name = imputer_name.lower()
    if imputer_name == "ice":
        return ICEImputer
    elif imputer_name == "missforest":
        return MissForestImputer
    elif imputer_name == "ot":
        return OTImputer
    else:
        raise ValueError(f"Imputer {imputer_name} not found")
