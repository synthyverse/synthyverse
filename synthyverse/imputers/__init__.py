try:
    from .ice_imputer import ICEImputer
except:
    ICEImputer = None

try:
    from .missforest_imputer import MissForestImputer
except:
    MissForestImputer = None

try:
    from .ot_imputer import OTImputer
except:
    OTImputer = None


def get_imputer(imputer_name: str):
    """Get an imputer class by name.

    Args:
        imputer_name: Name of the imputer to retrieve (case-insensitive).

    Returns:
        class: Imputer class corresponding to the name.

    Raises:
        ValueError: If imputer name is not found.
    """
    imputer_name = imputer_name.lower()
    if imputer_name == "ice":
        return ICEImputer
    elif imputer_name == "missforest":
        return MissForestImputer
    elif imputer_name == "ot":
        return OTImputer
    else:
        raise ValueError(f"Imputer {imputer_name} not found")
