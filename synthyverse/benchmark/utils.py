import pandas as pd


def format_results(data: dict, value_at_level: int = 3) -> pd.DataFrame:
    """
    A generalistic function that formats nested dictionary data into a dataframe.
    Each row represents one metric with columns for metric name, value, and context.

    Args:
        data: Nested dictionary containing metric values
        value_at_level: The dictionary level where metric key-value pairs reside

    Returns:
        pd.DataFrame: Formatted dataframe with metric, value, and context columns
    """

    def extract_metrics_recursive(d, current_level=0, context_dict=None):
        """Recursively extract metrics at the target level"""
        if context_dict is None:
            context_dict = {}

        rows = []

        for key, value in d.items():
            if isinstance(value, dict):
                if current_level == value_at_level - 1:
                    # We've reached the level where metrics are stored
                    # Each key-value pair becomes a row
                    for metric_key, metric_value in value.items():
                        row = context_dict.copy()
                        row[f"context{current_level + 1}"] = (
                            key  # Include current level key as context
                        )
                        row["metric"] = metric_key
                        row["value"] = metric_value
                        rows.append(row)
                else:
                    # Continue traversing and collect context
                    new_context = context_dict.copy()
                    new_context[f"context{current_level + 1}"] = key
                    rows.extend(
                        extract_metrics_recursive(value, current_level + 1, new_context)
                    )
            else:
                # Leaf node - if we're at the right level, this is a metric
                if current_level == value_at_level - 1:
                    row = context_dict.copy()
                    row[f"context{current_level + 1}"] = (
                        key  # Include current level key as context
                    )
                    row["metric"] = key
                    row["value"] = value
                    rows.append(row)
                else:
                    # This is context information at a higher level
                    context_dict[f"context{current_level + 1}"] = key

        return rows

    # Extract all rows
    all_rows = extract_metrics_recursive(data)

    if not all_rows:
        # Fallback: if no rows found, try to find the deepest level automatically
        def get_max_depth(d, level=0):
            if not isinstance(d, dict):
                return level
            return max(get_max_depth(v, level + 1) for v in d.values())

        max_depth = get_max_depth(data)
        if max_depth > 0:
            # Try with the deepest level
            all_rows = extract_metrics_recursive(
                data, current_level=0, context_dict=None
            )

    # Create dataframe
    df = pd.DataFrame(all_rows)

    # Ensure 'metric' and 'value' columns are first
    if "metric" in df.columns and "value" in df.columns:
        # Reorder columns to put metric and value first, then context columns
        context_cols = [col for col in df.columns if col not in ["metric", "value"]]
        df = df[["metric", "value"] + context_cols]

    return df
