import openml
import argparse
import yaml
import pandas as pd
import os

parser = argparse.ArgumentParser()
parser.add_argument("--suite-id", type=int, default=457)
parser.add_argument("--output-dir", type=str, default="data/")
args = parser.parse_args()


def _get_discrete_cols(dataset) -> list[str]:
    discrete_columns = []
    for feature in dataset.features.values():
        if feature.data_type == "nominal":
            discrete_columns.append(feature.name)

    return discrete_columns


def load_data(suite_id: int = 45, output_dir: str = "data/"):
    os.makedirs(output_dir, exist_ok=True)
    # download datasets with metadata from openml
    suite = openml.study.get_suite(suite_id)
    dataset_ids = sorted(int(dataset_id) for dataset_id in suite.data)
    metadata = {"suite_id": suite_id}
    for dataset_id in dataset_ids:
        dataset = openml.datasets.get_dataset(dataset_id)

        discrete_columns = _get_discrete_cols(dataset)
        target_column = dataset.default_target_attribute
        X, y, _, _ = dataset.get_data(
            target=target_column,
            dataset_format="dataframe",
        )
        y = y.squeeze().reset_index(drop=True)
        X = X.drop(columns=[target_column], errors="ignore").reset_index(
            drop=True
        )  # to be safe
        X = pd.concat([X, y], axis=1)

        # fully-safe discrete/numerical detection
        for col in X.columns:
            if col not in discrete_columns:
                try:
                    X[col] = X[col].astype(float)
                except:
                    discrete_columns.append(col)

            if col in discrete_columns:
                X[col] = X[col].astype(str)
        dataset_store = f"{output_dir}/datasets/"
        os.makedirs(dataset_store, exist_ok=True)
        X.to_parquet(f"{dataset_store}/{dataset.name}.parquet")

        # extract metadata
        min_cat = float("inf")
        max_cat = 0
        for col in discrete_columns:
            n_cat = X[col].nunique()
            min_cat = min(min_cat, n_cat)
            max_cat = max(max_cat, n_cat)

        missing_count = int(
            X[[col for col in X.columns if col not in discrete_columns]]
            .isnull()
            .sum()
            .sum()
        )

        metadata_ = {
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.name,
            "row_count": len(X),
            "missing_count": missing_count,
            "discrete_columns": discrete_columns,
            "target_column": target_column,
            "min_categories": min_cat if len(discrete_columns) > 0 else 0,
            "max_categories": max_cat if len(discrete_columns) > 0 else 0,
        }
        metadata[dataset.name] = metadata_

    # write metadata to yaml
    metadata_store = f"{output_dir}/metadata/"
    os.makedirs(metadata_store, exist_ok=True)
    with open(f"{metadata_store}/metadata.yaml", "w") as f:
        yaml.dump(metadata, f)


if __name__ == "__main__":
    args = parser.parse_args()
    load_data(suite_id=args.suite_id, output_dir=args.output_dir)
