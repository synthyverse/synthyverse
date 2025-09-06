from synthyverse.benchmark import TabularBenchmark
import json
import pandas as pd
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tabular data generation benchmark"
    )

    parser.add_argument(
        "--generator",
        type=str,
        default="xgenboost",
        help="Name of the generator to use",
    )
    parser.add_argument(
        "--n_random_splits",
        type=int,
        default=1,
        help="Number of random splits for evaluation",
    )
    parser.add_argument(
        "--n_inits", type=int, default=1, help="Number of initializations"
    )
    parser.add_argument(
        "--n_generated_datasets",
        type=int,
        default=20,
        help="Number of generated datasets",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="churn",
        help="Dataset name to use",
    )
    parser.add_argument("--test_size", type=float, default=0.3, help="Test size ratio")
    parser.add_argument(
        "--workspace",
        type=str,
        default="workspace",
        help="Workspace directory name",
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    benchmark = TabularBenchmark(
        generator_name=args.generator,
        generator_params=json.load(open("configs/model_configs.json"))[args.generator],
        n_random_splits=args.n_random_splits,
        n_inits=args.n_inits,
        n_generated_datasets=args.n_generated_datasets,
        metrics=json.load(open("configs/metric_configs.json")),
        test_size=args.test_size,
        workspace=args.workspace,
    )

    data_config = json.load(open("configs/data_configs.json"))[args.dataset]
    target_column = data_config["target"]
    discrete_features = data_config["cat_features"]
    X = pd.read_csv(data_config["path"])

    # cast data types?

    results = benchmark.run(X, target_column, discrete_features, result_format="frame")

    os.makedirs("results", exist_ok=True)
    results.to_csv(f"results/{args.dataset}_{args.generator}.csv", index=False)
