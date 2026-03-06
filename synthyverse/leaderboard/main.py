from leaderboard import LeaderboardBenchmark
import argparse
import json
from typing import Any
import os

DEFAULT_METRICS = {"classifier_test": {}, "mle": {}, "dcr": {}}


def parse_json_dict(
    s: str | None, default: dict[str, Any] | None = None
) -> dict[str, Any]:
    if s is None:
        return {} if default is None else default

    s = s.strip()
    if not s.startswith("{"):
        raise argparse.ArgumentTypeError(
            'must be a JSON object/dict (e.g. \'{"param1": 5, "param2": {"x": 1}}\')'
        )

    try:
        obj = json.loads(s)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"must be valid JSON: {e}") from e

    if not isinstance(obj, dict):
        raise argparse.ArgumentTypeError("must be a JSON object/dict")

    return obj


def parse_metrics(s: str | None) -> list[Any] | dict[str, Any]:
    if s is None:
        return DEFAULT_METRICS

    s = s.strip()

    if s.startswith("{"):
        return parse_json_dict(s)

    if s.startswith("["):
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            raise argparse.ArgumentTypeError(f"must be valid JSON: {e}") from e
        if not isinstance(obj, list):
            raise argparse.ArgumentTypeError("must be a JSON list when using '['")
        return obj

    return [tok.strip() for tok in s.split(",") if tok.strip()]


parser = argparse.ArgumentParser()
parser.add_argument("--run-type", type=str, default="train", choices=["train", "eval"])

parser.add_argument("--generator", type=str, default="permutation")
parser.add_argument("--generator-params", type=parse_json_dict, default=None)
parser.add_argument("--dataset", type=str, default="QSAR_fish_toxicity")

parser.add_argument("--metrics", type=parse_metrics, default=None)

parser.add_argument("--val-size", type=float, default=0.2)
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--n-random-splits", type=int, default=2)
parser.add_argument("--n-inits", type=int, default=2)
parser.add_argument("--n-generated-datasets", type=int, default=2)
parser.add_argument(
    "--max-eval-size",
    dest="max_eval_size",
    type=int,
    default=50000,
)

parser.add_argument("--data-dir", type=str, default="data/")
parser.add_argument("--workspace", type=str, default="workspace/")
parser.add_argument("--generator-save-dir", type=str, default="models/")

parser.add_argument("--seed", type=int, default=42)


if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise ValueError(
            f"Data directory {args.data_dir} does not exist. Please run load_data to download the datasets and metadata, or specify the correct data directory."
        )

    leaderboard = LeaderboardBenchmark(
        generator=args.generator,
        generator_params=args.generator_params,
        dataset=args.dataset,
        data_dir=args.data_dir,
        generator_save_dir=args.generator_save_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        n_splits=args.n_random_splits,
        n_inits=args.n_inits,
        n_generated_datasets=args.n_generated_datasets,
        workspace=args.workspace,
        seed=args.seed,
    )

    if args.run_type == "train":
        leaderboard.train()
    elif args.run_type == "eval":
        results = leaderboard.eval(
            metrics=args.metrics,
            max_eval_size=args.max_eval_size,
        )
    else:
        raise ValueError(f"Invalid run type: {args.run_type}")
