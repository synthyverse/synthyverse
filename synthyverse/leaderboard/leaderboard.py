from typing import Union
from ..generators.base import TabularBaseGenerator
import pandas as pd
import yaml
import os
from ..benchmark.synthesis import TabularSynthesisBenchmark


class LeaderboardBenchmark:

    def __init__(
        self,
        generator: Union[str, TabularBaseGenerator] = "arf",
        generator_params: dict = {},
        metrics: list = ["classifier_test", "mle", "dcr"],
        dataset: str = "QSAR_fish_toxicity",
        data_dir: str = "data/",
        generator_save_dir: str = "models/",
        test_size: float = 0.2,
        val_size: float = 0.2,
        n_splits: int = 1,
        n_inits: int = 1,
        n_generated_datasets: int = 10,
        max_eval_size: int = 50000,
        workspace: str = "workspace/",
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.dataset = dataset
        self.generator_save_dir = generator_save_dir
        self.generator = generator
        self.generator_params = generator_params
        self.test_size = test_size
        self.val_size = val_size
        self.workspace = workspace
        self.seed = seed
        self.n_splits = n_splits
        self.n_generated_datasets = n_generated_datasets
        self.n_inits = n_inits

        if isinstance(self.generator, str):
            model_name = self.generator
        else:
            try:
                model_name = self.generator.name
            except:
                model_name = self.generator.__class__.__name__
                model_name = model_name.replace(" ", "_")

        self.model_name = model_name

        self.benchmark = TabularSynthesisBenchmark(
            generator=self.generator,
            generator_params=self.generator_params,
            n_random_splits=self.n_splits,
            n_inits=self.n_inits,
            test_size=self.test_size,
            val_size=self.val_size,
            missing_imputation_method="missforest",
            retain_missingness=True,
            constraints=None,
            workspace=self.workspace,
            random_state=self.seed,
        )

    def train(self):
        # load dataset
        X = pd.read_parquet(f"{self.data_dir}/datasets/{self.dataset}.parquet")

        try:
            metadata = yaml.safe_load(
                open(f"{self.data_dir}/metadata/metadata.yaml", "r")
            )[self.dataset]
        except:
            raise ValueError(
                f"Metadata for dataset {self.dataset} not found. Please run load_data.py to download the dataset and metadata."
            )

        # perform benchmark training and save models
        models = self.benchmark.train(
            X=X,
            target_column=metadata["target_column"],
            discrete_columns=metadata["discrete_columns"],
        )

        model_output_path = os.path.join(self.generator_save_dir, self.model_name)

        if isinstance(models, dict):
            for split_key, split_models in models.items():
                for init_key, m in split_models.items():
                    p = model_output_path
                    p = os.path.join(p, split_key, init_key)
                    m.save_model(f"{p}/model.pkl")
        else:
            models.save_model(f"{model_output_path}/model.pkl")

    def eval(
        self,
        metrics: list = ["classifier_test", "mle", "dcr"],
        max_eval_size: int = 50000,
    ):
        # load dataset
        X = pd.read_parquet(f"{self.data_dir}/datasets/{self.dataset}.parquet")

        model_path = os.path.join(self.generator_save_dir, self.model_name)
        if self.n_splits == 1 and self.n_inits == 1:
            model_path = os.path.join(model_path, "model.pkl")
            models = TabularBaseGenerator.load_model(model_path)
        else:
            models = {}
            for split_i in os.listdir(model_path):
                models[split_i] = {}
                for init_i in os.listdir(os.path.join(model_path, split_i)):
                    models[split_i][init_i] = TabularBaseGenerator.load_model(
                        os.path.join(model_path, split_i, init_i, "model.pkl")
                    )

        results = self.benchmark.eval(
            X=X,
            trained_models=models,
            metrics=metrics,
            n_generated_datasets=self.n_generated_datasets,
            max_eval_size=max_eval_size,
            result_format="dict",
        )

        return results
