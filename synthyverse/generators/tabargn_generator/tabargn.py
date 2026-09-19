# Copyright 2025 MOSTLY AI
# Licensed under the Apache License, Version 2.0. See LICENSES/Apache-2.0.txt.
"""Single-table TabARGN generator."""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm

from ..base import BaseGenerator
from ..dgm_utils import FastTensorDataLoader, split_validation
from .encoding import ColumnEncoder
from .model import FlatModel

TABARGN_LEGACY_MODEL_SIZE_CONFIGS = {
    "S": {
        "embedding_dim_multiplier": 2.0,
        "embedding_dim_exponent": 0.15,
        "column_compression_dim": 4,
        "regressor_layer_units": (4,),
    },
    "M": {
        "embedding_dim_multiplier": 3.0,
        "embedding_dim_exponent": 0.25,
        "column_compression_dim": 10,
        "regressor_layer_units": (16,),
    },
    "L": {
        "embedding_dim_multiplier": 4.0,
        "embedding_dim_exponent": 0.33,
        "column_compression_dim": 16,
        "regressor_layer_units": (16, 16),
    },
}


class TabARGNGenerator(BaseGenerator):
    """Flat TabARGN model from mostlyai-engine.

    `fit(X, discrete_features)` models BaseGenerator-encoded categorical features
    with the upstream categorical encoding and continuous features with the
    upstream numeric-auto encoding. `generate(n)` samples the autoregressive network.
    Upstream's default Medium-sized settings, value protection, flexible column order,
    AdamW optimizer, and validation-loss checkpoint selection are retained.
    ``TABARGN_LEGACY_MODEL_SIZE_CONFIGS`` stores the low-level parameter mapping
    for the original ``"S"``, ``"M"``, and ``"L"`` presets.

    Args:
        model_size: Optional ``"S"``, ``"M"``, or ``"L"`` preset. When set, it
            overrides the four corresponding low-level architecture arguments.
        embedding_dim_multiplier: Multiplier used for per-subcolumn embedding sizes.
        embedding_dim_exponent: Cardinality exponent used for per-subcolumn embedding sizes.
        min_embedding_dim: Minimum embedding size before capping at cardinality.
        column_compression_dim: Base dimension for compressed multi-subcolumn embeddings.
        column_compression_min_columns: Minimum encoded subcolumns before compression is enabled.
        column_compression_min_subcolumns: Minimum subcolumns in a column before compression is enabled.
        regressor_layer_units: Width multipliers for autoregressive regressor layers.
        dropout: Dropout applied inside the autoregressive regressors.
        epochs: Maximum training epochs used when ``training_steps`` is not set.
        training_steps: Number of optimizer steps. Overrides ``epochs`` when set.
        cap_train_time: Time limit in seconds for training. Default: None.
        batch_size: Physical training batch size.
        gradient_accumulation_steps: Optimizer accumulation steps.
        lr: Learning rate. ``None`` uses the upstream batch-size scaling.
        weight_decay: AdamW weight decay.
        scheduler_factor: Multiplicative LR reduction factor after validation plateaus.
        scheduler_patience: Validation checks before reducing learning rate.
        min_lr_factor: Minimum learning rate as a fraction of the initial learning rate.
        val_size: Fraction of rows reserved for validation-loss checkpoint selection.
        val_steps: Epochs between validation checks, or training steps when
            ``training_steps`` is provided. ``None`` validates once per epoch.
        patience: Number of consecutive non-improving validation checks before early stopping.
        max_validation_rows: Maximum rows reserved for validation. Negative values remove the cap.
        enable_flexible_generation: Train with a random column order per batch.
        random_generation_order: Generate each batch with a random column order.
        target_column: Discrete column used to stratify the validation split.
        value_protection: Protect rare categories and numeric extremes.
        sampling_temperature: Sampling temperature.
        sampling_top_p: Nucleus sampling threshold.

    Example:
        >>> from synthyverse.generators import TabARGNGenerator
        >>> generator = TabARGNGenerator(model_size="S", epochs=10)
        >>> generator.fit(data, discrete_features=["category"])
        >>> synthetic = generator.generate(1000)
    """

    name = "tabargn"
    legacy_model_size_configs = TABARGN_LEGACY_MODEL_SIZE_CONFIGS

    def __init__(
        self,
        model_size: Optional[str] = None,
        embedding_dim_multiplier: float = 3.0,
        embedding_dim_exponent: float = 0.25,
        min_embedding_dim: int = 10,
        column_compression_dim: int = 10,
        column_compression_min_columns: int = 50,
        column_compression_min_subcolumns: int = 2,
        regressor_layer_units: tuple = (16,),
        dropout: float = 0.25,
        epochs: int = 100,
        training_steps: Optional[int] = None,
        cap_train_time: Optional[float] = None,
        batch_size: int = 4096,
        gradient_accumulation_steps: int = 1,
        lr: Optional[float] = None,
        weight_decay: float = 0.01,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 2,
        min_lr_factor: float = 0.1,
        val_size: float = 0.2,
        val_steps: Optional[int] = None,
        patience: int = 5,
        max_validation_rows: Optional[int] = 30_000,
        enable_flexible_generation: bool = True,
        random_generation_order: bool = False,
        target_column: Optional[str] = None,
        value_protection: bool = True,
        sampling_temperature: float = 1.0,
        sampling_top_p: float = 1.0,
        random_state: int = 0,
        full_determinism: bool = False,
    ):
        super().__init__(random_state=random_state, full_determinism=full_determinism)
        if model_size is not None:
            if model_size not in self.legacy_model_size_configs:
                raise ValueError(
                    f"model_size must be one of {tuple(self.legacy_model_size_configs)}."
                )
            size_config = self.legacy_model_size_configs[model_size]
            embedding_dim_multiplier = size_config["embedding_dim_multiplier"]
            embedding_dim_exponent = size_config["embedding_dim_exponent"]
            column_compression_dim = size_config["column_compression_dim"]
            regressor_layer_units = size_config["regressor_layer_units"]
        self.model_size = model_size
        self.embedding_dim_multiplier = embedding_dim_multiplier
        self.embedding_dim_exponent = embedding_dim_exponent
        self.min_embedding_dim = min_embedding_dim
        self.column_compression_dim = column_compression_dim
        self.column_compression_min_columns = column_compression_min_columns
        self.column_compression_min_subcolumns = column_compression_min_subcolumns
        self.regressor_layer_units = regressor_layer_units
        self.dropout = dropout
        self.epochs = epochs
        self.training_steps = training_steps
        self.cap_train_time = cap_train_time
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.min_lr_factor = min_lr_factor
        self.val_size = val_size
        self.val_steps = val_steps
        self.patience = patience
        self.max_validation_rows = max_validation_rows
        self.enable_flexible_generation = enable_flexible_generation
        self.random_generation_order = random_generation_order
        self.target_column = target_column
        self.value_protection = value_protection
        self.sampling_temperature = sampling_temperature
        self.sampling_top_p = sampling_top_p

    def _fit(self, X: pd.DataFrame, discrete_features: list):
        if self.val_size >= 1:
            raise ValueError("TabARGN requires val_size to be less than 1.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be a positive integer.")
        if self.training_steps is not None and self.training_steps <= 0:
            raise ValueError("training_steps must be a positive integer.")
        if self.training_steps is None and self.epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if self.patience < 1:
            raise ValueError("patience must be at least 1.")
        if self.max_validation_rows == 0:
            raise ValueError("max_validation_rows must be nonzero.")
        if len(self.regressor_layer_units) == 0:
            raise ValueError("regressor_layer_units must contain at least one value.")

        self.column_order = list(X.columns)
        self.discrete_features = list(discrete_features)
        X = X.reset_index(drop=True)
        X_train, X_val = split_validation(
            X,
            self.val_size,
            self.target_column,
            discrete_features,
            random_state=self.random_state,
            max_validation_rows=self.max_validation_rows,
        )
        train_indices = X_train.index.to_numpy()
        val_indices = (
            np.array([], dtype=int) if X_val is None else X_val.index.to_numpy()
        )
        self.encoders = {}
        self.rare_values = {}
        self.category_values = {}
        self.columns = {}
        self.cardinalities = {}
        encoded = {}
        for i, col in enumerate(self.column_order):
            encoder = ColumnEncoder(
                col in discrete_features, self.value_protection
            ).fit(X[col])
            self.encoders[col] = encoder
            if encoder.categorical:
                non_null = X[col].dropna()
                self.category_values[col] = dict(zip(non_null.astype(str), non_null))
                rare = X.loc[
                    X[col].notna() & ~X[col].astype("string").isin(encoder.codes), col
                ]
                self.rare_values[col] = (
                    rare.to_numpy() if len(rare) else X[col].dropna().to_numpy()
                )
            frame = encoder.encode(X[col])
            name = f"c{i}"
            self.columns[name] = [f"{name}__{sub}" for sub in encoder.cardinalities]
            for sub, cardinality in encoder.cardinalities.items():
                key = f"{name}__{sub}"
                self.cardinalities[key] = cardinality
                encoded[key] = torch.tensor(frame[sub].to_numpy(), dtype=torch.long)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Mostly AI uses the first encoded training partition for this bias.
        self.empirical_probs = {}
        for sub, cardinality in self.cardinalities.items():
            col = sub.split("__")[0]
            nan_sub = f"{col}__nan"
            vals = encoded[sub][train_indices]
            if nan_sub in encoded and sub != nan_sub:
                valid = encoded[nan_sub][train_indices] == 0
                if valid.any():
                    vals = vals[valid]
            counts = torch.bincount(vals, minlength=cardinality).numpy()
            probs = (counts + 1) / (counts.sum() + cardinality)
            self.empirical_probs[sub] = np.clip(probs, 1e-12, None)

        self.model = FlatModel(
            self.columns,
            self.cardinalities,
            self.empirical_probs,
            self.device,
            self.embedding_dim_multiplier,
            self.embedding_dim_exponent,
            self.min_embedding_dim,
            self.column_compression_dim,
            self.column_compression_min_columns,
            self.column_compression_min_subcolumns,
            self.regressor_layer_units,
            self.dropout,
        )
        batch_size = self.batch_size
        batch_size = max(1, min(batch_size, len(train_indices)))
        accumulation = max(
            1, min(self.gradient_accumulation_steps, len(train_indices) // batch_size)
        )
        steps_per_epoch = max(1, len(train_indices) // (batch_size * accumulation))
        lr = self.lr
        if lr is None:
            lr = float(np.round(0.001 * np.sqrt(batch_size * accumulation / 32), 5))

        names = list(self.cardinalities)
        loader = FastTensorDataLoader(
            *(encoded[sub][train_indices] for sub in names),
            shuffle=True,
            batch_size=batch_size,
        )
        val_data = {sub: encoded[sub][val_indices] for sub in names}
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=self.min_lr_factor * lr,
        )
        use_validation = len(val_indices) > 0 and (
            self.val_steps is None or self.val_steps > 0
        )
        validation_interval = self.val_steps or (
            steps_per_epoch if self.training_steps is not None else 1
        )
        best_loss, bad_checks, best_state = float("inf"), 0, None
        train_time = 0.0
        timed_out = False
        self.trained_steps_ = 0
        self.trained_epochs_ = 0
        iterator = iter(loader)
        fixed_order = (
            list(self.columns) if not self.enable_flexible_generation else None
        )
        last_validation_step = -1

        def train_step():
            nonlocal iterator, train_time
            step_start_time = time.monotonic()
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accumulation):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                data = {
                    name: values.to(self.device) for name, values in zip(names, batch)
                }
                logits = self.model(data, fixed_order)
                loss = (
                    sum(F.cross_entropy(logits[sub], data[sub]) for sub in names)
                    / accumulation
                )
                loss.backward()
            optimizer.step()
            self.trained_steps_ += 1
            train_time += time.monotonic() - step_start_time

        def validate():
            nonlocal best_loss, bad_checks, best_state, last_validation_step
            self.model.eval()
            with torch.no_grad():
                losses = []
                for indices in torch.arange(len(val_indices)).split(batch_size):
                    data = {
                        sub: vals[indices].to(self.device)
                        for sub, vals in val_data.items()
                    }
                    logits = self.model(data, fixed_order)
                    losses.append(
                        sum(
                            F.cross_entropy(logits[sub], data[sub], reduction="none")
                            for sub in names
                        )
                    )
                val_loss = torch.cat(losses).mean().item()
            if val_loss < best_loss:
                best_loss, bad_checks = val_loss, 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
            else:
                bad_checks += 1
            scheduler.step(val_loss)
            last_validation_step = self.trained_steps_
            return bad_checks >= self.patience

        if self.training_steps is None:
            for epoch in tqdm(range(self.epochs), desc="Training", unit="epoch"):
                for _ in range(steps_per_epoch):
                    train_step()
                    if (
                        self.cap_train_time is not None
                        and train_time > self.cap_train_time
                    ):
                        tqdm.write(
                            f"Training timed out after {self.cap_train_time} seconds."
                        )
                        timed_out = True
                        break
                self.trained_epochs_ = epoch + 1
                if (
                    use_validation
                    and self.trained_epochs_ % validation_interval == 0
                    and validate()
                ):
                    break
                if timed_out:
                    break
        else:
            for _ in tqdm(range(self.training_steps), desc="Training", unit="step"):
                train_step()
                self.trained_epochs_ = int(
                    np.ceil(self.trained_steps_ / steps_per_epoch)
                )
                if self.cap_train_time is not None and train_time > self.cap_train_time:
                    tqdm.write(f"Training timed out after {self.cap_train_time} seconds.")
                    timed_out = True
                    break
                if (
                    use_validation
                    and self.trained_steps_ % validation_interval == 0
                    and validate()
                ):
                    break

        if use_validation and last_validation_step != self.trained_steps_:
            validate()
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def _generate(self, n: int):
        suppressed = {}
        for i, col in enumerate(self.column_order):
            encoder = self.encoders[col]
            if encoder.kind in {"discrete", "binned"} or (
                encoder.kind == "categorical" and encoder.rare_count == 0
            ):
                suppressed[f"c{i}__{next(iter(encoder.cardinalities))}"] = 0
        batch_size = self.batch_size or 2048
        chunks = []
        for start in tqdm(range(0, n, batch_size), desc="Sampling", unit="batch"):
            count = min(batch_size, n - start)
            order = (
                np.random.permutation(list(self.columns))
                if self.random_generation_order
                else None
            )
            outputs = self.model.sample(
                count, self.sampling_temperature, self.sampling_top_p, suppressed, order
            )
            frame = pd.DataFrame(
                {sub: values.cpu().numpy() for sub, values in outputs.items()}
            )
            decoded = {}
            for i, col in enumerate(self.column_order):
                subs = self.columns[f"c{i}"]
                source = frame[subs].copy()
                source.columns = list(self.encoders[col].cardinalities)
                values = self.encoders[col].decode(source)
                if col in self.rare_values:
                    unknown = (values == "_RARE_").fillna(False)
                    values = values.map(self.category_values[col])
                    if unknown.any():
                        values.loc[unknown] = np.random.choice(
                            self.rare_values[col], size=unknown.sum()
                        )
                decoded[col] = values
            chunks.append(pd.DataFrame(decoded))
        return pd.concat(chunks, ignore_index=True)[self.column_order]

    def get_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_trained_steps_epochs(self):
        return {
            "trained_steps": self.trained_steps_,
            "trained_epochs": self.trained_epochs_,
        }

    def _state(self):
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in {"model", "device"}
        }

    def _save_extra(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path / "model.pt")

    def _load_extra(self, path: Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FlatModel(
            self.columns,
            self.cardinalities,
            self.empirical_probs,
            self.device,
            self.embedding_dim_multiplier,
            self.embedding_dim_exponent,
            self.min_embedding_dim,
            self.column_compression_dim,
            self.column_compression_min_columns,
            self.column_compression_min_subcolumns,
            self.regressor_layer_units,
            self.dropout,
        )
        self.model.load_state_dict(
            torch.load(path / "model.pt", map_location=self.device, weights_only=True)
        )
        self.model.eval()
