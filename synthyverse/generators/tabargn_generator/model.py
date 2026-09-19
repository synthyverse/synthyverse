# Copyright 2025 MOSTLY AI
# Licensed under the Apache License, Version 2.0. See LICENSES/Apache-2.0.txt.
"""The single-table, flat TabARGN network from mostlyai-engine."""

import numpy as np
import torch
from torch import nn


class FlatModel(nn.Module):
    def __init__(
        self,
        columns,
        cardinalities,
        empirical_probs,
        device,
        embedding_dim_multiplier=3.0,
        embedding_dim_exponent=0.25,
        min_embedding_dim=10,
        column_compression_dim=10,
        column_compression_min_columns=50,
        column_compression_min_subcolumns=2,
        regressor_layer_units=(16,),
        dropout=0.25,
    ):
        super().__init__()
        self.columns = columns
        self.cardinalities = cardinalities
        self.device = device
        self.sub_columns = [sub for subs in columns.values() for sub in subs]
        self.embedders = nn.ModuleDict()
        self.column_embedders = nn.ModuleDict()
        self.regressors = nn.ModuleDict()
        self.predictors = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout)

        embedding_dims = {}
        for sub, cardinality in cardinalities.items():
            size = max(min_embedding_dim, int(embedding_dim_multiplier * np.ceil(cardinality**embedding_dim_exponent)))
            embedding_dims[sub] = min(cardinality, size)
            self.embedders[sub] = nn.Embedding(cardinality, embedding_dims[sub], device=device)

        self.column_dims = {}
        for col, subs in columns.items():
            dim = sum(embedding_dims[sub] for sub in subs)
            compressed = min(dim, column_compression_dim + len(subs))
            if (
                len(cardinalities) > column_compression_min_columns
                and len(subs) > column_compression_min_subcolumns
                and compressed < dim
            ):
                self.column_embedders[col] = nn.Linear(dim, compressed, device=device)
                self.column_dims[col] = compressed
            else:
                self.column_embedders[col] = nn.Identity()
                self.column_dims[col] = dim

        all_col_dim = sum(self.column_dims.values())
        regressor_layer_units = list(regressor_layer_units)
        for col, subs in columns.items():
            prev_dim = 0
            for sub in subs:
                dim_in = all_col_dim + prev_dim
                card = cardinalities[sub]
                dims = [dim_in]
                for unit in regressor_layer_units[:-1]:
                    dims.append(int(unit * round(np.log(max(dims[-1], np.e)))))
                dims.append(int(regressor_layer_units[-1] * round(np.log(max(card, np.e)))))
                self.regressors[sub] = nn.ModuleList([
                    nn.Linear(a, b, device=device) for a, b in zip(dims[:-1], dims[1:])
                ])
                predictor = nn.Linear(dims[-1], card, device=device)
                nn.init.xavier_uniform_(predictor.weight)
                with torch.no_grad():
                    predictor.bias.copy_(torch.as_tensor(np.log(empirical_probs[sub]), dtype=predictor.bias.dtype, device=device))
                self.predictors[sub] = predictor
                prev_dim += embedding_dims[sub]

    def _logits(self, sub, column_embeddings, previous):
        x = torch.cat([column_embeddings] + previous, dim=-1)
        for layer in self.regressors[sub]:
            x = layer(self.dropout(x))
        return self.predictors[sub](torch.relu(x))

    def forward(self, x, order=None):
        """Return logits for every sub-column using one batch-wide column order."""
        embedded = {sub: self.embedders[sub](x[sub]) for sub in self.sub_columns}
        col_embedded = {
            col: self.column_embedders[col](torch.cat([embedded[sub] for sub in subs], dim=-1))
            for col, subs in self.columns.items()
        }
        all_columns = torch.cat(list(col_embedded.values()), dim=-1)
        names = list(self.columns)
        permutation = torch.tensor([names.index(c) for c in order], device=self.device) if order else torch.randperm(len(names), device=self.device)
        idx = torch.argsort(permutation)
        mask = torch.tril(torch.ones(len(names), len(names), device=self.device), diagonal=-1)
        mask = mask[idx, :][:, idx]
        mask = torch.repeat_interleave(mask, torch.tensor(list(self.column_dims.values()), device=self.device), dim=1)
        outputs = {}
        for row, (col, subs) in enumerate(self.columns.items()):
            masked = all_columns * mask[row]
            for i, sub in enumerate(subs):
                outputs[sub] = self._logits(sub, masked, [embedded[s] for s in subs[:i]])
        return outputs

    @torch.no_grad()
    def sample(self, n, temperature=1.0, top_p=1.0, suppressed_codes=None, order=None):
        self.eval()
        outputs = {}
        embedded_columns = {col: torch.zeros(n, dim, device=self.device) for col, dim in self.column_dims.items()}
        for col in order if order is not None else self.columns:
            subs = self.columns[col]
            previous = []
            for sub in subs:
                logits = self._logits(sub, torch.cat(list(embedded_columns.values()), dim=-1), previous)
                probs = torch.softmax(logits, dim=-1)
                if temperature != 1.0:
                    probs = torch.softmax(torch.log(probs) / max(temperature, 1e-3), dim=-1)
                if top_p < 1.0:
                    indices = torch.argsort(probs, descending=True, dim=-1)
                    sorted_probs = probs.gather(-1, indices)
                    remove = sorted_probs.cumsum(-1) > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    probs = probs.scatter(-1, indices, sorted_probs.masked_fill(remove, 0))
                    probs = probs / probs.sum(-1, keepdim=True)
                if suppressed_codes and sub in suppressed_codes:
                    probs = probs + 1e-20
                    probs[:, suppressed_codes[sub]] = 0
                    probs = probs / probs.sum(-1, keepdim=True)
                out = torch.multinomial(probs, 1).squeeze(-1)
                outputs[sub] = out
                previous.append(self.embedders[sub](out))
            embedded_columns[col] = self.column_embedders[col](torch.cat(previous, dim=-1))
        return outputs
