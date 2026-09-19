# Copyright 2025 MOSTLY AI
# Licensed under the Apache License, Version 2.0. See LICENSES/Apache-2.0.txt.
"""Flat TabARGN categorical and numeric-auto column encodings."""

import numpy as np
import pandas as pd


UNKNOWN = "_RARE_"
NULL = "<<NULL>>"


def _digit_parts(values, max_decimal=18, min_decimal=-8):
    columns = [f"E{i}" for i in range(max_decimal, min_decimal - 1, -1)]
    if values.isna().all():
        parts = pd.DataFrame({col: [0] * len(values) for col in columns})
    else:
        text = (values.astype("float64").abs()
                .apply(lambda v: np.format_float_positional(v, unique=True, pad_left=50, pad_right=20, precision=20))
                .astype("string[pyarrow]").replace("nan", pd.NA))
        text = text.str.replace(" ", "0").str.replace(".", "", n=1, regex=False)
        text = text.str[(49 - max_decimal):(49 - min_decimal + 1)]
        parts = text.str.split("", n=max_decimal - min_decimal + 2, expand=True)
        parts = parts.drop(columns=[0, max_decimal - min_decimal + 2]).fillna("0")
        parts.columns = columns
    parts.insert(0, "nan", values.isna().to_numpy())
    parts.insert(1, "neg", ((~values.isna()) & (values < 0)).to_numpy())
    return parts.astype(int)


def _distinct_bins(values, n=100):
    if len(values) <= n or len(set(values)) <= n:
        return sorted(set(values))
    count = n
    while count <= 1000:
        quantiles = np.quantile(values, np.linspace(0, 1, count + 1), method="closest_observation")
        distinct = sorted(set(quantiles))
        if len(distinct) >= n + 1:
            return distinct[: n // 2 + 1] + distinct[-n // 2:] if len(distinct) > n + 1 else distinct
        count += 1 + max(0, n - len(distinct))
    return distinct


class ColumnEncoder:
    def __init__(self, categorical=False, value_protection=True):
        self.categorical = categorical
        self.value_protection = value_protection

    def fit(self, values):
        values = values.reset_index(drop=True)
        if self.categorical:
            strings = values.astype("string").copy()
            escaped = strings.str.startswith("\x01", na=False)
            strings.loc[escaped] = "\x01" + strings.loc[escaped]
            strings = strings.replace({UNKNOWN: "\x01" + UNKNOWN, NULL: "\x01" + NULL})
            counts = strings.value_counts().to_dict()
            categories = sorted(counts)
            if self.value_protection:
                threshold = 5 + int(3 * np.random.uniform())
                categories = [v for v in categories if counts[v] >= threshold]
            self.rare_count = len(counts) - len(categories)
            self.codes = {v: i for i, v in enumerate([UNKNOWN] + ([NULL] if values.isna().any() else []) + categories)}
            self.cardinalities = {"cat": len(self.codes)}
            self.kind = "categorical"
            return self

        values = pd.to_numeric(values, errors="coerce")
        non_null = values.dropna()
        parts = _digit_parts(values)
        digit_cols = [col for col in parts if col.startswith("E")]
        self.min_digits = {col: int(parts.loc[parts.nan == 0, col].min()) if len(non_null) else 0 for col in digit_cols}
        self.max_digits = {col: int(parts.loc[parts.nan == 0, col].max()) if len(non_null) else 0 for col in digit_cols}
        nonzero = [col for col in digit_cols if self.max_digits[col] > 0]
        self.min_decimal = min((int(col[1:]) for col in nonzero), default=0)
        self.has_nan = values.isna().any()
        self.has_neg = (values < 0).any()
        if self.value_protection and len(non_null) < 20:
            self.minimum = self.maximum = None
        elif len(non_null):
            ordered = np.sort(non_null.to_numpy(dtype=float))
            low = 5 + int(3 * np.random.uniform()) if self.value_protection else 0
            high = 5 + int(3 * np.random.uniform()) if self.value_protection else 0
            self.minimum, self.maximum = ordered[low], ordered[-high - 1]
        else:
            self.minimum = self.maximum = None

        counts = non_null.value_counts().sort_index().to_dict() if non_null.nunique() < 100 else {}
        if counts:
            total = sum(counts.values())
            if self.value_protection:
                threshold = 5 + int(3 * np.random.uniform())
                counts = {v: n for v, n in counts.items() if n >= threshold}
            retained = sum(counts.values()) / total
        else:
            retained = 0.0
        if retained > 0.999:
            self.kind = "discrete"
            categories = [str(int(v)) if self.min_decimal >= 0 else str(v) for v in counts]
            self.codes = {v: i for i, v in enumerate([UNKNOWN] + ([NULL] if self.has_nan else []) + categories)}
            self.cardinalities = {"cat": len(self.codes)}
        elif len(nonzero) <= 3:
            self.kind = "digit"
            max_abs = max(abs(self.minimum), abs(self.maximum)) if self.minimum is not None else 0
            self.max_decimal = max(self.min_decimal, int(np.floor(np.log10(max_abs))) if max_abs >= 10 else 0)
            self.max_decimal = min(self.max_decimal, 18)
            self.cardinalities = {}
            if self.has_nan:
                self.cardinalities["nan"] = 2
            if self.has_neg:
                self.cardinalities["neg"] = 2
            for d in range(self.max_decimal, self.min_decimal - 1, -1):
                key = f"E{d}"
                self.cardinalities[key] = self.max_digits[key] + 1 - self.min_digits[key]
        else:
            self.kind = "binned"
            if self.minimum is None:
                self.bins = [0]
                self.min_decimal = 0
            else:
                quantiles = np.quantile(non_null, np.linspace(0, 1, 1001), method="closest_observation")
                self.bins = _distinct_bins(np.clip(quantiles, self.minimum, self.maximum))
            tokens = ["<<UNK>>"] + ([NULL] if self.has_nan else [])
            if self.minimum is not None:
                tokens += ["<<MIN>>", "<<MAX>>"]
            self.codes = {v: i for i, v in enumerate(tokens)}
            self.cardinalities = {"bin": len(tokens) + len(self.bins) - 1}
        return self

    def encode(self, values):
        values = values.reset_index(drop=True)
        if self.kind == "categorical":
            strings = values.astype("string").copy()
            escaped = strings.str.startswith("\x01", na=False)
            strings.loc[escaped] = "\x01" + strings.loc[escaped]
            strings = strings.replace({UNKNOWN: "\x01" + UNKNOWN, NULL: "\x01" + NULL}).fillna(NULL)
            return pd.DataFrame({"cat": strings.map(self.codes).fillna(0).astype(int)})
        values = pd.to_numeric(values, errors="coerce")
        if self.kind == "discrete":
            strings = values.round().astype("Int64").astype("string") if self.min_decimal >= 0 else values.astype("string")
            return pd.DataFrame({"cat": strings.fillna(NULL).map(self.codes).fillna(0).astype(int)})
        if self.kind == "binned":
            bins = self.bins.copy()
            bins[0], bins[-1] = -np.inf, np.inf
            codes = pd.cut(values, bins=bins, right=False).cat.codes + len(self.codes)
            codes = codes.mask(values.isna(), self.codes.get(NULL, 0))
            if "<<MIN>>" in self.codes:
                codes = codes.mask(values == self.bins[0], self.codes["<<MIN>>"])
                codes = codes.mask(values == self.bins[-1], self.codes["<<MAX>>"])
            return pd.DataFrame({"bin": codes.astype(int)})
        values = values.round().astype("Int64") if self.min_decimal >= 0 else values.astype("Float64")
        if self.minimum is not None:
            values = values.clip(self.minimum, self.maximum)
        missing = values.isna()
        counts = values.value_counts(normalize=True)
        if missing.any() and len(counts):
            values.loc[missing] = np.random.choice(counts.index, size=missing.sum(), p=counts.values)
        parts = _digit_parts(values, self.max_decimal, self.min_decimal)
        encoded = pd.DataFrame(index=parts.index)
        if self.has_nan:
            encoded["nan"] = missing.astype(int)
        if self.has_neg:
            encoded["neg"] = parts["neg"]
        for d in range(self.max_decimal, self.min_decimal - 1, -1):
            key = f"E{d}"
            encoded[key] = (parts[key] - self.min_digits[key]).clip(0, self.max_digits[key] - self.min_digits[key])
        return encoded

    def decode(self, encoded):
        if self.kind in ("categorical", "discrete"):
            inverse = {v: k for k, v in self.codes.items()}
            values = encoded["cat"].map(inverse).replace(NULL, pd.NA)
            if self.kind == "categorical":
                values = values.astype("string")
                mask = values.str.startswith("\x01", na=False)
                values.loc[mask] = values.loc[mask].str[1:]
                return values
            values = values.replace(UNKNOWN, pd.NA)
            return pd.to_numeric(values, errors="coerce")
        if self.kind == "binned":
            codes = encoded["bin"]
            values = pd.Series(np.nan, index=codes.index, dtype=float)
            if "<<MIN>>" in self.codes:
                values.loc[codes == self.codes["<<MIN>>"]] = self.bins[0]
                values.loc[codes == self.codes["<<MAX>>"]] = self.bins[-1]
            for i, (left, right) in enumerate(zip(self.bins[:-1], self.bins[1:])):
                mask = codes == i + len(self.codes)
                draws = np.random.uniform(left, right, mask.sum())
                scale = 10 ** -self.min_decimal
                values.loc[mask] = np.floor(draws * scale) / scale
            return values
        values = sum((encoded[f"E{d}"] + self.min_digits[f"E{d}"]).to_numpy("uint64") * 10 ** d
                     for d in range(self.max_decimal, self.min_decimal - 1, -1))
        values = pd.Series(values, dtype="Float64" if self.min_decimal < 0 else "Int64")
        if self.has_neg:
            values = values.where(encoded["neg"] == 0, -values)
        if self.has_nan:
            values = values.mask(encoded["nan"] == 1)
        if self.minimum is not None:
            values = values.clip(self.minimum, self.maximum)
        return values.round(-self.min_decimal)
