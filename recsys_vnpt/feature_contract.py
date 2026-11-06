from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MISSING_TOKEN = "__MISSING__"
UNKNOWN_TOKEN = "__UNK__"


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _clean_token(value: Any) -> str | None:
    if pd.isna(value):
        return None
    token = str(value).strip()
    return token or None


@dataclass(frozen=True)
class NumericFeature:
    name: str
    default: float = 0.0
    mean: float = 0.0
    std: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "default": self.default,
            "mean": self.mean,
            "std": self.std,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NumericFeature":
        return cls(
            name=str(data["name"]),
            default=float(data.get("default", 0.0)),
            mean=float(data.get("mean", 0.0)),
            std=max(float(data.get("std", 1.0)), 1e-12),
        )

    def normalize(self, values: pd.Series) -> np.ndarray:
        numeric = pd.to_numeric(values, errors="coerce").astype("float64").fillna(self.default)
        return ((numeric - self.mean) / self.std).to_numpy(dtype=np.float32)


@dataclass(frozen=True)
class VocabularyFeature:
    name: str
    vocabulary: dict[str, int]
    missing_token: str = MISSING_TOKEN
    unknown_token: str = UNKNOWN_TOKEN
    delimiter: str = ","

    @property
    def missing_id(self) -> int:
        return self.vocabulary[self.missing_token]

    @property
    def unknown_id(self) -> int:
        return self.vocabulary[self.unknown_token]

    def encode_value(self, value: Any) -> int:
        token = _clean_token(value)
        if token is None:
            return self.missing_id
        return self.vocabulary.get(token, self.unknown_id)

    def encode_multivalue(self, value: Any) -> list[int]:
        token = _clean_token(value)
        if token is None:
            return [self.missing_id]
        ids = [
            self.vocabulary.get(part.strip(), self.unknown_id)
            for part in token.split(self.delimiter)
            if part.strip()
        ]
        return ids or [self.missing_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vocabulary": self.vocabulary,
            "missing_token": self.missing_token,
            "unknown_token": self.unknown_token,
            "delimiter": self.delimiter,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VocabularyFeature":
        return cls(
            name=str(data["name"]),
            vocabulary={str(key): int(value) for key, value in data["vocabulary"].items()},
            missing_token=str(data.get("missing_token", MISSING_TOKEN)),
            unknown_token=str(data.get("unknown_token", UNKNOWN_TOKEN)),
            delimiter=str(data.get("delimiter", ",")),
        )


@dataclass(frozen=True)
class EncodedBatch:
    numeric_feature_names: list[str]
    numeric_values: np.ndarray
    categorical_feature_names: list[str]
    categorical_values: dict[str, np.ndarray]
    multivalue_feature_names: list[str]
    multivalue_values: dict[str, list[list[int]]]

    def slice(self, start: int, stop: int) -> "EncodedBatch":
        return EncodedBatch(
            numeric_feature_names=list(self.numeric_feature_names),
            numeric_values=self.numeric_values[start:stop],
            categorical_feature_names=list(self.categorical_feature_names),
            categorical_values={
                name: values[start:stop]
                for name, values in self.categorical_values.items()
            },
            multivalue_feature_names=list(self.multivalue_feature_names),
            multivalue_values={
                name: values[start:stop]
                for name, values in self.multivalue_values.items()
            },
        )

    def to_torch(self, device: str | None = None) -> dict[str, Any]:
        import torch

        target_device = torch.device(device) if device is not None else None
        categorical_columns = [
            self.categorical_values[name] for name in self.categorical_feature_names
        ]
        if categorical_columns:
            categorical = np.stack(categorical_columns, axis=1)
        else:
            categorical = np.zeros((self.numeric_values.shape[0], 0), dtype=np.int64)

        multivalue: dict[str, dict[str, Any]] = {}
        for name in self.multivalue_feature_names:
            rows = self.multivalue_values[name]
            offsets: list[int] = []
            values: list[int] = []
            offset = 0
            for row in rows:
                offsets.append(offset)
                values.extend(row)
                offset += len(row)
            multivalue[name] = {
                "values": torch.tensor(values, dtype=torch.long, device=target_device),
                "offsets": torch.tensor(offsets, dtype=torch.long, device=target_device),
            }

        return {
            "numeric": torch.tensor(self.numeric_values, dtype=torch.float32, device=target_device),
            "numeric_feature_names": list(self.numeric_feature_names),
            "categorical": torch.tensor(categorical, dtype=torch.long, device=target_device),
            "categorical_feature_names": list(self.categorical_feature_names),
            "multivalue": multivalue,
            "multivalue_feature_names": list(self.multivalue_feature_names),
        }


@dataclass(frozen=True)
class FeatureContract:
    domain: str
    schema_version: int
    label_policy: dict[str, Any]
    model_family: str
    model_config: dict[str, Any]
    model_config_hash: str
    numeric_features: dict[str, NumericFeature]
    categorical_features: dict[str, VocabularyFeature]
    multivalue_features: dict[str, VocabularyFeature]

    @classmethod
    def fit(
        cls,
        domain: str,
        label_policy: dict[str, Any],
        model_family: str,
        model_config: dict[str, Any],
        numeric_columns: list[str],
        categorical_columns: list[str],
        multivalue_columns: list[str],
        frame: pd.DataFrame | None = None,
        schema_version: int = 1,
        multivalue_delimiter: str = ",",
    ) -> "FeatureContract":
        data = pd.DataFrame() if frame is None else frame
        numeric: dict[str, NumericFeature] = {}
        for column in numeric_columns:
            values = pd.to_numeric(data[column], errors="coerce") if column in data else pd.Series(dtype=float)
            finite = values[np.isfinite(values)]
            feature_mean = float(finite.mean()) if not finite.empty else 0.0
            feature_std = float(finite.std(ddof=0)) if len(finite) > 1 else 1.0
            if not np.isfinite(feature_std) or feature_std < 1e-12:
                feature_std = 1.0
            numeric[column] = NumericFeature(
                name=column,
                default=feature_mean,
                mean=feature_mean,
                std=feature_std,
            )
        categorical = {
            column: VocabularyFeature(
                name=column,
                vocabulary=_fit_vocabulary(data[column] if column in data else []),
            )
            for column in categorical_columns
        }
        multivalue = {
            column: VocabularyFeature(
                name=column,
                vocabulary=_fit_multivalue_vocabulary(
                    data[column] if column in data else [],
                    delimiter=multivalue_delimiter,
                ),
                delimiter=multivalue_delimiter,
            )
            for column in multivalue_columns
        }
        return cls(
            domain=domain,
            schema_version=schema_version,
            label_policy=dict(label_policy),
            model_family=model_family,
            model_config=dict(model_config),
            model_config_hash=_stable_hash(dict(model_config)),
            numeric_features=numeric,
            categorical_features=categorical,
            multivalue_features=multivalue,
        )

    def transform(self, frame: pd.DataFrame) -> EncodedBatch:
        numeric_arrays: list[np.ndarray] = []
        for feature in self.numeric_features.values():
            if feature.name in frame:
                values = frame[feature.name]
            else:
                values = pd.Series([feature.default] * len(frame), index=frame.index)
            numeric_arrays.append(feature.normalize(values))
        if numeric_arrays:
            numeric_values = np.stack(numeric_arrays, axis=1)
        else:
            numeric_values = np.zeros((len(frame), 0), dtype=np.float32)

        categorical_values = {
            name: np.array(
                [
                    feature.encode_value(value)
                    for value in (frame[name] if name in frame else [None] * len(frame))
                ],
                dtype=np.int64,
            )
            for name, feature in self.categorical_features.items()
        }
        multivalue_values = {
            name: [
                feature.encode_multivalue(value)
                for value in (frame[name] if name in frame else [None] * len(frame))
            ]
            for name, feature in self.multivalue_features.items()
        }
        return EncodedBatch(
            numeric_feature_names=list(self.numeric_features),
            numeric_values=numeric_values,
            categorical_feature_names=list(self.categorical_features),
            categorical_values=categorical_values,
            multivalue_feature_names=list(self.multivalue_features),
            multivalue_values=multivalue_values,
        )

    def assert_model_compatible(self, model_family: str, model_config: dict[str, Any]) -> None:
        if model_family != self.model_family:
            raise ValueError(
                f"Feature contract model family is {self.model_family}, "
                f"but checkpoint model family is {model_family}"
            )
        config_hash = _stable_hash(dict(model_config))
        if config_hash != self.model_config_hash:
            raise ValueError(
                "Feature contract model config does not match checkpoint model config"
            )

    @property
    def artifact_hash(self) -> str:
        return _stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "schema_version": self.schema_version,
            "label_policy": self.label_policy,
            "model_family": self.model_family,
            "model_config": self.model_config,
            "model_config_hash": self.model_config_hash,
            "numeric_features": [feature.to_dict() for feature in self.numeric_features.values()],
            "categorical_features": [
                feature.to_dict() for feature in self.categorical_features.values()
            ],
            "multivalue_features": [
                feature.to_dict() for feature in self.multivalue_features.values()
            ],
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureContract":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        numeric = {
            feature.name: feature
            for feature in (
                NumericFeature.from_dict(item) for item in data.get("numeric_features", [])
            )
        }
        categorical = {
            feature.name: feature
            for feature in (
                VocabularyFeature.from_dict(item)
                for item in data.get("categorical_features", [])
            )
        }
        multivalue = {
            feature.name: feature
            for feature in (
                VocabularyFeature.from_dict(item)
                for item in data.get("multivalue_features", [])
            )
        }
        return cls(
            domain=str(data["domain"]),
            schema_version=int(data.get("schema_version", 1)),
            label_policy=dict(data.get("label_policy", {})),
            model_family=str(data["model_family"]),
            model_config=dict(data.get("model_config", {})),
            model_config_hash=str(data.get("model_config_hash", _stable_hash(data.get("model_config", {})))),
            numeric_features=numeric,
            categorical_features=categorical,
            multivalue_features=multivalue,
        )


def _fit_vocabulary(values: Any) -> dict[str, int]:
    vocabulary = {MISSING_TOKEN: 0, UNKNOWN_TOKEN: 1}
    tokens = sorted({token for value in values if (token := _clean_token(value)) is not None})
    for token in tokens:
        vocabulary.setdefault(token, len(vocabulary))
    return vocabulary


def _fit_multivalue_vocabulary(values: Any, delimiter: str) -> dict[str, int]:
    vocabulary = {MISSING_TOKEN: 0, UNKNOWN_TOKEN: 1}
    tokens: set[str] = set()
    for value in values:
        token = _clean_token(value)
        if token is None:
            continue
        tokens.update(part.strip() for part in token.split(delimiter) if part.strip())
    for token in sorted(tokens):
        vocabulary.setdefault(token, len(vocabulary))
    return vocabulary
