"""Gradient-boosted regression heads for the two learned scorers.

Both scorers fit the same kind of head: squared-error trees on a few numeric and
categorical features, with early stopping on a held-out tenth of the rows. The
``sklearn`` backend is ``HistGradientBoostingRegressor`` and runs on CPU only.
The ``xgboost`` backend grows the same histogram trees and runs on a CUDA GPU
when one is present, so training and scoring can move to GPU hosts.

A head trained on a GPU stays servable on a CPU-only host: with ``device="auto"``
prediction uses the GPU when there is one and the CPU otherwise.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

GBDT_BACKENDS = ("sklearn", "xgboost")
GBDT_DEVICES = ("auto", "cuda", "cpu")


@functools.lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Whether XGBoost can train on a CUDA device here (probed once per process)."""
    try:
        import xgboost as xgb

        booster = xgb.train(
            {"tree_method": "hist", "device": "cuda", "verbosity": 0},
            xgb.DMatrix(np.zeros((2, 1), dtype=np.float32), label=np.zeros(2)),
            num_boost_round=1,
        )
        return booster is not None
    except Exception:  # noqa: BLE001 - any failure means no usable GPU
        return False


def resolve_device(device: str, *, strict: bool) -> str:
    """``auto`` picks the GPU when present. An explicit ``cuda`` without a GPU is
    an error when ``strict`` (training), and falls back to CPU otherwise
    (serving a bundle trained elsewhere)."""
    if device not in GBDT_DEVICES:
        raise ValueError(f"unsupported gbdt device: {device}")
    if device == "cpu":
        return "cpu"
    if cuda_available():
        return "cuda"
    if device == "cuda" and strict:
        raise RuntimeError("gbdt device 'cuda' requested but no CUDA GPU is usable")
    return "cpu"


class XGBoostHead:
    """An XGBoost regressor with the sklearn head's fit/predict surface."""

    def __init__(
        self,
        categorical: list[bool],
        device: str,
        seed: int,
        learning_rate: float,
        max_iter: int,
        max_leaf_nodes: int,
        min_samples_leaf: int,
        l2_regularization: float,
    ) -> None:
        self.categorical = list(categorical)
        self.device = device
        self.seed = seed
        self.params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "learning_rate": learning_rate,
            "n_estimators": max_iter,
            "max_leaves": max_leaf_nodes,
            "grow_policy": "lossguide",
            "max_depth": 0,
            # Squared error has unit hessians, so the hessian floor is the
            # (sample-weighted) leaf size floor.
            "min_child_weight": float(min_samples_leaf),
            "reg_lambda": l2_regularization,
            "max_bin": 255,
            "enable_categorical": True,
            "feature_types": ["c" if flag else "q" for flag in self.categorical],
            "early_stopping_rounds": 8,
            "random_state": seed,
            "verbosity": 0,
        }
        self.model: Any = None
        self.trained_on: str | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "XGBoostHead":
        import xgboost as xgb

        self.trained_on = resolve_device(self.device, strict=True)
        weights = np.ones(len(y), dtype=np.float32) if sample_weight is None else sample_weight
        # Same early-stopping protocol as the sklearn head: a seeded 10% split.
        order = np.random.default_rng(self.seed).permutation(len(y))
        cut = max(1, int(round(len(y) * 0.1)))
        held, kept = order[:cut], order[cut:]
        self.model = xgb.XGBRegressor(**self.params, device=self.trained_on)
        self.model.fit(
            x[kept],
            y[kept],
            sample_weight=weights[kept],
            eval_set=[(x[held], y[held])],
            sample_weight_eval_set=[weights[held]],
            verbose=False,
        )
        return self

    @property
    def n_iter_(self) -> int:
        return int(self.model.get_booster().num_boosted_rounds())

    def predict(self, x: np.ndarray) -> np.ndarray:
        device = resolve_device(self.device, strict=False)
        self.model.set_params(device=device)
        with warnings.catch_warnings():
            # Host arrays on a CUDA booster go through a DMatrix copy: correct,
            # and XGBoost says so once per process.
            warnings.filterwarnings("ignore", message=".*mismatched devices.*")
            # Every round that was grown, as the sklearn head does: early
            # stopping ends training but does not roll back to the best round.
            # The scorers' hyperparameters were tuned under that convention.
            return self.model.predict(x, iteration_range=(0, self.n_iter_))


def make_head(
    backend: str,
    device: str,
    categorical: list[bool],
    seed: int,
    learning_rate: float,
    max_iter: int,
    max_leaf_nodes: int,
    min_samples_leaf: int,
    l2_regularization: float,
):
    """A fresh regression head for one gain target."""
    if backend == "sklearn":
        if device not in GBDT_DEVICES:
            raise ValueError(f"unsupported gbdt device: {device}")
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            categorical_features=categorical,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8,
            random_state=seed,
        )
    if backend == "xgboost":
        return XGBoostHead(
            categorical, device, seed, learning_rate, max_iter, max_leaf_nodes, min_samples_leaf, l2_regularization
        )
    raise ValueError(f"unsupported gbdt backend: {backend}")


def serve_on(model: Any, device: str) -> Any:
    """Score a fitted scorer's XGBoost heads on ``device`` (sklearn heads are CPU)."""
    if device not in GBDT_DEVICES:
        raise ValueError(f"unsupported gbdt device: {device}")
    for head in getattr(model, "heads", {}).values():
        if isinstance(head, XGBoostHead):
            head.device = device
    return model


def head_device(head: Any) -> str:
    """Where a fitted head trained: always CPU for the sklearn backend."""
    return getattr(head, "trained_on", None) or "cpu"
