from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score, roc_auc_score

from recsys_vnpt.feature_contract import EncodedBatch, FeatureContract
from recsys_vnpt.models import (
    DenseFCN,
    DenseModelConfig,
    EmbeddingDCNv3,
    EmbeddingFCN,
    EmbeddingModelConfig,
    HierarchicalFCN,
    OfficialECN,
    OfficialFCN,
)
from recsys_vnpt.temporal import require_leakage_safe_features


MODEL_FAMILY_ALIASES = {
    "embedding_fcn": "embedding_mlp_baseline",
    "embedding_dcnv3": "legacy_embedding_dcnv3",
}
SUPPORTED_MODEL_FAMILIES = {
    "dense_repaired_baseline",
    "embedding_fcn",
    "embedding_dcnv3",
    "embedding_mlp_baseline",
    "legacy_embedding_dcnv3",
    "fcn_official",
    "ecn_official",
    "hierarchical_fcn",
}
HIERARCHICAL_RULE_SEMANTICS = "actual_membership_contribution_v2"


@dataclass(frozen=True)
class TrainingConfig:
    model_family: str = "fcn_official"
    epochs: int = 100
    patience: int = 5
    batch_size: int = 4096
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    embedding_dim: int = 16
    numeric_projection_dim: int = 16
    hidden_dims: tuple[int, ...] = (64, 32)
    deep_cross_layers: int = 3
    shallow_cross_layers: int = 2
    dropout: float = 0.1
    deep_dropout: float = 0.1
    shallow_dropout: float = 0.1
    num_heads: int = 1
    layer_norm: bool = True
    batch_norm: bool = True
    rule_loss_weight: float = 0.4
    content_loss_weight: float = 0.4
    auxiliary_loss_weight: float = 0.2
    checkpoint_selection: str = "hierarchical"
    amp: bool = False

    def __post_init__(self) -> None:
        if self.model_family not in SUPPORTED_MODEL_FAMILIES:
            raise ValueError(f"unsupported model family: {self.model_family}")
        if self.epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("epochs, patience, and batch_size must be positive")
        if self.num_heads < 1 or self.embedding_dim < 2 or self.embedding_dim % 2:
            raise ValueError("num_heads must be positive and embedding_dim must be positive and even")
        weights = (
            self.rule_loss_weight,
            self.content_loss_weight,
            self.auxiliary_loss_weight,
        )
        if any(value < 0 for value in weights) or not np.isclose(sum(weights), 1.0):
            raise ValueError("hierarchical loss weights must be nonnegative and sum to 1")
        if self.model_family == "hierarchical_fcn" and (
            round(self.rule_loss_weight, 1),
            round(self.content_loss_weight, 1),
        ) not in {(0.3, 0.5), (0.4, 0.4), (0.5, 0.3)}:
            raise ValueError("unsupported hierarchical rule/content loss weight pair")
        if self.checkpoint_selection not in {"hierarchical", "log_loss"}:
            raise ValueError("checkpoint_selection must be hierarchical or log_loss")


@dataclass(frozen=True)
class FeatureRoles:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    multivalue: tuple[str, ...]


def infer_feature_roles(
    frame: pd.DataFrame,
    label_column: str = "label",
    model_family: str | None = None,
) -> FeatureRoles:
    excluded = {
        label_column,
        "label_reason",
        "sample_type",
        "sample_weight",
        "target_semantics",
        "completion_gain",
        "watch_time_gain",
        "episode_count",
        "total_watch_seconds",
        "max_episode_watch_seconds",
        "create_date",
        "_event_time",
        "temporal_window",
    }
    candidates = [
        column
        for column in frame.columns
        if column not in excluded and not column.startswith("_")
    ]
    require_leakage_safe_features(candidates)
    multivalue_allowlist = {"tag_names", "tag_ids", "historical_rule_ids", "content_category_name"}
    categorical_allowlist = {
        "package_code",
        "sex",
        "province",
        "tentinh",
        "type_id",
        "content_family",
        "VOD_CODE",
        "product_id",
        "profile_type",
        "rule_id",
    }
    if model_family == "hierarchical_fcn":
        categorical_allowlist.discard("rule_id")
    if model_family != "dense_repaired_baseline":
        categorical_allowlist.update({"profile_id", "content_id"})
    numeric_allowlist = {
        "age",
        "content_duration",
        "content_publish_year",
        "content_single",
        "content_hot",
        "total_episode",
    }
    multivalue = tuple(column for column in candidates if column in multivalue_allowlist)
    categorical = tuple(
        column
        for column in candidates
        if column not in multivalue and column in categorical_allowlist
    )
    numeric = tuple(
        column
        for column in candidates
        if column not in multivalue
        and column not in categorical
        and pd.api.types.is_numeric_dtype(frame[column])
        and (column.startswith("historical_") or column in numeric_allowlist)
    )
    if not numeric and not categorical and not multivalue:
        raise ValueError("no inference-safe model features were found")
    return FeatureRoles(numeric=numeric, categorical=categorical, multivalue=multivalue)


def _contract_config(config: TrainingConfig, roles: FeatureRoles) -> dict[str, Any]:
    payload = {
        "model_family": config.model_family,
        "embedding_dim": config.embedding_dim,
        "numeric_projection_dim": config.numeric_projection_dim,
        "hidden_dims": list(config.hidden_dims),
        "deep_cross_layers": config.deep_cross_layers,
        "shallow_cross_layers": config.shallow_cross_layers,
        "dropout": config.dropout,
        "deep_dropout": config.deep_dropout,
        "shallow_dropout": config.shallow_dropout,
        "num_heads": config.num_heads,
        "layer_norm": config.layer_norm,
        "batch_norm": config.batch_norm,
        "rule_loss_weight": config.rule_loss_weight,
        "content_loss_weight": config.content_loss_weight,
        "auxiliary_loss_weight": config.auxiliary_loss_weight,
        "checkpoint_selection": config.checkpoint_selection,
        "numeric_features": list(roles.numeric),
        "categorical_features": list(roles.categorical),
        "multivalue_features": list(roles.multivalue),
    }
    if config.model_family == "hierarchical_fcn":
        payload["hierarchical_rule_semantics"] = HIERARCHICAL_RULE_SEMANTICS
    return payload


def fit_feature_contract(
    frame: pd.DataFrame,
    domain: str,
    config: TrainingConfig,
    roles: FeatureRoles | None = None,
) -> FeatureContract:
    feature_roles = roles or infer_feature_roles(frame, model_family=config.model_family)
    return FeatureContract.fit(
        domain=domain,
        frame=frame,
        label_policy={"temporal": True, "target_pair_features": "excluded"},
        model_family=config.model_family,
        model_config=_contract_config(config, feature_roles),
        numeric_columns=list(feature_roles.numeric),
        categorical_columns=list(feature_roles.categorical),
        multivalue_columns=list(feature_roles.multivalue),
        schema_version=3,
    )


def dense_matrix(contract: FeatureContract, frame: pd.DataFrame) -> np.ndarray:
    encoded = contract.transform(frame)
    pieces: list[np.ndarray] = [encoded.numeric_values]
    for name in encoded.categorical_feature_names:
        values = encoded.categorical_values[name]
        size = len(contract.categorical_features[name].vocabulary)
        pieces.append(np.eye(size, dtype=np.float32)[values])
    for name in encoded.multivalue_feature_names:
        size = len(contract.multivalue_features[name].vocabulary)
        matrix = np.zeros((len(frame), size), dtype=np.float32)
        for row_index, values in enumerate(encoded.multivalue_values[name]):
            matrix[row_index, values] = 1.0
        pieces.append(matrix)
    return np.concatenate(pieces, axis=1) if pieces else np.zeros((len(frame), 0), dtype=np.float32)


def _embedding_config(contract: FeatureContract, config: TrainingConfig) -> EmbeddingModelConfig:
    return EmbeddingModelConfig.from_contract(
        contract,
        embedding_dim=config.embedding_dim,
        numeric_projection_dim=config.numeric_projection_dim,
        hidden_dims=config.hidden_dims,
        deep_cross_layers=config.deep_cross_layers,
        shallow_cross_layers=config.shallow_cross_layers,
        dropout=config.dropout,
        deep_dropout=config.deep_dropout,
        shallow_dropout=config.shallow_dropout,
        num_heads=config.num_heads,
        layer_norm=config.layer_norm,
        batch_norm=config.batch_norm,
    )


def build_model(contract: FeatureContract, config: TrainingConfig) -> torch.nn.Module:
    if config.model_family == "dense_repaired_baseline":
        input_dim = dense_matrix(contract, pd.DataFrame(index=[0])).shape[1]
        return DenseFCN(DenseModelConfig(input_dim=input_dim, hidden_dims=config.hidden_dims, dropout=config.dropout))
    embedding_config = _embedding_config(contract, config)
    if config.model_family in {"embedding_fcn", "embedding_mlp_baseline"}:
        return EmbeddingFCN(embedding_config)
    if config.model_family in {"embedding_dcnv3", "legacy_embedding_dcnv3"}:
        return EmbeddingDCNv3(embedding_config)
    if config.model_family == "fcn_official":
        return OfficialFCN(embedding_config)
    if config.model_family == "ecn_official":
        return OfficialECN(embedding_config)
    if config.model_family == "hierarchical_fcn":
        return HierarchicalFCN(embedding_config)
    raise ValueError(f"unsupported model family: {config.model_family}")


def _device(requested: str | None = None, require_cuda: bool = False) -> torch.device:
    target = torch.device(requested) if requested else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and target.type != "cuda":
        raise RuntimeError("CUDA was required but the active PyTorch build cannot use CUDA")
    return target


def _autocast_dtype(model_family: str, target: torch.device) -> torch.dtype:
    official = {"fcn_official", "ecn_official", "hierarchical_fcn"}
    if (
        target.type == "cuda"
        and model_family in official
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float16


def _batch_output(
    model: torch.nn.Module,
    contract: FeatureContract,
    frame: pd.DataFrame,
    family: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if family == "dense_repaired_baseline":
        tensor = torch.tensor(dense_matrix(contract, frame), dtype=torch.float32, device=device)
        return model(tensor)
    return model(contract.transform(frame).to_torch(str(device)))


def prepare_inference_input(
    contract: FeatureContract,
    frame: pd.DataFrame,
    model_family: str,
) -> np.ndarray | EncodedBatch:
    if model_family == "dense_repaired_baseline":
        return dense_matrix(contract, frame)
    return contract.transform(frame)


def _prepared_batch_output(
    model: torch.nn.Module,
    prepared: np.ndarray | EncodedBatch,
    family: str,
    start: int,
    stop: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if family == "dense_repaired_baseline":
        assert isinstance(prepared, np.ndarray)
        tensor = torch.tensor(
            prepared[start:stop], dtype=torch.float32, device=device
        )
        return model(tensor)
    assert isinstance(prepared, EncodedBatch)
    return model(prepared.slice(start, stop).to_torch(str(device)))


from recsys_vnpt.losses import (  # re-exported: tests import them from here
    TRI_BCE_FAMILIES,
    _grouped_listwise_loss,
    _grouped_pairwise_loss,
    _loss,
    _rule_listwise_loss,
    _single_target_loss,
    _soft_target_log_loss,
    _weighted_bce,
)



def predict(
    model: torch.nn.Module,
    contract: FeatureContract,
    frame: pd.DataFrame,
    model_family: str,
    batch_size: int = 4096,
    device: str | None = None,
    amp: bool = False,
    output_key: str = "y_pred",
    prepared: np.ndarray | EncodedBatch | None = None,
) -> np.ndarray:
    target = _device(device)
    model.to(target).eval()
    scores: list[np.ndarray] = []
    model_input = (
        prepared
        if prepared is not None
        else prepare_inference_input(contract, frame, model_family)
    )
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            with torch.autocast(
                device_type=target.type,
                dtype=_autocast_dtype(model_family, target),
                enabled=amp and target.type == "cuda",
            ):
                outputs = _prepared_batch_output(
                    model,
                    model_input,
                    model_family,
                    start,
                    min(start + batch_size, len(frame)),
                    target,
                )
            if output_key not in outputs:
                raise ValueError(
                    f"model family {model_family} does not provide output {output_key}"
                )
            scores.append(
                outputs[output_key].float().detach().cpu().numpy().reshape(-1)
            )
    return np.concatenate(scores) if scores else np.array([], dtype=np.float32)


def classification_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    truth = np.asarray(labels).astype(int)
    values = np.asarray(scores).astype(float)
    predicted = (values >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "log_loss": float(log_loss(truth, np.clip(values, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }
    metrics["auc"] = float(roc_auc_score(truth, values)) if len(np.unique(truth)) > 1 else 0.0
    return metrics


def _config_hash(config: TrainingConfig) -> str:
    return _config_payload_hash(asdict(config))


def _config_payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=list).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frame_data_hash(frame: pd.DataFrame) -> str:
    """Hash schema, row order, and values used by deterministic training."""
    columns = sorted(map(str, frame.columns))
    normalized = frame.loc[:, columns].copy()
    for column in columns:
        normalized[column] = normalized[column].astype("string").fillna("<NA>")
    row_hashes = pd.util.hash_pandas_object(
        normalized, index=False, categorize=True
    ).to_numpy()
    schema = json.dumps(
        {
            "columns": columns,
            "dtypes": {column: str(frame[column].dtype) for column in columns},
            "rows": len(frame),
        },
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(schema)
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()




def _hierarchical_slate_batches(
    frame: pd.DataFrame,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], dict[str, int]]:
    """Pack complete temporal profile slates without crossing batch boundaries."""
    if "profile_id" not in frame:
        raise ValueError("hierarchical batching requires profile_id")
    windows = (
        frame["temporal_window"].astype(str)
        if "temporal_window" in frame
        else pd.Series("__NO_WINDOW__", index=frame.index, dtype="string")
    )
    slate_frame = pd.DataFrame(
        {
            "temporal_window": windows.to_numpy(),
            "profile_id": frame["profile_id"].astype(str).to_numpy(),
            "_row": np.arange(len(frame), dtype=np.int64),
        }
    )
    slates = [
        group["_row"].to_numpy(dtype=np.int64)
        for _, group in slate_frame.groupby(
            ["temporal_window", "profile_id"], sort=False, observed=True
        )
    ]
    batches: list[np.ndarray] = []
    current: list[np.ndarray] = []
    current_rows = 0
    for slate_index in rng.permutation(len(slates)):
        slate = slates[int(slate_index)]
        if current and current_rows + len(slate) > batch_size:
            batches.append(np.concatenate(current))
            current = []
            current_rows = 0
        current.append(slate)
        current_rows += len(slate)
        if current_rows >= batch_size:
            batches.append(np.concatenate(current))
            current = []
            current_rows = 0
    if current:
        batches.append(np.concatenate(current))
    sizes = [len(slate) for slate in slates]
    return batches, {
        "slates": int(len(slates)),
        "maximum_slate_rows": int(max(sizes, default=0)),
        "oversized_slates": int(sum(size > batch_size for size in sizes)),
        "split_slates": 0,
        "batches": int(len(batches)),
    }


def train_model(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    domain: str,
    output_dir: str | Path,
    config: TrainingConfig = TrainingConfig(),
    label_column: str = "label",
    resume: bool = False,
    device: str | None = None,
    require_cuda: bool = False,
    validation_selector: Callable[
        [torch.nn.Module, FeatureContract], tuple[float, dict[str, Any]]
    ]
    | None = None,
) -> tuple[torch.nn.Module, FeatureContract, dict[str, Any]]:
    if label_column not in train_frame or label_column not in validation_frame:
        raise ValueError("train and validation frames require a label column")
    if train_frame.empty or validation_frame.empty:
        raise ValueError("train and validation frames must not be empty")
    graded_target_columns = tuple(
        column
        for column in ("completion_gain", "watch_time_gain")
        if column in train_frame and column in validation_frame
    )
    target_columns = (
        graded_target_columns
        if len(graded_target_columns) == 2
        else (label_column,)
    )
    uses_graded_targets = target_columns != (label_column,)
    target_semantics = (
        str(train_frame["target_semantics"].dropna().iloc[0])
        if "target_semantics" in train_frame
        and len(train_frame["target_semantics"].dropna())
        else "legacy_binary"
    )
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    roles = infer_feature_roles(train_frame, label_column, config.model_family)
    contract = fit_feature_contract(train_frame, domain, config, roles)
    model = build_model(contract, config)
    validation_prepared = prepare_inference_input(
        contract, validation_frame, config.model_family
    )
    target = _device(device, require_cuda)
    model.to(target)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.epochs, 1)
    )
    amp_dtype = _autocast_dtype(config.model_family, target)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            config.amp
            and target.type == "cuda"
            and amp_dtype == torch.float16
        ),
    )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / "best_model_v2.pth"
    training_data_hash = _frame_data_hash(train_frame)
    validation_data_hash = _frame_data_hash(validation_frame)
    start_epoch = 0
    best_loss = float("inf")
    best_selection_score = float("-inf")
    best_selection_metrics: dict[str, Any] = {}
    initialization = "random"
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=target, weights_only=False)
        config_payload = asdict(config)
        pre_amp_payload = {
            key: value for key, value in config_payload.items() if key != "amp"
        }
        compatible_hashes = {
            _config_hash(config),
            _config_payload_hash(pre_amp_payload),
        }
        if checkpoint.get("model_family") != config.model_family or checkpoint.get("config_hash") not in compatible_hashes:
            raise ValueError("checkpoint family/config does not match requested training config")
        if config.model_family == "hierarchical_fcn" and checkpoint.get(
            "hierarchical_rule_semantics"
        ) != HIERARCHICAL_RULE_SEMANTICS:
            raise ValueError(
                "hierarchical checkpoint uses obsolete rule semantics; retrain from initialization"
            )
        if checkpoint.get("contract_hash") != contract.artifact_hash:
            raise ValueError("checkpoint feature contract does not match current training data")
        if checkpoint.get("training_data_hash") != training_data_hash:
            raise ValueError(
                "checkpoint training data hash does not match; restart without --resume"
            )
        if checkpoint.get("validation_data_hash") != validation_data_hash:
            raise ValueError(
                "checkpoint validation data hash does not match; restart without --resume"
            )
        if tuple(checkpoint.get("target_columns", (label_column,))) != tuple(
            target_columns
        ):
            raise ValueError(
                "checkpoint target semantics do not match; restart without --resume"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["validation_loss"])
        best_selection_score = float(
            checkpoint.get("selection_score", -checkpoint["validation_loss"])
        )
        best_selection_metrics = dict(checkpoint.get("selection_metrics", {}))
        initialization = "resumed_checkpoint"

    def target_values(frame: pd.DataFrame) -> np.ndarray:
        values = np.column_stack(
            [
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(0.0)
                .clip(0.0, 1.0)
                .to_numpy(dtype=np.float32)
                for column in target_columns
            ]
        )
        return values[:, 0] if values.shape[1] == 1 else values

    train_labels = target_values(train_frame)
    val_labels = target_values(validation_frame)
    train_weights = pd.to_numeric(
        train_frame.get("sample_weight", pd.Series(1.0, index=train_frame.index)),
        errors="coerce",
    ).fillna(1.0).to_numpy(dtype=np.float32)
    validation_weights = pd.to_numeric(
        validation_frame.get(
            "sample_weight", pd.Series(1.0, index=validation_frame.index)
        ),
        errors="coerce",
    ).fillna(1.0).to_numpy(dtype=np.float32)
    rng = np.random.default_rng(config.seed)
    patience_count = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    best_epoch = start_epoch - 1
    hierarchical_batching: dict[str, int] = {}
    for epoch in range(start_epoch, config.epochs):
        model.train()
        if config.model_family == "hierarchical_fcn" and "profile_id" in train_frame:
            epoch_batches, hierarchical_batching = _hierarchical_slate_batches(
                train_frame,
                config.batch_size,
                rng,
            )
        else:
            indices = rng.permutation(len(train_frame))
            epoch_batches = [
                indices[start : start + config.batch_size]
                for start in range(0, len(indices), config.batch_size)
            ]
        running = 0.0
        batches = 0
        for batch_indices in epoch_batches:
            batch = train_frame.iloc[batch_indices]
            labels = torch.tensor(train_labels[batch_indices], dtype=torch.float32, device=target)
            sample_weights = torch.tensor(
                train_weights[batch_indices], dtype=torch.float32, device=target
            )
            optimizer.zero_grad()
            with torch.autocast(
                device_type=target.type,
                dtype=amp_dtype,
                enabled=config.amp and target.type == "cuda",
            ):
                outputs = _batch_output(model, contract, batch, config.model_family, target)
            float_outputs = {
                name: value.float() for name, value in outputs.items()
            }
            loss = _loss(
                float_outputs,
                labels,
                config.model_family,
                sample_weights=sample_weights,
                batch_frame=batch,
                config=config,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())
            batches += 1
        val_scores = predict(
            model,
            contract,
            validation_frame,
            config.model_family,
            config.batch_size,
            str(target),
            amp=config.amp,
            prepared=validation_prepared,
        )
        target_matrix = (
            val_labels[:, None] if val_labels.ndim == 1 else val_labels
        )
        validation_target_losses = {
            column: _soft_target_log_loss(
                target_matrix[:, index], val_scores, validation_weights
            )
            for index, column in enumerate(target_columns)
        }
        val_loss = float(np.mean(list(validation_target_losses.values())))
        history.append(
            {
                "epoch": epoch,
                "training_loss": running / max(batches, 1),
                "validation_loss": val_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "batches": batches,
            }
        )
        if validation_selector is not None:
            selection_score, selection_metrics = validation_selector(model, contract)
            selection_source = "hierarchical_validation"
        else:
            selection_score = -val_loss
            if uses_graded_targets:
                selection_metrics = {
                    "graded_objective_loss": val_loss,
                    "target_losses": validation_target_losses,
                }
                selection_source = "graded_objective_loss_fallback"
            else:
                selection_metrics = {"log_loss": val_loss}
                selection_source = "log_loss_fallback"
        scheduler.step()
        if selection_score > best_selection_score:
            best_loss = val_loss
            best_selection_score = float(selection_score)
            best_selection_metrics = dict(selection_metrics)
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "schema_version": (
                        4 if config.model_family == "hierarchical_fcn" else 3
                    ),
                    "model_family": config.model_family,
                    "hierarchical_rule_semantics": (
                        HIERARCHICAL_RULE_SEMANTICS
                        if config.model_family == "hierarchical_fcn"
                        else None
                    ),
                    "config": asdict(config),
                    "config_hash": _config_hash(config),
                    "contract_hash": contract.artifact_hash,
                    "training_data_hash": training_data_hash,
                    "validation_data_hash": validation_data_hash,
                    "epoch": epoch,
                    "validation_loss": val_loss,
                    "selection_score": float(selection_score),
                    "selection_source": selection_source,
                    "selection_metrics": selection_metrics,
                    "target_columns": list(target_columns),
                    "target_semantics": target_semantics,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                checkpoint_path,
            )
        else:
            patience_count += 1
            if patience_count >= config.patience:
                break
    elapsed = time.perf_counter() - started
    if checkpoint_path.exists():
        best = torch.load(checkpoint_path, map_location=target, weights_only=False)
        model.load_state_dict(best["model_state_dict"])
    contract.save(directory / "feature_contract_v2.json")
    contract.save(directory / "feature_contract_v3.json")
    scores = predict(
        model,
        contract,
        validation_frame,
        config.model_family,
        config.batch_size,
        str(target),
        amp=config.amp,
        prepared=validation_prepared,
    )
    metrics: dict[str, Any] = {
        "model_family": config.model_family,
        "initialization": initialization,
        "training_time_seconds": elapsed,
        "training_rows": len(train_frame),
        "validation_rows": len(validation_frame),
        "training_data_hash": training_data_hash,
        "validation_data_hash": validation_data_hash,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "best_selection_score": best_selection_score,
        "best_selection_metrics": best_selection_metrics,
        "checkpoint_selection_requested": config.checkpoint_selection,
        "checkpoint_selection_source": (
            "hierarchical_validation"
            if validation_selector is not None
            else (
                "graded_objective_loss_fallback"
                if uses_graded_targets
                else "log_loss_fallback"
            )
        ),
        "target_columns": list(target_columns),
        "target_semantics": target_semantics,
        "hierarchical_rule_semantics": (
            HIERARCHICAL_RULE_SEMANTICS
            if config.model_family == "hierarchical_fcn"
            else None
        ),
        "classification_metrics_valid": False,
        "device": str(target),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(target)) if target.type == "cuda" else 0,
        "history": history,
        "hierarchical_batching": hierarchical_batching,
        "validation_objective": {
            column: _soft_target_log_loss(
                (val_labels[:, None] if val_labels.ndim == 1 else val_labels)[
                    :, index
                ],
                scores,
                validation_weights,
            )
            for index, column in enumerate(target_columns)
        },
        "gain_distributions": {
            split: {
                column: {
                    "mean": float(
                        pd.to_numeric(frame[column], errors="coerce")
                        .fillna(0.0)
                        .mean()
                    ),
                    "p50": float(
                        pd.to_numeric(frame[column], errors="coerce")
                        .fillna(0.0)
                        .quantile(0.5)
                    ),
                    "p95": float(
                        pd.to_numeric(frame[column], errors="coerce")
                        .fillna(0.0)
                        .quantile(0.95)
                    ),
                }
                for column in target_columns
            }
            for split, frame in (
                ("train", train_frame),
                ("validation", validation_frame),
            )
        },
    }
    return model, contract, metrics


def load_v2_checkpoint(
    checkpoint_path: str | Path,
    contract_path: str | Path,
    device: str | None = None,
) -> tuple[torch.nn.Module, FeatureContract, dict[str, Any]]:
    target = _device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target, weights_only=False)
    checkpoint_schema = int(checkpoint.get("schema_version", 0))
    if checkpoint_schema not in {2, 3, 4}:
        raise ValueError("unsupported checkpoint schema; expected version 2, 3, or 4")
    raw_config = dict(checkpoint["config"])
    if checkpoint.get("config_hash") != _config_payload_hash(raw_config):
        raise ValueError("checkpoint config hash is invalid")
    raw_config["hidden_dims"] = tuple(raw_config.get("hidden_dims", (64, 32)))
    config = TrainingConfig(**raw_config)
    contract = FeatureContract.load(contract_path)
    if checkpoint.get("model_family") != contract.model_family:
        raise ValueError("checkpoint model family does not match feature contract")
    if config.model_family == "hierarchical_fcn" and checkpoint.get(
        "hierarchical_rule_semantics"
    ) != HIERARCHICAL_RULE_SEMANTICS:
        raise ValueError(
            "hierarchical checkpoint uses obsolete rule semantics; retrain from initialization"
        )
    if checkpoint_schema in {3, 4}:
        contract.assert_model_compatible(
            config.model_family,
            _contract_config(config, infer_feature_roles_from_contract(contract)),
        )
    if checkpoint.get("contract_hash") != contract.artifact_hash:
        raise ValueError("checkpoint and feature contract hashes do not match")
    model = build_model(contract, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(target).eval()
    return model, contract, checkpoint


def infer_feature_roles_from_contract(contract: FeatureContract) -> FeatureRoles:
    return FeatureRoles(tuple(contract.numeric_features), tuple(contract.categorical_features), tuple(contract.multivalue_features))
