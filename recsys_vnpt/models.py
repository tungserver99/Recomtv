from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from recsys_vnpt.feature_contract import FeatureContract


@dataclass(frozen=True)
class DenseModelConfig:
    input_dim: int
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.1


class DenseFCN(nn.Module):
    """Leakage-safe dense baseline used by the shared v2 training pipeline."""

    def __init__(self, config: DenseModelConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend(
                [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(config.dropout)]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        logit = self.network(features.float()).squeeze(-1)
        return {"y_pred": torch.sigmoid(logit)}


@dataclass(frozen=True)
class EmbeddingModelConfig:
    numeric_feature_names: tuple[str, ...]
    categorical_vocab_sizes: dict[str, int]
    multivalue_vocab_sizes: dict[str, int]
    embedding_dim: int = 8
    numeric_projection_dim: int = 8
    hidden_dims: tuple[int, ...] = (64, 32)
    deep_cross_layers: int = 2
    shallow_cross_layers: int = 4
    dropout: float = 0.1
    deep_dropout: float | None = None
    shallow_dropout: float | None = None
    num_heads: int = 1
    layer_norm: bool = True
    batch_norm: bool = True

    @classmethod
    def from_contract(
        cls,
        contract: FeatureContract,
        embedding_dim: int = 8,
        numeric_projection_dim: int = 8,
        hidden_dims: tuple[int, ...] = (64, 32),
        deep_cross_layers: int = 2,
        shallow_cross_layers: int = 3,
        dropout: float = 0.1,
        deep_dropout: float | None = None,
        shallow_dropout: float | None = None,
        num_heads: int = 1,
        layer_norm: bool = True,
        batch_norm: bool = True,
    ) -> "EmbeddingModelConfig":
        return cls(
            numeric_feature_names=tuple(contract.numeric_features),
            categorical_vocab_sizes={
                name: len(feature.vocabulary)
                for name, feature in contract.categorical_features.items()
            },
            multivalue_vocab_sizes={
                name: len(feature.vocabulary)
                for name, feature in contract.multivalue_features.items()
            },
            embedding_dim=embedding_dim,
            numeric_projection_dim=numeric_projection_dim,
            hidden_dims=tuple(hidden_dims),
            deep_cross_layers=deep_cross_layers,
            shallow_cross_layers=shallow_cross_layers,
            dropout=dropout,
            deep_dropout=deep_dropout,
            shallow_dropout=shallow_dropout,
            num_heads=num_heads,
            layer_norm=layer_norm,
            batch_norm=batch_norm,
        )

    @property
    def encoded_dim(self) -> int:
        numeric_dim = self.numeric_projection_dim if self.numeric_feature_names else 0
        categorical_dim = len(self.categorical_vocab_sizes) * self.embedding_dim
        multivalue_dim = len(self.multivalue_vocab_sizes) * self.embedding_dim
        return numeric_dim + categorical_dim + multivalue_dim


class EmbeddingFeatureEncoder(nn.Module):
    def __init__(self, config: EmbeddingModelConfig) -> None:
        super().__init__()
        self.config = config
        self.numeric_projection = (
            nn.Linear(len(config.numeric_feature_names), config.numeric_projection_dim)
            if config.numeric_feature_names
            else None
        )
        self.categorical_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_size, config.embedding_dim)
                for name, vocab_size in config.categorical_vocab_sizes.items()
            }
        )
        self.multivalue_embeddings = nn.ModuleDict(
            {
                name: nn.EmbeddingBag(vocab_size, config.embedding_dim, mode="mean")
                for name, vocab_size in config.multivalue_vocab_sizes.items()
            }
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        numeric = batch["numeric"].float()
        if self.numeric_projection is not None:
            outputs.append(F.relu(self.numeric_projection(numeric)))

        categorical = batch["categorical"].long()
        categorical_names = batch.get("categorical_feature_names", list(self.categorical_embeddings))
        for index, name in enumerate(categorical_names):
            outputs.append(self.categorical_embeddings[name](categorical[:, index]))

        multivalue = batch.get("multivalue", {})
        for name, embedding in self.multivalue_embeddings.items():
            values = multivalue[name]["values"].long()
            offsets = multivalue[name]["offsets"].long()
            outputs.append(embedding(values, offsets))

        if not outputs:
            return numeric.new_zeros((numeric.shape[0], 0))
        return torch.cat(outputs, dim=1)


class EmbeddingFCN(nn.Module):
    """Legacy embedding plus MLP baseline (formerly named embedding_fcn)."""
    def __init__(self, config: EmbeddingModelConfig) -> None:
        super().__init__()
        self.encoder = EmbeddingFeatureEncoder(config)
        layers: list[nn.Module] = []
        input_dim = config.encoded_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(config.dropout)])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        encoded = self.encoder(batch)
        logit = self.network(encoded).squeeze(-1)
        return {"y_pred": torch.sigmoid(logit)}


class _CrossTower(nn.Module):
    def __init__(self, input_dim: int, num_layers: int, exponential: bool, dropout: float) -> None:
        super().__init__()
        self.exponential = exponential
        self.input_dim = input_dim
        half_dim = max(1, (input_dim + 1) // 2)
        self.projections = nn.ModuleList(
            nn.Linear(input_dim, half_dim, bias=False) for _ in range(num_layers)
        )
        self.gates = nn.ModuleList(nn.Linear(half_dim, half_dim) for _ in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(half_dim) for _ in range(num_layers))
        self.biases = nn.ParameterList(nn.Parameter(torch.zeros(input_dim)) for _ in range(num_layers))
        self.dropouts = nn.ModuleList(nn.Dropout(dropout) for _ in range(num_layers))
        self.output = nn.Linear(input_dim, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        for projection, gate, norm, bias, dropout in zip(
            self.projections, self.gates, self.norms, self.biases, self.dropouts
        ):
            projected = projection(x)
            mask = F.relu(norm(projected))
            gated = projected * torch.sigmoid(gate(projected))
            interaction = torch.cat([gated, gated * mask], dim=-1)
            if interaction.shape[-1] < self.input_dim:
                interaction = F.pad(interaction, (0, self.input_dim - interaction.shape[-1]))
            interaction = interaction[:, : self.input_dim]
            if self.exponential:
                x = x * (interaction + bias) + x
            else:
                x = x0 * (interaction + bias) + x
            x = F.normalize(torch.clamp(x, min=-10.0, max=10.0), p=2, dim=-1)
            x = dropout(x)
        return self.output(x).squeeze(-1)


class EmbeddingDCNv3(nn.Module):
    """Legacy DCNv3 approximation kept for checkpoint compatibility."""

    def __init__(self, config: EmbeddingModelConfig) -> None:
        super().__init__()
        self.encoder = EmbeddingFeatureEncoder(config)
        self.deep_cross = _CrossTower(
            config.encoded_dim,
            num_layers=config.deep_cross_layers,
            exponential=True,
            dropout=config.dropout,
        )
        self.shallow_cross = _CrossTower(
            config.encoded_dim,
            num_layers=config.shallow_cross_layers,
            exponential=False,
            dropout=config.dropout,
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        encoded = self.encoder(batch)
        dlogit = self.deep_cross(encoded)
        slogit = self.shallow_cross(encoded)
        logit = 0.5 * (dlogit + slogit)
        return {
            "y_pred": torch.sigmoid(logit),
            "y_d": torch.sigmoid(dlogit),
            "y_s": torch.sigmoid(slogit),
        }


# Clear names for the two historical implementations. The old class names remain
# importable because schema-v2 checkpoints and downstream scripts use them.
EmbeddingMLPBaseline = EmbeddingFCN
LegacyEmbeddingDCNv3 = EmbeddingDCNv3


class FieldAlignedFeatureEncoder(nn.Module):
    """Encode each numeric, categorical and multi-value input as one field."""

    def __init__(self, config: EmbeddingModelConfig) -> None:
        super().__init__()
        self.config = config
        table_dim = config.embedding_dim * config.num_heads
        self.numeric_vectors = nn.Parameter(
            torch.empty(len(config.numeric_feature_names), table_dim)
        )
        self.numeric_bias = nn.Parameter(
            torch.zeros(len(config.numeric_feature_names), table_dim)
        )
        self.categorical_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(vocab_size, table_dim)
                for name, vocab_size in config.categorical_vocab_sizes.items()
            }
        )
        self.multivalue_embeddings = nn.ModuleDict(
            {
                name: nn.EmbeddingBag(vocab_size, table_dim, mode="mean")
                for name, vocab_size in config.multivalue_vocab_sizes.items()
            }
        )
        if self.numeric_vectors.numel():
            nn.init.xavier_uniform_(self.numeric_vectors)
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
                nn.init.xavier_uniform_(module.weight)

    @property
    def num_fields(self) -> int:
        return (
            len(self.config.numeric_feature_names)
            + len(self.config.categorical_vocab_sizes)
            + len(self.config.multivalue_vocab_sizes)
        )

    @property
    def head_input_dim(self) -> int:
        return self.num_fields * self.config.embedding_dim

    def forward_fields(self, batch: dict[str, Any]) -> torch.Tensor:
        fields: list[torch.Tensor] = []
        numeric = batch["numeric"].float()
        if self.config.numeric_feature_names:
            fields.extend(
                (
                    numeric[:, index : index + 1] * self.numeric_vectors[index]
                    + self.numeric_bias[index]
                )
                for index in range(len(self.config.numeric_feature_names))
            )

        categorical = batch["categorical"].long()
        categorical_names = batch.get(
            "categorical_feature_names", list(self.categorical_embeddings)
        )
        for index, name in enumerate(categorical_names):
            fields.append(self.categorical_embeddings[name](categorical[:, index]))

        multivalue = batch.get("multivalue", {})
        for name, embedding in self.multivalue_embeddings.items():
            fields.append(
                embedding(
                    multivalue[name]["values"].long(),
                    multivalue[name]["offsets"].long(),
                )
            )
        if not fields:
            raise ValueError("official FCN requires at least one encoded feature field")
        return torch.stack(fields, dim=1)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        feature_emb = self.forward_fields(batch)
        # Match the author's MultiHeadFeatureEmbedding ordering.
        heads = torch.tensor_split(feature_emb, self.config.num_heads, dim=-1)
        multihead = torch.stack(heads, dim=1)
        first, second = torch.tensor_split(multihead, 2, dim=-1)
        return torch.cat(
            [first.flatten(start_dim=2), second.flatten(start_dim=2)], dim=-1
        )


class SelfMask(nn.Module):
    def __init__(self, half_dim: int, layer_norm: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(half_dim) if layer_norm else nn.Identity()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        mask = F.relu(self.norm(hidden))
        return torch.cat([hidden, hidden * mask], dim=-1)


class OfficialCrossNetwork(nn.Module):
    """Source-faithful ECN/LCN recurrence from FCN, formerly DCNv3."""

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        exponential: bool,
        dropout: float,
        num_heads: int,
        layer_norm: bool,
        batch_norm: bool,
    ) -> None:
        super().__init__()
        if input_dim < 2 or input_dim % 2:
            raise ValueError("official FCN head input dimension must be positive and even")
        self.exponential = exponential
        half_dim = input_dim // 2
        self.projections = nn.ModuleList(
            nn.Linear(input_dim, half_dim, bias=False) for _ in range(num_layers)
        )
        self.masks = nn.ModuleList(
            SelfMask(half_dim, layer_norm=layer_norm) for _ in range(num_layers)
        )
        self.batch_norms = nn.ModuleList(
            nn.BatchNorm1d(num_heads) if batch_norm else nn.Identity()
            for _ in range(num_layers)
        )
        self.dropouts = nn.ModuleList(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            for _ in range(num_layers)
        )
        self.biases = nn.ParameterList(
            nn.Parameter(torch.empty(input_dim)) for _ in range(num_layers)
        )
        for bias in self.biases:
            nn.init.uniform_(bias)
        self.output = nn.Linear(input_dim, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        for projection, batch_norm, masker, bias, dropout in zip(
            self.projections,
            self.batch_norms,
            self.masks,
            self.biases,
            self.dropouts,
        ):
            interaction = masker(batch_norm(projection(x)))
            if self.exponential:
                x = x * (interaction + bias) + x
            else:
                x = x0 * (interaction + bias) + x
            x = dropout(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.forward_features(x))


class OfficialFCN(nn.Module):
    """Fusing Exponential and Linear Cross Network."""

    def __init__(self, config: EmbeddingModelConfig, include_lcn: bool = True) -> None:
        super().__init__()
        self.encoder = FieldAlignedFeatureEncoder(config)
        input_dim = self.encoder.head_input_dim
        deep_dropout = config.dropout if config.deep_dropout is None else config.deep_dropout
        shallow_dropout = (
            config.dropout if config.shallow_dropout is None else config.shallow_dropout
        )
        self.ecn = OfficialCrossNetwork(
            input_dim,
            config.deep_cross_layers,
            exponential=True,
            dropout=deep_dropout,
            num_heads=config.num_heads,
            layer_norm=config.layer_norm,
            batch_norm=config.batch_norm,
        )
        self.lcn = (
            OfficialCrossNetwork(
                input_dim,
                config.shallow_cross_layers,
                exponential=False,
                dropout=shallow_dropout,
                num_heads=config.num_heads,
                layer_norm=config.layer_norm,
                batch_norm=config.batch_norm,
            )
            if include_lcn
            else None
        )

    def logits(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(batch)
        dlogit = self.ecn(encoded).mean(dim=1).squeeze(-1)
        slogit = (
            self.lcn(encoded).mean(dim=1).squeeze(-1)
            if self.lcn is not None
            else dlogit
        )
        return dlogit, slogit

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        dlogit, slogit = self.logits(batch)
        logit = (dlogit + slogit) * 0.5
        return {
            "y_pred": torch.sigmoid(logit),
            "y_d": torch.sigmoid(dlogit),
            "y_s": torch.sigmoid(slogit),
        }


class OfficialECN(OfficialFCN):
    def __init__(self, config: EmbeddingModelConfig) -> None:
        super().__init__(config, include_lcn=False)


class HierarchicalFCN(OfficialFCN):
    """FCN backbone with separate rule and within-rule content predictions."""

    def __init__(self, config: EmbeddingModelConfig) -> None:
        super().__init__(config, include_lcn=True)
        self.rule_output = nn.Linear(self.encoder.head_input_dim * 2, 1)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        encoded = self.encoder(batch)
        deep_features = self.ecn.forward_features(encoded)
        shallow_features = self.lcn.forward_features(encoded)
        dlogit = self.ecn.output(deep_features).mean(dim=1).squeeze(-1)
        slogit = self.lcn.output(shallow_features).mean(dim=1).squeeze(-1)
        content_logit = (dlogit + slogit) * 0.5
        joint = torch.cat([deep_features, shallow_features], dim=-1)
        rule_logit = self.rule_output(joint).mean(dim=1).squeeze(-1)
        return {
            "y_pred": torch.sigmoid(content_logit),
            "content_score": torch.sigmoid(content_logit),
            "rule_score": torch.sigmoid(rule_logit),
            "y_d": torch.sigmoid(dlogit),
            "y_s": torch.sigmoid(slogit),
        }
