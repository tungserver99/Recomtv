"""Every row ordering and content ordering a page can be built with.

A page orders two things: its rows (rules) and the contents inside each row.
Both are built here from shared parts -- the hand-built heuristic, the two GBDT
scorers, the FCN models and a segment heuristic for cold profiles -- so the
offline evaluation, the retraining gate and serving build the same page from
the same configuration. ``COMBINATIONS`` names every combination measured on
2026-11-09, so one config key selects one of them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


RULE_ORDERINGS = ("count", "learned_rrf", "fcn", "segment")
CONTENT_ORDERINGS = (
    "hand_sum",
    "learned_rrf",
    "fcn",
    "fcn_rrf",
    "ensemble",
    "routed",
    "segment",
)
CONTENT_RANK_SCOPES = ("rule", "profile")
FCN_FAMILIES = ("fcn_official", "hierarchical_fcn")
ROUTED_FALLBACKS = ("learned_rrf", "segment")
# Only this family has a rule head, so it is the one that can order rows.
RULE_FCN_FAMILY = "hierarchical_fcn"
RRF_CONSTANT = 60.0
FCN_CONTENT_ORDERINGS = {"fcn", "fcn_rrf", "ensemble", "routed"}

_XGBOOST = {"gbdt_backend": "xgboost"}
# Name -> settings. H heuristic, G GBDT on XGBoost, Gs GBDT on scikit-learn,
# F fcn_official, Fh hierarchical_fcn, Hseg segment heuristic; suffix p ranks
# inside the profile instead of inside each (profile, rule) block.
COMBINATIONS: dict[str, dict[str, str]] = {
    "HxH": {"rule_ordering": "count", "content_ordering": "hand_sum"},
    "HxG": {"rule_ordering": "count", "content_ordering": "learned_rrf", "content_rank_scope": "rule", **_XGBOOST},
    "HxF": {"rule_ordering": "count", "content_ordering": "fcn", "content_fcn_family": "fcn_official"},
    "GxH": {"rule_ordering": "learned_rrf", "content_ordering": "hand_sum", **_XGBOOST},
    "GxG": {"rule_ordering": "learned_rrf", "content_ordering": "learned_rrf", "content_rank_scope": "rule", **_XGBOOST},
    "GxF": {"rule_ordering": "learned_rrf", "content_ordering": "fcn", "content_fcn_family": "fcn_official", **_XGBOOST},
    "FhxH": {"rule_ordering": "fcn", "content_ordering": "hand_sum"},
    "FhxG": {"rule_ordering": "fcn", "content_ordering": "learned_rrf", "content_rank_scope": "rule", **_XGBOOST},
    "FhxF": {"rule_ordering": "fcn", "content_ordering": "fcn", "content_fcn_family": "fcn_official"},
    "GsxGs": {"rule_ordering": "learned_rrf", "content_ordering": "learned_rrf", "content_rank_scope": "rule", "gbdt_backend": "sklearn"},
    "FhxFh": {"rule_ordering": "fcn", "content_ordering": "fcn", "content_fcn_family": "hierarchical_fcn"},
    "GxGp": {"rule_ordering": "learned_rrf", "content_ordering": "learned_rrf", "content_rank_scope": "profile", **_XGBOOST},
    "GxFp": {"rule_ordering": "learned_rrf", "content_ordering": "fcn_rrf", "content_rank_scope": "profile", "content_fcn_family": "fcn_official", **_XGBOOST},
    "GxFrrf": {"rule_ordering": "learned_rrf", "content_ordering": "fcn_rrf", "content_rank_scope": "rule", "content_fcn_family": "fcn_official", **_XGBOOST},
    "GxEp": {"rule_ordering": "learned_rrf", "content_ordering": "ensemble", "content_rank_scope": "profile", "content_fcn_family": "fcn_official", **_XGBOOST},
    "GxRp": {"rule_ordering": "learned_rrf", "content_ordering": "routed", "content_rank_scope": "profile", "content_fcn_family": "fcn_official", "routed_fallback": "learned_rrf", **_XGBOOST},
    "HxFp": {"rule_ordering": "count", "content_ordering": "fcn_rrf", "content_rank_scope": "profile", "content_fcn_family": "fcn_official"},
    "FhxFp": {"rule_ordering": "fcn", "content_ordering": "fcn_rrf", "content_rank_scope": "profile", "content_fcn_family": "fcn_official"},
    "HsegxHseg": {"rule_ordering": "segment", "content_ordering": "segment"},
    "HsegxRseg": {"rule_ordering": "segment", "content_ordering": "routed", "content_rank_scope": "profile", "content_fcn_family": "fcn_official", "routed_fallback": "segment"},
    "HsegxFp": {"rule_ordering": "segment", "content_ordering": "fcn_rrf", "content_rank_scope": "profile", "content_fcn_family": "fcn_official"},
    "Gx_cpuxGx_cpu": {"rule_ordering": "learned_rrf", "content_ordering": "learned_rrf", "content_rank_scope": "rule", "gbdt_backend": "xgboost", "gbdt_device": "cpu"},
}


def expand_combination(name: str) -> dict[str, str]:
    if name not in COMBINATIONS:
        raise ValueError(f"unknown combination {name!r}; known: {sorted(COMBINATIONS)}")
    return dict(COMBINATIONS[name])


@dataclass(frozen=True)
class OrderingConfig:
    rule_ordering: str = "count"
    content_ordering: str = "hand_sum"
    content_rank_scope: str = "rule"
    content_fcn_family: str = "fcn_official"
    routed_fallback: str = "learned_rrf"
    rule_scorer_fusion_weight: float = 1.25
    content_scorer_fusion_weight: float = 5.0

    def __post_init__(self) -> None:
        for name, allowed in (
            ("rule_ordering", RULE_ORDERINGS),
            ("content_ordering", CONTENT_ORDERINGS),
            ("content_rank_scope", CONTENT_RANK_SCOPES),
            ("content_fcn_family", FCN_FAMILIES),
            ("routed_fallback", ROUTED_FALLBACKS),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if self.rule_scorer_fusion_weight <= 0 or self.content_scorer_fusion_weight <= 0:
            raise ValueError("fusion weights must be positive")

    @property
    def rule_gbdt(self) -> bool:
        return self.rule_ordering == "learned_rrf"

    @property
    def content_gbdt(self) -> bool:
        return self.content_ordering in {"learned_rrf", "ensemble"} or (
            self.content_ordering == "routed" and self.routed_fallback == "learned_rrf"
        )

    @property
    def content_fcn(self) -> bool:
        return self.content_ordering in FCN_CONTENT_ORDERINGS

    @property
    def segment(self) -> bool:
        return "segment" in (self.rule_ordering, self.content_ordering) or (
            self.content_ordering == "routed" and self.routed_fallback == "segment"
        )

    @property
    def fcn_families(self) -> tuple[str, ...]:
        families = {RULE_FCN_FAMILY} if self.rule_ordering == "fcn" else set()
        if self.content_fcn:
            families.add(self.content_fcn_family)
        return tuple(sorted(families))

    @property
    def needs_models(self) -> bool:
        return self.rule_gbdt or self.content_gbdt or bool(self.fcn_families)


# --- CLI --------------------------------------------------------------------------


def add_ordering_arguments(parser: argparse.ArgumentParser) -> None:
    """Row and content ordering flags, shared by every entrypoint that builds a
    page. Defaults are the production combination, GxGp."""
    parser.add_argument("--combination", choices=sorted(COMBINATIONS),
                        help="preset that overrides the ordering flags below")
    parser.add_argument("--rule-ordering", choices=RULE_ORDERINGS, default="learned_rrf")
    parser.add_argument("--rule-scorer-fusion-weight", type=float, default=1.25)
    parser.add_argument("--content-ordering", choices=CONTENT_ORDERINGS, default="learned_rrf")
    parser.add_argument("--content-scorer-fusion-weight", type=float, default=5.0)
    parser.add_argument("--content-rank-scope", choices=CONTENT_RANK_SCOPES, default="profile")
    parser.add_argument("--content-fcn-family", choices=FCN_FAMILIES, default="fcn_official")
    parser.add_argument("--routed-fallback", choices=ROUTED_FALLBACKS, default="learned_rrf")
    parser.add_argument("--fcn-model-dir",
                        help="directory holding <family>/best_model_v2.pth; default: the trained model directory")


def apply_combination(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "combination", None):
        for name, value in expand_combination(args.combination).items():
            setattr(args, name, value)
    return args


# --- reciprocal-rank fusion --------------------------------------------------------


def reciprocal_rank(
    frame: pd.DataFrame, values: Any, scope: str = "rule", constant: float = RRF_CONSTANT
) -> np.ndarray:
    """1 / (constant + rank) of ``values``, descending, inside each (profile,
    rule) block or inside the whole profile. Ties keep row order."""
    keys = (
        [frame["profile_id"], frame["candidate_rule_id"]]
        if scope == "rule"
        else [frame["profile_id"]]
    )
    ranks = (
        pd.Series(np.asarray(values, dtype=float), index=frame.index)
        .groupby(keys, sort=False, observed=True)
        .rank(method="first", ascending=False)
    )
    return (1.0 / (constant + ranks)).to_numpy(dtype=np.float64)


def rrf_fuse(
    frame: pd.DataFrame, signals: list[np.ndarray], scope: str, weight: float
) -> np.ndarray:
    """The hand-built personal rank plus ``weight`` times each signal's rank.

    Inside a rule block the rule term of the hand sum is kept as an offset, so
    rows stay comparable to how retrieval scored them. Ranked over the profile,
    a strong rule's content keeps its lead over a weak rule's head, which is
    what the competitive slot round of the page compares.
    """
    rule = frame["rule_score"].astype(float).to_numpy()
    personal = frame["retrieval_score"].astype(float).to_numpy() - rule
    fused = rule.copy() if scope == "rule" else np.zeros(len(frame))
    fused = fused + reciprocal_rank(frame, personal, scope)
    for signal in signals:
        fused = fused + weight * reciprocal_rank(frame, signal, scope)
    return fused


# --- FCN -------------------------------------------------------------------------------


@dataclass(frozen=True)
class FcnModel:
    family: str
    model: Any
    contract: Any

    @property
    def profiles(self) -> set[str]:
        """Profiles with a learned embedding: the ones routing sends to FCN."""
        return set(map(str, self.contract.categorical_features["profile_id"].vocabulary))


def load_fcn(directory: str | Path, family: str) -> FcnModel:
    from recsys_vnpt.model_runner import load_v2_checkpoint

    directory = Path(directory)
    model, contract, _ = load_v2_checkpoint(
        directory / "best_model_v2.pth", directory / "feature_contract_v2.json"
    )
    if contract.model_family != family:
        raise ValueError(f"{directory} holds {contract.model_family}, expected {family}")
    return FcnModel(family, model, contract)


def load_fcn_models(root: str | Path, families: tuple[str, ...]) -> dict[str, FcnModel]:
    """``root/<family>/`` for each family, the layout `pipeline_cli train` writes."""
    return {family: load_fcn(Path(root) / family, family) for family in families}


def fit_fcn(
    canonical: Path,
    domain: str,
    family: str,
    output_dir: str | Path,
    epochs: int = 2,
    schedule_epochs: int = 30,
    seed: int = 42,
    batch_size: int = 4096,
) -> dict[str, Any]:
    """Fit on train_1..3 plus validation, the windows the GBDT scorers serve
    from, for a fixed number of epochs and keep the last one.

    Nothing is held out to pick an epoch, so the count is fixed: 2 epochs of a
    30-epoch cosine schedule is what checkpoint selection picked when the same
    model was trained on train_1..3 and selected on validation.
    """
    from recsys_vnpt.content_scorer import _normal_membership
    from recsys_vnpt.model_runner import TrainingConfig, train_model

    frames = []
    for name in ("train_1", "train_2", "train_3", "validation"):
        frame = pd.read_parquet(canonical / "temporal" / f"{name}.parquet")
        frame["temporal_window"] = name
        frames.append(frame)
    train, validation = pd.concat(frames, ignore_index=True), frames[-1]
    if family == RULE_FCN_FAMILY:
        membership = _normal_membership(canonical)
        train = train.drop(columns=["rule_id"], errors="ignore").merge(membership, on="content_id", how="inner")
        validation = validation.drop(columns=["rule_id"], errors="ignore").merge(membership, on="content_id", how="inner")
    config = TrainingConfig(
        model_family=family,
        epochs=epochs,
        lr_schedule_epochs=schedule_epochs,
        batch_size=batch_size,
        seed=seed,
    )
    calls = iter(range(1 << 30))
    _, _, metrics = train_model(
        train,
        validation,
        domain,
        output_dir,
        config,
        # Every epoch "improves", so the last one is the one kept.
        validation_selector=lambda model, contract: (float(next(calls)), {}),
    )
    return metrics


@dataclass(frozen=True)
class FcnFeatureScope:
    """The FCN's pair features at one cutoff, built exactly as the neural
    evaluation builds them, shared by every chunk of candidates."""

    cutoff: pd.Timestamp
    profile_features: pd.DataFrame
    items: pd.DataFrame
    users: pd.DataFrame
    usernames: pd.DataFrame
    sequence: pd.DataFrame | None


def fcn_feature_scope(
    canonical: Path, cutoff: pd.Timestamp, interactions: pd.DataFrame
) -> FcnFeatureScope:
    from recsys_vnpt.canonical import select_content_metadata_as_of
    from recsys_vnpt.engagement import historical_sequence_features
    from recsys_vnpt.temporal import historical_profile_features

    versions = canonical / "content_versions.parquet"
    if versions.exists():
        # Tag text is not a model feature; see _serving_candidates_for_evaluation.
        items = select_content_metadata_as_of(pd.read_parquet(versions), cutoff).drop(
            columns=["tag_names"], errors="ignore"
        )
    else:
        items = pd.read_parquet(canonical / "contents.parquet")
    playback = canonical / "playback_events.parquet"
    usernames = pd.read_parquet(canonical / "profile_mapping.parquet", columns=["profile_id", "username"])
    return FcnFeatureScope(
        cutoff=pd.Timestamp(cutoff),
        profile_features=historical_profile_features(interactions, cutoff),
        items=items,
        users=pd.read_parquet(canonical / "users.parquet"),
        usernames=usernames.astype({"profile_id": str}).drop_duplicates("profile_id"),
        sequence=historical_sequence_features(pd.read_parquet(playback), cutoff) if playback.exists() else None,
    )


def score_fcn(
    fcn: FcnModel,
    scope: FcnFeatureScope,
    pairs: pd.DataFrame,
    outputs: tuple[str, ...] = ("y_pred",),
    chunk_profiles: int = 1000,
    batch_size: int = 4096,
) -> dict[str, np.ndarray]:
    """FCN outputs for every row of ``pairs``, built a chunk of profiles at a
    time so a shard never materialises its whole feature frame."""
    from recsys_vnpt.model_runner import predict, prepare_inference_input
    from recsys_vnpt.temporal import build_pair_features

    scores = {output: np.empty(len(pairs), dtype=np.float32) for output in outputs}
    codes, profiles = pd.factorize(pairs["profile_id"].astype(str))
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(0, len(profiles) + chunk_profiles, chunk_profiles))
    for low, high in zip(bounds[:-1], bounds[1:]):
        if low == high:
            continue
        positions = order[low:high]
        frame = pairs.iloc[positions].drop(columns=["username"], errors="ignore").copy()
        frame["profile_id"] = frame["profile_id"].astype(str)
        frame = frame.merge(scope.usernames, on="profile_id", how="left")
        features = build_pair_features(
            frame, None, scope.items, scope.users, scope.cutoff,
            profile_features=scope.profile_features,
        )
        if scope.sequence is not None:
            features = features.merge(scope.sequence, on="profile_id", how="left")
            for column in ("historical_session_count", "historical_playback_count", "historical_sequence_depth"):
                features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
        if len(features) != len(positions):
            raise ValueError("FCN feature joins changed the number of candidate rows")
        prepared = prepare_inference_input(fcn.contract, features, fcn.family)
        for output in outputs:
            scores[output][positions] = predict(
                fcn.model, fcn.contract, features, fcn.family,
                batch_size=batch_size, output_key=output, prepared=prepared,
            )
    return scores


# --- segment heuristic for cold profiles ------------------------------------------------

SEGMENT_ATTRIBUTES = ("package_code", "province", "age_band")


@dataclass(frozen=True)
class SegmentScope:
    """Popularity inside each subscriber segment (package, province, age band),
    for profiles with no VOD history before the cutoff."""

    warm: frozenset[str]
    attributes: pd.DataFrame
    contents: dict[str, pd.DataFrame]
    rules: dict[str, pd.Series]
    rule_global: pd.Series
    global_norm: pd.Series


def segment_scope(
    canonical: Path,
    cutoff: pd.Timestamp,
    interactions_loader: Callable[[set[str], list[str]], pd.DataFrame],
    popularity: pd.Series,
) -> SegmentScope:
    from recsys_vnpt.rule_assembly import SPECIAL_RULES

    catalog = set(pd.read_parquet(canonical / "contents.parquet", columns=["content_id"])["content_id"].astype(str))
    events = interactions_loader(catalog, ["profile_id", "content_id", "create_date"])
    events = events[pd.to_datetime(events["create_date"], errors="coerce") <= pd.Timestamp(cutoff)]
    events = events[["profile_id", "content_id"]].astype(str)
    users = pd.read_parquet(canonical / "users.parquet", columns=["username", "province", "package_code", "age"])
    age = pd.to_numeric(users["age"], errors="coerce")
    users["age_band"] = (
        pd.cut(age, [-1, 24, 34, 44, 54, 200], labels=["<25", "25-34", "35-44", "45-54", "55+"])
        .astype("string")
        .fillna("unknown")
    )
    mapping = pd.read_parquet(canonical / "profile_mapping.parquet", columns=["profile_id", "username"]).astype(str)
    attributes = mapping.drop_duplicates("profile_id").merge(
        users.astype({"username": str}).drop_duplicates("username"), on="username", how="left"
    )
    attributes = attributes[["profile_id", *SEGMENT_ATTRIBUTES]].astype("string").fillna("unknown")
    events = events.merge(attributes, on="profile_id", how="left").fillna("unknown")
    counts = {
        attribute: events.groupby([attribute, "content_id"]).size().rename("events").reset_index()
        for attribute in SEGMENT_ATTRIBUTES
    }
    global_pop = pd.Series(popularity, dtype=float)
    global_pop.index = global_pop.index.astype(str)
    membership = pd.read_parquet(canonical / "content_rule_membership.parquet")
    normal = (
        membership[~membership["rule_id"].astype(str).isin(SPECIAL_RULES)][["content_id", "rule_id"]]
        .astype(str)
        .drop_duplicates()
    )
    rule_global = normal.assign(pop=normal["content_id"].map(global_pop).fillna(0.0)).groupby("rule_id")["pop"].sum()
    contents, rules = {}, {}
    for attribute, table in counts.items():
        maxima = table.groupby(attribute)["events"].transform("max").clip(lower=1)
        contents[attribute] = table.assign(norm=table["events"] / maxima)[[attribute, "content_id", "norm"]]
        per_rule = (
            normal.merge(table, on="content_id")
            .groupby([attribute, "rule_id"])["events"].sum().rename("events").reset_index()
        )
        per_rule["norm"] = per_rule["events"] / per_rule.groupby(attribute)["events"].transform("max").clip(lower=1)
        rules[attribute] = per_rule.set_index([attribute, "rule_id"])["norm"]
    return SegmentScope(
        warm=frozenset(events["profile_id"]),
        attributes=attributes.set_index("profile_id"),
        contents=contents,
        rules=rules,
        rule_global=rule_global / max(float(rule_global.max()), 1e-12) if len(rule_global) else rule_global,
        global_norm=global_pop / max(float(global_pop.max()), 1.0) if len(global_pop) else global_pop,
    )


def segment_content_scores(scope: SegmentScope, pairs: pd.DataFrame) -> np.ndarray:
    """Cold profiles: the rule term plus half global, half segment popularity.
    Profiles with VOD history keep the hand sum."""
    profiles = pairs["profile_id"].astype(str)
    frame = pd.DataFrame({"profile_id": profiles.to_numpy(), "content_id": pairs["content_id"].astype(str).to_numpy()})
    segment = np.zeros(len(pairs))
    for attribute, table in scope.contents.items():
        frame[attribute] = frame["profile_id"].map(scope.attributes[attribute]).fillna("unknown")
        segment += frame.merge(table, on=[attribute, "content_id"], how="left")["norm"].fillna(0.0).to_numpy()
    segment /= len(scope.contents)
    blended = 0.5 * frame["content_id"].map(scope.global_norm).fillna(0.0).to_numpy() + 0.5 * segment
    cold = ~profiles.isin(scope.warm).to_numpy()
    return np.where(
        cold,
        pairs["rule_score"].astype(float).to_numpy() + blended,
        pairs["retrieval_score"].astype(float).to_numpy(),
    )


def segment_rule_scores(scope: SegmentScope, rule_scores: pd.DataFrame) -> pd.DataFrame:
    """Cold profiles' rows by half global, half segment rule popularity,
    scaled to 1 per profile; warm profiles keep the hand-built rows."""
    scored = rule_scores.copy()
    scored["profile_id"] = scored["profile_id"].astype(str)
    cold_mask = ~scored["profile_id"].isin(scope.warm)
    cold = scored[cold_mask].copy()
    segment = np.zeros(len(cold))
    for attribute, series in scope.rules.items():
        keys = pd.MultiIndex.from_arrays(
            [cold["profile_id"].map(scope.attributes[attribute]).fillna("unknown"), cold["rule_id"].astype(str)]
        )
        segment += series.reindex(keys).fillna(0.0).to_numpy()
    cold["rule_score"] = (
        0.5 * cold["rule_id"].astype(str).map(scope.rule_global).fillna(0.0).to_numpy()
        + 0.5 * segment / len(scope.rules)
    )
    cold["rule_score"] = cold["rule_score"] / cold.groupby("profile_id")["rule_score"].transform("max").clip(lower=1e-12)
    scored.loc[cold_mask, "rule_score"] = cold["rule_score"].to_numpy()
    return scored


# --- assembling the parts and applying an ordering -------------------------------------


@dataclass(frozen=True)
class OrderingParts:
    """Everything an ordering reads at one cutoff besides the candidates."""

    rule_model: Any = None
    rule_scope: Any = None
    content_model: Any = None
    content_scope: Any = None
    fcn: dict[str, FcnModel] = field(default_factory=dict)
    fcn_scope: FcnFeatureScope | None = None
    segment: SegmentScope | None = None


def ordering_parts(
    config: OrderingConfig,
    canonical: Path,
    cutoff: pd.Timestamp,
    interactions_loader: Callable[[set[str], list[str]], pd.DataFrame],
    interactions: pd.DataFrame,
    popularity: pd.Series,
    rule_model: Any = None,
    content_model: Any = None,
    fcn: dict[str, FcnModel] | None = None,
) -> OrderingParts:
    """The serving scopes ``config`` needs at ``cutoff``. ``interactions`` are
    the ones retrieval reads; they feed the FCN's profile features."""
    from recsys_vnpt.content_scorer import content_serving_scope
    from recsys_vnpt.rule_scorer import rule_serving_scope

    fcn = fcn or {}
    missing = (
        (["rule scorer"] if config.rule_gbdt and rule_model is None else [])
        + (["content scorer"] if config.content_gbdt and content_model is None else [])
        + [family for family in config.fcn_families if family not in fcn]
    )
    if missing:
        raise ValueError(f"the configured ordering needs {missing}")
    cutoff = pd.Timestamp(cutoff)
    return OrderingParts(
        rule_model=rule_model if config.rule_gbdt else None,
        rule_scope=rule_serving_scope(canonical, cutoff, rule_model, interactions_loader) if config.rule_gbdt else None,
        content_model=content_model if config.content_gbdt else None,
        content_scope=(
            content_serving_scope(canonical, cutoff, content_model, interactions_loader)
            if config.content_gbdt
            else None
        ),
        fcn={family: fcn[family] for family in config.fcn_families},
        fcn_scope=fcn_feature_scope(canonical, cutoff, interactions) if config.fcn_families else None,
        segment=segment_scope(canonical, cutoff, interactions_loader, popularity) if config.segment else None,
    )


def apply_orderings(
    config: OrderingConfig,
    parts: OrderingParts,
    profiles: pd.Series,
    candidate_pairs: pd.DataFrame,
    rule_scores: pd.DataFrame,
    membership: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Order rows into ``rule_scores`` and contents into ``retrieval_score``."""
    from recsys_vnpt.content_scorer import apply_content_scorer, content_scorer_signal
    from recsys_vnpt.rule_assembly import aggregate_model_rule_scores
    from recsys_vnpt.rule_scorer import apply_rule_scorer

    metrics: dict[str, Any] = {}
    fcn_scores: dict[str, dict[str, np.ndarray]] = {}
    for family in config.fcn_families:
        outputs = ("y_pred",) if config.content_fcn and family == config.content_fcn_family else ()
        if config.rule_ordering == "fcn" and family == RULE_FCN_FAMILY:
            outputs += ("rule_score",)
        fcn_scores[family] = score_fcn(parts.fcn[family], parts.fcn_scope, candidate_pairs, outputs)

    if config.rule_ordering == "learned_rrf":
        rule_scores, rule_metrics = apply_rule_scorer(
            parts.rule_model, parts.rule_scope, profiles, rule_scores, config.rule_scorer_fusion_weight
        )
        metrics.update(rule_metrics)
    elif config.rule_ordering == "fcn":
        contributions = pd.DataFrame(
            {
                "profile_id": candidate_pairs["profile_id"].astype(str).to_numpy(),
                "content_id": candidate_pairs["content_id"].astype(str).to_numpy(),
                "model_rule_score": fcn_scores[RULE_FCN_FAMILY]["rule_score"],
            }
        )
        rule_scores = pd.concat(
            [rule_scores, aggregate_model_rule_scores(contributions, membership)], ignore_index=True
        ).drop_duplicates(["profile_id", "rule_id"], keep="last")
    elif config.rule_ordering == "segment":
        rule_scores = segment_rule_scores(parts.segment, rule_scores)

    ordering = config.content_ordering
    scope, weight = config.content_rank_scope, config.content_scorer_fusion_weight
    pairs = candidate_pairs.copy()
    pairs["profile_id"] = pairs["profile_id"].astype(str)
    fcn_signal = fcn_scores[config.content_fcn_family]["y_pred"] if config.content_fcn else None
    if ordering == "learned_rrf":
        pairs, content_metrics = apply_content_scorer(
            parts.content_model, parts.content_scope, pairs, weight, rank_scope=scope
        )
        metrics.update(content_metrics)
    elif ordering == "fcn":
        pairs["retrieval_score"] = fcn_signal.astype(float)
    elif ordering == "fcn_rrf":
        pairs["retrieval_score"] = rrf_fuse(pairs, [fcn_signal], scope, weight)
    elif ordering == "ensemble":
        gbdt_signal, _, content_metrics = content_scorer_signal(
            parts.content_model, parts.content_scope, pairs, scope
        )
        metrics.update(content_metrics)
        pairs["retrieval_score"] = rrf_fuse(pairs, [gbdt_signal, fcn_signal], scope, weight)
    elif ordering == "routed":
        in_vocabulary = pairs["profile_id"].isin(parts.fcn[config.content_fcn_family].profiles).to_numpy()
        if config.routed_fallback == "learned_rrf":
            fallback_pairs, content_metrics = apply_content_scorer(
                parts.content_model, parts.content_scope, pairs, weight, rank_scope=scope
            )
            metrics.update(content_metrics)
            fallback = fallback_pairs["retrieval_score"].to_numpy(dtype=float)
        else:
            fallback = segment_content_scores(parts.segment, pairs)
        pairs["retrieval_score"] = np.where(in_vocabulary, rrf_fuse(pairs, [fcn_signal], scope, weight), fallback)
        metrics["routed_profile_rate"] = float(pairs.loc[in_vocabulary, "profile_id"].nunique() / max(pairs["profile_id"].nunique(), 1))
    elif ordering == "segment":
        pairs["retrieval_score"] = segment_content_scores(parts.segment, pairs)

    metrics.update(
        {
            "rule_ordering": config.rule_ordering,
            "content_ordering": ordering,
            "content_rank_scope": scope,
            "fcn_families": list(config.fcn_families),
        }
    )
    if parts.segment is not None:
        metrics["segment_cold_profile_rate"] = float(
            (~profiles.astype(str).isin(parts.segment.warm)).mean()
        ) if len(profiles) else 0.0
    return pairs, rule_scores, metrics
