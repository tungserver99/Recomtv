from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainConfig:
    name: str
    content_patterns: tuple[str, ...]
    item_output_name: str
    watch_percent_threshold: float | None
    repeat_watch_positive: bool
    uses_family_label_policy: bool = False
    top_n: int = 200


DOMAIN_CONFIGS: dict[str, DomainConfig] = {
    "clip": DomainConfig(
        name="clip",
        content_patterns=("content_clip_2*.json",),
        item_output_name="clip_item_data.parquet",
        watch_percent_threshold=0.3,
        repeat_watch_positive=False,
    ),
    "movie": DomainConfig(
        name="movie",
        content_patterns=("content_movie_2*.json",),
        item_output_name="movie_item_data.parquet",
        watch_percent_threshold=0.9,
        repeat_watch_positive=True,
    ),
    "music": DomainConfig(
        name="music",
        content_patterns=("music_video_2*.json",),
        item_output_name="music_item_data.parquet",
        watch_percent_threshold=1.0,
        repeat_watch_positive=True,
    ),
    "all": DomainConfig(
        name="all",
        content_patterns=("content_movie_2*.json", "content_clip_2*.json", "music_video_2*.json"),
        item_output_name="all_item_data.parquet",
        watch_percent_threshold=None,
        repeat_watch_positive=False,
        uses_family_label_policy=True,
    ),
}


def get_domain_config(domain: str) -> DomainConfig:
    try:
        return DOMAIN_CONFIGS[domain]
    except KeyError as exc:
        known = ", ".join(sorted(DOMAIN_CONFIGS))
        raise ValueError(f"Unknown domain '{domain}'. Expected one of: {known}") from exc
