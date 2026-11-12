"""A small synthetic input batch in the production raw layout.

Invented ids only, no customer data: safe for tests, the Kaggle smoke run and a
first rehearsal of the batch commands on a new machine. Large enough for every
stage to run for real, including retraining on relative temporal windows.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from recsys_vnpt.batch_contracts import write_ready_manifest


CATEGORIES = 4


def write_smoke_sources(
    batch_dir: str | Path,
    profiles: int = 30,
    contents: int = 120,
    days: int = 60,
    start: str = "2025-01-01",
) -> datetime:
    """Write raw sources into ``batch_dir``; returns the data cutoff."""
    root = Path(batch_dir)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "username": [f"u{index:03d}" for index in range(profiles)],
            "province": ["01"] * profiles,
            "tentinh": ["A"] * profiles,
            "sex": [index % 2 for index in range(profiles)],
            "birthday": ["19900101"] * profiles,
            "package_code": [f"P{index % 3}" for index in range(profiles)],
        }
    ).to_parquet(root / "month_mytv_info.parquet", index=False)

    first_day = pd.Timestamp(start)
    rows = []
    for profile in range(profiles):
        for day in range(0, days, 2):
            # Each profile keeps to one category, the kind of preference the
            # learned scorer can pick up, and moves on to titles it has not
            # seen, so every window has novel positives.
            category = profile % CATEGORIES
            content = category + CATEGORIES * ((profile * 7 + day // 2) % (contents // CATEGORIES))
            moment = first_day + pd.Timedelta(days=day, hours=12 + profile % 6)
            rows.append(
                {
                    "profile_id": f"p{profile:03d}",
                    "username": f"u{profile:03d}",
                    "member_id": f"u{profile:03d}",
                    "content_id": f"c{content:03d}",
                    "duration": 10 + (profile * 13 + day * 7) % 90,
                    "create_date": moment.strftime("%Y-%m-%d %H:%M:%S"),
                    "filename_date": moment.strftime("%Y%m%d"),
                }
            )
    (root / "duration").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "duration" / "duration_part_1.csv", index=False)

    snapshot = first_day.strftime("%Y%m%d")
    content_dir = root / "mytv_vmp_content" / snapshot
    content_dir.mkdir(parents=True, exist_ok=True)
    (content_dir / f"content_movie_{snapshot}_000000.json").write_text(
        json.dumps(
            [
                {
                    "content_id": f"c{index:03d}",
                    "content_name": f"Content {index}",
                    "type_id": 2,
                    "content_duration": 100,
                    "content_status": 1,
                    "content_publish_year": 2020 + index % 5,
                    "VOD_CODE": "AVOD",
                    "total_episode": 1 + index % 4,
                    "content_category_name": f"Category {index % CATEGORIES}",
                    "content_country": ["VN", "KR", "US"][index % 3],
                    "createdate": "2024-12-01 08:00:00",
                    "publish_date": "2024-12-01 09:00:00",
                }
                for index in range(contents)
            ]
        ),
        encoding="utf-8",
    )

    tag_dir = root / "tags" / snapshot
    tag_dir.mkdir(parents=True, exist_ok=True)
    (tag_dir / "TAG.json").write_text(
        json.dumps(
            [{"TAG_ID": tag, "TAG_NAME": f"tag{tag}", "STATUS": 1} for tag in (1, 2, 3)]
        ),
        encoding="utf-8",
    )
    # Tag 1 covers everything, tag 2 every other content, tag 3 every third, so
    # rules differ in size and overlap the way real rows do.
    content_tags = [
        {"CONTENT_ID": f"c{index:03d}", "TAG_ID": tag}
        for index in range(contents)
        for tag in (1, 2, 3)
        if tag == 1 or (tag == 2 and index % 2 == 0) or (tag == 3 and index % 3 == 0)
    ]
    content_tag_dir = root / "content_tags" / snapshot
    content_tag_dir.mkdir(parents=True, exist_ok=True)
    (content_tag_dir / "CONTENT_TO_TAG.json").write_text(json.dumps(content_tags), encoding="utf-8")
    pd.DataFrame(
        {
            "RULE_ID": [str(rule) for rule in range(1, 25)],
            "LIST_TAGS_ID": ["1"] * 8 + ["2"] * 8 + ["3"] * 8,
        }
    ).to_parquet(root / "rule_info.parquet", index=False)

    last_event = first_day + pd.Timedelta(days=days)
    return datetime(last_event.year, last_event.month, last_event.day, tzinfo=timezone(timedelta(hours=7)))


def write_smoke_batch(incoming_root: str | Path, batch_id: str, **options) -> Path:
    """A complete committed batch: sources first, ``_READY.json`` last."""
    batch_dir = Path(incoming_root) / batch_id
    write_smoke_sources(batch_dir, **options)
    write_ready_manifest(batch_dir, batch_id=batch_id)
    return batch_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming-root", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--start", default="2025-01-01", help="first event day; later batches need a later start")
    args = parser.parse_args(argv)
    print(write_smoke_batch(args.incoming_root, args.batch_id, start=args.start))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
