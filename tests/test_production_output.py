from __future__ import annotations

import json

from recsys_vnpt.export import (
    build_profile_grouping,
    production_output_names,
    write_production_named_outputs,
)
from recsys_vnpt.output_validation import parse_homepage, parse_rule_content


def test_production_output_names_match_rs_output_convention() -> None:
    names = production_output_names("all", "20260630")
    assert names["homepage"] == "f_personalized_homepage_rule_ms_all_0_700000_DL_MyTV_20260630.json"
    assert names["rule_content"] == "f_personalized_rule_content_ms_0_700000_DL_MyTV_20260630.csv"
    assert names["grouping"] == "f_profile_grouping_ms_DL_MyTV_0_20260630.json"


def test_profile_grouping_falls_back_to_the_profile_id() -> None:
    homepage = [{"pid": "4078", "d": "60,148,112,109,108,110"}]
    grouping = build_profile_grouping(homepage)
    assert grouping == [{"pid": "4078", "mid": "4078", "gid": 3}]


def test_write_production_named_outputs_matches_rs_output(tmp_path) -> None:
    homepage = [
        {"pid": "4078", "d": "60,148,112,109,108,110"},
        {"pid": "4309", "d": "144,118,112,109,108,110"},
    ]
    rule_content = "4078|60,2#111$2#222;148,12#333\n4309|144,6#444\n"
    paths = write_production_named_outputs(
        tmp_path, homepage, rule_content, domain="all", date="20260630"
    )
    # exact filenames
    assert paths["homepage"].name == "f_personalized_homepage_rule_ms_all_0_700000_DL_MyTV_20260630.json"
    assert paths["rule_content"].name == "f_personalized_rule_content_ms_0_700000_DL_MyTV_20260630.csv"
    assert paths["grouping"].name == "f_profile_grouping_ms_DL_MyTV_0_20260630.json"
    # homepage JSON parses to the same {pid: [rules]} structure the validator expects
    hp = json.loads(paths["homepage"].read_text(encoding="utf-8"))
    assert parse_homepage(hp) == {
        "4078": ["60", "148", "112", "109", "108", "110"],
        "4309": ["144", "118", "112", "109", "108", "110"],
    }
    # one JSON object per line, like the rs_output fixture
    assert paths["homepage"].read_text(encoding="utf-8").splitlines()[1].startswith('{"pid": "4078"')
    # rule_content is the pipe/hash/dollar format, no header
    parsed = parse_rule_content(paths["rule_content"].read_text(encoding="utf-8"))
    assert parsed["4078"]["60"] == [
        {"type_id": "2", "content_id": "111"},
        {"type_id": "2", "content_id": "222"},
    ]
    # grouping: identity mid, constant gid, one per homepage pid
    grouping = json.loads(paths["grouping"].read_text(encoding="utf-8"))
    assert grouping == [
        {"pid": "4078", "mid": "4078", "gid": 3},
        {"pid": "4309", "mid": "4309", "gid": 3},
    ]


def test_profile_grouping_uses_the_resolved_member_id() -> None:
    # A member owns one or more profiles; only the default profile carries the
    # member id as its own id, so `mid` must come from the mapping, not `pid`.
    homepage = [
        {"pid": "4078", "d": "60,148,112,109,108,110"},
        {"pid": "11367785", "d": "144,118,112,109,108,110"},
    ]
    grouping = build_profile_grouping(
        homepage, member_ids={"11367785": "38510153"}
    )
    assert grouping == [
        {"pid": "4078", "mid": "4078", "gid": 3},
        {"pid": "11367785", "mid": "38510153", "gid": 3},
    ]


def test_profile_grouping_takes_the_group_id_per_profile() -> None:
    homepage = [{"pid": "4078", "d": "60,112,109,108,110"}]
    grouping = build_profile_grouping(homepage, group_ids={"4078": 7})
    assert grouping == [{"pid": "4078", "mid": "4078", "gid": 7}]


def test_write_production_named_outputs_carries_the_member_map(tmp_path) -> None:
    homepage = [{"pid": "11367785", "d": "144,118,112,109,108,110"}]
    paths = write_production_named_outputs(
        tmp_path,
        homepage,
        "11367785|144,6#444\n",
        domain="all",
        date="20260630",
        member_ids={"11367785": "38510153"},
    )
    grouping = json.loads(paths["grouping"].read_text(encoding="utf-8"))
    assert grouping == [{"pid": "11367785", "mid": "38510153", "gid": 3}]
