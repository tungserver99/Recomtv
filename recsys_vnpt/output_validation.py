from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from recsys_vnpt.rule_assembly import RuleAssemblyConfig


@dataclass
class OutputValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    metrics: dict[str, float | int] = field(default_factory=dict)


def parse_rule_content(text: str) -> dict[str, dict[str, list[dict[str, str]]]]:
    output: dict[str, dict[str, list[dict[str, str]]]] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line:
            continue
        if "|" not in raw_line:
            raise ValueError(f"line {line_number} is missing profile delimiter")
        profile_id, payload = raw_line.split("|", 1)
        if profile_id in output:
            raise ValueError(f"duplicate profile line: {profile_id}")
        rules: dict[str, list[dict[str, str]]] = {}
        for chunk in filter(None, payload.split(";")):
            if "," not in chunk:
                raise ValueError(f"line {line_number} rule chunk is missing comma")
            rule_id, content_payload = chunk.split(",", 1)
            if rule_id in rules:
                raise ValueError(f"line {line_number} has duplicate rule {rule_id}")
            contents: list[dict[str, str]] = []
            for token in filter(None, content_payload.split("$")):
                if "#" not in token:
                    raise ValueError(f"line {line_number} content token is missing #")
                type_id, content_id = token.split("#", 1)
                contents.append({"type_id": type_id, "content_id": content_id})
            rules[rule_id] = contents
        output[profile_id] = rules
    return output


def parse_homepage(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for record in records:
        profile_id = str(record["pid"])
        if profile_id in output:
            raise ValueError(f"duplicate homepage profile: {profile_id}")
        output[profile_id] = [rule for rule in str(record.get("d", "")).split(",") if rule]
    return output


def validate_client_output(
    homepage_records: list[dict[str, Any]],
    rule_content_text: str,
    expected_profiles: set[str] | None = None,
    catalog: dict[str, dict[str, Any]] | None = None,
    valid_membership: set[tuple[str, str]] | None = None,
    config: RuleAssemblyConfig = RuleAssemblyConfig(),
) -> OutputValidationResult:
    errors: list[str] = []
    homepage = parse_homepage(homepage_records)
    content = parse_rule_content(rule_content_text)
    profiles = set(homepage)
    expected = set(map(str, expected_profiles)) if expected_profiles is not None else profiles
    if profiles != expected:
        errors.append(f"homepage profile mismatch: missing={sorted(expected-profiles)}, extra={sorted(profiles-expected)}")
    if set(content) != profiles:
        errors.append(f"rule-content profile mismatch: missing={sorted(profiles-set(content))}, extra={sorted(set(content)-profiles)}")

    exact_rules = 0
    exact_contents = 0
    special_complete = 0
    duplicate_count = 0
    in_range_contents = 0
    cross_rule_duplicates = 0
    page_slot_total = 0
    invalid_count = 0
    unmatched_count = 0
    total_content = 0
    consistent_profiles = 0
    special_without_content = 0
    for profile_id in sorted(profiles):
        rules = homepage[profile_id]
        normal = rules[: config.normal_rule_count]
        special = rules[config.normal_rule_count :]
        if len(normal) == config.normal_rule_count and len(set(normal)) == config.normal_rule_count:
            exact_rules += 1
        else:
            errors.append(f"{profile_id}: expected {config.normal_rule_count} unique normal rules")
        if special == list(config.special_rules):
            special_complete += 1
        else:
            errors.append(f"{profile_id}: special suffix mismatch")
        content_rules = content.get(profile_id, {})
        rules_consistent = set(content_rules) == set(normal)
        if not rules_consistent:
            errors.append(f"{profile_id}: homepage and content normal rules differ")
            unmatched_count += len(set(normal) ^ set(content_rules))
        else:
            consistent_profiles += 1
        if not (set(content_rules) & set(config.special_rules)):
            special_without_content += 1
        profile_exact = True
        profile_in_range = True
        profile_contents: list[str] = []
        for rule_id, items in content_rules.items():
            total_content += len(items)
            ids = [item["content_id"] for item in items]
            if rule_id not in config.special_rules:
                profile_contents.extend(ids)
            duplicates = len(ids) - len(set(ids))
            duplicate_count += duplicates
            in_range = config.min_contents <= len(items) <= config.max_contents
            if len(items) != config.contents_per_rule or duplicates:
                profile_exact = False
            if not in_range or duplicates:
                profile_in_range = False
                errors.append(
                    f"{profile_id}/{rule_id}: expected {config.min_contents}"
                    f"-{config.max_contents} unique contents, got {len(items)}"
                )
            if rule_id in config.special_rules:
                errors.append(f"{profile_id}: special rule {rule_id} has content")
            for item in items:
                if catalog is not None:
                    meta = catalog.get(item["content_id"])
                    if meta is None or str(meta.get("type_id")) != item["type_id"] or str(meta.get("content_status", "1")) != "1" or meta.get("recommendable", True) is False:
                        invalid_count += 1
                        if len(errors) < 100:
                            errors.append(
                                f"{profile_id}/{rule_id}/{item['content_id']}: invalid catalog content"
                            )
                if valid_membership is not None and (item["content_id"], rule_id) not in valid_membership:
                    unmatched_count += 1
                    if len(errors) < 100:
                        errors.append(
                            f"{profile_id}/{rule_id}/{item['content_id']}: content-rule membership mismatch"
                        )
        if profile_exact and len(content_rules) == config.normal_rule_count:
            exact_contents += 1
        if profile_in_range and len(content_rules) == config.normal_rule_count:
            in_range_contents += 1
        # A content repeated under two different rules wastes a page slot; the
        # per-rule duplicate check above cannot see it.
        cross_rule_duplicates += len(profile_contents) - len(set(profile_contents))
        page_slot_total += len(profile_contents)

    denominator = len(expected) or 1
    metrics = {
        "profiles_expected": len(expected),
        "profiles_emitted": len(profiles),
        "profile_output_coverage": len(profiles & expected) / denominator,
        "exact_20_normal_rule_rate": exact_rules / denominator,
        "exact_10_content_rate": exact_contents / denominator,
        "content_count_within_range_rate": in_range_contents / denominator,
        "duplicate_across_rules_count": cross_rule_duplicates,
        "duplicate_across_rules_rate": cross_rule_duplicates / max(page_slot_total, 1),
        "special_rule_complete_rate": special_complete / denominator,
        "cross_file_consistency_rate": consistent_profiles / denominator,
        "special_no_content_rate": special_without_content / denominator,
        "duplicate_within_rule_count": duplicate_count,
        "duplicate_within_rule_rate": duplicate_count / max(total_content, 1),
        "invalid_content_count": invalid_count,
        "invalid_content_rate": invalid_count / max(total_content, 1),
        "unmatched_rule_count": unmatched_count,
        "unmatched_rule_rate": unmatched_count / max(total_content, 1),
    }
    return OutputValidationResult(passed=not errors, errors=errors, metrics=metrics)
