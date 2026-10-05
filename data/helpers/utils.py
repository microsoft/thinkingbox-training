from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from thinkingbox.common.eval_utils import beta_post_params, cred_int, prob_in_zone

BANDS: tuple[tuple[str, float, float], ...] = (
    ("too_hard", 0.0, 0.08),
    ("hard", 0.08, 0.40),
    ("medium", 0.40, 0.60),
    ("easy", 0.60, 0.90),
    ("too_easy", 0.90, 1.0),
)
TRAINING_VALUE_BASES: dict[str, float] = {
    "highest_value": 1.0,
    "high_value": 0.85,
    "easy_value": 0.6,
    "low_value": 0.25,
    "insufficient_data": 0.3,
    "very_hard": 0.2,
}
AGGREGATE_OUTPUT_KEYS: tuple[str, ...] = (
    "uid",
    "total_rows",
    "clean_runs",
    "successes",
    "failures",
    "system_errors",
    "pass_rate",
    "standard_error",
    "posterior_alpha",
    "posterior_beta",
    "posterior_band_masses",
    "credible_interval",
    "credible_interval_width",
    "bernoulli_variance",
    "difficulty",
    "difficulty_mass",
    "training_value",
    "priority_score",
    "reference_total_rows",
    "reference_clean_runs",
    "reference_successes",
    "reference_failures",
    "reference_system_errors",
    "reference_probability_above_threshold",
    "reference_solvable",
    "eligible",
    "rejection_reasons",
)


def extract_dotted(row: Any, field_path: str) -> Any:
    if not field_path or not isinstance(field_path, str):
        raise ValueError("field path must be a non-empty string")
    value = row
    for part in field_path.split("."):
        if isinstance(value, Mapping):
            if part not in value:
                raise KeyError(field_path)
            value = value[part]
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            try:
                index = int(part)
            except ValueError as exc:
                raise KeyError(field_path) from exc
            if index < 0 or index >= len(value):
                raise KeyError(field_path)
            value = value[index]
        else:
            raise KeyError(field_path)
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed: int, key: str) -> int:
    payload = f"{seed}\0{key}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _new_counts(uid: str) -> dict[str, Any]:
    return {
        "uid": uid,
        "total_rows": 0,
        "clean_runs": 0,
        "successes": 0,
        "failures": 0,
        "system_errors": 0,
    }


def aggregate_jsonl(
    paths: Iterable[str | Path],
    *,
    uid_field: str,
    success_field: str,
    system_error_field: str | None = None,
) -> dict[str, dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    saw_row = False
    for path_value in paths:
        path = Path(path_value)
        if not path.is_file():
            raise ValueError("an input JSONL file does not exist")
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                saw_row = True
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"malformed JSONL at input {path.name}, line {line_number}"
                    ) from exc
                if not isinstance(row, Mapping):
                    raise TypeError(
                        f"JSONL row must be an object at input {path.name}, "
                        f"line {line_number}"
                    )
                try:
                    uid = extract_dotted(row, uid_field)
                except KeyError as exc:
                    raise ValueError(
                        f"missing UID field at input {path.name}, line {line_number}"
                    ) from exc
                if not isinstance(uid, str) or not uid:
                    raise TypeError(
                        f"UID must be a non-empty string at input {path.name}, "
                        f"line {line_number}"
                    )
                is_system_error = False
                if system_error_field:
                    try:
                        is_system_error = extract_dotted(row, system_error_field)
                    except KeyError as exc:
                        raise ValueError(
                            f"missing system-error field at input {path.name}, "
                            f"line {line_number}"
                        ) from exc
                    if not isinstance(is_system_error, bool):
                        raise TypeError(
                            f"system-error field must be boolean at input {path.name}, "
                            f"line {line_number}"
                        )
                counts = aggregates.setdefault(uid, _new_counts(uid))
                counts["total_rows"] += 1
                if is_system_error:
                    counts["system_errors"] += 1
                    continue
                try:
                    success = extract_dotted(row, success_field)
                except KeyError as exc:
                    raise ValueError(
                        f"missing success field at input {path.name}, "
                        f"line {line_number}"
                    ) from exc
                if not isinstance(success, bool):
                    raise TypeError(
                        f"success field must be boolean at input {path.name}, "
                        f"line {line_number}"
                    )
                counts["clean_runs"] += 1
                if success:
                    counts["successes"] += 1
                else:
                    counts["failures"] += 1
    if not saw_row:
        raise ValueError("input JSONL files contain no rows")
    return aggregates


def combine_counts(
    *aggregate_sets: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}
    for aggregate_set in aggregate_sets:
        for uid, source in aggregate_set.items():
            target = combined.setdefault(uid, _new_counts(uid))
            for key in (
                "total_rows",
                "clean_runs",
                "successes",
                "failures",
                "system_errors",
            ):
                value = source.get(key)
                if not isinstance(value, int) or value < 0:
                    raise ValueError(f"invalid aggregate count for {key}")
                target[key] += value
    for counts in combined.values():
        _validate_counts(counts)
    return combined


def _validate_counts(counts: Mapping[str, Any]) -> None:
    total = counts["total_rows"]
    clean = counts["clean_runs"]
    successes = counts["successes"]
    failures = counts["failures"]
    errors = counts["system_errors"]
    if clean != successes + failures or total != clean + errors:
        raise ValueError("aggregate counts are inconsistent")


def posterior_band_masses(successes: int, clean_runs: int) -> dict[str, float]:
    if clean_runs <= 0:
        raise ValueError("posterior requires at least one clean run")
    return {
        label: prob_in_zone(successes, clean_runs, lower, upper)
        for label, lower, upper in BANDS
    }


def probability_above(
    successes: int,
    clean_runs: int,
    threshold: float,
) -> float:
    if not 0.0 <= threshold < 1.0:
        raise ValueError("probability threshold must be in [0, 1)")
    if clean_runs <= 0:
        return 0.0
    return prob_in_zone(successes, clean_runs, threshold, 1.0)


def _training_value(
    difficulty: str,
    successes: int,
    clean_runs: int,
    interval: tuple[float, float],
) -> str:
    if difficulty == "too_easy":
        return "low_value"
    if difficulty == "easy":
        return "easy_value"
    if difficulty == "medium":
        return "high_value"
    if difficulty not in {"hard", "too_hard"}:
        return "insufficient_data"
    if successes == 0 and clean_runs >= 3:
        upper = interval[1]
        mass_below = prob_in_zone(successes, clean_runs, 0.0, 0.15)
        if upper <= 0.15 or mass_below >= 0.9:
            return "very_hard"
    if difficulty == "too_hard":
        return "highest_value" if clean_runs >= 3 else "insufficient_data"
    return "highest_value" if clean_runs >= 3 else "insufficient_data"


def assess_counts(counts: Mapping[str, Any]) -> dict[str, Any]:
    _validate_counts(counts)
    clean_runs = int(counts["clean_runs"])
    if clean_runs <= 0:
        raise ValueError("task has no clean runs")
    successes = int(counts["successes"])
    pass_rate = successes / clean_runs
    variance = pass_rate * (1.0 - pass_rate)
    standard_error = math.sqrt(variance / clean_runs)
    alpha, beta = beta_post_params(successes, clean_runs)
    masses = posterior_band_masses(successes, clean_runs)
    difficulty, difficulty_mass = max(
        masses.items(), key=lambda item: (item[1], -list(masses).index(item[0]))
    )
    interval_values = cred_int(successes, clean_runs)
    interval = (float(interval_values[0]), float(interval_values[1]))
    interval_width = interval[1] - interval[0]
    training_value = _training_value(difficulty, successes, clean_runs, interval)
    rarity = 1.0 - pass_rate
    normalized_width = min(interval_width / 0.5, 1.0)
    base = TRAINING_VALUE_BASES[training_value]
    priority = round(base * (0.55 + 0.30 * rarity + 0.15 * normalized_width), 4)
    return {
        **dict(counts),
        "pass_rate": pass_rate,
        "standard_error": standard_error,
        "posterior_alpha": float(alpha),
        "posterior_beta": float(beta),
        "posterior_band_masses": masses,
        "credible_interval": [interval[0], interval[1]],
        "credible_interval_width": interval_width,
        "bernoulli_variance": variance,
        "difficulty": difficulty,
        "difficulty_mass": difficulty_mass,
        "training_value": training_value,
        "priority_score": priority,
    }


def assess_aggregates(
    aggregates: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        assess_counts(aggregates[uid])
        for uid in sorted(aggregates)
        if aggregates[uid]["clean_runs"] > 0
    ]


def load_exclusion_uids(path_value: str | Path | None) -> set[str]:
    if path_value is None:
        return set()
    path = Path(path_value)
    if not path.is_file():
        raise ValueError("exclusion file does not exist")
    try:
        if path.suffix.lower() == ".json":
            document = json.loads(path.read_text(encoding="utf-8"))
        else:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError("exclusion file is malformed") from exc
    if isinstance(document, Mapping):
        if set(document) != {"uids"}:
            raise ValueError("exclusion object must contain only a 'uids' field")
        document = document["uids"]
    if not isinstance(document, list):
        raise TypeError("exclusion file must contain a UID list")
    if any(not isinstance(uid, str) or not uid for uid in document):
        raise ValueError("every exclusion UID must be a non-empty string")
    if len(document) != len(set(document)):
        raise ValueError("exclusion UIDs must be unique")
    return set(document)


def apply_reference_gates(
    rows: list[dict[str, Any]],
    reference: Mapping[str, Mapping[str, Any]] | None,
    *,
    min_clean_runs: int,
    min_successes: int,
    solvability_threshold: float | None,
    min_probability_above: float | None,
) -> None:
    if min_clean_runs < 0 or min_successes < 0:
        raise ValueError("reference count thresholds must be non-negative")
    if solvability_threshold is not None and not 0.0 <= solvability_threshold < 1.0:
        raise ValueError("reference solvability threshold must be in [0, 1)")
    if min_probability_above is not None and not 0.0 <= min_probability_above <= 1.0:
        raise ValueError("reference posterior probability must be in [0, 1]")
    if (solvability_threshold is None) != (min_probability_above is None):
        raise ValueError(
            "reference posterior gate requires both threshold and probability"
        )
    for row in rows:
        source = reference.get(row["uid"]) if reference is not None else None
        reference_counts = source or _new_counts(row["uid"])
        clean = int(reference_counts["clean_runs"])
        successes = int(reference_counts["successes"])
        probability = (
            probability_above(successes, clean, solvability_threshold)
            if solvability_threshold is not None
            else None
        )
        solvable = clean >= min_clean_runs and successes >= min_successes
        if min_probability_above is not None:
            solvable = solvable and probability is not None
            solvable = solvable and probability >= min_probability_above
        row.update(
            {
                "reference_total_rows": int(reference_counts["total_rows"]),
                "reference_clean_runs": clean,
                "reference_successes": successes,
                "reference_failures": int(reference_counts["failures"]),
                "reference_system_errors": int(reference_counts["system_errors"]),
                "reference_probability_above_threshold": probability,
                "reference_solvable": solvable,
            }
        )


def select_eligible(
    rows: list[dict[str, Any]],
    *,
    exclusions: set[str],
    min_clean_runs: int,
    retained_bands: set[str],
    require_reference: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    valid_bands = {band[0] for band in BANDS}
    if min_clean_runs < 1:
        raise ValueError("minimum clean runs must be at least one")
    if not retained_bands or not retained_bands <= valid_bands:
        raise ValueError("retained bands must be a non-empty subset of fixed bands")
    selected: list[dict[str, Any]] = []
    rejected = Counter()
    for row in rows:
        reasons: list[str] = []
        if row["uid"] in exclusions:
            reasons.append("excluded")
        if row["clean_runs"] < min_clean_runs:
            reasons.append("insufficient_clean_runs")
        if row["difficulty"] not in retained_bands:
            reasons.append("difficulty_not_retained")
        if require_reference and not row["reference_solvable"]:
            reasons.append("reference_gate")
        row["eligible"] = not reasons
        row["rejection_reasons"] = reasons
        if reasons:
            rejected.update(reasons)
        else:
            selected.append(row)
    if exclusions & {row["uid"] for row in selected}:
        raise RuntimeError("excluded UID passed eligibility")
    if not selected:
        raise ValueError("no tasks passed the configured gates")
    return selected, dict(sorted(rejected.items()))


def _dataset_group(uid: str) -> str:
    return uid.split(":", 1)[0] if ":" in uid else "single_dataset"


def _pass_rate_bin(value: float) -> str:
    if value == 0.0:
        return "zero"
    if value < 0.25:
        return "low"
    if value < 0.75:
        return "medium"
    return "high"


def _run_count_bin(value: int) -> str:
    if value <= 3:
        return "few"
    if value <= 10:
        return "medium"
    return "many"


def _stratum(row: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(row["difficulty"]),
            str(row["training_value"]),
            _pass_rate_bin(float(row["pass_rate"])),
            _run_count_bin(int(row["clean_runs"])),
        )
    )


def _rank(seed: int, key: str, uid: str) -> str:
    return hashlib.sha256(f"{stable_seed(seed, key)}\0{uid}".encode()).hexdigest()


def balanced_split(
    rows: list[dict[str, Any]],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test fraction must be between zero and one")
    if not rows:
        raise ValueError("cannot split an empty selection")
    if len({row["uid"] for row in rows}) != len(rows):
        raise ValueError("split input contains duplicate UIDs")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_dataset_group(row["uid"]), []).append(row)
    test_uids: set[str] = set()
    for group_key in sorted(groups):
        group = groups[group_key]
        if len(group) == 1:
            continue
        target = max(1, math.ceil(len(group) * test_fraction))
        target = min(target, len(group) - 1)
        strata: dict[str, list[dict[str, Any]]] = {}
        for row in group:
            strata.setdefault(_stratum(row), []).append(row)
        selected: list[str] = []
        if all(len(items) >= 2 for items in strata.values()) and target >= len(strata):
            quotas = {
                key: min(
                    len(items) - 1,
                    max(1, math.floor(len(items) * test_fraction)),
                )
                for key, items in strata.items()
            }
            while sum(quotas.values()) > target:
                choices = [key for key, quota in quotas.items() if quota > 1]
                if not choices:
                    break
                key = max(choices, key=lambda item: (quotas[item], item))
                quotas[key] -= 1
            for stratum_key in sorted(strata):
                ordered = sorted(
                    strata[stratum_key],
                    key=lambda row: _rank(
                        seed, f"{group_key}\0{stratum_key}", row["uid"]
                    ),
                )
                selected.extend(row["uid"] for row in ordered[: quotas[stratum_key]])
        remaining = target - len(selected)
        if remaining > 0:
            candidates = [row for row in group if row["uid"] not in selected]
            ordered = sorted(
                candidates,
                key=lambda row: _rank(seed, f"{group_key}\0fallback", row["uid"]),
            )
            selected.extend(row["uid"] for row in ordered[:remaining])
        test_uids.update(selected[:target])
    train = sorted(
        (row for row in rows if row["uid"] not in test_uids),
        key=lambda row: row["uid"],
    )
    test = sorted(
        (row for row in rows if row["uid"] in test_uids),
        key=lambda row: row["uid"],
    )
    train_uids = {row["uid"] for row in train}
    actual_test_uids = {row["uid"] for row in test}
    all_uids = {row["uid"] for row in rows}
    if not train or not test:
        raise ValueError(
            "configured split cannot produce non-empty train and test sets"
        )
    if train_uids & actual_test_uids:
        raise RuntimeError("train/test overlap detected")
    if train_uids | actual_test_uids != all_uids:
        raise RuntimeError("train/test union does not cover the selection")
    for group_key, group in groups.items():
        group_uids = {row["uid"] for row in group}
        if group_uids and not group_uids & train_uids:
            raise RuntimeError(f"split left no training item in group {group_key}")
    return train, test


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
    }


def balance_audit(
    train: list[Mapping[str, Any]],
    test: list[Mapping[str, Any]],
) -> dict[str, Any]:
    total = len(train) + len(test)
    if total == 0:
        raise ValueError("cannot audit empty splits")

    def distribution(
        rows: list[Mapping[str, Any]], key: str
    ) -> dict[str, dict[str, float | int]]:
        counts = Counter(str(row[key]) for row in rows)
        return {
            label: {
                "count": count,
                "fraction": count / len(rows) if rows else 0.0,
            }
            for label, count in sorted(counts.items())
        }

    return {
        "split_sizes": {
            "train": len(train),
            "test": len(test),
            "total": total,
            "train_fraction": len(train) / total,
            "test_fraction": len(test) / total,
        },
        "difficulty_distribution": {
            "train": distribution(train, "difficulty"),
            "test": distribution(test, "difficulty"),
        },
        "training_value_distribution": {
            "train": distribution(train, "training_value"),
            "test": distribution(test, "training_value"),
        },
        "pass_rate": {
            "train": _mean_std([float(row["pass_rate"]) for row in train]),
            "test": _mean_std([float(row["pass_rate"]) for row in test]),
        },
        "clean_runs": {
            "train": _mean_std([float(row["clean_runs"]) for row in train]),
            "test": _mean_std([float(row["clean_runs"]) for row in test]),
        },
    }


def ensure_output_outside_repository(
    output_directory: str | Path,
    repository_root: str | Path,
) -> Path:
    output = Path(output_directory).expanduser().resolve()
    repository = Path(repository_root).resolve()
    try:
        output.relative_to(repository)
    except ValueError:
        pass
    else:
        raise ValueError("runtime output directory must be outside the repository")
    if output.exists() and not output.is_dir():
        raise ValueError("runtime output path must be a directory")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _json_bytes(document: Any) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _yaml_uid_bytes(uids: Iterable[str]) -> bytes:
    return yaml.safe_dump(
        sorted(uids),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).encode()


def export_outputs(
    *,
    output_directory: Path,
    mode: str,
    selected: list[dict[str, Any]],
    aggregate_rows: list[dict[str, Any]],
    train: list[dict[str, Any]] | None,
    test: list[dict[str, Any]] | None,
    manifest: dict[str, Any],
) -> dict[str, str]:
    if mode not in {"selection", "split"}:
        raise ValueError("mode must be 'selection' or 'split'")
    files: dict[str, bytes] = {}
    if mode == "selection":
        files["selected.yaml"] = _yaml_uid_bytes(row["uid"] for row in selected)
    else:
        if train is None or test is None:
            raise ValueError("split mode requires train and test rows")
        files["train.yaml"] = _yaml_uid_bytes(row["uid"] for row in train)
        files["test.yaml"] = _yaml_uid_bytes(row["uid"] for row in test)
    aggregate_lines = b"".join(
        _json_bytes({key: row.get(key) for key in AGGREGATE_OUTPUT_KEYS})
        for row in sorted(aggregate_rows, key=lambda item: item["uid"])
    )
    files["aggregates.jsonl"] = aggregate_lines
    output_hashes: dict[str, str] = {}
    for name in sorted(files):
        _atomic_write(output_directory / name, files[name])
        output_hashes[name] = hashlib.sha256(files[name]).hexdigest()
    manifest["output_sha256"] = output_hashes
    manifest_bytes = _json_bytes(manifest)
    _atomic_write(output_directory / "manifest.json", manifest_bytes)
    output_hashes["manifest.json"] = hashlib.sha256(manifest_bytes).hexdigest()
    return output_hashes


def validate_configuration(
    *,
    min_clean_runs: int,
    retained_bands: set[str],
    test_fraction: float,
    mode: str,
) -> None:
    if min_clean_runs < 1:
        raise ValueError("minimum clean runs must be at least one")
    known_bands = {label for label, _, _ in BANDS}
    if not retained_bands or not retained_bands <= known_bands:
        raise ValueError("retained bands must be a non-empty subset of fixed bands")
    if mode not in {"selection", "split"}:
        raise ValueError("mode must be 'selection' or 'split'")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test fraction must be between zero and one")


def run_preparation(
    *,
    policy_paths: list[str | Path],
    reference_paths: list[str | Path],
    exclusion_path: str | Path | None,
    output_directory: str | Path,
    repository_root: str | Path,
    uid_field: str,
    success_field: str,
    system_error_field: str | None,
    reference_uid_field: str,
    reference_success_field: str,
    reference_system_error_field: str | None,
    min_clean_runs: int,
    retained_bands: set[str],
    reference_min_clean_runs: int,
    reference_min_successes: int,
    reference_solvability_threshold: float | None,
    reference_min_probability_above: float | None,
    mode: str,
    test_fraction: float,
    seed: int,
) -> dict[str, Any]:
    validate_configuration(
        min_clean_runs=min_clean_runs,
        retained_bands=retained_bands,
        test_fraction=test_fraction,
        mode=mode,
    )
    if not policy_paths:
        raise ValueError("at least one policy probe file is required")
    output = ensure_output_outside_repository(output_directory, repository_root)
    exclusions = load_exclusion_uids(exclusion_path)
    policy = aggregate_jsonl(
        policy_paths,
        uid_field=uid_field,
        success_field=success_field,
        system_error_field=system_error_field,
    )
    rows = assess_aggregates(policy)
    if not rows:
        raise ValueError("policy inputs contain no clean task outcomes")
    excluded_uids_present = exclusions & {row["uid"] for row in rows}
    rows = [row for row in rows if row["uid"] not in exclusions]
    if not rows:
        raise ValueError("no tasks remain after exclusions")
    reference = None
    if reference_paths:
        reference = aggregate_jsonl(
            reference_paths,
            uid_field=reference_uid_field,
            success_field=reference_success_field,
            system_error_field=reference_system_error_field,
        )
    apply_reference_gates(
        rows,
        reference,
        min_clean_runs=reference_min_clean_runs,
        min_successes=reference_min_successes,
        solvability_threshold=reference_solvability_threshold,
        min_probability_above=reference_min_probability_above,
    )
    selected, rejected = select_eligible(
        rows,
        exclusions=set(),
        min_clean_runs=min_clean_runs,
        retained_bands=retained_bands,
        require_reference=bool(reference_paths),
    )
    train: list[dict[str, Any]] | None = None
    test: list[dict[str, Any]] | None = None
    audit = None
    if mode == "split":
        train, test = balanced_split(selected, test_fraction=test_fraction, seed=seed)
        audit = balance_audit(train, test)
    input_hashes = {
        "policy": [sha256_file(path) for path in policy_paths],
        "reference": [sha256_file(path) for path in reference_paths],
        "exclusions": sha256_file(exclusion_path) if exclusion_path else None,
    }
    summary = {
        "input_rows": sum(row["total_rows"] for row in policy.values()),
        "aggregated_tasks": len(policy),
        "clean_tasks": len(rows),
        "excluded_uids_present": len(excluded_uids_present),
        "selected_tasks": len(selected),
        "train_tasks": len(train) if train is not None else None,
        "test_tasks": len(test) if test is not None else None,
        "rejections": rejected,
    }
    manifest = {
        "schema_version": 1,
        "mode": mode,
        "fixed_bands": {label: [lower, upper] for label, lower, upper in BANDS},
        "configuration": {
            "min_clean_runs": min_clean_runs,
            "retained_bands": sorted(retained_bands),
            "reference_required": bool(reference_paths),
            "reference_min_clean_runs": reference_min_clean_runs,
            "reference_min_successes": reference_min_successes,
            "reference_solvability_threshold": reference_solvability_threshold,
            "reference_min_probability_above": reference_min_probability_above,
            "test_fraction": test_fraction if mode == "split" else None,
            "seed": seed if mode == "split" else None,
        },
        "summary": summary,
        "balance_audit": audit,
        "input_sha256": input_hashes,
    }
    output_hashes = export_outputs(
        output_directory=output,
        mode=mode,
        selected=selected,
        aggregate_rows=rows,
        train=train,
        test=test,
        manifest=manifest,
    )
    return {
        "summary": summary,
        "output_sha256": output_hashes,
        "output_directory": output,
    }
