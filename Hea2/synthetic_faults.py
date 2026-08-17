"""Deterministic synthetic faults for label-free PSAP model evaluation.

The injector treats the supplied hourly KPI frame as an untouched baseline.  It
returns a separate injected frame and two separate truth tables; model-facing
data therefore never contains ground-truth columns.

The deployed directional scorer regards a positive call-volume residual and a
negative success-rate residual as anomalous.  Volume drops are intentionally
kept in the suite as ``DETECT`` challenge events, because failure to detect them
reveals that directional blind spot.  Success-rate improvements are controls
whose expected behaviour is ``NO_INCREASE``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd


TIME_COL = "HOUR_START_UTC"
ID_COL = "PSAP_ID"
VOLUME_COL = "CALL_VOLUME"
SR_COLS = ("LSR_SR", "ASR_SR", "BID_SR", "CSR_SR")
KPI_COLS = (VOLUME_COL, *SR_COLS)
REQUIRED_COLUMNS = (TIME_COL, ID_COL, *KPI_COLS)

ALLOWED_DURATIONS = frozenset({1, 2, 3, 6})
SEVERITY_FRACTIONS = {
    "mild": 0.10,
    "moderate": 0.25,
    "severe": 0.50,
}

_PATTERN_ALIASES = {
    "single_kpi_degradation": "kpi_degradation",
    "single_kpi_failure": "kpi_degradation",
    "multi_kpi_failure": "multi_kpi_degradation",
    "improvement_control": "benign_improvement",
    "slow_burn_ramp": "slow_burn",
    "intermittent": "flapping",
    "intermittent_flapping": "flapping",
    "correlated_failure": "correlated_volume_sr",
    "correlated_volume_success_failure": "correlated_volume_sr",
}

_PATTERN_METADATA = {
    "volume_spike": ("DETECT", "CALL_VOLUME_UP", "SUPPORTED"),
    "volume_drop": ("DETECT", "CALL_VOLUME_DOWN", "CHALLENGE"),
    "kpi_degradation": ("DETECT", "SUCCESS_RATE_DOWN", "SUPPORTED"),
    "multi_kpi_degradation": (
        "DETECT",
        "MULTIPLE_SUCCESS_RATES_DOWN",
        "SUPPORTED",
    ),
    "benign_improvement": ("NO_INCREASE", "SUCCESS_RATE_UP", "CONTROL"),
    "slow_burn": ("DETECT", "SUCCESS_RATE_RAMP_DOWN", "SUPPORTED"),
    "flapping": ("DETECT", "SUCCESS_RATE_INTERMITTENT_DOWN", "SUPPORTED"),
    "correlated_volume_sr": (
        "DETECT",
        "CALL_VOLUME_UP_AND_SUCCESS_RATE_DOWN",
        "SUPPORTED",
    ),
}


def _canonical_pattern(value: str) -> str:
    pattern = str(value).strip().lower().replace("-", "_")
    pattern = _PATTERN_ALIASES.get(pattern, pattern)
    if pattern not in _PATTERN_METADATA:
        allowed = ", ".join(sorted(_PATTERN_METADATA))
        raise ValueError(f"Unknown fault pattern {value!r}; expected one of: {allowed}")
    return pattern


def _canonical_stratum(value: str | None) -> str | None:
    if value is None:
        return None
    stratum = str(value).strip().lower()
    if stratum == "mid":
        stratum = "medium"
    if stratum not in {"low", "medium", "high"}:
        raise ValueError("volume_stratum must be 'low', 'medium', 'high', or None")
    return stratum


def _severity_fraction(value: str | float) -> tuple[str, float]:
    if isinstance(value, str):
        label = value.strip().lower()
        if label not in SEVERITY_FRACTIONS:
            allowed = ", ".join(SEVERITY_FRACTIONS)
            raise ValueError(
                f"Unknown severity {value!r}; use {allowed}, or a fraction in (0, 1]"
            )
        return label, SEVERITY_FRACTIONS[label]
    if isinstance(value, Real) and not isinstance(value, bool):
        fraction = float(value)
        if np.isfinite(fraction) and 0.0 < fraction <= 1.0:
            return "custom", fraction
    raise ValueError("severity must be mild/moderate/severe or a fraction in (0, 1]")


@dataclass(frozen=True)
class FaultSpec:
    """Description of one synthetic event.

    ``psap_id`` and ``start_time`` may be omitted.  In that case
    :func:`inject_faults` deterministically selects a matching PSAP and/or a
    contiguous time window from the baseline using ``seed``.
    """

    pattern: str
    severity: str | float = "moderate"
    duration_hours: int = 1
    psap_id: Any | None = None
    volume_stratum: str | None = None
    start_time: Any | None = None
    kpis: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        pattern = _canonical_pattern(self.pattern)
        object.__setattr__(self, "pattern", pattern)

        _severity_fraction(self.severity)
        if isinstance(self.severity, str):
            object.__setattr__(self, "severity", self.severity.strip().lower())

        if isinstance(self.duration_hours, bool):
            raise ValueError(f"duration_hours must be one of {sorted(ALLOWED_DURATIONS)}")
        duration = int(self.duration_hours)
        if duration != self.duration_hours or duration not in ALLOWED_DURATIONS:
            raise ValueError(f"duration_hours must be one of {sorted(ALLOWED_DURATIONS)}")
        object.__setattr__(self, "duration_hours", duration)
        object.__setattr__(self, "volume_stratum", _canonical_stratum(self.volume_stratum))

        kpis = tuple(str(kpi).strip().upper() for kpi in self.kpis)
        invalid_kpis = sorted(set(kpis).difference(SR_COLS))
        if invalid_kpis:
            raise ValueError(f"kpis may contain only {SR_COLS}; invalid: {invalid_kpis}")
        if len(kpis) != len(set(kpis)):
            raise ValueError("kpis must not contain duplicates")

        if pattern in {"volume_spike", "volume_drop"}:
            if kpis:
                raise ValueError(f"{pattern} does not accept success-rate kpis")
        elif pattern == "kpi_degradation":
            kpis = kpis or ("LSR_SR",)
            if len(kpis) != 1:
                raise ValueError("kpi_degradation requires exactly one success-rate KPI")
        elif pattern == "multi_kpi_degradation":
            kpis = kpis or SR_COLS
            if len(kpis) < 2:
                raise ValueError("multi_kpi_degradation requires at least two KPIs")
        else:
            kpis = kpis or SR_COLS
        object.__setattr__(self, "kpis", kpis)


@dataclass(frozen=True)
class InjectionResult:
    """Baseline, injected data, and separate synthetic ground truth."""

    baseline_data: pd.DataFrame
    injected_data: pd.DataFrame
    row_truth: pd.DataFrame
    event_truth: pd.DataFrame
    seed: int
    baseline_fingerprint: str


ROW_TRUTH_COLUMNS = (
    "event_id",
    "event_ordinal",
    "psap_id",
    "hour_start_utc",
    "offset_hour",
    "pattern",
    "severity",
    "severity_fraction",
    "duration_hours",
    "volume_stratum",
    "expected_behavior",
    "direction",
    "directional_model_support",
    "is_active",
    "has_numeric_change",
    "changed_columns",
    *tuple(f"baseline_{column.lower()}" for column in KPI_COLS),
    *tuple(f"injected_{column.lower()}" for column in KPI_COLS),
    *tuple(f"delta_{column.lower()}" for column in KPI_COLS),
)

EVENT_TRUTH_COLUMNS = (
    "event_id",
    "event_ordinal",
    "psap_id",
    "volume_stratum",
    "pattern",
    "severity",
    "severity_fraction",
    "duration_hours",
    "event_start_utc",
    "event_end_utc",
    "active_hours",
    "expected_behavior",
    "direction",
    "directional_model_support",
    "target_kpis",
    "changed_columns",
    "changed_row_count",
    "seed",
    "baseline_fingerprint",
)


def _stable_scalar(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return f"bool:{int(value)}"
    if isinstance(value, int):
        return f"int:{value}"
    if isinstance(value, float):
        return f"float:{value.hex()}"
    return f"{type(value).__name__}:{value}"


def _parse_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    try:
        parsed = pd.to_datetime(value, utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid timestamp: {value!r}") from exc
    if isinstance(parsed, pd.DatetimeIndex):
        raise ValueError(f"{name} must be one timestamp, not a sequence")
    timestamp = pd.Timestamp(parsed)
    if timestamp != timestamp.floor("h"):
        raise ValueError(f"{name} must be aligned to the start of an hour: {value!r}")
    return timestamp


def _validate_baseline(
    baseline: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, float]:
    if not isinstance(baseline, pd.DataFrame):
        raise TypeError("baseline must be a pandas DataFrame")
    if bool(baseline.columns.duplicated().any()):
        raise ValueError("baseline must not contain duplicate column names")
    missing = [column for column in REQUIRED_COLUMNS if column not in baseline.columns]
    if missing:
        raise ValueError(f"baseline is missing required columns: {missing}")
    if baseline.empty:
        raise ValueError("baseline must contain at least one hourly row")
    if baseline[ID_COL].isna().any():
        raise ValueError(f"{ID_COL} must not contain null values")

    try:
        timestamps = pd.to_datetime(baseline[TIME_COL], utc=True, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{TIME_COL} contains an invalid timestamp") from exc
    if timestamps.isna().any():
        raise ValueError(f"{TIME_COL} must not contain null timestamps")
    off_hour = timestamps != timestamps.dt.floor("h")
    if bool(off_hour.any()):
        raise ValueError(f"{TIME_COL} values must be aligned to the start of an hour")

    keys = pd.DataFrame(
        {
            "_psap": baseline[ID_COL].map(_stable_scalar),
            "_timestamp": timestamps,
        }
    )
    if bool(keys.duplicated().any()):
        raise ValueError(f"baseline contains duplicate ({ID_COL}, {TIME_COL}) rows")

    numeric = pd.DataFrame(index=baseline.index)
    for column in KPI_COLS:
        numeric[column] = pd.to_numeric(baseline[column], errors="coerce")
        values = numeric[column].to_numpy(dtype="float64")
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain only finite numeric values")
    if bool((numeric[VOLUME_COL] < 0).any()):
        raise ValueError(f"{VOLUME_COL} must be non-negative")

    sr_values = numeric[list(SR_COLS)].to_numpy(dtype="float64")
    rate_scale = 1.0 if float(sr_values.max()) <= 1.0 + 1e-12 else 100.0
    if float(sr_values.min()) < 0.0 or float(sr_values.max()) > rate_scale + 1e-9:
        raise ValueError(f"success rates must be between 0 and {rate_scale:g}")
    return pd.Series(timestamps, index=baseline.index), numeric, rate_scale


def _baseline_fingerprint(
    baseline: pd.DataFrame, timestamps: pd.Series, numeric: pd.DataFrame
) -> str:
    records: list[tuple[str, ...]] = []
    for position, index in enumerate(baseline.index):
        record = (
            _stable_scalar(baseline.iloc[position][ID_COL]),
            pd.Timestamp(timestamps.iloc[position]).isoformat(),
            *tuple(float(numeric.iloc[position][column]).hex() for column in KPI_COLS),
        )
        records.append(record)
    records.sort()
    payload = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _volume_strata(baseline: pd.DataFrame, numeric: pd.DataFrame) -> dict[str, str]:
    rows = pd.DataFrame(
        {
            "_psap_key": baseline[ID_COL].map(_stable_scalar).to_numpy(),
            "_volume": numeric[VOLUME_COL].to_numpy(dtype="float64"),
        }
    )
    medians = rows.groupby("_psap_key", sort=False)["_volume"].median()
    ordered = sorted(medians.items(), key=lambda item: (float(item[1]), item[0]))
    n_psaps = len(ordered)
    if n_psaps == 1:
        return {ordered[0][0]: "medium"}
    labels = ("low", "medium", "high")
    result: dict[str, str] = {}
    for position, (psap_key, _) in enumerate(ordered):
        bucket = int(round((position * 2) / (n_psaps - 1)))
        result[psap_key] = labels[bucket]
    return result


def _spec_payload(spec: FaultSpec) -> dict[str, Any]:
    return {
        "pattern": spec.pattern,
        "severity": spec.severity,
        "duration_hours": spec.duration_hours,
        "psap_id": None if spec.psap_id is None else _stable_scalar(spec.psap_id),
        "volume_stratum": spec.volume_stratum,
        "start_time": None
        if spec.start_time is None
        else _parse_timestamp(spec.start_time, name="start_time").isoformat(),
        "kpis": spec.kpis,
    }


def _resolve_event_window(
    baseline: pd.DataFrame,
    timestamps: pd.Series,
    strata: dict[str, str],
    spec: FaultSpec,
    *,
    ordinal: int,
    seed: int,
    reserved: set[tuple[str, pd.Timestamp]],
    allow_overlaps: bool,
) -> tuple[Any, str, pd.Timestamp, list[int]]:
    psap_keys = baseline[ID_COL].map(_stable_scalar).reset_index(drop=True)
    timestamp_values = timestamps.reset_index(drop=True)
    requested_key = None if spec.psap_id is None else _stable_scalar(spec.psap_id)
    requested_start = (
        None
        if spec.start_time is None
        else _parse_timestamp(spec.start_time, name="FaultSpec.start_time")
    )

    candidates: list[tuple[str, Any, pd.Timestamp, list[int]]] = []
    unique_keys = sorted(psap_keys.unique().tolist())
    for psap_key in unique_keys:
        if requested_key is not None and psap_key != requested_key:
            continue
        if spec.volume_stratum is not None and strata[psap_key] != spec.volume_stratum:
            continue

        positions = np.flatnonzero(psap_keys.to_numpy() == psap_key).tolist()
        by_timestamp = {pd.Timestamp(timestamp_values.iloc[pos]): pos for pos in positions}
        starts = [requested_start] if requested_start is not None else sorted(by_timestamp)
        for start in starts:
            if start is None:
                continue
            hours = [start + pd.Timedelta(hours=offset) for offset in range(spec.duration_hours)]
            if not all(hour in by_timestamp for hour in hours):
                continue
            if not allow_overlaps and any((psap_key, hour) in reserved for hour in hours):
                continue
            row_positions = [by_timestamp[hour] for hour in hours]
            psap_value = baseline.iloc[row_positions[0]][ID_COL]
            candidates.append((psap_key, psap_value, start, row_positions))

    if not candidates:
        target = "any PSAP" if spec.psap_id is None else f"PSAP {spec.psap_id!r}"
        stratum = "" if spec.volume_stratum is None else f" in {spec.volume_stratum!r} stratum"
        start = "" if requested_start is None else f" at {requested_start.isoformat()}"
        overlap_note = "" if allow_overlaps else " (non-overlapping)"
        raise ValueError(
            f"No contiguous {spec.duration_hours}-hour{overlap_note} window for "
            f"{target}{stratum}{start}"
        )

    spec_json = json.dumps(_spec_payload(spec), sort_keys=True, separators=(",", ":"))

    def candidate_order(candidate: tuple[str, Any, pd.Timestamp, list[int]]) -> str:
        psap_key, _, start, _ = candidate
        text = f"{seed}|{ordinal}|{spec_json}|{psap_key}|{start.isoformat()}"
        return sha256(text.encode("utf-8")).hexdigest()

    psap_key, psap_value, start, row_positions = min(candidates, key=candidate_order)
    return psap_value, strata[psap_key], start, row_positions


def _event_id(
    spec: FaultSpec,
    *,
    ordinal: int,
    seed: int,
    psap_id: Any,
    start: pd.Timestamp,
    fingerprint: str,
) -> str:
    payload = {
        "baseline": fingerprint,
        "ordinal": ordinal,
        "seed": seed,
        "resolved_psap_id": _stable_scalar(psap_id),
        "resolved_start": start.isoformat(),
        "spec": _spec_payload(spec),
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"SYN-{digest[:16].upper()}"


def _changed_columns(spec: FaultSpec) -> tuple[str, ...]:
    if spec.pattern in {"volume_spike", "volume_drop"}:
        return (VOLUME_COL,)
    if spec.pattern == "correlated_volume_sr":
        return (VOLUME_COL, *spec.kpis)
    return spec.kpis


def _apply_volume(value: float, fraction: float, *, increase: bool) -> float:
    if increase:
        injected = float(np.ceil(value * (1.0 + fraction)))
        if injected <= value:
            injected = value + 1.0
        return injected
    return max(0.0, float(np.floor(value * (1.0 - fraction))))


def _apply_rate(
    value: float,
    fraction: float,
    *,
    rate_scale: float,
    improve: bool,
) -> float:
    if improve:
        return float(np.clip(value + ((rate_scale - value) * fraction), 0.0, rate_scale))
    return float(np.clip(value * (1.0 - fraction), 0.0, rate_scale))


def inject_faults(
    baseline: pd.DataFrame,
    specs: Iterable[FaultSpec],
    *,
    seed: int = 42,
    allow_overlaps: bool = False,
) -> InjectionResult:
    """Inject deterministic events into a deep copy of hourly baseline data.

    Parameters
    ----------
    baseline:
        Untouched hourly PSAP data.  Duplicate PSAP/hour rows, off-hour
        timestamps, non-finite KPI values, and invalid rate ranges are rejected.
    specs:
        Event descriptions.  Mapping objects are accepted and converted to
        :class:`FaultSpec` for convenient configuration-driven use.
    seed:
        Controls deterministic PSAP/window selection and event IDs.  No global
        random state is read or modified.
    allow_overlaps:
        False by default so each truth row has one unambiguous generating event.

    Returns
    -------
    InjectionResult
        A preserved deep baseline copy, a separate injected frame with the same
        columns/order, row truth, event truth, and a baseline fingerprint.
    """

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    timestamps, numeric, rate_scale = _validate_baseline(baseline)
    original_fingerprint = _baseline_fingerprint(baseline, timestamps, numeric)
    baseline_copy = baseline.copy(deep=True)
    injected = baseline.copy(deep=True)
    for column in KPI_COLS:
        # Synthetic rates commonly become fractional even if an input CSV was
        # inferred as integer-valued.  Use a stable float dtype up front rather
        # than relying on pandas' deprecated implicit dtype widening.
        injected[column] = numeric[column].to_numpy(dtype="float64", copy=True)

    normalized_specs: list[FaultSpec] = []
    for spec in specs:
        if isinstance(spec, FaultSpec):
            normalized_specs.append(spec)
        elif isinstance(spec, Mapping):
            normalized_specs.append(FaultSpec(**spec))
        else:
            raise TypeError("each spec must be a FaultSpec or a dict of FaultSpec fields")

    strata = _volume_strata(baseline, numeric)
    reserved: set[tuple[str, pd.Timestamp]] = set()
    row_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []

    for ordinal, spec in enumerate(normalized_specs, start=1):
        psap_id, stratum, start, row_positions = _resolve_event_window(
            baseline,
            timestamps,
            strata,
            spec,
            ordinal=ordinal,
            seed=seed,
            reserved=reserved,
            allow_overlaps=allow_overlaps,
        )
        psap_key = _stable_scalar(psap_id)
        window_hours = [start + pd.Timedelta(hours=offset) for offset in range(spec.duration_hours)]
        if not allow_overlaps:
            reserved.update((psap_key, hour) for hour in window_hours)

        severity_label, severity_fraction = _severity_fraction(spec.severity)
        expected_behavior, direction, direction_support = _PATTERN_METADATA[spec.pattern]
        event_id = _event_id(
            spec,
            ordinal=ordinal,
            seed=seed,
            psap_id=psap_id,
            start=start,
            fingerprint=original_fingerprint,
        )
        event_columns = _changed_columns(spec)
        active_count = 0
        changed_row_count = 0
        injected_column_positions = {
            column: int(injected.columns.get_loc(column)) for column in event_columns
        }

        for offset, row_position in enumerate(row_positions):
            active = spec.pattern != "flapping" or offset % 2 == 0
            shape = (offset + 1) / spec.duration_hours if spec.pattern == "slow_burn" else 1.0
            effective_fraction = severity_fraction * shape if active else 0.0
            before = {
                column: float(numeric.iloc[row_position][column]) for column in KPI_COLS
            }

            if active:
                active_count += 1
                if spec.pattern == "volume_spike":
                    injected.iat[row_position, injected_column_positions[VOLUME_COL]] = _apply_volume(
                        before[VOLUME_COL], effective_fraction, increase=True
                    )
                elif spec.pattern == "volume_drop":
                    injected.iat[row_position, injected_column_positions[VOLUME_COL]] = _apply_volume(
                        before[VOLUME_COL], effective_fraction, increase=False
                    )
                elif spec.pattern == "correlated_volume_sr":
                    injected.iat[row_position, injected_column_positions[VOLUME_COL]] = _apply_volume(
                        before[VOLUME_COL], effective_fraction, increase=True
                    )
                    for column in spec.kpis:
                        injected.iat[row_position, injected_column_positions[column]] = _apply_rate(
                            before[column],
                            effective_fraction,
                            rate_scale=rate_scale,
                            improve=False,
                        )
                else:
                    improve = spec.pattern == "benign_improvement"
                    for column in spec.kpis:
                        injected.iat[row_position, injected_column_positions[column]] = _apply_rate(
                            before[column],
                            effective_fraction,
                            rate_scale=rate_scale,
                            improve=improve,
                        )

            after = {
                column: float(injected.iloc[row_position][column]) for column in KPI_COLS
            }
            changed = tuple(
                column
                for column in event_columns
                if not np.isclose(before[column], after[column], rtol=0.0, atol=1e-12)
            )
            has_numeric_change = bool(changed)
            changed_row_count += int(has_numeric_change)
            row_record: dict[str, Any] = {
                "event_id": event_id,
                "event_ordinal": ordinal,
                "psap_id": psap_id,
                "hour_start_utc": window_hours[offset],
                "offset_hour": offset,
                "pattern": spec.pattern,
                "severity": severity_label,
                "severity_fraction": severity_fraction,
                "duration_hours": spec.duration_hours,
                "volume_stratum": stratum,
                "expected_behavior": expected_behavior,
                "direction": direction,
                "directional_model_support": direction_support,
                "is_active": active,
                "has_numeric_change": has_numeric_change,
                "changed_columns": ",".join(changed),
            }
            for column in KPI_COLS:
                lower = column.lower()
                row_record[f"baseline_{lower}"] = before[column]
                row_record[f"injected_{lower}"] = after[column]
                row_record[f"delta_{lower}"] = after[column] - before[column]
            row_records.append(row_record)

        event_records.append(
            {
                "event_id": event_id,
                "event_ordinal": ordinal,
                "psap_id": psap_id,
                "volume_stratum": stratum,
                "pattern": spec.pattern,
                "severity": severity_label,
                "severity_fraction": severity_fraction,
                "duration_hours": spec.duration_hours,
                "event_start_utc": start,
                "event_end_utc": start + pd.Timedelta(hours=spec.duration_hours - 1),
                "active_hours": active_count,
                "expected_behavior": expected_behavior,
                "direction": direction,
                "directional_model_support": direction_support,
                "target_kpis": ",".join(spec.kpis),
                "changed_columns": ",".join(event_columns),
                "changed_row_count": changed_row_count,
                "seed": seed,
                "baseline_fingerprint": original_fingerprint,
            }
        )

    # Defensive proof that neither validation nor injection touched caller data.
    final_timestamps, final_numeric, _ = _validate_baseline(baseline)
    final_fingerprint = _baseline_fingerprint(baseline, final_timestamps, final_numeric)
    if final_fingerprint != original_fingerprint:  # pragma: no cover - defensive invariant
        raise RuntimeError("baseline changed during fault injection")

    row_truth = pd.DataFrame.from_records(row_records, columns=ROW_TRUTH_COLUMNS)
    event_truth = pd.DataFrame.from_records(event_records, columns=EVENT_TRUTH_COLUMNS)
    return InjectionResult(
        baseline_data=baseline_copy,
        injected_data=injected,
        row_truth=row_truth,
        event_truth=event_truth,
        seed=seed,
        baseline_fingerprint=original_fingerprint,
    )


def default_fault_suite(
    baseline: pd.DataFrame,
    *,
    severities: Sequence[str | float] = ("mild", "moderate", "severe"),
    durations: Sequence[int] = (1, 2, 3, 6),
    volume_strata: Sequence[str] = ("low", "medium", "high"),
) -> tuple[FaultSpec, ...]:
    """Return a compact suite covering simple, complex, and control patterns.

    Targets and starts remain unresolved so :func:`inject_faults` can place the
    suite deterministically while avoiding overlaps.  The baseline is validated
    here to fail early when the requested evaluation source is unsuitable.
    """

    _validate_baseline(baseline)
    if not severities:
        raise ValueError("severities must not be empty")
    if not durations:
        raise ValueError("durations must not be empty")
    if not volume_strata:
        raise ValueError("volume_strata must not be empty")

    for severity in severities:
        _severity_fraction(severity)
    normalized_durations: list[int] = []
    for duration in durations:
        if isinstance(duration, bool) or int(duration) != duration or int(duration) not in ALLOWED_DURATIONS:
            raise ValueError(f"durations may contain only {sorted(ALLOWED_DURATIONS)}")
        normalized_durations.append(int(duration))
    normalized_strata = [_canonical_stratum(value) for value in volume_strata]

    templates: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("volume_spike", ()),
        ("volume_drop", ()),
        ("kpi_degradation", ("LSR_SR",)),
        ("kpi_degradation", ("ASR_SR",)),
        ("kpi_degradation", ("BID_SR",)),
        ("kpi_degradation", ("CSR_SR",)),
        ("multi_kpi_degradation", SR_COLS),
        ("slow_burn", ("LSR_SR", "ASR_SR")),
        ("flapping", ("BID_SR", "CSR_SR")),
        ("correlated_volume_sr", SR_COLS),
        ("benign_improvement", SR_COLS),
    )
    longest_duration = max(normalized_durations)
    correlated_duration = 3 if 3 in normalized_durations else longest_duration
    suite: list[FaultSpec] = []
    for index, (pattern, kpis) in enumerate(templates):
        duration = normalized_durations[index % len(normalized_durations)]
        if pattern in {"slow_burn", "flapping"}:
            duration = longest_duration
        elif pattern == "correlated_volume_sr":
            duration = correlated_duration
        suite.append(
            FaultSpec(
                pattern=pattern,
                severity=severities[index % len(severities)],
                duration_hours=duration,
                volume_stratum=normalized_strata[index % len(normalized_strata)],
                kpis=kpis,
            )
        )
    return tuple(suite)


__all__ = [
    "ALLOWED_DURATIONS",
    "EVENT_TRUTH_COLUMNS",
    "FaultSpec",
    "InjectionResult",
    "KPI_COLS",
    "REQUIRED_COLUMNS",
    "ROW_TRUTH_COLUMNS",
    "SEVERITY_FRACTIONS",
    "default_fault_suite",
    "inject_faults",
]
