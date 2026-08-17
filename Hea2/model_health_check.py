"""Pure, label-free and synthetic-ground-truth health checks for PSAP KPI models.

The module deliberately performs no I/O.  It accepts scored pandas data frames and
returns a JSON-serialisable report.  Real production accuracy is *not* claimed:
the baseline alert rate is explicitly reported as an alert-burden proxy, while
precision/recall are restricted to controlled synthetic events.

The deployed model uses a strict ``score > threshold`` decision.  When score and
threshold columns are present this module always reconstructs that decision, even
if an explicit alert column is also supplied.  A mismatch with an explicit alert
column is reported as an integrity error.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import asdict, dataclass, fields
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


STATUS_RANK = {"PASS": 0, "UNKNOWN": 0, "WARN": 1, "FAIL": 2}


@dataclass(frozen=True)
class HealthCheckConfig:
    """Thresholds used to grade evidence.

    All thresholds are configurable because the production limits should
    ultimately be frozen from a chronological reference/calibration window.
    Defaults are conservative starting points, not estimates of real accuracy.
    """

    detection_tolerance_hours: float = 1.0
    episode_gap_hours: float = 1.5
    event_recall_pass: float = 0.80
    event_recall_fail: float = 0.60
    event_precision_pass: float = 0.80
    event_precision_fail: float = 0.60
    row_recall_pass: float = 0.75
    row_recall_fail: float = 0.50
    row_precision_pass: float = 0.75
    row_precision_fail: float = 0.50
    temporal_coverage_pass: float = 0.70
    temporal_coverage_fail: float = 0.40
    attributable_recall_pass: float = 0.70
    attributable_recall_fail: float = 0.50
    time_to_detection_pass_hours: float = 1.0
    time_to_detection_fail_hours: float = 3.0
    median_ratio_lift_pass: float = 0.10
    median_ratio_lift_fail: float = 0.0
    monotonicity_pass: float = 0.80
    monotonicity_fail: float = 0.50
    minimum_detectable_severity_recall: float = 0.80
    minimum_detectable_severity_pass: float = 2.0
    minimum_detectable_severity_fail: float = 3.0
    baseline_alerts_per_1000_pass: float = 50.0
    baseline_alerts_per_1000_fail: float = 100.0
    scoring_coverage_pass: float = 0.99
    scoring_coverage_fail: float = 0.95
    missing_component_pass: float = 0.0
    missing_component_fail: float = 0.05
    head_agreement_pass: float = 0.80
    head_agreement_fail: float = 0.60
    head_ratio_dispersion_pass: float = 0.50
    head_ratio_dispersion_fail: float = 1.00
    drift_psi_pass: float = 0.10
    drift_psi_fail: float = 0.25
    min_drift_group_rows: int = 12
    isolated_alert_fraction_pass: float = 0.50
    isolated_alert_fraction_fail: float = 0.80
    flapping_rate_pass: float = 0.25
    flapping_rate_fail: float = 0.50
    cluster_change_rate_pass: float = 0.02
    cluster_change_rate_fail: float = 0.10
    mapping_change_rate_pass: float = 0.02
    mapping_change_rate_fail: float = 0.10
    mapping_coverage_pass: float = 0.98
    mapping_coverage_fail: float = 0.90
    benign_new_alert_rate_pass: float = 0.01
    benign_new_alert_rate_fail: float = 0.05
    benign_score_increase_rate_pass: float = 0.10
    benign_score_increase_rate_fail: float = 0.30
    benign_score_delta_tolerance: float = 1e-6

    @classmethod
    def from_value(
        cls, value: Optional["HealthCheckConfig | Mapping[str, Any]"],
    ) -> "HealthCheckConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown health-check config keys: {unknown}")
        return cls(**dict(value))


ALIASES: dict[str, tuple[str, ...]] = {
    "psap": ("psap_id", "PSAP_ID", "member_psap_id"),
    "timestamp": (
        "timestamp", "TIMESTAMP", "eval_timestamp", "EVAL_TIMESTAMP",
        "eval_ts_utc", "EVAL_TS_UTC", "run_ts_utc", "RUN_TS_UTC",
        "hour_start_utc", "HOUR_START_UTC", "datetime", "DATETIME",
        "date_hour", "DATE_HOUR",
    ),
    "score": (
        "anomaly_score", "ANOMALY_SCORE", "score", "SCORE",
        "raw_score", "RAW_SCORE",
    ),
    "threshold": (
        "anomaly_thresh", "ANOMALY_THRESH", "anomaly_threshold",
        "ANOMALY_THRESHOLD", "threshold", "THRESHOLD",
        "raw_threshold", "RAW_THRESHOLD",
    ),
    "raw_alert": (
        "raw_alert", "RAW_ALERT", "is_anomaly", "IS_ANOMALY",
        "predicted_anomaly", "PREDICTED_ANOMALY", "alert", "ALERT",
        "raw_threshold_crossing", "RAW_THRESHOLD_CROSSING", "vote", "VOTE",
    ),
    "sanity_alert": (
        "sanity_eligible_alert", "SANITY_ELIGIBLE_ALERT", "eligible_alert",
        "ELIGIBLE_ALERT", "policy_alert", "POLICY_ALERT",
    ),
    "hard_pass": ("hard_sanity_pass", "HARD_SANITY_PASS"),
    "third_pass": ("third_level_pass", "THIRD_LEVEL_PASS"),
    "event": ("event_id", "EVENT_ID", "injection_id", "INJECTION_ID"),
    "row_id": (
        "evaluation_row_id", "EVALUATION_ROW_ID", "scenario_row_id",
        "SCENARIO_ROW_ID",
    ),
    "pattern": (
        "pattern", "PATTERN", "injection_type", "INJECTION_TYPE",
        "fault_type", "FAULT_TYPE",
    ),
    "severity": ("severity", "SEVERITY", "severity_level", "SEVERITY_LEVEL"),
    "expected": (
        "expected_behavior", "EXPECTED_BEHAVIOR", "expected", "EXPECTED",
    ),
    "should_detect": (
        "should_detect", "SHOULD_DETECT", "expected_alert", "EXPECTED_ALERT",
        "is_ground_truth_anomaly", "IS_GROUND_TRUTH_ANOMALY",
    ),
    "affected": ("affected", "AFFECTED", "is_affected", "IS_AFFECTED"),
    "active": ("is_active", "IS_ACTIVE", "active", "ACTIVE"),
    "numeric_change": (
        "has_numeric_change", "HAS_NUMERIC_CHANGE", "numeric_change",
        "NUMERIC_CHANGE",
    ),
    "directional_support": (
        "directional_model_support", "DIRECTIONAL_MODEL_SUPPORT",
        "model_directional_support", "MODEL_DIRECTIONAL_SUPPORT",
    ),
    "start": (
        "start_time", "START_TIME", "event_start", "EVENT_START",
        "event_start_utc", "EVENT_START_UTC", "start_timestamp", "START_TIMESTAMP",
    ),
    "end": (
        "end_time", "END_TIME", "event_end", "EVENT_END",
        "event_end_utc", "EVENT_END_UTC", "end_timestamp", "END_TIMESTAMP",
    ),
    "duration": ("duration_hours", "DURATION_HOURS", "duration", "DURATION"),
    "volume_stratum": (
        "volume_stratum", "VOLUME_STRATUM", "volume_band", "VOLUME_BAND",
    ),
    "target_kpis": ("target_kpis", "TARGET_KPIS", "kpis", "KPIS"),
    "tolerance": (
        "detection_tolerance_hours", "DETECTION_TOLERANCE_HOURS",
        "tolerance_hours", "TOLERANCE_HOURS",
    ),
    "cluster": (
        "cluster_id", "CLUSTER_ID", "cluster", "CLUSTER",
        "market_cluster", "MARKET_CLUSTER",
    ),
    "market": ("market", "MARKET", "market_name", "MARKET_NAME"),
    "mapping": (
        "cluster_head_psap_id", "CLUSTER_HEAD_PSAP_ID", "mapped_head_id",
        "MAPPED_HEAD_ID", "mapping_id", "MAPPING_ID",
    ),
    "cluster_mapping_found": (
        "cluster_mapping_found", "CLUSTER_MAPPING_FOUND",
        "has_cluster_mapping", "HAS_CLUSTER_MAPPING",
    ),
    "head_mapping_found": (
        "head_mapping_found", "HEAD_MAPPING_FOUND",
        "has_head_mapping", "HAS_HEAD_MAPPING",
    ),
    "head": ("head_id", "HEAD_ID", "detector_head_id", "DETECTOR_HEAD_ID"),
    "missing_head": ("missing_head", "MISSING_HEAD", "head_missing", "HEAD_MISSING"),
    "head_loaded": ("head_loaded", "HEAD_LOADED", "head_available", "HEAD_AVAILABLE"),
    "missing_detector": (
        "missing_detector", "MISSING_DETECTOR", "detector_missing",
        "DETECTOR_MISSING",
    ),
    "detector_loaded": (
        "detector_loaded", "DETECTOR_LOADED", "head_available", "HEAD_AVAILABLE",
    ),
    "head_scores": ("head_scores", "HEAD_SCORES", "per_head_scores", "PER_HEAD_SCORES"),
    "head_thresholds": (
        "head_thresholds", "HEAD_THRESHOLDS", "per_head_thresholds",
        "PER_HEAD_THRESHOLDS",
    ),
    "head_alerts": ("head_alerts", "HEAD_ALERTS", "per_head_alerts", "PER_HEAD_ALERTS"),
    "call_volume": ("call_volume", "CALL_VOLUME", "CALL_VOLUME_LIST"),
    "lsr": ("lsr_sr", "LSR_SR", "LSR_SR_LIST"),
    "asr": ("asr_sr", "ASR_SR", "ASR_SR_LIST"),
    "bid": ("bid_sr", "BID_SR", "BID_SR_LIST"),
    "csr": ("csr_sr", "CSR_SR", "CSR_SR_LIST"),
}


SEVERITY_MAP = {
    "NONE": 0.0, "CONTROL": 0.0, "BENIGN": 0.0,
    "LOW": 1.0, "MILD": 1.0,
    "MEDIUM": 2.0, "MODERATE": 2.0,
    "HIGH": 3.0, "SEVERE": 3.0,
    "CRITICAL": 4.0, "EXTREME": 4.0,
}


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _find_col(df: pd.DataFrame, kind: str) -> Optional[str]:
    if df is None or df.empty and not len(df.columns):
        return None
    exact = {str(c): c for c in df.columns}
    normalised = {_normalise_name(c): c for c in df.columns}
    for alias in ALIASES[kind]:
        if alias in exact:
            return exact[alias]
        found = normalised.get(_normalise_name(alias))
        if found is not None:
            return found
    return None


def _as_frame(value: Optional[pd.DataFrame]) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if not isinstance(value, pd.DataFrame):
        raise TypeError("Health-check inputs must be pandas DataFrames or None")
    return value.copy()


def _as_bool_scalar(value: Any) -> Optional[bool]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return bool(value)
    text = str(value).strip().upper()
    if text in {"1", "TRUE", "T", "YES", "Y", "PASS", "DETECT", "ALERT", "ANOMALY"}:
        return True
    if text in {"0", "FALSE", "F", "NO", "N", "FAIL", "NO_INCREASE", "CONTROL", "BENIGN"}:
        return False
    return None


def _bool_series(series: pd.Series) -> pd.Series:
    return series.map(_as_bool_scalar).astype("boolean")


def _key_string(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _normalise_pattern(value: Any) -> str:
    if value is None or pd.isna(value):
        return "UNKNOWN"
    return re.sub(r"[^A-Z0-9]+", "_", str(value).strip().upper()).strip("_") or "UNKNOWN"


def _normalise_expected(value: Any) -> Optional[str]:
    flag = _as_bool_scalar(value)
    if flag is True:
        return "DETECT"
    if flag is False:
        return "NO_INCREASE"
    text = str(value).strip().upper() if value is not None else ""
    if text in {"DETECT", "EXPECTED_DETECT", "BAD", "FAULT"}:
        return "DETECT"
    if text in {"NO_INCREASE", "BENIGN", "CONTROL", "IMPROVEMENT", "IGNORE"}:
        return "NO_INCREASE"
    return None


def _severity_number(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return SEVERITY_MAP.get(str(value).strip().upper())


def _parse_sequence(value: Any) -> list[Any]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, Mapping):
                    return list(parsed.values())
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
        return [part.strip() for part in text.split(",") if part.strip()]
    return [value]


def _prepare_scored(value: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = _as_frame(value)
    if df.empty and not len(df.columns):
        return df

    def assign(kind: str, target: str, transform=None) -> None:
        col = _find_col(df, kind)
        if col is not None:
            df[target] = transform(df[col]) if transform else df[col]

    assign("psap", "_psap", lambda s: s.map(_key_string))
    assign("timestamp", "_timestamp", lambda s: pd.to_datetime(s, errors="coerce", utc=True))
    assign("event", "_event", lambda s: s.map(_key_string))
    assign("row_id", "_row_id", lambda s: s.map(_key_string))
    assign("score", "_score", lambda s: pd.to_numeric(s, errors="coerce"))
    assign("threshold", "_threshold", lambda s: pd.to_numeric(s, errors="coerce"))
    assign("cluster", "_cluster", lambda s: s.map(_key_string))
    assign("market", "_market", lambda s: s.map(_key_string))
    assign("mapping", "_mapping", lambda s: s.map(_key_string))
    assign("cluster_mapping_found", "_cluster_mapping_found", _bool_series)
    assign("head_mapping_found", "_head_mapping_found", _bool_series)
    assign("head", "_head", lambda s: s.map(_key_string))
    assign("pattern", "_pattern", lambda s: s.map(_normalise_pattern))
    assign("severity", "_severity", lambda s: s.map(_severity_number))

    raw_col = _find_col(df, "raw_alert")
    if raw_col is not None:
        df["_explicit_raw_alert"] = _bool_series(df[raw_col])
    if "_score" in df and "_threshold" in df:
        valid = np.isfinite(df["_score"]) & np.isfinite(df["_threshold"]) & (df["_threshold"] > 0)
        derived = pd.Series(pd.NA, index=df.index, dtype="boolean")
        # Strict production semantics: equality is not an alert.
        derived.loc[valid] = df.loc[valid, "_score"] > df.loc[valid, "_threshold"]
        df["_raw_alert"] = derived
        df["_ratio"] = np.where(valid, df["_score"] / df["_threshold"], np.nan)
    elif raw_col is not None:
        df["_raw_alert"] = df["_explicit_raw_alert"]
        df["_ratio"] = np.nan
    else:
        df["_raw_alert"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
        df["_ratio"] = np.nan

    sanity_col = _find_col(df, "sanity_alert")
    hard_col = _find_col(df, "hard_pass")
    third_col = _find_col(df, "third_pass")
    if sanity_col is not None:
        df["_sanity_alert"] = _bool_series(df[sanity_col])
    elif hard_col is not None and third_col is not None:
        hard = _bool_series(df[hard_col])
        third = _bool_series(df[third_col])
        df["_sanity_alert"] = (df["_raw_alert"] & hard & third).astype("boolean")
    else:
        df["_sanity_alert"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    return df


def _logical_rows(prepared: pd.DataFrame) -> pd.DataFrame:
    """Collapse optional long per-head records into one evaluated PSAP-hour."""
    if prepared.empty:
        return prepared.copy()
    if "_psap" not in prepared or "_timestamp" not in prepared:
        return prepared.copy()
    valid = prepared[prepared["_psap"].notna() & prepared["_timestamp"].notna()].copy()
    if valid.empty:
        return valid
    keys = ["_psap", "_timestamp"]
    if "_event" in valid and valid["_event"].notna().any():
        keys.append("_event")
    if not valid.duplicated(keys).any():
        return valid.sort_values(keys).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for key, group in valid.groupby(keys, dropna=False, sort=False):
        key_values = key if isinstance(key, tuple) else (key,)
        row = {name: value for name, value in zip(keys, key_values)}
        ratios = pd.to_numeric(group.get("_ratio"), errors="coerce")
        ratios = ratios[np.isfinite(ratios)]
        if len(ratios):
            # Production multi-head ratio aggregation is mean(score/threshold).
            row["_ratio"] = float(ratios.mean())
            row["_score"] = float(ratios.mean())
            row["_threshold"] = 1.0
            row["_raw_alert"] = bool(row["_ratio"] > 1.0)
        else:
            row.update({"_ratio": np.nan, "_score": np.nan, "_threshold": np.nan, "_raw_alert": pd.NA})
        sanity = group.get("_sanity_alert")
        row["_sanity_alert"] = (
            bool(pd.Series(sanity).fillna(False).any()) if sanity is not None and pd.Series(sanity).notna().any()
            else pd.NA
        )
        for name in ("_cluster", "_market", "_mapping", "_pattern", "_severity"):
            if name in group:
                nonnull = group[name].dropna()
                row[name] = nonnull.iloc[0] if len(nonnull) else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _prepare_event_truth(value: Optional[pd.DataFrame], cfg: HealthCheckConfig) -> pd.DataFrame:
    df = _as_frame(value)
    if df.empty:
        return df

    def source(kind: str) -> Optional[pd.Series]:
        col = _find_col(df, kind)
        return df[col] if col is not None else None

    out = pd.DataFrame(index=df.index)
    event = source("event")
    out["event_id"] = event.map(_key_string) if event is not None else [f"event_{i}" for i in range(len(df))]
    psap = source("psap")
    out["psap"] = psap.map(_key_string) if psap is not None else None
    start = source("start")
    if start is None:
        start = source("timestamp")
    out["start"] = pd.to_datetime(start, errors="coerce", utc=True) if start is not None else pd.NaT
    end = source("end")
    out["end"] = pd.to_datetime(end, errors="coerce", utc=True) if end is not None else pd.NaT
    duration = source("duration")
    out["duration_hours"] = pd.to_numeric(duration, errors="coerce") if duration is not None else np.nan
    missing_end = out["end"].isna() & out["start"].notna()
    out.loc[missing_end, "end"] = out.loc[missing_end, "start"] + pd.to_timedelta(
        out.loc[missing_end, "duration_hours"].fillna(0), unit="h",
    )
    expected = source("expected")
    should = source("should_detect")
    if expected is not None:
        out["expected"] = expected.map(_normalise_expected)
    elif should is not None:
        out["expected"] = should.map(lambda x: "DETECT" if _as_bool_scalar(x) else "NO_INCREASE")
    else:
        out["expected"] = "DETECT"
    pattern = source("pattern")
    out["pattern"] = pattern.map(_normalise_pattern) if pattern is not None else "UNKNOWN"
    severity = source("severity")
    out["severity"] = severity.map(_severity_number) if severity is not None else np.nan
    volume = source("volume_stratum")
    out["volume_stratum"] = volume.map(_key_string) if volume is not None else None
    target_kpis = source("target_kpis")
    out["target_kpis"] = target_kpis.map(str) if target_kpis is not None else None
    directional = source("directional_support")
    out["directional_model_support"] = directional if directional is not None else None
    tolerance = source("tolerance")
    out["tolerance_hours"] = (
        pd.to_numeric(tolerance, errors="coerce").fillna(cfg.detection_tolerance_hours)
        if tolerance is not None else cfg.detection_tolerance_hours
    )
    return out.reset_index(drop=True)


def _prepare_row_truth(value: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = _as_frame(value)
    if df.empty:
        return df

    def source(kind: str) -> Optional[pd.Series]:
        col = _find_col(df, kind)
        return df[col] if col is not None else None

    out = pd.DataFrame(index=df.index)
    event = source("event")
    out["event_id"] = event.map(_key_string) if event is not None else None
    psap = source("psap")
    out["psap"] = psap.map(_key_string) if psap is not None else None
    timestamp = source("timestamp")
    out["timestamp"] = pd.to_datetime(timestamp, errors="coerce", utc=True) if timestamp is not None else pd.NaT
    should = source("should_detect")
    expected = source("expected")
    expected_normalised = expected.map(_normalise_expected) if expected is not None else None
    active = source("active")
    numeric_change = source("numeric_change")
    active_flag = _bool_series(active).fillna(False) if active is not None else pd.Series(True, index=out.index)
    numeric_flag = (
        _bool_series(numeric_change).fillna(False)
        if numeric_change is not None else pd.Series(True, index=out.index)
    )
    if should is not None:
        out["should_detect"] = _bool_series(should)
    elif expected is not None:
        # Flapping injections intentionally include inactive guard hours.  Those
        # rows are negatives, not missed positives.  Likewise a requested change
        # that clips to the original value has no numeric fault to detect.
        out["should_detect"] = (
            (expected_normalised == "DETECT") & active_flag & numeric_flag
        ).astype("boolean")
    else:
        out["should_detect"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["expected"] = (
        expected_normalised if expected is not None
        else out["should_detect"].map(lambda x: "DETECT" if x is True else "NO_INCREASE" if x is False else None)
    )
    out["is_active"] = active_flag.astype("boolean")
    out["has_numeric_change"] = numeric_flag.astype("boolean")
    affected = source("affected")
    out["affected"] = _bool_series(affected) if affected is not None else True
    pattern = source("pattern")
    out["pattern"] = pattern.map(_normalise_pattern) if pattern is not None else "UNKNOWN"
    severity = source("severity")
    out["severity"] = severity.map(_severity_number) if severity is not None else np.nan
    volume = source("volume_stratum")
    out["volume_stratum"] = volume.map(_key_string) if volume is not None else None
    directional = source("directional_support")
    out["directional_model_support"] = directional if directional is not None else None
    return out.reset_index(drop=True)


def _ratio_summary(values: Iterable[Any]) -> dict[str, Any]:
    series = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce")
    finite = series[np.isfinite(series)]
    if finite.empty:
        return {"count": 0, "mean": None, "std": None, "min": None, "q05": None,
                "q25": None, "median": None, "q75": None, "q95": None, "q99": None, "max": None}
    q = finite.quantile([0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "count": int(len(finite)), "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)), "min": float(finite.min()),
        "q05": float(q.loc[0.05]), "q25": float(q.loc[0.25]),
        "median": float(q.loc[0.50]), "q75": float(q.loc[0.75]),
        "q95": float(q.loc[0.95]), "q99": float(q.loc[0.99]),
        "max": float(finite.max()),
    }


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    return float(numerator / denominator) if denominator else None


def _f1(precision: Optional[float], recall: Optional[float]) -> Optional[float]:
    if precision is None or recall is None or precision + recall == 0:
        return 0.0 if precision == 0 or recall == 0 else None
    return float(2 * precision * recall / (precision + recall))


def _higher_status(value: Optional[float], pass_at: float, fail_below: float) -> str:
    if value is None or not np.isfinite(value):
        return "UNKNOWN"
    if value >= pass_at:
        return "PASS"
    if value >= fail_below:
        return "WARN"
    return "FAIL"


def _lower_status(value: Optional[float], pass_at: float, fail_above: float) -> str:
    if value is None or not np.isfinite(value):
        return "UNKNOWN"
    if value <= pass_at:
        return "PASS"
    if value <= fail_above:
        return "WARN"
    return "FAIL"


def _worst(*statuses: str) -> str:
    known = [s for s in statuses if s != "UNKNOWN"]
    if not known:
        return "UNKNOWN"
    return max(known, key=lambda s: STATUS_RANK[s])


def _check(name: str, status: str, message: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "message": message, "details": details}


def _logical_lookup(logical: pd.DataFrame) -> tuple[dict[tuple, dict], dict[tuple, dict]]:
    by_event: dict[tuple, dict] = {}
    by_time: dict[tuple, dict] = {}
    for _, row in logical.iterrows():
        psap = row.get("_psap")
        ts = row.get("_timestamp")
        if psap is None or pd.isna(ts):
            continue
        rec = row.to_dict()
        by_time[(psap, ts)] = rec if (psap, ts) not in by_time else _max_ratio_row(by_time[(psap, ts)], rec)
        event = row.get("_event")
        if event is not None and not pd.isna(event):
            by_event[(str(event), psap, ts)] = rec
    return by_event, by_time


def _max_ratio_row(first: dict, second: dict) -> dict:
    a = first.get("_ratio")
    b = second.get("_ratio")
    if b is not None and np.isfinite(b) and (a is None or not np.isfinite(a) or b > a):
        return second
    return first


def _lookup_truth_row(
    truth: pd.Series, by_event: Mapping[tuple, dict], by_time: Mapping[tuple, dict],
) -> Optional[dict]:
    event, psap, ts = truth.get("event_id"), truth.get("psap"), truth.get("timestamp")
    if event is not None and (str(event), psap, ts) in by_event:
        return by_event[(str(event), psap, ts)]
    return by_time.get((psap, ts))


def _row_detection_metrics(
    injected: pd.DataFrame, row_truth: pd.DataFrame, alert_col: str = "_raw_alert",
) -> dict[str, Any]:
    if row_truth.empty or "should_detect" not in row_truth:
        return {"available": False, "reason": "row ground truth is missing"}
    by_event, by_time = _logical_lookup(injected)
    records: list[dict[str, Any]] = []
    for _, truth in row_truth.iterrows():
        if pd.isna(truth.get("timestamp")) or truth.get("psap") is None or pd.isna(truth.get("should_detect")):
            continue
        scored = _lookup_truth_row(truth, by_event, by_time)
        predicted = None if scored is None else _as_bool_scalar(scored.get(alert_col))
        records.append({
            "actual": bool(truth["should_detect"]), "predicted": predicted,
            "pattern": truth.get("pattern", "UNKNOWN"), "event_id": truth.get("event_id"),
        })
    if not records:
        return {"available": False, "reason": "no ground-truth rows could be matched"}
    frame = pd.DataFrame(records)
    matched = frame[frame["predicted"].notna()].copy()
    if matched.empty:
        return {
            "available": True, "truth_rows": int(len(frame)), "scored_truth_rows": 0,
            "truth_scoring_coverage": 0.0, "tp": 0, "fp": 0, "tn": 0,
            "fn": int(frame["actual"].sum()), "precision": None, "recall": 0.0, "f1": None,
            "by_pattern": {},
        }
    actual = matched["actual"].astype(bool)
    predicted = matched["predicted"].astype(bool)
    tp = int((actual & predicted).sum())
    fp = int((~actual & predicted).sum())
    tn = int((~actual & ~predicted).sum())
    fn = int((actual & ~predicted).sum()) + int((frame["predicted"].isna() & frame["actual"]).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    breakdown: dict[str, Any] = {}
    for pattern, group in frame.groupby("pattern", dropna=False):
        group_matched = group[group["predicted"].notna()]
        positives = group[group["actual"]]
        detected = group_matched[group_matched["actual"] & group_matched["predicted"].astype(bool)]
        negatives = group[~group["actual"]]
        false_alerts = group_matched[~group_matched["actual"] & group_matched["predicted"].astype(bool)]
        breakdown[str(pattern)] = {
            "rows": int(len(group)), "positive_rows": int(len(positives)),
            "detected_positive_rows": int(len(detected)),
            "positive_row_recall": _safe_div(len(detected), len(positives)),
            "control_rows": int(len(negatives)), "control_alert_rows": int(len(false_alerts)),
        }
    return {
        "available": True, "truth_rows": int(len(frame)), "scored_truth_rows": int(len(matched)),
        "truth_scoring_coverage": float(len(matched) / len(frame)),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": _f1(precision, recall),
        "by_pattern": breakdown,
    }


def _alert_episodes(logical: pd.DataFrame, cfg: HealthCheckConfig, alert_col: str) -> list[dict[str, Any]]:
    if logical.empty or alert_col not in logical or "_psap" not in logical or "_timestamp" not in logical:
        return []
    alerted = logical[logical[alert_col].fillna(False).astype(bool)].copy()
    if alerted.empty:
        return []
    group_cols = ["_psap"]
    if "_event" in alerted and alerted["_event"].notna().any():
        group_cols.append("_event")
    episodes: list[dict[str, Any]] = []
    for group_key, group in alerted.groupby(group_cols, dropna=False, sort=False):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        key_map = dict(zip(group_cols, key_values))
        times = sorted(pd.Series(group["_timestamp"].dropna().unique()).tolist())
        if not times:
            continue
        start = previous = pd.Timestamp(times[0])
        points = 1
        for raw_time in times[1:]:
            current = pd.Timestamp(raw_time)
            gap = (current - previous).total_seconds() / 3600.0
            if gap > cfg.episode_gap_hours:
                episodes.append({
                    "psap": key_map["_psap"], "event_id": key_map.get("_event"),
                    "start": start, "end": previous, "alert_hours": points,
                })
                start, points = current, 1
            else:
                points += 1
            previous = current
        episodes.append({
            "psap": key_map["_psap"], "event_id": key_map.get("_event"),
            "start": start, "end": previous, "alert_hours": points,
        })
    return episodes


def _event_match_metrics(
    injected: pd.DataFrame, event_truth: pd.DataFrame, cfg: HealthCheckConfig,
    alert_col: str = "_raw_alert",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    detects = event_truth[event_truth.get("expected", pd.Series(index=event_truth.index, dtype=object)) == "DETECT"].copy()
    if detects.empty:
        return {"available": False, "reason": "no DETECT events in event ground truth"}, []
    episodes = _alert_episodes(injected, cfg, alert_col)
    used_episodes: set[int] = set()
    event_results: list[dict[str, Any]] = []
    for _, event in detects.iterrows():
        start, end = event.get("start"), event.get("end")
        tolerance = float(event.get("tolerance_hours", cfg.detection_tolerance_hours))
        if pd.isna(start) or pd.isna(end) or event.get("psap") is None:
            event_results.append({
                "event_id": event.get("event_id"), "psap": event.get("psap"),
                "pattern": event.get("pattern"), "severity": event.get("severity"),
                "duration_hours": event.get("duration_hours"),
                "volume_stratum": event.get("volume_stratum"),
                "target_kpis": event.get("target_kpis"),
                "directional_model_support": event.get("directional_model_support"),
                "valid": False, "detected": False, "time_to_detection_hours": None,
            })
            continue
        candidates: list[tuple[int, dict]] = []
        for index, episode in enumerate(episodes):
            same_event = (
                episode.get("event_id") is not None and event.get("event_id") is not None
                and str(episode["event_id"]) == str(event["event_id"])
            )
            if not same_event and episode["psap"] != event["psap"]:
                continue
            if episode.get("event_id") is not None and event.get("event_id") is not None and not same_event:
                continue
            allowed_end = end + pd.Timedelta(hours=tolerance)
            if episode["end"] >= start and episode["start"] <= allowed_end:
                candidates.append((index, episode))
        candidates.sort(key=lambda item: item[1]["start"])
        chosen = candidates[0] if candidates else None
        if chosen is not None:
            used_episodes.add(chosen[0])
            first = max(chosen[1]["start"], start)
            delay = max(0.0, (first - start).total_seconds() / 3600.0)
        else:
            delay = None
        event_results.append({
            "event_id": event.get("event_id"), "psap": event.get("psap"),
            "pattern": event.get("pattern"), "severity": event.get("severity"),
            "duration_hours": event.get("duration_hours"),
            "volume_stratum": event.get("volume_stratum"),
            "target_kpis": event.get("target_kpis"),
            "directional_model_support": event.get("directional_model_support"),
            "valid": True, "detected": chosen is not None,
            "time_to_detection_hours": delay,
        })

    valid = [result for result in event_results if result["valid"]]
    tp = sum(result["detected"] for result in valid)
    fn = len(valid) - tp
    fp = len(episodes) - len(used_episodes)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    delays = [r["time_to_detection_hours"] for r in valid if r["detected"]]
    pattern_breakdown: dict[str, Any] = {}
    severity_breakdown: dict[str, Any] = {}
    duration_breakdown: dict[str, Any] = {}
    volume_breakdown: dict[str, Any] = {}
    for label_name, target in (
        ("pattern", pattern_breakdown), ("severity", severity_breakdown),
        ("duration_hours", duration_breakdown), ("volume_stratum", volume_breakdown),
    ):
        labels = sorted({str(r.get(label_name)) for r in valid})
        for label in labels:
            group = [r for r in valid if str(r.get(label_name)) == label]
            detected = sum(r["detected"] for r in group)
            target[label] = {
                "events": len(group), "detected_events": detected,
                "recall": _safe_div(detected, len(group)),
                "missed_event_ids": [r["event_id"] for r in group if not r["detected"]],
            }
    cross_breakdown: dict[str, Any] = {}
    for result in valid:
        key = (
            str(result.get("pattern")), str(result.get("severity")),
            str(result.get("duration_hours")), str(result.get("volume_stratum")),
        )
        label = "|".join(key)
        item = cross_breakdown.setdefault(label, {
            "pattern": key[0], "severity": key[1], "duration_hours": key[2],
            "volume_stratum": key[3], "events": 0, "detected_events": 0,
            "missed_event_ids": [],
        })
        item["events"] += 1
        item["detected_events"] += int(result["detected"])
        if not result["detected"]:
            item["missed_event_ids"].append(result["event_id"])
    for item in cross_breakdown.values():
        item["recall"] = _safe_div(item["detected_events"], item["events"])
    metrics = {
        "available": True, "truth_events": len(valid), "invalid_truth_events": len(event_results) - len(valid),
        "predicted_alert_episodes": len(episodes), "tp_events": int(tp), "fp_episodes": int(fp),
        "fn_events": int(fn), "precision": precision, "recall": recall,
        "f1": _f1(precision, recall), "by_pattern": pattern_breakdown,
        "by_severity": severity_breakdown, "by_duration_hours": duration_breakdown,
        "by_volume_stratum": volume_breakdown,
        "by_pattern_severity_duration_volume": cross_breakdown,
        "time_to_detection_hours": {
            "count": len(delays), "mean": float(np.mean(delays)) if delays else None,
            "median": float(np.median(delays)) if delays else None,
            "p90": float(np.quantile(delays, 0.90)) if delays else None,
            "max": float(np.max(delays)) if delays else None,
        },
        "missed_event_ids": [r["event_id"] for r in valid if not r["detected"]],
    }
    return metrics, event_results


def _paired_rows(
    baseline: pd.DataFrame, injected: pd.DataFrame, row_truth: pd.DataFrame,
) -> pd.DataFrame:
    baseline_event, baseline_time = _logical_lookup(baseline)
    injected_event, injected_time = _logical_lookup(injected)
    records: list[dict[str, Any]] = []
    for _, truth in row_truth.iterrows():
        if truth.get("psap") is None or pd.isna(truth.get("timestamp")):
            continue
        base = _lookup_truth_row(truth, baseline_event, baseline_time)
        changed = _lookup_truth_row(truth, injected_event, injected_time)
        if base is None or changed is None:
            continue
        records.append({
            "event_id": truth.get("event_id"), "psap": truth.get("psap"),
            "timestamp": truth.get("timestamp"), "pattern": truth.get("pattern"),
            "severity": truth.get("severity"), "should_detect": truth.get("should_detect"),
            "expected": truth.get("expected"),
            "baseline_ratio": base.get("_ratio"), "injected_ratio": changed.get("_ratio"),
            "baseline_alert": _as_bool_scalar(base.get("_raw_alert")),
            "injected_alert": _as_bool_scalar(changed.get("_raw_alert")),
            "baseline_sanity_alert": _as_bool_scalar(base.get("_sanity_alert")),
            "injected_sanity_alert": _as_bool_scalar(changed.get("_sanity_alert")),
        })
    frame = pd.DataFrame(records)
    if not frame.empty:
        frame["ratio_lift"] = pd.to_numeric(frame["injected_ratio"], errors="coerce") - pd.to_numeric(
            frame["baseline_ratio"], errors="coerce",
        )
    return frame


def _score_lift_metrics(paired: pd.DataFrame) -> dict[str, Any]:
    if paired.empty:
        return {"available": False, "reason": "baseline/injected rows could not be paired"}
    positive = paired[paired["should_detect"] == True]  # noqa: E712
    lift = pd.to_numeric(positive.get("ratio_lift"), errors="coerce")
    lift = lift[np.isfinite(lift)]
    if lift.empty:
        return {"available": False, "reason": "no finite paired DETECT score ratios"}
    by_pattern: dict[str, Any] = {}
    for pattern, group in positive.groupby("pattern", dropna=False):
        values = pd.to_numeric(group["ratio_lift"], errors="coerce")
        values = values[np.isfinite(values)]
        by_pattern[str(pattern)] = {
            "rows": int(len(values)), "median_ratio_lift": float(values.median()) if len(values) else None,
            "positive_lift_rate": float((values > 0).mean()) if len(values) else None,
        }
    return {
        "available": True, "paired_detect_rows": int(len(lift)),
        "mean_ratio_lift": float(lift.mean()), "median_ratio_lift": float(lift.median()),
        "p25_ratio_lift": float(lift.quantile(0.25)), "p75_ratio_lift": float(lift.quantile(0.75)),
        "positive_lift_rate": float((lift > 0).mean()),
        # Counterfactual responsiveness gives partial credit to useful sub-threshold movement.
        "counterfactual_win_rate": float((lift > 0).mean()),
        "by_pattern": by_pattern,
    }


def _monotonicity_metrics(paired: pd.DataFrame) -> dict[str, Any]:
    if paired.empty:
        return {"available": False, "reason": "paired score lift is unavailable"}
    detect = paired[(paired["should_detect"] == True)].copy()  # noqa: E712
    detect["severity"] = pd.to_numeric(detect["severity"], errors="coerce")
    detect["ratio_lift"] = pd.to_numeric(detect["ratio_lift"], errors="coerce")
    detect = detect[np.isfinite(detect["severity"]) & np.isfinite(detect["ratio_lift"])]
    pattern_results: dict[str, Any] = {}
    correlations: list[float] = []
    for pattern, group in detect.groupby("pattern", dropna=False):
        levels = group.groupby("severity")["ratio_lift"].median().sort_index()
        if len(levels) < 2:
            continue
        severity_rank = pd.Series(levels.index, dtype=float).rank(method="average")
        response_rank = pd.Series(levels.values, dtype=float).rank(method="average")
        corr = float(severity_rank.corr(response_rank))
        diffs = np.diff(levels.values)
        inversion_rate = float((diffs < 0).mean()) if len(diffs) else None
        correlations.append(corr)
        pattern_results[str(pattern)] = {
            "severity_levels": {str(k): float(v) for k, v in levels.items()},
            "spearman": corr, "adjacent_inversion_rate": inversion_rate,
        }
    if not correlations:
        return {"available": False, "reason": "need at least two severities within a pattern"}
    return {
        "available": True, "patterns_evaluated": len(correlations),
        "median_pattern_spearman": float(np.median(correlations)),
        "minimum_pattern_spearman": float(np.min(correlations)),
        "by_pattern": pattern_results,
    }


def _minimum_detectable_severity(
    event_results: Sequence[dict[str, Any]], cfg: HealthCheckConfig,
) -> dict[str, Any]:
    """Find the lowest severity whose recall target holds at it and above.

    Requiring every observed higher severity to meet the target prevents a
    chance success at mild severity from hiding non-monotonic failure later.
    """
    valid = [
        result for result in event_results
        if result.get("valid") and _severity_number(result.get("severity")) is not None
    ]
    if not valid:
        return {"available": False, "reason": "no valid events with ordered severity"}

    def calculate(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        levels: dict[float, list[bool]] = {}
        for result in group:
            severity = _severity_number(result.get("severity"))
            if severity is not None:
                levels.setdefault(float(severity), []).append(bool(result.get("detected")))
        recalls = {
            level: float(np.mean(detected)) for level, detected in sorted(levels.items())
        }
        minimum = None
        ordered = sorted(recalls)
        for index, level in enumerate(ordered):
            if all(recalls[higher] >= cfg.minimum_detectable_severity_recall for higher in ordered[index:]):
                minimum = level
                break
        return {
            "severity_recall": {str(level): recall for level, recall in recalls.items()},
            "minimum_detectable_severity": minimum,
            "recall_target": cfg.minimum_detectable_severity_recall,
        }

    overall = calculate(valid)
    by_pattern = {
        str(pattern): calculate(group)
        for pattern, group in _group_records(valid, "pattern").items()
    }
    by_pattern_volume = {
        f"{pattern}|{volume}": calculate(group)
        for (pattern, volume), group in _group_records(valid, ("pattern", "volume_stratum")).items()
    }
    missing_patterns = [
        pattern for pattern, result in by_pattern.items()
        if result["minimum_detectable_severity"] is None
    ]
    observed_minima = [
        result["minimum_detectable_severity"] for result in by_pattern.values()
        if result["minimum_detectable_severity"] is not None
    ]
    return {
        "available": True, **overall, "by_pattern": by_pattern,
        "by_pattern_volume_stratum": by_pattern_volume,
        "patterns_without_detectable_severity": missing_patterns,
        "worst_pattern_minimum_detectable_severity": max(observed_minima) if observed_minima else None,
        "all_patterns_have_detectable_severity": not missing_patterns,
    }


def _group_records(
    records: Sequence[dict[str, Any]], keys: str | tuple[str, ...],
) -> dict[Any, list[dict[str, Any]]]:
    key_tuple = (keys,) if isinstance(keys, str) else keys
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for record in records:
        values = tuple(record.get(key) for key in key_tuple)
        group_key: Any = values[0] if len(values) == 1 else values
        grouped.setdefault(group_key, []).append(record)
    return grouped


def _event_temporal_coverage(
    event_results: Sequence[dict[str, Any]], row_truth: pd.DataFrame, injected: pd.DataFrame,
) -> dict[str, Any]:
    if row_truth.empty:
        return {"available": False, "reason": "row truth is required for duration coverage"}
    by_event, by_time = _logical_lookup(injected)
    per_event: dict[str, Any] = {}
    coverages: list[float] = []
    for result in event_results:
        if not result.get("valid"):
            continue
        event_id = result.get("event_id")
        rows = row_truth[(row_truth["event_id"] == event_id) & (row_truth["should_detect"] == True)]  # noqa: E712
        if rows.empty:
            continue
        detected = 0
        scored = 0
        for _, truth in rows.iterrows():
            row = _lookup_truth_row(truth, by_event, by_time)
            if row is None or _as_bool_scalar(row.get("_raw_alert")) is None:
                continue
            scored += 1
            detected += bool(_as_bool_scalar(row.get("_raw_alert")))
        coverage = _safe_div(detected, len(rows))
        if coverage is not None:
            coverages.append(coverage)
        per_event[str(event_id)] = {
            "affected_hours": int(len(rows)), "scored_hours": int(scored),
            "alerted_hours": int(detected), "temporal_coverage": coverage,
        }
    if not coverages:
        return {"available": False, "reason": "no event rows could be evaluated"}
    return {
        "available": True, "events": len(coverages),
        "mean_event_temporal_coverage": float(np.mean(coverages)),
        "median_event_temporal_coverage": float(np.median(coverages)),
        "fully_covered_event_rate": float(np.mean(np.asarray(coverages) == 1.0)),
        "by_event": per_event,
    }


def _counterfactual_event_metrics(
    event_truth: pd.DataFrame, event_results: Sequence[dict[str, Any]],
    baseline: pd.DataFrame,
) -> dict[str, Any]:
    if event_truth.empty or baseline.empty:
        return {"available": False, "reason": "baseline and event truth are required"}
    baseline_by_event, baseline_by_time = _logical_lookup(baseline)
    result_by_id = {str(r.get("event_id")): r for r in event_results if r.get("valid")}
    rows: list[dict[str, Any]] = []
    for _, event in event_truth[event_truth["expected"] == "DETECT"].iterrows():
        if pd.isna(event["start"]) or pd.isna(event["end"]):
            continue
        event_id = str(event["event_id"])
        baseline_alert = False
        # Baseline normally has no event ID, so time/PSAP matching is primary.
        for (psap, timestamp), record in baseline_by_time.items():
            if psap == event["psap"] and event["start"] <= timestamp <= event["end"]:
                baseline_alert = baseline_alert or bool(_as_bool_scalar(record.get("_raw_alert")) or False)
        result = result_by_id.get(event_id, {})
        rows.append({
            "event_id": event_id, "pattern": event["pattern"],
            "baseline_contaminated": baseline_alert,
            "injected_detected": bool(result.get("detected", False)),
        })
    if not rows:
        return {"available": False, "reason": "no valid events overlap baseline evidence"}
    frame = pd.DataFrame(rows)
    clean = frame[~frame["baseline_contaminated"]]
    attributable = int(clean["injected_detected"].sum())
    by_pattern: dict[str, Any] = {}
    for pattern, group in frame.groupby("pattern", dropna=False):
        group_clean = group[~group["baseline_contaminated"]]
        by_pattern[str(pattern)] = {
            "events": int(len(group)), "baseline_contaminated_events": int(group["baseline_contaminated"].sum()),
            "clean_events": int(len(group_clean)), "attributably_detected_events": int(group_clean["injected_detected"].sum()),
            "attributable_recall": _safe_div(group_clean["injected_detected"].sum(), len(group_clean)),
        }
    return {
        "available": True, "events": int(len(frame)),
        "baseline_contaminated_events": int(frame["baseline_contaminated"].sum()),
        "baseline_contamination_rate": float(frame["baseline_contaminated"].mean()),
        "clean_events": int(len(clean)), "attributably_detected_events": attributable,
        "attributable_event_recall": _safe_div(attributable, len(clean)),
        "by_pattern": by_pattern,
        "meaning": "Detection credit after excluding events already alerting in the untouched baseline.",
    }


def _benign_metrics(paired: pd.DataFrame, cfg: HealthCheckConfig) -> dict[str, Any]:
    if paired.empty:
        return {"available": False, "reason": "paired rows are unavailable"}
    # Inactive hours inside a DETECT/flapping event are guard negatives for row
    # precision, but they are not improvement-direction controls.
    controls = paired[paired["expected"] == "NO_INCREASE"].copy()
    if controls.empty:
        return {"available": False, "reason": "no NO_INCREASE controls"}
    controls["new_alert"] = (
        controls["injected_alert"].fillna(False).astype(bool)
        & ~controls["baseline_alert"].fillna(False).astype(bool)
    )
    lift = pd.to_numeric(controls["ratio_lift"], errors="coerce")
    finite = lift[np.isfinite(lift)]
    increase = finite > cfg.benign_score_delta_tolerance
    return {
        "available": True, "control_rows": int(len(controls)),
        "new_alert_rows": int(controls["new_alert"].sum()),
        "new_alert_rate": float(controls["new_alert"].mean()),
        "finite_score_pairs": int(len(finite)),
        "score_increase_rate": float(increase.mean()) if len(increase) else None,
        "median_ratio_delta": float(finite.median()) if len(finite) else None,
        "max_ratio_delta": float(finite.max()) if len(finite) else None,
    }


def _meta_find(metadata: Optional[Mapping[str, Any]], names: Sequence[str]) -> Any:
    if not isinstance(metadata, Mapping):
        return None
    wanted = {_normalise_name(name) for name in names}
    queue: list[Mapping[str, Any]] = [metadata]
    while queue:
        current = queue.pop(0)
        for key, value in current.items():
            if _normalise_name(key) in wanted:
                return value
            if isinstance(value, Mapping):
                queue.append(value)
    return None


def _coverage_metrics(
    baseline_prepared: pd.DataFrame, baseline: pd.DataFrame,
    injected_prepared: pd.DataFrame, injected: pd.DataFrame,
    metadata: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    def one(label: str, prepared: pd.DataFrame, logical: pd.DataFrame) -> dict[str, Any]:
        planned_meta = _meta_find(metadata, (f"{label}_planned_psap_hours",))
        scored_meta = _meta_find(metadata, (f"{label}_scored_psap_hours",))
        if label == "baseline":
            planned_meta = planned_meta if planned_meta is not None else _meta_find(metadata, ("planned_psap_hours", "planned"))
            scored_meta = scored_meta if scored_meta is not None else _meta_find(metadata, ("scored_psap_hours", "scored"))
        valid_scores = (
            np.isfinite(pd.to_numeric(logical.get("_score"), errors="coerce"))
            & np.isfinite(pd.to_numeric(logical.get("_threshold"), errors="coerce"))
            & (pd.to_numeric(logical.get("_threshold"), errors="coerce") > 0)
            if not logical.empty and "_score" in logical and "_threshold" in logical
            else pd.Series(False, index=logical.index)
        )
        planned = int(planned_meta) if planned_meta is not None else int(len(logical) or len(prepared))
        scored = int(scored_meta) if scored_meta is not None else int(valid_scores.sum())
        return {
            "planned_psap_hours": planned, "scored_psap_hours": scored,
            "coverage": _safe_div(scored, planned),
            "denominator_source": "metadata" if planned_meta is not None else "dataframe_rows",
        }
    return {
        "baseline": one("baseline", baseline_prepared, baseline),
        "injected": one("injected", injected_prepared, injected),
    }


def _component_availability(prepared: pd.DataFrame, metadata: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"available": False}
    for component in ("detector", "head"):
        missing_col = _find_col(prepared, f"missing_{component}")
        loaded_col = _find_col(prepared, f"{component}_loaded")
        missing_count_meta = _meta_find(metadata, (f"missing_{component}s", f"missing_{component}_count"))
        total_meta = _meta_find(metadata, (f"planned_{component}s", f"total_{component}s"))
        if missing_col is not None:
            flags = _bool_series(prepared[missing_col])
            missing, total = int(flags.fillna(False).sum()), int(flags.notna().sum())
        elif loaded_col is not None:
            flags = _bool_series(prepared[loaded_col])
            missing, total = int((~flags.fillna(True)).sum()), int(flags.notna().sum())
        elif missing_count_meta is not None:
            missing = int(missing_count_meta)
            total = int(total_meta) if total_meta is not None else None
        else:
            result[component] = {"available": False}
            continue
        result["available"] = True
        result[component] = {
            "available": True, "missing": missing, "total": total,
            "missing_rate": _safe_div(missing, total) if total is not None else None,
        }
    return result


def _head_consistency(prepared: pd.DataFrame) -> dict[str, Any]:
    groups: list[list[float]] = []
    score_col, threshold_col = _find_col(prepared, "head_scores"), _find_col(prepared, "head_thresholds")
    alert_col = _find_col(prepared, "head_alerts")
    if score_col is not None and threshold_col is not None:
        for scores_raw, thresholds_raw in zip(prepared[score_col], prepared[threshold_col]):
            scores, thresholds = _parse_sequence(scores_raw), _parse_sequence(thresholds_raw)
            ratios = [
                float(s) / float(t) for s, t in zip(scores, thresholds)
                if _is_finite_number(s) and _is_finite_number(t) and float(t) > 0
            ]
            if len(ratios) >= 2:
                groups.append(ratios)
    elif alert_col is not None:
        for raw in prepared[alert_col]:
            alerts = [_as_bool_scalar(x) for x in _parse_sequence(raw)]
            ratios = [1.000001 if x is True else 0.999999 for x in alerts if x is not None]
            if len(ratios) >= 2:
                groups.append(ratios)
    elif "_head" in prepared and "_psap" in prepared and "_timestamp" in prepared:
        key_cols = (
            ["_row_id"] if "_row_id" in prepared and prepared["_row_id"].notna().any()
            else ["_psap", "_timestamp"] + (["_event"] if "_event" in prepared else [])
        )
        for _, group in prepared.groupby(key_cols, dropna=False):
            if group["_head"].nunique(dropna=True) < 2:
                continue
            ratios = pd.to_numeric(group["_ratio"], errors="coerce")
            ratios = ratios[np.isfinite(ratios)].tolist()
            if len(ratios) >= 2:
                groups.append([float(x) for x in ratios])
    if not groups:
        return {"available": False, "reason": "no observations with at least two detector heads"}
    agreements, dispersions, all_agree = [], [], []
    for ratios in groups:
        alerts = np.asarray(ratios) > 1.0
        agreements.append(float(max(alerts.mean(), 1.0 - alerts.mean())))
        dispersions.append(float(np.std(ratios, ddof=0)))
        all_agree.append(bool(alerts.all() or (~alerts).all()))
    return {
        "available": True, "multi_head_psap_hours": len(groups),
        "mean_majority_agreement": float(np.mean(agreements)),
        "all_heads_agree_rate": float(np.mean(all_agree)),
        "mean_ratio_dispersion": float(np.mean(dispersions)),
        "p95_ratio_dispersion": float(np.quantile(dispersions, 0.95)),
        "head_count_distribution": _ratio_summary([len(x) for x in groups]),
    }


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _score_integrity(prepared: pd.DataFrame, logical: pd.DataFrame) -> dict[str, Any]:
    if prepared.empty or "_score" not in prepared or "_threshold" not in prepared:
        return {"available": False, "reason": "score and threshold columns are required"}
    score = pd.to_numeric(prepared["_score"], errors="coerce")
    threshold = pd.to_numeric(prepared["_threshold"], errors="coerce")
    finite_score = np.isfinite(score)
    valid_threshold = np.isfinite(threshold) & (threshold > 0)
    mismatch = 0
    explicit_rows = 0
    if "_explicit_raw_alert" in prepared:
        comparable = prepared["_explicit_raw_alert"].notna() & finite_score & valid_threshold
        explicit_rows = int(comparable.sum())
        mismatch = int((prepared.loc[comparable, "_explicit_raw_alert"].astype(bool) != (score[comparable] > threshold[comparable])).sum())
    raw_count = int(logical.get("_raw_alert", pd.Series(dtype=bool)).fillna(False).sum())
    sanity_available = "_sanity_alert" in logical and logical["_sanity_alert"].notna().any()
    sanity_count = int(logical["_sanity_alert"].fillna(False).sum()) if sanity_available else None
    return {
        "available": True, "rows": int(len(prepared)),
        "nonfinite_score_rows": int((~finite_score).sum()),
        "nonpositive_or_nonfinite_threshold_rows": int((~valid_threshold).sum()),
        "explicit_alert_comparison_rows": explicit_rows,
        "explicit_vs_strict_decision_mismatches": mismatch,
        "strict_threshold_semantics": "ANOMALY_SCORE > ANOMALY_THRESH",
        "raw_alerts": raw_count, "sanity_eligible_alerts": sanity_count,
        "raw_to_sanity_suppression_rate": (
            _safe_div(raw_count - sanity_count, raw_count) if sanity_count is not None else None
        ),
        "normalised_score_distribution": _ratio_summary(logical.get("_ratio", [])),
    }


def _baseline_alert_burden(logical: pd.DataFrame) -> dict[str, Any]:
    if logical.empty or "_raw_alert" not in logical:
        return {"available": False, "reason": "baseline scored rows are missing"}
    scored = logical[logical["_raw_alert"].notna()]
    if scored.empty:
        return {"available": False, "reason": "baseline alert decisions are missing"}
    raw = int(scored["_raw_alert"].astype(bool).sum())
    sanity_available = scored["_sanity_alert"].notna().any() if "_sanity_alert" in scored else False
    sanity = int(scored["_sanity_alert"].fillna(False).sum()) if sanity_available else None
    return {
        "available": True, "scored_psap_hours": int(len(scored)), "raw_alerts": raw,
        "raw_alerts_per_1000_psap_hours": float(raw * 1000.0 / len(scored)),
        "sanity_eligible_alerts": sanity,
        "sanity_alerts_per_1000_psap_hours": (
            float(sanity * 1000.0 / len(scored)) if sanity is not None else None
        ),
        "interpretation": "Alert-burden proxy only; this is not a false-positive rate without real labels.",
    }


def _alert_stability(logical: pd.DataFrame, cfg: HealthCheckConfig, alert_col: str = "_raw_alert") -> dict[str, Any]:
    if logical.empty or "_psap" not in logical or "_timestamp" not in logical or alert_col not in logical:
        return {"available": False, "reason": "chronological alert rows are missing"}
    episodes = _alert_episodes(logical, cfg, alert_col)
    if not episodes and not logical[alert_col].notna().any():
        return {"available": False, "reason": "alert decisions are missing"}
    isolated = [episode for episode in episodes if episode["alert_hours"] == 1]
    isolated_by_psap: dict[str, int] = {}
    for episode in isolated:
        isolated_by_psap[episode["psap"]] = isolated_by_psap.get(episode["psap"], 0) + 1
    repeated_isolated_psaps = sum(count >= 2 for count in isolated_by_psap.values())
    transitions = opportunities = 0
    for _, group in logical.sort_values("_timestamp").groupby("_psap"):
        group = group[group[alert_col].notna()].sort_values("_timestamp")
        if len(group) < 2:
            continue
        times = group["_timestamp"].tolist()
        alerts = group[alert_col].astype(bool).tolist()
        for index in range(1, len(group)):
            gap = (times[index] - times[index - 1]).total_seconds() / 3600.0
            if 0 < gap <= cfg.episode_gap_hours:
                opportunities += 1
                transitions += alerts[index] != alerts[index - 1]
    return {
        "available": True, "alert_episodes": len(episodes),
        "one_hour_alert_episodes": len(isolated),
        "one_hour_episode_fraction": _safe_div(len(isolated), len(episodes)),
        "psaps_with_repeated_one_hour_alerts": int(repeated_isolated_psaps),
        "adjacent_hour_state_transitions": int(transitions),
        "adjacent_hour_transition_opportunities": int(opportunities),
        "flapping_rate": _safe_div(transitions, opportunities),
    }


def _psi(reference_values: Iterable[Any], current_values: Iterable[Any]) -> Optional[float]:
    reference = pd.to_numeric(pd.Series(list(reference_values), dtype="object"), errors="coerce")
    current = pd.to_numeric(pd.Series(list(current_values), dtype="object"), errors="coerce")
    reference = reference[np.isfinite(reference)].to_numpy(dtype=float)
    current = current[np.isfinite(current)].to_numpy(dtype=float)
    if len(reference) < 2 or len(current) < 2:
        return None
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        low, high = min(reference.min(), current.min()), max(reference.max(), current.max())
        if low == high:
            return 0.0
        edges = np.linspace(low, high, 11)
    edges = np.r_[-np.inf, edges[1:-1], np.inf]
    ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
    cur_counts = np.histogram(current, bins=edges)[0].astype(float)
    epsilon = 1e-6
    ref_pct = np.clip(ref_counts / ref_counts.sum(), epsilon, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), epsilon, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _add_time_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "_timestamp" not in result:
        result["_time_bucket"] = None
        return result
    ts = result["_timestamp"]
    day = np.where(ts.dt.dayofweek >= 5, "WEEKEND", "WEEKDAY")
    block = (ts.dt.hour // 4) * 4
    result["_time_bucket"] = [f"{d}_H{int(h):02d}_{int(h + 3):02d}" for d, h in zip(day, block)]
    return result


def _drift_metrics(current: pd.DataFrame, reference_value: Any, cfg: HealthCheckConfig) -> dict[str, Any]:
    if isinstance(reference_value, Mapping):
        reference_value = reference_value.get("scored_dataframe", reference_value.get("dataframe"))
    if not isinstance(reference_value, pd.DataFrame):
        return {"available": False, "reason": "reference scored DataFrame was not supplied"}
    reference = _logical_rows(_prepare_scored(reference_value))
    if current.empty or reference.empty:
        return {"available": False, "reason": "current or reference scored rows are empty"}
    if "_ratio" not in current or "_ratio" not in reference:
        return {"available": False, "reason": "normalised score is unavailable"}
    current = _add_time_bucket(current)
    reference = _add_time_bucket(reference)
    global_psi = _psi(reference["_ratio"], current["_ratio"])
    by_dimension: dict[str, Any] = {}
    dimension_columns = {
        "cluster": "_cluster", "market": "_market", "psap": "_psap", "time_bucket": "_time_bucket",
    }
    all_group_values: list[float] = []
    for label, column in dimension_columns.items():
        if column not in current or column not in reference:
            by_dimension[label] = {"available": False}
            continue
        records: list[dict[str, Any]] = []
        common = sorted(set(current[column].dropna()) & set(reference[column].dropna()), key=str)
        for group_value in common:
            cur = current[current[column] == group_value]["_ratio"]
            ref = reference[reference[column] == group_value]["_ratio"]
            if len(cur) < cfg.min_drift_group_rows or len(ref) < cfg.min_drift_group_rows:
                continue
            value = _psi(ref, cur)
            if value is not None:
                records.append({
                    "group": str(group_value), "psi": value,
                    "current_rows": int(len(cur)), "reference_rows": int(len(ref)),
                })
                all_group_values.append(value)
        records.sort(key=lambda item: item["psi"], reverse=True)
        by_dimension[label] = {
            "available": bool(records), "groups_evaluated": len(records),
            "max_psi": max((x["psi"] for x in records), default=None),
            "p95_psi": float(np.quantile([x["psi"] for x in records], 0.95)) if records else None,
            "top_groups": records[:20],
        }
    kpi_drift: dict[str, Any] = {}
    for label, kind in (("CALL_VOLUME", "call_volume"), ("LSR", "lsr"), ("ASR", "asr"), ("BID", "bid"), ("CSR", "csr")):
        cur_col, ref_col = _find_col(current, kind), _find_col(reference, kind)
        if cur_col is not None and ref_col is not None:
            kpi_drift[label] = _psi(reference[ref_col], current[cur_col])
    p95 = float(np.quantile(all_group_values, 0.95)) if all_group_values else None
    assessed = max([x for x in (global_psi, p95) if x is not None], default=None)
    return {
        "available": global_psi is not None or bool(all_group_values),
        "global_normalised_score_psi": global_psi,
        "p95_group_normalised_score_psi": p95,
        "assessed_psi": assessed, "by_dimension": by_dimension,
        "global_kpi_psi": kpi_drift,
    }


def _cluster_mapping_stability(current: pd.DataFrame, reference_value: Any) -> dict[str, Any]:
    if isinstance(reference_value, Mapping):
        reference_value = reference_value.get("scored_dataframe", reference_value.get("dataframe"))
    if not isinstance(reference_value, pd.DataFrame):
        return {"available": False, "reason": "reference scored DataFrame was not supplied"}
    reference = _prepare_scored(reference_value)
    if current.empty or reference.empty or "_psap" not in current or "_psap" not in reference:
        return {"available": False, "reason": "PSAP identity is unavailable"}

    def latest(frame: pd.DataFrame) -> pd.DataFrame:
        sort = frame.sort_values("_timestamp") if "_timestamp" in frame else frame
        columns = [
            c for c in (
                "_psap", "_cluster", "_mapping", "_cluster_mapping_found",
                "_head_mapping_found",
            ) if c in sort
        ]
        return sort[columns].dropna(subset=["_psap"]).drop_duplicates("_psap", keep="last")

    cur, ref = latest(current), latest(reference)
    joined = ref.merge(cur, on="_psap", how="inner", suffixes=("_reference", "_current"))
    metrics: dict[str, Any] = {
        "available": False, "current_psaps": int(cur["_psap"].nunique()),
        "reference_psaps": int(ref["_psap"].nunique()), "compared_psaps": int(len(joined)),
    }
    if "_cluster_reference" in joined and "_cluster_current" in joined:
        comparable = joined["_cluster_reference"].notna() & joined["_cluster_current"].notna()
        metrics["cluster_compared_psaps"] = int(comparable.sum())
        metrics["cluster_change_rate"] = (
            float((joined.loc[comparable, "_cluster_reference"] != joined.loc[comparable, "_cluster_current"]).mean())
            if comparable.any() else None
        )
        metrics["available"] = metrics["available"] or comparable.any()
    if "_mapping_reference" in joined and "_mapping_current" in joined:
        comparable = joined["_mapping_reference"].notna() & joined["_mapping_current"].notna()
        metrics["mapping_compared_psaps"] = int(comparable.sum())
        metrics["mapping_change_rate"] = (
            float((joined.loc[comparable, "_mapping_reference"] != joined.loc[comparable, "_mapping_current"]).mean())
            if comparable.any() else None
        )
        metrics["available"] = metrics["available"] or comparable.any()
    if "_mapping" in cur:
        metrics["current_mapping_coverage"] = float(cur["_mapping"].notna().mean()) if len(cur) else None
        metrics["reference_mapping_coverage"] = float(ref["_mapping"].notna().mean()) if "_mapping" in ref and len(ref) else None
        metrics["available"] = True
    for label, column in (
        ("cluster_mapping", "_cluster_mapping_found"),
        ("head_mapping", "_head_mapping_found"),
    ):
        current_column = f"{column}_current"
        reference_column = f"{column}_reference"
        if current_column in joined:
            current_values = _bool_series(joined[current_column])
            reference_values = _bool_series(joined[reference_column]) if reference_column in joined else None
            current_coverage = float(current_values.fillna(False).mean()) if len(current_values) else None
            reference_coverage = (
                float(reference_values.fillna(False).mean())
                if reference_values is not None and len(reference_values) else None
            )
            metrics[f"current_{label}_coverage"] = current_coverage
            metrics[f"reference_{label}_coverage"] = reference_coverage
            metrics[f"{label}_coverage_change"] = (
                current_coverage - reference_coverage
                if current_coverage is not None and reference_coverage is not None else None
            )
            metrics["available"] = True
    if not metrics["available"]:
        metrics["reason"] = "cluster and mapping columns are unavailable"
    return metrics


def _directional_breakdown(event_metrics: Mapping[str, Any]) -> dict[str, Any]:
    patterns = event_metrics.get("by_pattern", {}) if event_metrics.get("available") else {}
    drop = patterns.get("CALL_VOLUME_DROP")
    if drop is None:
        for name, metrics in patterns.items():
            if "CALL" in name and "VOLUME" in name and "DROP" in name:
                drop = metrics
                break
    return {
        "production_directional_semantics": {
            "call_volume_spike": "bad direction",
            "service_rate_drop": "bad direction for LSR/ASR/BID/CSR",
            "call_volume_drop": "not targeted by directional score",
        },
        "call_volume_drop_result": drop,
        "note": (
            "CALL_VOLUME_DROP is shown separately because aggregate recall can hide this known directional blind spot."
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate_model_health(
    baseline_scored: Optional[pd.DataFrame],
    injected_scored: Optional[pd.DataFrame],
    event_ground_truth: Optional[pd.DataFrame] = None,
    row_ground_truth: Optional[pd.DataFrame] = None,
    *,
    paired_baseline_scored: Optional[pd.DataFrame] = None,
    per_head_scored: Optional[pd.DataFrame] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    reference: Any = None,
    config: Optional[HealthCheckConfig | Mapping[str, Any]] = None,
    # Backward-compatible aliases used by an earlier collector draft.
    event_truth: Optional[pd.DataFrame] = None,
    row_truth: Optional[pd.DataFrame] = None,
    collection_metadata: Optional[Mapping[str, Any]] = None,
    model_metadata: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Evaluate scored baseline and controlled-injection evidence.

    Parameters are in-memory objects only.  Missing optional evidence produces an
    ``UNKNOWN`` check instead of an exception or a fabricated score.
    """
    cfg = HealthCheckConfig.from_value(config)
    if event_ground_truth is None:
        event_ground_truth = event_truth
    if row_ground_truth is None:
        row_ground_truth = row_truth
    combined_metadata: dict[str, Any] = dict(metadata or {})
    if collection_metadata is not None:
        combined_metadata.setdefault("collection", dict(collection_metadata))
    if model_metadata is not None:
        combined_metadata.setdefault("model", dict(model_metadata))
    baseline_prepared = _prepare_scored(baseline_scored)
    injected_prepared = _prepare_scored(injected_scored)
    baseline = _logical_rows(baseline_prepared)
    injected = _logical_rows(injected_prepared)
    paired_baseline_prepared = _prepare_scored(
        paired_baseline_scored if paired_baseline_scored is not None else baseline_scored,
    )
    paired_baseline = _logical_rows(paired_baseline_prepared)
    per_head_prepared = _prepare_scored(per_head_scored) if per_head_scored is not None else injected_prepared
    events = _prepare_event_truth(event_ground_truth, cfg)
    rows = _prepare_row_truth(row_ground_truth)

    event_metrics, event_results = _event_match_metrics(injected, events, cfg)
    row_metrics = _row_detection_metrics(injected, rows, "_raw_alert")
    sanity_row_metrics = _row_detection_metrics(injected, rows, "_sanity_alert")
    paired = _paired_rows(paired_baseline, injected, rows)
    score_lift = _score_lift_metrics(paired)
    monotonicity = _monotonicity_metrics(paired)
    minimum_severity = _minimum_detectable_severity(event_results, cfg)
    temporal_coverage = _event_temporal_coverage(event_results, rows, injected)
    counterfactual_events = _counterfactual_event_metrics(events, event_results, baseline)
    benign = _benign_metrics(paired, cfg)
    burden = _baseline_alert_burden(baseline)
    coverage = _coverage_metrics(
        baseline_prepared, baseline, injected_prepared, injected, combined_metadata,
    )
    components = _component_availability(per_head_prepared, combined_metadata)
    heads = _head_consistency(per_head_prepared)
    baseline_integrity = _score_integrity(baseline_prepared, baseline)
    injected_integrity = _score_integrity(injected_prepared, injected)
    stability = _alert_stability(baseline, cfg)
    drift = _drift_metrics(baseline, reference, cfg)
    mapping_stability = _cluster_mapping_stability(baseline_prepared, reference)

    checks: list[dict[str, Any]] = []
    if event_metrics.get("available"):
        status = _worst(
            _higher_status(event_metrics.get("recall"), cfg.event_recall_pass, cfg.event_recall_fail),
            _higher_status(event_metrics.get("precision"), cfg.event_precision_pass, cfg.event_precision_fail),
        )
        checks.append(_check(
            "synthetic_event_detection", status,
            "Precision/recall against controlled synthetic event ground truth only.",
            precision=event_metrics.get("precision"), recall=event_metrics.get("recall"), f1=event_metrics.get("f1"),
        ))
    else:
        checks.append(_check("synthetic_event_detection", "UNKNOWN", event_metrics.get("reason", "event evidence missing")))

    if row_metrics.get("available"):
        status = _worst(
            _higher_status(row_metrics.get("recall"), cfg.row_recall_pass, cfg.row_recall_fail),
            _higher_status(row_metrics.get("precision"), cfg.row_precision_pass, cfg.row_precision_fail),
        )
        checks.append(_check(
            "synthetic_row_detection", status,
            "PSAP-hour precision/recall against controlled row truth.",
            precision=row_metrics.get("precision"), recall=row_metrics.get("recall"), f1=row_metrics.get("f1"),
        ))
    else:
        checks.append(_check("synthetic_row_detection", "UNKNOWN", row_metrics.get("reason", "row evidence missing")))

    ttd = event_metrics.get("time_to_detection_hours", {}).get("median") if event_metrics.get("available") else None
    checks.append(_check(
        "time_to_detection", _lower_status(ttd, cfg.time_to_detection_pass_hours, cfg.time_to_detection_fail_hours),
        "Median delay from injected event start to first strict threshold crossing.", median_hours=ttd,
    ))

    lift_value = score_lift.get("median_ratio_lift") if score_lift.get("available") else None
    checks.append(_check(
        "score_lift", _higher_status(lift_value, cfg.median_ratio_lift_pass, cfg.median_ratio_lift_fail),
        "Paired increase in score/threshold ratio on DETECT rows.", median_ratio_lift=lift_value,
        counterfactual_win_rate=score_lift.get("counterfactual_win_rate"),
    ))

    monotonic_value = monotonicity.get("median_pattern_spearman") if monotonicity.get("available") else None
    checks.append(_check(
        "severity_monotonicity", _higher_status(monotonic_value, cfg.monotonicity_pass, cfg.monotonicity_fail),
        "Within-pattern rank relationship between injected severity and response lift.",
        median_pattern_spearman=monotonic_value,
    ))

    if not minimum_severity.get("available"):
        minimum_status = "UNKNOWN"
    elif not minimum_severity.get("all_patterns_have_detectable_severity"):
        minimum_status = "FAIL"
    else:
        minimum_status = _lower_status(
            minimum_severity.get("worst_pattern_minimum_detectable_severity"),
            cfg.minimum_detectable_severity_pass,
            cfg.minimum_detectable_severity_fail,
        )
    checks.append(_check(
        "minimum_detectable_severity", minimum_status,
        "Lowest severity where event recall reaches target at that and every higher observed severity.",
        worst_pattern_minimum_detectable_severity=minimum_severity.get(
            "worst_pattern_minimum_detectable_severity"
        ),
        patterns_without_detectable_severity=minimum_severity.get(
            "patterns_without_detectable_severity", []
        ),
    ))

    temporal_value = temporal_coverage.get("mean_event_temporal_coverage") if temporal_coverage.get("available") else None
    checks.append(_check(
        "event_temporal_coverage", _higher_status(
            temporal_value, cfg.temporal_coverage_pass, cfg.temporal_coverage_fail,
        ),
        "Added metric: fraction of each event's affected PSAP-hours that remain detected, beyond one-hit event recall.",
        mean_event_temporal_coverage=temporal_value,
    ))

    attributable = counterfactual_events.get("attributable_event_recall") if counterfactual_events.get("available") else None
    checks.append(_check(
        "counterfactual_detection_gain", _higher_status(
            attributable, cfg.attributable_recall_pass, cfg.attributable_recall_fail,
        ),
        "Added causal guard: detection credit excludes events already alerting in untouched baseline.",
        attributable_event_recall=attributable,
        baseline_contamination_rate=counterfactual_events.get("baseline_contamination_rate"),
    ))

    burden_value = burden.get("raw_alerts_per_1000_psap_hours") if burden.get("available") else None
    checks.append(_check(
        "baseline_alert_burden_proxy", _lower_status(
            burden_value, cfg.baseline_alerts_per_1000_pass, cfg.baseline_alerts_per_1000_fail,
        ),
        "Untouched baseline alerts per 1,000 PSAP-hours; not a false-positive rate.",
        raw_alerts_per_1000_psap_hours=burden_value,
        sanity_alerts_per_1000_psap_hours=burden.get("sanity_alerts_per_1000_psap_hours"),
    ))

    coverage_values = [coverage[k].get("coverage") for k in ("baseline", "injected")]
    coverage_status = _worst(*[
        _higher_status(v, cfg.scoring_coverage_pass, cfg.scoring_coverage_fail) for v in coverage_values
    ])
    checks.append(_check(
        "planned_vs_scored_coverage", coverage_status,
        "Coverage preserves the denominator of every planned PSAP-hour.", **coverage,
    ))

    component_statuses: list[str] = []
    for component in ("detector", "head"):
        evidence = components.get(component, {})
        rate = evidence.get("missing_rate") if evidence.get("available") else None
        component_statuses.append(_lower_status(rate, cfg.missing_component_pass, cfg.missing_component_fail))
    checks.append(_check(
        "detector_head_availability", _worst(*component_statuses),
        "Missing detector/head evidence; UNKNOWN when the collector did not supply it.", **components,
    ))

    head_status = _worst(
        _higher_status(heads.get("mean_majority_agreement"), cfg.head_agreement_pass, cfg.head_agreement_fail),
        _lower_status(heads.get("p95_ratio_dispersion"), cfg.head_ratio_dispersion_pass, cfg.head_ratio_dispersion_fail),
    ) if heads.get("available") else "UNKNOWN"
    checks.append(_check(
        "multi_head_consistency", head_status,
        "Agreement and score/threshold dispersion where at least two heads scored the same PSAP-hour.", **heads,
    ))

    invalid = sum(
        int(metrics.get("nonfinite_score_rows", 0))
        + int(metrics.get("nonpositive_or_nonfinite_threshold_rows", 0))
        + int(metrics.get("explicit_vs_strict_decision_mismatches", 0))
        for metrics in (baseline_integrity, injected_integrity) if metrics.get("available")
    )
    integrity_available = baseline_integrity.get("available") or injected_integrity.get("available")
    checks.append(_check(
        "score_threshold_integrity", "PASS" if integrity_available and invalid == 0 else "FAIL" if integrity_available else "UNKNOWN",
        "Scores must be finite, thresholds positive, and decisions must use strict score > threshold.",
        invalid_or_mismatched_rows=invalid,
    ))

    drift_value = drift.get("assessed_psi") if drift.get("available") else None
    checks.append(_check(
        "reference_drift", _lower_status(drift_value, cfg.drift_psi_pass, cfg.drift_psi_fail),
        "PSI compares untouched current evidence with the optional frozen reference.", assessed_psi=drift_value,
    ))

    stability_status = _worst(
        _lower_status(
            stability.get("one_hour_episode_fraction"),
            cfg.isolated_alert_fraction_pass, cfg.isolated_alert_fraction_fail,
        ),
        _lower_status(stability.get("flapping_rate"), cfg.flapping_rate_pass, cfg.flapping_rate_fail),
    ) if stability.get("available") else "UNKNOWN"
    checks.append(_check(
        "alert_stability", stability_status,
        "Repeated isolated one-hour episodes and adjacent-hour alert/no-alert flipping.", **stability,
    ))

    mapping_statuses = []
    if mapping_stability.get("available"):
        mapping_statuses.extend([
            _lower_status(mapping_stability.get("cluster_change_rate"), cfg.cluster_change_rate_pass, cfg.cluster_change_rate_fail),
            _lower_status(mapping_stability.get("mapping_change_rate"), cfg.mapping_change_rate_pass, cfg.mapping_change_rate_fail),
            _higher_status(mapping_stability.get("current_mapping_coverage"), cfg.mapping_coverage_pass, cfg.mapping_coverage_fail),
            _higher_status(
                mapping_stability.get("current_cluster_mapping_coverage"),
                cfg.mapping_coverage_pass, cfg.mapping_coverage_fail,
            ),
            _higher_status(
                mapping_stability.get("current_head_mapping_coverage"),
                cfg.mapping_coverage_pass, cfg.mapping_coverage_fail,
            ),
            _lower_status(
                abs(mapping_stability["cluster_mapping_coverage_change"])
                if mapping_stability.get("cluster_mapping_coverage_change") is not None else None,
                cfg.cluster_change_rate_pass, cfg.cluster_change_rate_fail,
            ),
            _lower_status(
                abs(mapping_stability["head_mapping_coverage_change"])
                if mapping_stability.get("head_mapping_coverage_change") is not None else None,
                cfg.mapping_change_rate_pass, cfg.mapping_change_rate_fail,
            ),
        ])
    checks.append(_check(
        "cluster_mapping_stability", _worst(*mapping_statuses) if mapping_statuses else "UNKNOWN",
        "Cluster membership, detector-head mapping, and mapping coverage versus reference.", **mapping_stability,
    ))

    benign_status = _worst(
        _lower_status(benign.get("new_alert_rate"), cfg.benign_new_alert_rate_pass, cfg.benign_new_alert_rate_fail),
        _lower_status(
            benign.get("score_increase_rate"),
            cfg.benign_score_increase_rate_pass, cfg.benign_score_increase_rate_fail,
        ),
    ) if benign.get("available") else "UNKNOWN"
    checks.append(_check(
        "benign_controls", benign_status,
        "Improvement-direction controls should create no new alert and should not increase score.", **benign,
    ))

    sanity_available = sanity_row_metrics.get("available") and sanity_row_metrics.get("scored_truth_rows", 0) > 0
    checks.append(_check(
        "raw_vs_sanity_policy", "PASS" if sanity_available else "UNKNOWN",
        "Raw model alerts and post-sanity eligible alerts are reported separately; evaluation does not change production policy.",
        raw_row_recall=row_metrics.get("recall"), sanity_row_recall=sanity_row_metrics.get("recall") if sanity_available else None,
    ))

    fail_count = sum(check["status"] == "FAIL" for check in checks)
    warn_count = sum(check["status"] == "WARN" for check in checks)
    pass_count = sum(check["status"] == "PASS" for check in checks)
    unknown_count = sum(check["status"] == "UNKNOWN" for check in checks)
    overall = "FAIL" if fail_count else "WARN" if warn_count else "PASS" if pass_count else "UNKNOWN"

    report = {
        "schema_version": "1.0",
        "overall_status": overall,
        "scope": {
            "real_production_accuracy_available": False,
            "synthetic_precision_recall_scope": "controlled injections only",
            "baseline_alert_rate_scope": "alert-burden proxy only",
            "decision_rule": "strict score > threshold",
        },
        "summary": {
            "pass": pass_count, "warn": warn_count, "fail": fail_count, "unknown": unknown_count,
        },
        "checks": checks,
        "metrics": {
            "synthetic_event_detection": event_metrics,
            "synthetic_row_detection": row_metrics,
            "sanity_eligible_row_detection": sanity_row_metrics,
            "time_to_detection": event_metrics.get("time_to_detection_hours", {}),
            "score_lift": score_lift,
            "severity_monotonicity": monotonicity,
            "minimum_detectable_severity": minimum_severity,
            "event_temporal_coverage": temporal_coverage,
            "counterfactual_detection_gain": counterfactual_events,
            "directional_failure_breakdown": _directional_breakdown(event_metrics),
            "benign_controls": benign,
            "baseline_alert_burden_proxy": burden,
            "planned_vs_scored_coverage": coverage,
            "component_availability": components,
            "multi_head_consistency": heads,
            "baseline_score_threshold_health": baseline_integrity,
            "injected_score_threshold_health": injected_integrity,
            "reference_drift": drift,
            "alert_stability": stability,
            "cluster_mapping_stability": mapping_stability,
        },
        "metadata": combined_metadata,
        "config": asdict(cfg),
    }
    return _json_safe(report)


# Concise alias for orchestration code.
run_health_checks = evaluate_model_health


__all__ = ["HealthCheckConfig", "evaluate_model_health", "run_health_checks"]
