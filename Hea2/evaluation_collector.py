"""Chronological replay and synthetic-challenge collector for PSAP KPI models.

This module is deliberately read-only with respect to Oracle, MLflow, and the
production outage tables.  It reuses the existing KPI loader/SQL definitions,
scores a frozen chronological window with pinned local artifacts, creates an
independent synthetic challenge campaign, and writes evidence files for the
pure health checker.

No production thresholds, mappings, models, or input rows are modified.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
import pandas as pd

from .synthetic_faults import FaultSpec, inject_faults


LOG = logging.getLogger("psap_model_evaluation")

EVALUATION_VERSION = "1.1.0"
TIME_COL = "HOUR_START_UTC"
ID_COL = "PSAP_ID"
KPI_COLS = ("CALL_VOLUME", "LSR_SR", "ASR_SR", "BID_SR", "CSR_SR")
SR_COLS = KPI_COLS[1:]

TIME_ALIASES = ("HOUR_START_UTC", "EVAL_TS_UTC", "RUN_TS_UTC")
KPI_ALIASES = {
    "CALL_VOLUME_LIST": "CALL_VOLUME",
    "LSR_SR_LIST": "LSR_SR",
    "ASR_SR_LIST": "ASR_SR",
    "BID_SR_LIST": "BID_SR",
    "CSR_SR_LIST": "CSR_SR",
}

CAMPAIGN_TEMPLATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("volume_spike", ()),
    ("volume_drop", ()),
    ("kpi_degradation", ("LSR_SR",)),
    ("kpi_degradation", ("ASR_SR",)),
    ("kpi_degradation", ("BID_SR",)),
    ("kpi_degradation", ("CSR_SR",)),
    ("multi_kpi_degradation", SR_COLS),
    # More difficult temporal/combined challenges.
    ("slow_burn", ("LSR_SR", "ASR_SR")),
    ("flapping", ("BID_SR", "CSR_SR")),
    ("correlated_volume_sr", SR_COLS),
    # Negative control: a directional detector should not increase its score.
    ("benign_improvement", SR_COLS),
)


class ReplayScorer(Protocol):
    """Minimal boundary implemented by :class:`ProductionArtifactScorer`."""

    def score(self, frame: pd.DataFrame) -> Any:
        """Return an object with aggregate, per_head, and metadata attributes."""


@dataclass(frozen=True)
class CampaignConfig:
    severities: tuple[str, ...] = ("mild", "moderate", "severe")
    durations: tuple[int, ...] = (1, 2, 3, 6)
    volume_strata: tuple[str, ...] = ("low", "medium", "high")
    replicates_per_cell: int = 1
    psaps_per_stratum: int = 3
    seed: int = 42
    scenarios: tuple[FaultSpec, ...] | None = None

    def __post_init__(self) -> None:
        if self.scenarios is None:
            if not self.severities:
                raise ValueError("severities must not be empty")
            if not self.durations or any(int(x) not in {1, 2, 3, 6} for x in self.durations):
                raise ValueError("durations may contain only 1, 2, 3, and 6 hours")
            if not self.volume_strata or any(x not in {"low", "medium", "high"} for x in self.volume_strata):
                raise ValueError("volume_strata may contain only low, medium, and high")
        else:
            scenarios = tuple(self.scenarios)
            if not scenarios:
                raise ValueError("scenarios must contain at least one FaultSpec")
            if any(not isinstance(spec, FaultSpec) for spec in scenarios):
                raise TypeError("scenarios may contain only FaultSpec values")
            if self.replicates_per_cell != 1:
                raise ValueError(
                    "replicates_per_cell must be 1 with explicit scenarios; "
                    "repeat --scenario to request another independent test"
                )
            object.__setattr__(self, "scenarios", scenarios)
        if self.replicates_per_cell < 1:
            raise ValueError("replicates_per_cell must be >= 1")
        if self.psaps_per_stratum < 1:
            raise ValueError("psaps_per_stratum must be >= 1")


@dataclass(frozen=True)
class CollectedFrame:
    data: pd.DataFrame
    quality: dict[str, Any]


@dataclass(frozen=True)
class CampaignData:
    injected_rows: pd.DataFrame
    row_truth: pd.DataFrame
    event_truth: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EvaluationArtifacts:
    baseline_scored: pd.DataFrame
    paired_baseline_scored: pd.DataFrame
    injected_scored: pd.DataFrame
    per_head_scored: pd.DataFrame
    row_truth: pd.DataFrame
    event_truth: pd.DataFrame
    evidence: dict[str, Any]


def _utc_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a valid timestamp: {value!r}") from exc
    return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        stamp = pd.Timestamp(value)
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        return stamp.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, set)) else False:
        return None
    return value


def normalize_source_frame(
    source: pd.DataFrame,
    *,
    start_utc: Any | None = None,
    end_utc: Any | None = None,
) -> CollectedFrame:
    """Normalize the canonical SQL output without hiding data-quality defects."""

    if not isinstance(source, pd.DataFrame):
        raise TypeError("source must be a pandas DataFrame")
    if source.empty:
        raise ValueError("source contains no PSAP KPI rows")

    frame = source.copy(deep=True)
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    time_candidates = [column for column in TIME_ALIASES if column in frame.columns]
    if not time_candidates:
        raise ValueError(f"source requires one timestamp column from {TIME_ALIASES}")
    if time_candidates[0] != TIME_COL:
        frame = frame.rename(columns={time_candidates[0]: TIME_COL})
    for old, new in KPI_ALIASES.items():
        if old in frame.columns and new not in frame.columns:
            frame = frame.rename(columns={old: new})

    missing = [column for column in (TIME_COL, ID_COL, *KPI_COLS) if column not in frame.columns]
    if missing:
        raise ValueError(f"source is missing required columns: {missing}")

    source_rows = len(frame)
    parsed_time = pd.to_datetime(frame[TIME_COL], utc=True, errors="coerce")
    invalid_timestamp_rows = int(parsed_time.isna().sum())
    off_hour_rows = int((parsed_time.dropna() != parsed_time.dropna().dt.floor("h")).sum())
    frame[TIME_COL] = parsed_time

    if start_utc is not None:
        start = _utc_timestamp(start_utc, name="start_utc")
        frame = frame.loc[frame[TIME_COL] >= start].copy()
    else:
        start = frame[TIME_COL].min()
    if end_utc is not None:
        end = _utc_timestamp(end_utc, name="end_utc")
        frame = frame.loc[frame[TIME_COL] < end].copy()
    else:
        end = frame[TIME_COL].max() + pd.Timedelta(hours=1)
    if pd.isna(start) or pd.isna(end) or end <= start:
        raise ValueError("evaluation window must satisfy start_utc < end_utc")

    numeric = frame.loc[:, KPI_COLS].apply(pd.to_numeric, errors="coerce")
    nonfinite_mask = ~np.isfinite(numeric.to_numpy(dtype="float64")).all(axis=1)
    invalid_psap_mask = frame[ID_COL].isna() | (pd.to_numeric(frame[ID_COL], errors="coerce") == -1)
    negative_volume_mask = numeric["CALL_VOLUME"] < 0
    rate_range_mask = ((numeric.loc[:, SR_COLS] < 0) | (numeric.loc[:, SR_COLS] > 100)).any(axis=1)
    valid_mask = ~(
        frame[TIME_COL].isna()
        | nonfinite_mask
        | invalid_psap_mask
        | negative_volume_mask
    )

    valid = frame.loc[valid_mask].copy()
    valid.loc[:, KPI_COLS] = numeric.loc[valid_mask, KPI_COLS].to_numpy()
    # Production inference applies this exact clipping before feature scoring.
    valid.loc[:, SR_COLS] = valid.loc[:, SR_COLS].clip(lower=0.0, upper=100.0)
    valid[TIME_COL] = valid[TIME_COL].dt.floor("h")

    duplicate_mask = valid.duplicated([ID_COL, TIME_COL], keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    if duplicate_rows:
        sample = valid.loc[duplicate_mask, [ID_COL, TIME_COL]].head(5).to_dict("records")
        raise ValueError(
            "source contains duplicate PSAP-hour rows; aggregation would change model semantics. "
            f"duplicate_rows={duplicate_rows}, sample={sample}"
        )
    if valid.empty:
        raise ValueError("no valid PSAP KPI rows remain after validation")

    expected_hours = pd.date_range(start, end, freq="h", inclusive="left")
    observed_hours = pd.DatetimeIndex(valid[TIME_COL].drop_duplicates().sort_values())
    missing_hours = expected_hours.difference(observed_hours)
    distinct_psaps = int(valid[ID_COL].nunique())
    planned_psap_hours = int(distinct_psaps * len(expected_hours))
    observed_psap_hours = int(len(valid))

    success_count_violations: dict[str, int] = {}
    for prefix in ("LSR", "ASR", "BID", "CSR"):
        calls, successes = f"{prefix}_CALLS", f"{prefix}_SUCCESS"
        if calls in valid.columns and successes in valid.columns:
            call_values = pd.to_numeric(valid[calls], errors="coerce")
            success_values = pd.to_numeric(valid[successes], errors="coerce")
            success_count_violations[prefix] = int((success_values > call_values).sum())

    valid = valid.sort_values([TIME_COL, ID_COL], kind="stable").reset_index(drop=True)
    valid["EVALUATION_ROW_ID"] = [f"BASE-{index:010d}" for index in range(len(valid))]
    quality = {
        "window_start_utc": start,
        "window_end_utc": end,
        "source_rows": source_rows,
        "rows_after_window_filter": int(len(frame)),
        "valid_observed_psap_hours": observed_psap_hours,
        "invalid_timestamp_rows": invalid_timestamp_rows,
        "off_hour_rows": off_hour_rows,
        "nonfinite_kpi_rows": int(nonfinite_mask.sum()),
        "invalid_psap_rows": int(invalid_psap_mask.sum()),
        "negative_volume_rows": int(negative_volume_mask.sum()),
        "success_rate_range_violation_rows_clipped_like_production": int(rate_range_mask.sum()),
        "duplicate_psap_hour_rows": duplicate_rows,
        "distinct_psaps": distinct_psaps,
        "expected_hours": int(len(expected_hours)),
        "observed_hours": int(len(observed_hours)),
        "missing_hour_count": int(len(missing_hours)),
        "missing_hours_sample": list(missing_hours[:24]),
        "planned_psap_hours_grid": planned_psap_hours,
        "absent_psap_hours_in_grid": max(0, planned_psap_hours - observed_psap_hours),
        "success_greater_than_calls": success_count_violations,
    }
    return CollectedFrame(valid, quality)


def collect_source_data(
    *,
    input_csv: str | Path | None = None,
    start_utc: Any | None = None,
    end_utc: Any | None = None,
    profile: str = "PRD",
    market: str = "ALL",
    sqlite_path: str | Path | None = None,
    allow_cache: bool = False,
) -> CollectedFrame:
    """Collect from CSV or reuse ``psap_utils.data_loader.load_psap_kpis``."""

    if input_csv is not None:
        source = pd.read_csv(Path(input_csv))
        return normalize_source_frame(source, start_utc=start_utc, end_utc=end_utc)
    if start_utc is None or end_utc is None:
        raise ValueError("start_utc and end_utc are required when input_csv is omitted")

    # Lazy import keeps local CSV evaluation independent of Oracle packages.
    from psap_kpi.psap_utils.data_loader import load_psap_kpis

    kwargs: dict[str, Any] = {
        "start_date": _utc_timestamp(start_utc, name="start_utc").strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": _utc_timestamp(end_utc, name="end_utc").strftime("%Y-%m-%d %H:%M:%S"),
        "profile": profile,
        "market": market,
        # Evaluation defaults to source refresh so a partial/cross-scope cache
        # cannot silently become the test denominator.
        "force_refresh": not allow_cache,
        "require_complete_cache": True,
        "use_cache_write": True,
        "oracle_upsert_enabled": False,
        "cache_table_name": (
            "PSAP_KPI_HISTORY_"
            + re.sub(r"[^A-Z0-9]+", "_", f"{profile}_{market}".upper()).strip("_")
        ),
    }
    if sqlite_path is not None:
        kwargs["sqlite_path"] = sqlite_path
    source = load_psap_kpis(**kwargs)
    return normalize_source_frame(source, start_utc=start_utc, end_utc=end_utc)


def _rank_volume_strata(frame: pd.DataFrame) -> pd.DataFrame:
    medians = (
        frame.groupby(ID_COL, dropna=False)["CALL_VOLUME"]
        .median()
        .rename("MEDIAN_CALL_VOLUME")
        .reset_index()
        .sort_values(["MEDIAN_CALL_VOLUME", ID_COL], kind="stable")
        .reset_index(drop=True)
    )
    if len(medians) == 1:
        medians["VOLUME_STRATUM"] = "medium"
        return medians
    labels = ("low", "medium", "high")
    medians["VOLUME_STRATUM"] = [
        labels[int(round(position * 2 / (len(medians) - 1)))]
        for position in range(len(medians))
    ]
    return medians


def select_campaign_population(
    baseline: pd.DataFrame,
    *,
    psaps_per_stratum: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select representative PSAP histories while retaining all baseline scoring."""

    ranked = _rank_volume_strata(baseline)
    selected_parts: list[pd.DataFrame] = []
    for _, group in ranked.groupby("VOLUME_STRATUM", sort=False):
        group = group.reset_index(drop=True)
        centre = float(group["MEDIAN_CALL_VOLUME"].median())
        chosen = (
            group.assign(_DISTANCE=(group["MEDIAN_CALL_VOLUME"] - centre).abs())
            .sort_values(["_DISTANCE", ID_COL], kind="stable")
            .head(psaps_per_stratum)
            .drop(columns="_DISTANCE")
        )
        selected_parts.append(chosen)
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else ranked.iloc[0:0]
    population = baseline.loc[baseline[ID_COL].isin(selected[ID_COL])].copy()
    # Recalculate strata using the injector's rank rule on this compact cohort.
    cohort_rank = _rank_volume_strata(population)
    selected = selected.drop(columns="VOLUME_STRATUM").merge(
        cohort_rank[[ID_COL, "VOLUME_STRATUM"]], on=ID_COL, how="left"
    )
    return population.reset_index(drop=True), selected


def build_campaign_specs(config: CampaignConfig) -> tuple[FaultSpec, ...]:
    """Return exact requested scenarios, or build the default Cartesian campaign."""

    if config.scenarios is not None:
        return config.scenarios

    specs: list[FaultSpec] = []
    for pattern, kpis in CAMPAIGN_TEMPLATES:
        for severity in config.severities:
            for duration in config.durations:
                for stratum in config.volume_strata:
                    for _ in range(config.replicates_per_cell):
                        specs.append(
                            FaultSpec(
                                pattern=pattern,
                                severity=severity,
                                duration_hours=duration,
                                volume_stratum=stratum,
                                kpis=kpis,
                            )
                        )
    return tuple(specs)


def build_synthetic_campaign(
    baseline: pd.DataFrame,
    config: CampaignConfig,
) -> CampaignData:
    """Inject each event independently and return only its challenge rows."""

    population, representatives = select_campaign_population(
        baseline, psaps_per_stratum=config.psaps_per_stratum
    )
    specs = build_campaign_specs(config)
    injected_parts: list[pd.DataFrame] = []
    truth_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []

    for ordinal, spec in enumerate(specs, start=1):
        scenario_seed = config.seed + ordinal - 1
        try:
            result = inject_faults(population, [spec], seed=scenario_seed)
        except ValueError as exc:
            skipped.append({"ordinal": ordinal, "spec": asdict(spec), "reason": str(exc)})
            continue

        row_truth = result.row_truth.copy()
        event_truth = result.event_truth.copy()
        event_id = str(event_truth.iloc[0]["event_id"])
        event_truth["scenario_ordinal"] = ordinal
        row_truth["scenario_ordinal"] = ordinal
        row_truth["should_detect"] = (
            row_truth["expected_behavior"].eq("DETECT")
            & row_truth["is_active"].astype(bool)
            & row_truth["has_numeric_change"].astype(bool)
        )

        keys = row_truth[["psap_id", "hour_start_utc", "offset_hour"]].copy()
        keys["hour_start_utc"] = pd.to_datetime(keys["hour_start_utc"], utc=True)
        injected = result.injected_data.copy()
        injected[TIME_COL] = pd.to_datetime(injected[TIME_COL], utc=True)
        event_rows = injected.merge(
            keys,
            left_on=[ID_COL, TIME_COL],
            right_on=["psap_id", "hour_start_utc"],
            how="inner",
            validate="one_to_one",
        ).drop(columns=["psap_id", "hour_start_utc"])
        event_rows["EVENT_ID"] = event_id
        event_rows["SCENARIO_ORDINAL"] = ordinal
        event_rows["EVALUATION_ROW_ID"] = [
            f"{event_id}-{int(offset):02d}" for offset in event_rows["offset_hour"]
        ]
        event_rows = event_rows.drop(columns="offset_hour")

        injected_parts.append(event_rows)
        truth_parts.append(row_truth)
        event_parts.append(event_truth)

    injected_rows = pd.concat(injected_parts, ignore_index=True) if injected_parts else baseline.iloc[0:0].copy()
    row_truth = pd.concat(truth_parts, ignore_index=True) if truth_parts else pd.DataFrame()
    event_truth = pd.concat(event_parts, ignore_index=True) if event_parts else pd.DataFrame()
    metadata = {
        "configured_scenarios": len(specs),
        "created_scenarios": int(len(event_truth)),
        "skipped_scenarios": len(skipped),
        "skipped_sample": skipped[:25],
        "challenge_rows": int(len(injected_rows)),
        "representative_psaps": representatives.to_dict("records"),
        "config": asdict(config),
    }
    return CampaignData(injected_rows, row_truth, event_truth, metadata)


def _score_frames(scorer: ReplayScorer, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    result = scorer.score(frame)
    aggregate = getattr(result, "aggregate", None)
    per_head = getattr(result, "per_head", None)
    metadata = getattr(result, "metadata", {})
    if not isinstance(aggregate, pd.DataFrame) or not isinstance(per_head, pd.DataFrame):
        raise TypeError("scorer.score() must return aggregate and per_head DataFrames")
    return aggregate, per_head, dict(metadata or {})


def run_evaluation(
    collected: CollectedFrame,
    scorer: ReplayScorer,
    *,
    campaign_config: CampaignConfig | None = None,
    health_config: Any | None = None,
    reference_scored: pd.DataFrame | None = None,
    training_cutoff_utc: Any | None = None,
) -> EvaluationArtifacts:
    """Run untouched replay, independent injections, and pure health checks."""

    campaign_config = campaign_config or CampaignConfig()
    baseline = collected.data.copy(deep=True)
    if training_cutoff_utc is None:
        raise ValueError(
            "training_cutoff_utc is required to prove the evaluation window was untouched"
        )
    training_cutoff = _utc_timestamp(
        training_cutoff_utc, name="training_cutoff_utc"
    )
    first_evaluation_hour = pd.to_datetime(
        baseline[TIME_COL], utc=True, errors="raise"
    ).min()
    if first_evaluation_hour <= training_cutoff:
        raise ValueError(
            "chronological leakage: first evaluation hour "
            f"{first_evaluation_hour.isoformat()} must be after training cutoff "
            f"{training_cutoff.isoformat()}"
        )
    baseline_scored, baseline_heads, baseline_model_meta = _score_frames(scorer, baseline)

    campaign = build_synthetic_campaign(baseline, campaign_config)
    if campaign.injected_rows.empty:
        injected_scored = pd.DataFrame()
        injected_heads = pd.DataFrame()
        injected_model_meta: dict[str, Any] = {}
    else:
        injected_scored, injected_heads, injected_model_meta = _score_frames(
            scorer, campaign.injected_rows
        )

    # Pair every injected event-hour with the score of the *same untouched*
    # PSAP-hour.  Full baseline replay remains separate and is still the only
    # denominator used for production alert-burden/stability metrics.
    if not campaign.row_truth.empty:
        pair_keys = campaign.row_truth[
            ["event_id", "psap_id", "hour_start_utc", "offset_hour"]
        ].copy()
        pair_keys["hour_start_utc"] = pd.to_datetime(
            pair_keys["hour_start_utc"], utc=True
        )
        paired_baseline_input = baseline.drop(columns="EVALUATION_ROW_ID").merge(
            pair_keys,
            left_on=[ID_COL, TIME_COL],
            right_on=["psap_id", "hour_start_utc"],
            how="inner",
            validate="many_to_many",
        ).drop(columns=["psap_id", "hour_start_utc"])
        paired_baseline_input["EVENT_ID"] = paired_baseline_input["event_id"].astype(str)
        paired_baseline_input["EVALUATION_ROW_ID"] = [
            f"BASE-{event_id}-{int(offset):02d}"
            for event_id, offset in zip(
                paired_baseline_input["event_id"], paired_baseline_input["offset_hour"]
            )
        ]
        paired_baseline_input = paired_baseline_input.drop(
            columns=["event_id", "offset_hour"]
        )
        paired_baseline_scored, paired_baseline_heads, _ = _score_frames(
            scorer, paired_baseline_input
        )
    else:
        paired_baseline_scored = baseline_scored.iloc[0:0].copy()
        paired_baseline_heads = baseline_heads.iloc[0:0].copy()

    baseline_heads = baseline_heads.copy()
    baseline_heads["EVALUATION_SET"] = "UNTOUCHED_BASELINE"
    paired_baseline_heads = paired_baseline_heads.copy()
    if not paired_baseline_heads.empty:
        paired_baseline_heads["EVALUATION_SET"] = "PAIRED_UNTOUCHED_BASELINE"
    injected_heads = injected_heads.copy()
    if not injected_heads.empty:
        injected_heads["EVALUATION_SET"] = "SYNTHETIC_CHALLENGE"
    per_head = pd.concat(
        [baseline_heads, paired_baseline_heads, injected_heads],
        ignore_index=True,
        sort=False,
    )

    from .model_health_check import evaluate_model_health

    health_metadata = {
        "collection": collected.quality,
        "model": baseline_model_meta,
        "injected_model": injected_model_meta,
        "planned_psap_hours": int(len(baseline_scored)),
        "scored_psap_hours": int(
            baseline_scored.get("SCORING_STATUS", pd.Series(dtype=str))
            .astype(str)
            .str.startswith("SCORED")
            .sum()
        ),
        "injected_planned_psap_hours": int(len(injected_scored)),
        "injected_scored_psap_hours": int(
            injected_scored.get("SCORING_STATUS", pd.Series(dtype=str))
            .astype(str)
            .str.startswith("SCORED")
            .sum()
        ),
        "planned_heads": int(len(per_head)),
        "missing_heads": int(
            (~per_head.get("HEAD_AVAILABLE", pd.Series(dtype=bool)).fillna(False).astype(bool)).sum()
        ),
        "planned_detectors": int(len(per_head)),
        "missing_detectors": int(
            (~per_head.get("HEAD_AVAILABLE", pd.Series(dtype=bool)).fillna(False).astype(bool)).sum()
        ),
    }
    health = evaluate_model_health(
        baseline_scored=baseline_scored,
        injected_scored=injected_scored,
        event_ground_truth=campaign.event_truth,
        row_ground_truth=campaign.row_truth,
        paired_baseline_scored=paired_baseline_scored,
        per_head_scored=per_head,
        metadata=health_metadata,
        reference=reference_scored,
        config=health_config,
    )
    evidence = {
        "schema_version": "1.0",
        "evaluation_version": EVALUATION_VERSION,
        "generated_at_utc": datetime.now(timezone.utc),
        "method": {
            "name": "chronological_shadow_replay_plus_controlled_fault_injection",
            "production_labels_available": False,
            "accuracy_claim_allowed": False,
            "baseline_alert_rate_is_false_alert_proxy_only": True,
            "threshold_policy": "frozen_deployed_threshold_strict_score_gt_threshold",
            "training_cutoff_utc": training_cutoff,
            "first_evaluation_hour_utc": first_evaluation_hour,
            "chronological_leakage_check": "PASS",
            "model_temporal_scope": (
                "pointwise hourly KPIs plus hour/day cyclic features; duration, ramp, and "
                "flapping metrics evaluate repeated hourly behavior, not sequence learning"
            ),
            "directional_volume_drop_scope": (
                "challenge test: deployed directional heads ignore negative CALL_VOLUME residuals"
            ),
        },
        "collection": collected.quality,
        "campaign": campaign.metadata,
        "model": baseline_model_meta,
        "injected_model": injected_model_meta,
        "paired_untouched_rows": int(len(paired_baseline_scored)),
        "health": health,
    }
    return EvaluationArtifacts(
        baseline_scored=baseline_scored,
        paired_baseline_scored=paired_baseline_scored,
        injected_scored=injected_scored,
        per_head_scored=per_head,
        row_truth=campaign.row_truth,
        event_truth=campaign.event_truth,
        evidence=_json_value(evidence),
    )


def save_evaluation_artifacts(artifacts: EvaluationArtifacts, output_dir: str | Path) -> dict[str, str]:
    """Save the complete denominator, synthetic truth, scores, and health JSON."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    files = {
        "baseline_scored": destination / "baseline_scored_psap_hours.csv",
        "paired_baseline_scored": destination / "paired_untouched_scored_psap_hours.csv",
        "injected_scored": destination / "synthetic_scored_psap_hours.csv",
        "per_head_scored": destination / "per_head_score_evidence.csv",
        "row_truth": destination / "synthetic_row_truth.csv",
        "event_truth": destination / "synthetic_event_truth.csv",
        "evidence": destination / "psap_model_health_evidence.json",
    }
    artifacts.baseline_scored.to_csv(files["baseline_scored"], index=False)
    artifacts.paired_baseline_scored.to_csv(
        files["paired_baseline_scored"], index=False
    )
    artifacts.injected_scored.to_csv(files["injected_scored"], index=False)
    artifacts.per_head_scored.to_csv(files["per_head_scored"], index=False)
    artifacts.row_truth.to_csv(files["row_truth"], index=False)
    artifacts.event_truth.to_csv(files["event_truth"], index=False)
    files["evidence"].write_text(
        json.dumps(_json_value(artifacts.evidence), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {name: str(path.resolve()) for name, path in files.items()}


def _csv_tuple(value: str, cast=str) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


_SCENARIO_FIELD_ALIASES = {
    "pattern": "pattern",
    "kpi": "kpis",
    "kpis": "kpis",
    "severity": "severity",
    "duration": "duration_hours",
    "duration_hours": "duration_hours",
    "volume": "volume_stratum",
    "volume_stratum": "volume_stratum",
    "stratum": "volume_stratum",
}


def parse_scenario_spec(value: str) -> FaultSpec:
    """Parse one repeatable CLI scenario into exactly one :class:`FaultSpec`.

    The format is a comma-separated list of ``key=value`` fields. Multiple
    KPIs use ``+`` rather than a comma, for example::

        pattern=multi_kpi_degradation,kpis=LSR_SR+ASR_SR,severity=severe,duration=3,volume=high
    """

    fields: dict[str, str] = {}
    for raw_part in str(value).split(","):
        part = raw_part.strip()
        if not part or "=" not in part:
            raise ValueError(
                "scenario must be comma-separated key=value fields; "
                f"invalid field: {raw_part!r}"
            )
        raw_key, raw_field_value = part.split("=", 1)
        key = raw_key.strip().lower().replace("-", "_")
        field_name = _SCENARIO_FIELD_ALIASES.get(key)
        if field_name is None:
            allowed = ", ".join(sorted(_SCENARIO_FIELD_ALIASES))
            raise ValueError(f"unknown scenario field {raw_key!r}; expected one of: {allowed}")
        if field_name in fields:
            raise ValueError(f"scenario field {field_name!r} was provided more than once")
        field_value = raw_field_value.strip()
        if not field_value:
            raise ValueError(f"scenario field {raw_key!r} must not be empty")
        fields[field_name] = field_value

    if "pattern" not in fields:
        raise ValueError("scenario requires pattern=<fault-pattern>")

    severity: str | float = fields.get("severity", "moderate")
    if isinstance(severity, str) and severity.strip().lower() not in {
        "mild", "moderate", "severe"
    }:
        try:
            severity = float(severity)
        except ValueError:
            pass  # FaultSpec supplies the authoritative validation message.

    duration_raw = fields.get("duration_hours", "1")
    try:
        duration = int(duration_raw)
    except ValueError as exc:
        raise ValueError("scenario duration must be one of 1, 2, 3, or 6") from exc

    kpis = tuple(
        item.strip().upper()
        for item in re.split(r"[+|]", fields.get("kpis", ""))
        if item.strip()
    )
    return FaultSpec(
        pattern=fields["pattern"],
        kpis=kpis,
        severity=severity,
        duration_hours=duration,
        volume_stratum=fields.get("volume_stratum"),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only PSAP anomaly-model chronological replay and synthetic evaluation"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-csv", help="Untouched flat hourly KPI CSV")
    source.add_argument("--oracle", action="store_true", help="Collect with existing canonical Oracle loader")
    parser.add_argument("--start-utc", help="Inclusive UTC start; required for --oracle")
    parser.add_argument("--end-utc", help="Exclusive UTC end; required for --oracle")
    parser.add_argument("--profile", choices=("PRD", "NPR"), default="PRD")
    parser.add_argument("--market", default="ALL")
    parser.add_argument("--sqlite-path")
    parser.add_argument("--allow-cache", action="store_true")
    parser.add_argument("--run-dir", required=True, help="Pinned local deployed artifact run directory")
    parser.add_argument(
        "--training-cutoff-utc",
        required=True,
        help="Last timestamp included in training/calibration; evaluation must start later",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--reference-scored-csv",
        help="Optional frozen earlier baseline_scored_psap_hours.csv for drift/mapping stability",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        metavar="KEY=VALUE,...",
        help=(
            "Run exactly one scenario; repeat this option for two or more. "
            "Example: pattern=kpi_degradation,kpis=ASR_SR,severity=severe,"
            "duration=3,volume=high. Multiple KPIs use '+'. When supplied, "
            "the Cartesian --severities/--durations/--volume-strata grid is not used."
        ),
    )
    parser.add_argument("--severities", default="mild,moderate,severe")
    parser.add_argument("--durations", default="1,2,3,6")
    parser.add_argument("--volume-strata", default="low,medium,high")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--psaps-per-stratum", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    collected = collect_source_data(
        input_csv=args.input_csv,
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        profile=args.profile,
        market=args.market,
        sqlite_path=args.sqlite_path,
        allow_cache=args.allow_cache,
    )
    from .production_scorer import ProductionScorer

    scorer = ProductionScorer(run_dir=args.run_dir)
    try:
        selected_scenarios = (
            tuple(parse_scenario_spec(value) for value in args.scenario)
            if args.scenario
            else None
        )
        campaign_config = CampaignConfig(
            severities=_csv_tuple(args.severities),
            durations=_csv_tuple(args.durations, int),
            volume_strata=_csv_tuple(args.volume_strata),
            replicates_per_cell=args.replicates,
            psaps_per_stratum=args.psaps_per_stratum,
            seed=args.seed,
            scenarios=selected_scenarios,
        )
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    reference_scored = (
        pd.read_csv(args.reference_scored_csv)
        if args.reference_scored_csv
        else None
    )
    artifacts = run_evaluation(
        collected,
        scorer,
        campaign_config=campaign_config,
        reference_scored=reference_scored,
        training_cutoff_utc=args.training_cutoff_utc,
    )
    paths = save_evaluation_artifacts(artifacts, args.output_dir)
    LOG.info("Evaluation complete: %s", json.dumps(paths, indent=2))
    print(
        json.dumps(
            {
                "status": artifacts.evidence["health"].get("overall_status"),
                "configured_scenarios": artifacts.evidence["campaign"].get(
                    "configured_scenarios"
                ),
                "created_scenarios": artifacts.evidence["campaign"].get(
                    "created_scenarios"
                ),
                "files": paths,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAMPAIGN_TEMPLATES",
    "EVALUATION_VERSION",
    "CampaignConfig",
    "CampaignData",
    "CollectedFrame",
    "EvaluationArtifacts",
    "build_campaign_specs",
    "build_synthetic_campaign",
    "collect_source_data",
    "normalize_source_frame",
    "parse_scenario_spec",
    "run_evaluation",
    "save_evaluation_artifacts",
    "select_campaign_population",
]
