"""Read-only offline adapter for the deployed PSAP KPI anomaly scorer.

The adapter deliberately accepts a *pinned local* run directory.  It never
resolves a latest MLflow run, downloads artifacts, queries a database, or
writes inference results.  The production ``psap_kpi_inference`` module is
imported lazily when :class:`ProductionScorer` is instantiated (and can be
replaced by a small test double).

Unlike the production outage table, the returned frames retain the complete
denominator: every input PSAP-hour, every candidate head, missing detectors,
invalid inputs, and scoring failures.  ``RAW_THRESHOLD_CROSSING`` always uses
the deployed strict comparison (score > threshold).  Production hard and
third-level sanity rules are reported as diagnostics but do not alter that raw
decision.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import importlib
import inspect
import math
from pathlib import Path
import re
import sys
from typing import Any, Protocol

import numpy as np
import pandas as pd


INPUT_KPI_COLS = (
    "CALL_VOLUME",
    "LSR_SR",
    "ASR_SR",
    "BID_SR",
    "CSR_SR",
)
MODEL_KPI_COLS = tuple(f"{column}_LIST" for column in INPUT_KPI_COLS)
EXPECTED_FEATURE_COLS = (
    *MODEL_KPI_COLS,
    "hod_sin",
    "hod_cos",
    "dow_sin",
    "dow_cos",
)
TIMESTAMP_CANDIDATES = (
    "HOUR_START_UTC",
    "EVAL_TS_UTC",
    "EVAL_TIMESTAMP",
    "TIMESTAMP",
    "DATETIME",
)
ROW_ID_CANDIDATES = ("EVALUATION_ROW_ID", "SCENARIO_ROW_ID")


class DetectorProtocol(Protocol):
    """Minimum interface required from a deployed or fake detector."""

    def score_series(self, ts_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        ...


DetectorLoader = Callable[..., DetectorProtocol | None]


@dataclass(frozen=True)
class ScoreResult:
    """Complete offline scoring evidence."""

    aggregate: pd.DataFrame
    per_head: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _DetectorRecord:
    detector: DetectorProtocol | None
    available: bool
    source: str
    ae_score_mode: str | None
    error: str
    validation_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _MappingBundle:
    cluster_by_psap: dict[int, int]
    heads_by_psap: dict[int, tuple[int, ...]]
    head_mapping_source: dict[int, str]
    cluster_path: Path | None
    multi_head_path: Path | None
    cluster_rows: int
    multi_head_rows: int
    discovery: dict[str, Any]


class ArtifactValidationError(ValueError):
    """Raised when a pinned artifact package is internally inconsistent."""


class ProductionScorer:
    """Score chronological flat KPI data using pinned production artifacts.

    Parameters
    ----------
    run_dir:
        Local directory containing one immutable training/inference run.
    cluster_mapping_path, multi_head_mapping_path:
        Optional explicit mapping CSVs.  Otherwise files named
        ``training_cluster_mapping*.csv`` and
        ``cluster_multi_head_member_map*.csv`` are discovered recursively;
        the production ``*_k15.csv`` variant is supported.
    detector_loader:
        Optional callback/object used for tests.  It may accept ``head_id`` or
        ``(head_id, cluster_id)`` and return a detector (or ``None``).
    inference_module:
        Optional test double for ``psap_kpi_inference``.  Supplying it avoids
        importing TensorFlow, MLflow, Oracle, and other production packages.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        cluster_mapping_path: str | Path | None = None,
        multi_head_mapping_path: str | Path | None = None,
        detector_loader: DetectorLoader | Any | None = None,
        inference_module: Any | None = None,
        timestamp_col: str | None = None,
        aggregation_method: str = "ratio",
        strict_detector_metadata: bool | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        if not self.run_dir.exists() or not self.run_dir.is_dir():
            raise FileNotFoundError(f"Pinned run_dir is not a directory: {self.run_dir}")
        if aggregation_method != "ratio":
            raise ValueError(
                "Production evaluation is pinned to the deployed mean-ratio "
                "multi-head aggregation method ('ratio')."
            )

        # Lazy relative import: importing this file alone has no production
        # dependency side effects.  A supplied test double bypasses it.
        self._inference = inference_module or self._import_inference_module()
        self.timestamp_col = timestamp_col
        self.aggregation_method = aggregation_method
        self._external_loader = detector_loader
        self._strict_detector_metadata = (
            detector_loader is None
            if strict_detector_metadata is None
            else bool(strict_detector_metadata)
        )
        self._mapping = self._load_mappings(
            explicit_cluster=cluster_mapping_path,
            explicit_multi=multi_head_mapping_path,
        )
        self._detector_cache: dict[int, _DetectorRecord] = {}

    @staticmethod
    def _import_inference_module() -> Any:
        """Import the deployed module only after scorer instantiation."""
        errors: list[BaseException] = []
        for module_name in ("psap_kpi.psap_kpi_inference", "psap_kpi_inference"):
            try:
                return importlib.import_module(module_name)
            except (ImportError, ModuleNotFoundError) as exc:
                errors.append(exc)

        # The reconstructed production file uses script-style ``psap_utils``
        # imports.  Add its own directory only after the normal package imports
        # fail, then retry.  This does not read or write external state.
        package_root = Path(__file__).resolve().parents[1]
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
        try:
            return importlib.import_module("psap_kpi.psap_kpi_inference")
        except (ImportError, ModuleNotFoundError) as exc:
            errors.append(exc)
        detail = "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
        raise ImportError(
            "Could not lazily import the deployed psap_kpi_inference module. "
            "Install its production dependencies or supply inference_module "
            f"for an offline test double. Attempts: {detail}"
        ) from errors[-1]

    def score(self, frame: pd.DataFrame) -> ScoreResult:
        """Score a multi-hour flat frame without any external writes.

        Duplicate ``(PSAP_ID, timestamp)`` pairs are allowed.  Aggregation is
        keyed by a unique caller row identifier when supplied, and always by a
        private positional key internally, so paired baseline/injected or
        severity scenarios cannot overwrite one another.
        """
        prepared, timestamp_col, row_id_source, input_meta = self._prepare(frame)
        plans = self._build_plans(prepared)

        requested_heads = sorted(
            {head for plan in plans for head in plan["candidate_heads"]}
        )
        for head_id in requested_heads:
            cluster_ids = sorted(
                {
                    int(plan["cluster_id"])
                    for plan in plans
                    if head_id in plan["candidate_heads"]
                    and plan["cluster_id"] is not None
                }
            )
            cluster_id = cluster_ids[0] if len(cluster_ids) == 1 else None
            self._get_detector(head_id, cluster_id)

        per_head = self._initial_per_head(prepared, plans)
        per_head = self._score_heads(prepared, per_head)
        aggregate = self._aggregate(prepared, plans, per_head)

        detector_records = {
            int(head): record for head, record in self._detector_cache.items()
            if head in requested_heads
        }
        modes = sorted(
            {record.ae_score_mode for record in detector_records.values()
             if record.ae_score_mode}
        )
        metadata: dict[str, Any] = {
            "schema_version": "1.0",
            "adapter": "read_only_pinned_local_production_scorer",
            "run_dir": str(self.run_dir),
            "timestamp_column": timestamp_col,
            "row_id_source": row_id_source,
            "aggregation_method": "mean_score_threshold_ratio",
            "threshold_comparison": "strict_score_gt_threshold",
            "sanity_policy": "calculated_not_applied",
            "min_eval_volume_policy": (
                "artifact min_eval_volume is loaded by production but is not applied "
                "by the deployed scoring/alert path"
            ),
            "external_reads": {"database": False, "mlflow": False},
            "external_writes": {"database": False, "mlflow": False, "files": False},
            "input": input_meta,
            "mapping": {
                "cluster_mapping_path": (
                    str(self._mapping.cluster_path) if self._mapping.cluster_path else None
                ),
                "multi_head_mapping_path": (
                    str(self._mapping.multi_head_path) if self._mapping.multi_head_path else None
                ),
                "cluster_mapping_sha256": self._file_hash(self._mapping.cluster_path),
                "multi_head_mapping_sha256": self._file_hash(self._mapping.multi_head_path),
                "cluster_mapping_rows": self._mapping.cluster_rows,
                "multi_head_mapping_rows": self._mapping.multi_head_rows,
                "mapped_psaps": len(
                    set(self._mapping.cluster_by_psap)
                    | set(self._mapping.heads_by_psap)
                ),
                **self._mapping.discovery,
            },
            "scoring": {
                "planned_rows": int(len(aggregate)),
                "scored_rows": int(aggregate["SCORING_STATUS"].str.startswith("SCORED").sum()),
                "raw_threshold_crossings": int(
                    aggregate["RAW_THRESHOLD_CROSSING"].map(
                        lambda value: False if pd.isna(value) else bool(value)
                    ).sum()
                ),
                "candidate_head_rows": int(len(per_head)),
                "requested_heads": len(requested_heads),
                "available_heads": sum(r.available for r in detector_records.values()),
                "missing_or_invalid_heads": sum(not r.available for r in detector_records.values()),
                "ae_score_modes": modes,
                "head_errors": {
                    str(head): record.error
                    for head, record in detector_records.items()
                    if record.error
                },
                "head_metadata_warnings": {
                    str(head): list(record.validation_warnings)
                    for head, record in detector_records.items()
                    if record.validation_warnings
                },
            },
        }
        return ScoreResult(
            aggregate=aggregate.reset_index(drop=True),
            per_head=per_head.reset_index(drop=True),
            metadata=metadata,
        )

    def _prepare(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, str, str, dict[str, Any]]:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if frame.empty:
            raise ValueError("frame must contain at least one planned PSAP-hour")
        if bool(frame.columns.duplicated().any()):
            raise ValueError("frame must not contain duplicate column names")

        timestamp_col = self.timestamp_col or next(
            (column for column in TIMESTAMP_CANDIDATES if column in frame.columns),
            None,
        )
        if timestamp_col is None or timestamp_col not in frame.columns:
            raise ValueError(
                "No timestamp column found; pass timestamp_col or provide one of "
                f"{TIMESTAMP_CANDIDATES}"
            )
        missing = [column for column in ("PSAP_ID", *INPUT_KPI_COLS) if column not in frame]
        if missing:
            raise ValueError(f"frame is missing required columns: {missing}")

        data = frame.copy(deep=True).reset_index(names="__SOURCE_INDEX")
        data["__ROW_POSITION"] = np.arange(len(data), dtype="int64")

        row_id_source = "generated_positional"
        caller_row_id: pd.Series | None = None
        for candidate in ROW_ID_CANDIDATES:
            if candidate in data.columns:
                if data[candidate].isna().any() or data[candidate].duplicated().any():
                    raise ValueError(f"{candidate} must be non-null and unique when supplied")
                caller_row_id = data[candidate].copy()
                row_id_source = candidate
                break
        if caller_row_id is None:
            caller_row_id = pd.Series(
                [f"ROW-{position:012d}" for position in data["__ROW_POSITION"]],
                index=data.index,
                dtype="object",
            )
        data["__EVALUATION_ROW_ID"] = caller_row_id

        original_ts = data[timestamp_col]
        naive_count = self._count_naive_timestamps(original_ts)
        timestamps = pd.to_datetime(original_ts, errors="coerce", utc=True)
        data["__TIMESTAMP_UTC"] = timestamps

        issues: list[list[str]] = [[] for _ in range(len(data))]
        invalid_ts = timestamps.isna()
        off_hour = (~invalid_ts) & (timestamps != timestamps.dt.floor("h"))
        self._append_issue(issues, invalid_ts, "INVALID_TIMESTAMP")
        self._append_issue(issues, off_hour, "TIMESTAMP_NOT_HOUR_ALIGNED")

        psap_numeric = pd.to_numeric(data["PSAP_ID"], errors="coerce")
        invalid_psap = (~np.isfinite(psap_numeric.to_numpy(dtype="float64"))) | (
            psap_numeric.fillna(0) != np.floor(psap_numeric.fillna(0))
        )
        self._append_issue(issues, pd.Series(invalid_psap), "INVALID_PSAP_ID")
        data["__PSAP_ID_NUM"] = psap_numeric

        for column in INPUT_KPI_COLS:
            numeric = pd.to_numeric(data[column], errors="coerce")
            finite = np.isfinite(numeric.to_numpy(dtype="float64"))
            self._append_issue(issues, pd.Series(~finite), f"NON_FINITE_{column}")
            if column == "CALL_VOLUME":
                self._append_issue(issues, numeric < 0, "NEGATIVE_CALL_VOLUME")
            else:
                self._append_issue(
                    issues,
                    (numeric < 0) | (numeric > 100),
                    f"OUT_OF_RANGE_{column}",
                )
            data[f"__{column}"] = numeric

        data["__INPUT_ISSUES"] = [";".join(row_issues) for row_issues in issues]
        data["__INPUT_VALID"] = data["__INPUT_ISSUES"].eq("")
        # Preserve a valid PSAP identity even when a timestamp/KPI is invalid,
        # so the skipped row can still report all of its candidate heads and
        # their artifact availability.
        valid_psap = data["__PSAP_ID_NUM"].where(~pd.Series(invalid_psap, index=data.index))
        data["__PSAP_ID_INT"] = valid_psap.astype("Int64")

        key_frame = pd.DataFrame(
            {
                "psap": data["PSAP_ID"].astype(str),
                "timestamp": timestamps,
            }
        )
        duplicate_psap_hours = int(key_frame.duplicated(keep=False).sum())
        input_meta = {
            "rows": int(len(data)),
            "valid_rows": int(data["__INPUT_VALID"].sum()),
            "invalid_rows": int((~data["__INPUT_VALID"]).sum()),
            "distinct_psaps": int(data.loc[data["__INPUT_VALID"], "__PSAP_ID_INT"].nunique()),
            "duplicate_psap_timestamp_rows": duplicate_psap_hours,
            "duplicate_psap_timestamps_allowed": True,
            "naive_timestamps_assumed_utc": int(naive_count),
        }
        return data, timestamp_col, row_id_source, input_meta

    @staticmethod
    def _append_issue(
        issues: list[list[str]], mask: pd.Series | np.ndarray, issue: str
    ) -> None:
        mask_array = np.asarray(mask, dtype=bool)
        for position in np.flatnonzero(mask_array):
            issues[int(position)].append(issue)

    @staticmethod
    def _count_naive_timestamps(values: pd.Series) -> int:
        count = 0
        for value in values:
            if pd.isna(value):
                continue
            try:
                parsed = pd.Timestamp(value)
            except (TypeError, ValueError):
                continue
            if parsed.tzinfo is None:
                count += 1
        return count

    def _build_plans(self, data: pd.DataFrame) -> list[dict[str, Any]]:
        plans: list[dict[str, Any]] = []
        # Do not use ``itertuples`` here: pandas renames columns beginning with
        # underscores, which would make the private row key inaccessible.
        for position in range(len(data)):
            row = data.iloc[position]
            position = int(row["__ROW_POSITION"])
            input_valid = bool(row["__INPUT_VALID"])
            psap_value = row["__PSAP_ID_INT"]
            psap_id = None if pd.isna(psap_value) else int(psap_value)
            special = bool(input_valid and psap_id == -1)
            cluster_id = self._mapping.cluster_by_psap.get(psap_id) if psap_id is not None else None
            mapped_heads = self._mapping.heads_by_psap.get(psap_id, ()) if psap_id is not None else ()
            if special or psap_id is None:
                heads: tuple[int, ...] = ()
                source = "NONE"
            elif mapped_heads:
                heads = tuple(dict.fromkeys(int(head) for head in mapped_heads))
                source = self._mapping.head_mapping_source.get(psap_id, "MAPPING")
            else:
                # Exact deployed fallback: attempt a detector whose head ID is
                # the member PSAP ID when no explicit head mapping is present.
                heads = (int(psap_id),)
                source = (
                    "SELF_FALLBACK_CLUSTER_ONLY"
                    if cluster_id is not None
                    else "SELF_FALLBACK_UNMAPPED"
                )
            plans.append(
                {
                    "position": position,
                    "psap_id": psap_id,
                    "cluster_id": cluster_id,
                    "candidate_heads": heads,
                    "head_mapping_found": bool(mapped_heads),
                    "cluster_mapping_found": cluster_id is not None,
                    "mapping_source": source,
                    "special": special,
                }
            )
        return plans

    def _initial_per_head(
        self, data: pd.DataFrame, plans: list[dict[str, Any]]
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for plan in plans:
            source = data.iloc[plan["position"]]
            for rank, head_id in enumerate(plan["candidate_heads"], start=1):
                record = self._detector_cache[head_id]
                status = "READY"
                if not record.available:
                    status = "MISSING_OR_INVALID_DETECTOR"
                elif not bool(source["__INPUT_VALID"]):
                    status = "NOT_SCORED_INVALID_INPUT"
                rows.append(
                    {
                        "__ROW_POSITION": plan["position"],
                        "EVALUATION_ROW_ID": source["__EVALUATION_ROW_ID"],
                        "SOURCE_INDEX": source["__SOURCE_INDEX"],
                        "HOUR_START_UTC": source["__TIMESTAMP_UTC"],
                        "PSAP_ID": plan["psap_id"],
                        "CLUSTER_ID": plan["cluster_id"],
                        "HEAD_ID": int(head_id),
                        "HEAD_RANK": int(rank),
                        "MAPPING_SOURCE": plan["mapping_source"],
                        "HEAD_AVAILABLE": bool(record.available),
                        "HEAD_STATUS": status,
                        "RAW_SCORE": np.nan,
                        "RAW_THRESHOLD": np.nan,
                        "SCORE_THRESHOLD_RATIO": np.nan,
                        "VOTE": pd.NA,
                        "AE_SCORE_MODE": record.ae_score_mode,
                        "ARTIFACT_SOURCE": record.source,
                        "ERROR": record.error,
                    }
                )
        columns = [
            "__ROW_POSITION", "EVALUATION_ROW_ID", "SOURCE_INDEX", "HOUR_START_UTC",
            "PSAP_ID", "CLUSTER_ID", "HEAD_ID", "HEAD_RANK", "MAPPING_SOURCE",
            "HEAD_AVAILABLE", "HEAD_STATUS", "RAW_SCORE", "RAW_THRESHOLD",
            "SCORE_THRESHOLD_RATIO", "VOTE", "AE_SCORE_MODE", "ARTIFACT_SOURCE",
            "ERROR",
        ]
        return pd.DataFrame(rows, columns=columns)

    def _score_heads(self, data: pd.DataFrame, per_head: pd.DataFrame) -> pd.DataFrame:
        if per_head.empty:
            return per_head
        result = per_head.copy()
        for head_id, head_rows in result.groupby("HEAD_ID", sort=True):
            detector_record = self._detector_cache[int(head_id)]
            if not detector_record.available or detector_record.detector is None:
                continue
            eligible_index = head_rows.index[
                head_rows["HEAD_STATUS"].eq("READY")
            ]
            if len(eligible_index) == 0:
                continue
            positions = result.loc[eligible_index, "__ROW_POSITION"].astype(int).to_numpy()
            subset = data.iloc[positions]
            ts_frame = pd.DataFrame(
                {
                    model_column: subset[f"__{input_column}"].to_numpy(dtype="float32")
                    for input_column, model_column in zip(INPUT_KPI_COLS, MODEL_KPI_COLS)
                },
                index=pd.DatetimeIndex(subset["__TIMESTAMP_UTC"]),
            )
            try:
                scores, thresholds = detector_record.detector.score_series(ts_frame)
                scores = np.asarray(scores, dtype="float64").reshape(-1)
                thresholds = np.asarray(thresholds, dtype="float64").reshape(-1)
                if len(scores) != len(eligible_index) or len(thresholds) != len(eligible_index):
                    raise ArtifactValidationError(
                        f"head {head_id} returned shapes scores={scores.shape}, "
                        f"thresholds={thresholds.shape}; expected ({len(eligible_index)},)"
                    )
            except Exception as exc:  # evidence must retain all affected rows
                result.loc[eligible_index, "HEAD_STATUS"] = "SCORING_ERROR"
                result.loc[eligible_index, "ERROR"] = f"{type(exc).__name__}: {exc}"
                continue

            for output_position, row_index in enumerate(eligible_index):
                score = float(scores[output_position])
                threshold = float(thresholds[output_position])
                if not math.isfinite(score) or not math.isfinite(threshold) or threshold <= 0:
                    result.at[row_index, "HEAD_STATUS"] = "INVALID_SCORE_OR_THRESHOLD"
                    result.at[row_index, "ERROR"] = (
                        f"score={score!r}, threshold={threshold!r}; both must be finite "
                        "and threshold must be positive"
                    )
                    continue
                result.at[row_index, "RAW_SCORE"] = score
                result.at[row_index, "RAW_THRESHOLD"] = threshold
                result.at[row_index, "SCORE_THRESHOLD_RATIO"] = score / threshold
                result.at[row_index, "VOTE"] = bool(score > threshold)
                result.at[row_index, "HEAD_STATUS"] = "SCORED"
        return result

    def _aggregate(
        self,
        data: pd.DataFrame,
        plans: list[dict[str, Any]],
        per_head: pd.DataFrame,
    ) -> pd.DataFrame:
        by_position = {
            int(position): group
            for position, group in per_head.groupby("__ROW_POSITION", sort=False)
        } if not per_head.empty else {}
        plan_by_position = {plan["position"]: plan for plan in plans}
        output_rows: list[dict[str, Any]] = []

        diagnostic_columns = {
            "EVALUATION_ROW_ID", "SOURCE_INDEX", "CLUSTER_ID", "CLUSTER_MAPPING_FOUND",
            "HEAD_MAPPING_FOUND", "MAPPING_SOURCE", "CANDIDATE_HEAD_COUNT",
            "AVAILABLE_HEAD_COUNT", "SCORED_HEAD_COUNT", "MISSING_HEAD_COUNT",
            "SCORING_STATUS", "INPUT_VALID", "INPUT_ISSUES", "ANOMALY_SCORE",
            "ANOMALY_THRESH", "SCORE_THRESHOLD_RATIO", "RAW_THRESHOLD_CROSSING",
            "HEAD_VOTE_COUNT", "HEAD_VOTE_FRACTION", "HEAD_RATIO_STD",
            "HARD_SANITY_PASS", "HARD_SANITY_REASON", "THIRD_LEVEL_PASS",
            "THIRD_LEVEL_REASON", "SANITY_ELIGIBLE_ALERT", "AE_SCORE_MODE",
        }
        original_columns = [
            column for column in data.columns
            if not column.startswith("__") and column not in diagnostic_columns
        ]

        for position in range(len(data)):
            source = data.iloc[position]
            plan = plan_by_position[position]
            heads = by_position.get(position, pd.DataFrame())
            scored = heads[heads["HEAD_STATUS"].eq("SCORED")] if not heads.empty else heads
            candidate_count = len(heads)
            available_count = int(heads["HEAD_AVAILABLE"].sum()) if not heads.empty else 0
            scored_count = len(scored)

            aggregate_score = np.nan
            aggregate_threshold = np.nan
            raw_alert: Any = pd.NA
            hard_ok: Any = pd.NA
            hard_reason = "NOT_SCORED"
            third_ok: Any = pd.NA
            third_reason = "NOT_SCORED"

            if plan["special"]:
                status = "SKIPPED_SPECIAL_PSAP"
            elif not bool(source["__INPUT_VALID"]):
                status = "SKIPPED_INVALID_INPUT"
            elif scored_count == 0:
                if available_count == 0:
                    status = "SKIPPED_NO_DETECTOR"
                else:
                    status = "SCORING_ERROR"
            else:
                pairs = list(
                    zip(
                        scored["RAW_SCORE"].astype(float),
                        scored["RAW_THRESHOLD"].astype(float),
                    )
                )
                aggregate_score, aggregate_threshold, decision = self._aggregate_pairs(pairs)
                raw_alert = bool(decision)
                status = "SCORED" if scored_count == candidate_count else "SCORED_PARTIAL_HEADS"
                hard_ok, hard_reason = self._hard_sanity(
                    lsr_sr=float(source["__LSR_SR"]),
                    asr_sr=float(source["__ASR_SR"]),
                    bid_sr=float(source["__BID_SR"]),
                    csr_sr=float(source["__CSR_SR"]),
                    call_volume=float(source["__CALL_VOLUME"]),
                    anomaly_score=float(aggregate_score),
                    anomaly_thresh=float(aggregate_threshold),
                )
                if raw_alert:
                    third_ok, third_reason = self._third_level_sanity(
                        lsr_sr=float(source["__LSR_SR"]),
                        asr_sr=float(source["__ASR_SR"]),
                        bid_sr=float(source["__BID_SR"]),
                        csr_sr=float(source["__CSR_SR"]),
                        call_volume=float(source["__CALL_VOLUME"]),
                    )
                else:
                    # Exact current production flow: third-level starts true
                    # and is called only after a raw threshold crossing.
                    third_ok, third_reason = True, ""

            ratios = scored["SCORE_THRESHOLD_RATIO"].astype(float) if scored_count else pd.Series(dtype=float)
            votes = scored["VOTE"].astype(bool) if scored_count else pd.Series(dtype=bool)
            sanity_eligible = (
                bool(raw_alert) and bool(hard_ok) and bool(third_ok)
                if raw_alert is not pd.NA and hard_ok is not pd.NA and third_ok is not pd.NA
                else pd.NA
            )
            modes = sorted(set(heads["AE_SCORE_MODE"].dropna().astype(str))) if not heads.empty else []
            row = {column: source[column] for column in original_columns}
            row.update(
                {
                    "EVALUATION_ROW_ID": source["__EVALUATION_ROW_ID"],
                    "SOURCE_INDEX": source["__SOURCE_INDEX"],
                    "HOUR_START_UTC": source["__TIMESTAMP_UTC"],
                    "PSAP_ID": plan["psap_id"] if plan["psap_id"] is not None else source["PSAP_ID"],
                    "CLUSTER_ID": plan["cluster_id"],
                    "CLUSTER_MAPPING_FOUND": plan["cluster_mapping_found"],
                    "HEAD_MAPPING_FOUND": plan["head_mapping_found"],
                    "MAPPING_SOURCE": plan["mapping_source"],
                    "CANDIDATE_HEAD_COUNT": candidate_count,
                    "AVAILABLE_HEAD_COUNT": available_count,
                    "SCORED_HEAD_COUNT": scored_count,
                    "MISSING_HEAD_COUNT": candidate_count - available_count,
                    "SCORING_STATUS": status,
                    "INPUT_VALID": bool(source["__INPUT_VALID"]),
                    "INPUT_ISSUES": source["__INPUT_ISSUES"],
                    "ANOMALY_SCORE": aggregate_score,
                    "ANOMALY_THRESH": aggregate_threshold,
                    "SCORE_THRESHOLD_RATIO": (
                        aggregate_score / aggregate_threshold
                        if math.isfinite(float(aggregate_score))
                        and math.isfinite(float(aggregate_threshold))
                        and aggregate_threshold > 0
                        else np.nan
                    ),
                    "RAW_THRESHOLD_CROSSING": raw_alert,
                    "HEAD_VOTE_COUNT": int(votes.sum()) if scored_count else 0,
                    "HEAD_VOTE_FRACTION": float(votes.mean()) if scored_count else np.nan,
                    "HEAD_RATIO_STD": float(ratios.std(ddof=0)) if scored_count else np.nan,
                    "HARD_SANITY_PASS": hard_ok,
                    "HARD_SANITY_REASON": hard_reason,
                    "THIRD_LEVEL_PASS": third_ok,
                    "THIRD_LEVEL_REASON": third_reason,
                    "SANITY_ELIGIBLE_ALERT": sanity_eligible,
                    "AE_SCORE_MODE": ",".join(modes),
                }
            )
            output_rows.append(row)
        return pd.DataFrame(output_rows)

    def _aggregate_pairs(
        self, pairs: list[tuple[float, float]]
    ) -> tuple[float, float, bool]:
        function = getattr(self._inference, "_aggregate_multihead_decision", None)
        if callable(function):
            return function(pairs, method="ratio")
        ratios = [score / threshold for score, threshold in pairs]
        mean_ratio = float(np.mean(ratios))
        return mean_ratio, 1.0, bool(mean_ratio > 1.0)

    def _hard_sanity(self, **values: float) -> tuple[bool, str]:
        function = getattr(self._inference, "_passes_hard_sanity", None)
        if callable(function):
            return function(**values)
        for name, value in values.items():
            if not math.isfinite(value):
                return False, f"NON_FINITE_{name.upper()}"
            if value < 0:
                return False, f"NEGATIVE_{name.upper()}"
        for name in ("lsr_sr", "asr_sr", "bid_sr", "csr_sr"):
            if values[name] > 100:
                return False, f"OUT_OF_RANGE_{name.upper()}"
        if values["anomaly_thresh"] <= 0:
            return False, "NON_POSITIVE_THRESHOLD"
        if values["call_volume"] <= 0:
            return False, "ZERO_OR_NEGATIVE_VOLUME"
        if values["anomaly_score"] <= values["anomaly_thresh"]:
            return False, "NOT_AN_ANOMALY"
        return True, ""

    def _third_level_sanity(self, **values: float) -> tuple[bool, str]:
        function = getattr(self._inference, "_passes_third_level_sanity", None)
        if callable(function):
            return function(**values)
        lsr = values["lsr_sr"]
        asr = values["asr_sr"]
        bid = values["bid_sr"]
        csr = values["csr_sr"]
        volume = values["call_volume"]
        if bid >= 95 and csr >= 95 and asr >= 85 and lsr >= 85:
            return False, "TIER1_HEALTHY_PSAP_SUPPRESSED"
        if bid >= 90 and csr >= 90:
            return (
                (True, "") if lsr <= 45 and asr <= 45 and volume > 20
                else (False, "TIER2_BID_CSR_HIGH_INSUFFICIENT_DEGRADATION")
            )
        if bid >= 82 and csr >= 82:
            return (
                (True, "") if lsr <= 65 and asr <= 65 and volume > 15
                else (False, "TIER3_BID_CSR_MODERATE_INSUFFICIENT_DEGRADATION")
            )
        if bid < 82 and csr < 82:
            return (
                (True, "") if volume > 9
                else (False, "FALLBACK_BOTH_LOW_VOLUME_SUPPRESSED")
            )
        if (lsr < 70 or asr < 70) and volume > 12:
            return True, ""
        return False, "FALLBACK_SINGLE_LOW_INSUFFICIENT_EVIDENCE"

    def _get_detector(self, head_id: int, cluster_id: int | None) -> _DetectorRecord:
        if head_id in self._detector_cache:
            return self._detector_cache[head_id]
        try:
            if self._external_loader is None:
                detector, source = self._load_pinned_detector(head_id, cluster_id)
            else:
                detector = self._call_external_loader(head_id, cluster_id)
                source = "CALLBACK"
            if detector is None:
                record = _DetectorRecord(None, False, source, None, "DETECTOR_NOT_FOUND")
            elif not callable(getattr(detector, "score_series", None)):
                record = _DetectorRecord(
                    None, False, source, None,
                    "INVALID_DETECTOR: missing callable score_series",
                )
            else:
                warnings = self._validate_detector(detector)
                record = _DetectorRecord(
                    detector=detector,
                    available=True,
                    source=source,
                    ae_score_mode=str(getattr(detector, "ae_score_mode", "unknown")),
                    error="",
                    validation_warnings=tuple(warnings),
                )
        except Exception as exc:
            record = _DetectorRecord(
                detector=None,
                available=False,
                source="PINNED_LOCAL" if self._external_loader is None else "CALLBACK",
                ae_score_mode=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        self._detector_cache[head_id] = record
        return record

    def _call_external_loader(self, head_id: int, cluster_id: int | None) -> DetectorProtocol | None:
        loader = self._external_loader
        function = getattr(loader, "load_detector", loader)
        if not callable(function):
            raise TypeError("detector_loader must be callable or expose load_detector")
        try:
            signature = inspect.signature(function)
            positional = [
                parameter for parameter in signature.parameters.values()
                if parameter.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            has_varargs = any(
                parameter.kind == inspect.Parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            # Some extension callables do not expose a Python signature.  The
            # documented two-argument protocol is the safe default for them.
            has_varargs = True
            positional = []
        result = (
            function(head_id, cluster_id)
            if has_varargs or len(positional) >= 2
            else function(head_id)
        )
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
            return result[0]
        return result

    def _load_pinned_detector(
        self, head_id: int, cluster_id: int | None
    ) -> tuple[DetectorProtocol | None, str]:
        model_path, meta_path = self._resolve_head_artifacts(head_id, cluster_id)
        if model_path is None or meta_path is None:
            return None, "PINNED_LOCAL_NOT_FOUND"
        loader = getattr(self._inference, "_load_detector_from_paths", None)
        if not callable(loader):
            raise AttributeError(
                "deployed inference module does not expose _load_detector_from_paths"
            )
        detector = loader(
            model_path=model_path,
            meta_path=meta_path,
            psap_id=int(head_id),
            head_id=int(head_id),
            source="OFFLINE_PINNED_LOCAL",
        )
        relative_model = model_path.relative_to(self.run_dir)
        relative_meta = meta_path.relative_to(self.run_dir)
        return detector, f"PINNED_LOCAL:{relative_model}|{relative_meta}"

    def _resolve_head_artifacts(
        self, head_id: int, cluster_id: int | None
    ) -> tuple[Path | None, Path | None]:
        directories: list[Path] = []
        if cluster_id is not None:
            directories.extend(
                [
                    self.run_dir / "cluster_models" / f"cluster_{cluster_id}" / f"psap_id_{head_id}",
                    self.run_dir / "cluster_models" / f"cluster_{cluster_id}" / f"head_{head_id}",
                ]
            )
        directories.extend(
            [
                self.run_dir / "cluster_models" / f"cluster_{head_id}" / f"psap_id_{head_id}",
                self.run_dir / "cluster_heads" / f"psap_id_{head_id}",
                self.run_dir / f"psap_id_{head_id}",
            ]
        )
        directories.extend(
            path for path in self.run_dir.rglob(f"psap_id_{head_id}") if path.is_dir()
        )
        directories.extend(
            path for path in self.run_dir.rglob(f"head_{head_id}") if path.is_dir()
        )
        unique_dirs = list(dict.fromkeys(path.resolve() for path in directories))
        for directory in unique_dirs:
            if not directory.exists():
                continue
            models = self._rank_model_files(directory.glob("*.keras"))
            metadata = self._rank_metadata_files(
                [*directory.glob("*.pkl"), *directory.glob("*.json")]
            )
            if models and metadata:
                return models[0], metadata[0]

        # Last-resort filename discovery, with numeric token matching to avoid
        # confusing head 1 with head 11.
        token = re.compile(rf"(?<!\d){re.escape(str(head_id))}(?!\d)")
        models = self._rank_model_files(
            path for path in self.run_dir.rglob("*.keras")
            if token.search(str(path.relative_to(self.run_dir)))
        )
        for model_path in models:
            metadata = self._rank_metadata_files(
                [*model_path.parent.glob("*.pkl"), *model_path.parent.glob("*.json")]
            )
            if metadata:
                return model_path, metadata[0]
        return None, None

    @staticmethod
    def _rank_model_files(paths: Any) -> list[Path]:
        priority = {"ae_model.keras": 0, "model.keras": 1}
        return sorted(
            (Path(path) for path in paths),
            key=lambda path: (priority.get(path.name.lower(), 10), path.name.lower()),
        )

    @staticmethod
    def _rank_metadata_files(paths: Any) -> list[Path]:
        def rank(path: Path) -> tuple[int, str]:
            name = path.name.lower()
            if name in {"ae_meta.pkl", "ae_metadata.pkl", "ae_meta.json", "ae_metadata.json"}:
                return 0, name
            if "meta" in name and "training" not in name:
                return 1, name
            if path.suffix.lower() == ".pkl":
                return 2, name
            return 3, name
        return sorted((Path(path) for path in paths), key=rank)

    def _validate_detector(self, detector: DetectorProtocol) -> list[str]:
        errors: list[str] = []
        warnings: list[str] = []
        kpi_cols = getattr(detector, "kpi_cols_", None)
        feature_cols = getattr(detector, "feature_cols_", None)
        mean = getattr(detector, "scaler_mean_", None)
        scale = getattr(detector, "scaler_scale_", None)

        if kpi_cols is None:
            warnings.append("kpi_cols_ not exposed by detector")
        elif tuple(kpi_cols) != MODEL_KPI_COLS:
            errors.append(f"kpi_cols_={tuple(kpi_cols)!r}; expected {MODEL_KPI_COLS!r}")

        if feature_cols is None:
            warnings.append("feature_cols_ not exposed; deployed builder order will be assumed")
            expected_width = len(EXPECTED_FEATURE_COLS)
        else:
            feature_cols = tuple(feature_cols)
            expected_width = len(feature_cols)
            if feature_cols != EXPECTED_FEATURE_COLS:
                errors.append(
                    f"feature_cols_={feature_cols!r}; expected exact deployed order "
                    f"{EXPECTED_FEATURE_COLS!r}"
                )

        for name, values in (("scaler_mean_", mean), ("scaler_scale_", scale)):
            if values is None:
                warnings.append(f"{name} not exposed by detector")
                continue
            array = np.asarray(values)
            if array.ndim != 1 or len(array) != expected_width:
                errors.append(f"{name} shape={array.shape}; expected ({expected_width},)")
            elif not np.isfinite(array.astype("float64")).all():
                errors.append(f"{name} contains non-finite values")
            elif name == "scaler_scale_" and bool((array == 0).any()):
                errors.append("scaler_scale_ contains zero")

        threshold = getattr(detector, "threshold_", None)
        if threshold is None:
            warnings.append("threshold_ not exposed; runtime thresholds will still be validated")
        elif not math.isfinite(float(threshold)) or float(threshold) <= 0:
            errors.append(f"threshold_ must be finite and positive, got {threshold!r}")

        mode = str(getattr(detector, "ae_score_mode", "unknown")).lower()
        if mode not in {"mse", "directional", "unknown"}:
            errors.append(f"unsupported ae_score_mode={mode!r}")
        if mode == "directional":
            for name in ("ae_resid_centre_", "ae_resid_mad_"):
                values = getattr(detector, name, None)
                if values is None:
                    errors.append(f"directional detector missing {name}")
                    continue
                array = np.asarray(values)
                expected_kpis = len(kpi_cols) if kpi_cols is not None else len(MODEL_KPI_COLS)
                if array.ndim != 1 or len(array) != expected_kpis:
                    errors.append(f"{name} shape={array.shape}; expected ({expected_kpis},)")
                elif not np.isfinite(array.astype("float64")).all():
                    errors.append(f"{name} contains non-finite values")

        model = getattr(detector, "model", None)
        model_input = getattr(model, "input_shape", None)
        model_output = getattr(model, "output_shape", None)
        for name, shape in (("model.input_shape", model_input), ("model.output_shape", model_output)):
            if shape is None or isinstance(shape, list):
                continue
            if len(shape) and shape[-1] is not None and int(shape[-1]) != expected_width:
                errors.append(f"{name}={shape!r}; last dimension must be {expected_width}")

        if errors and self._strict_detector_metadata:
            raise ArtifactValidationError("; ".join(errors))
        warnings.extend(errors)
        return warnings

    def _load_mappings(
        self,
        *,
        explicit_cluster: str | Path | None,
        explicit_multi: str | Path | None,
    ) -> _MappingBundle:
        cluster_candidates = sorted(self.run_dir.rglob("training_cluster_mapping*.csv"))
        # Compatibility with older local packages while prioritizing the named
        # production artifact above.
        if not cluster_candidates:
            cluster_candidates = sorted(self.run_dir.rglob("cluster_mapping*.csv"))
        multi_candidates = sorted(self.run_dir.rglob("cluster_multi_head_member_map*.csv"))
        if not multi_candidates:
            multi_candidates = sorted(self.run_dir.rglob("multi_head_member_map*.csv"))

        cluster_path = self._resolve_mapping_path(
            explicit_cluster, cluster_candidates, kind="cluster"
        )
        multi_path = self._resolve_mapping_path(
            explicit_multi, multi_candidates, kind="multi_head"
        )
        cluster_by_psap: dict[int, int] = {}
        heads_by_psap: dict[int, tuple[int, ...]] = {}
        head_source: dict[int, str] = {}
        cluster_rows = 0
        multi_rows = 0

        if cluster_path is not None:
            cluster = self._read_csv_case_insensitive(cluster_path)
            cluster_rows = len(cluster)
            psap_col = self._required_column(cluster, "psap_id", cluster_path)
            cluster_col = self._optional_column(cluster, "cluster", "cluster_id", "phase2_cluster")
            head_col = self._optional_column(cluster, "cluster_head_psap_id", "head_id")
            psaps = self._integer_series(cluster[psap_col], f"{cluster_path}:PSAP_ID")
            if cluster_col:
                clusters = self._integer_series(cluster[cluster_col], f"{cluster_path}:cluster")
                cluster_by_psap = self._consistent_scalar_map(psaps, clusters, "cluster")
            if head_col:
                heads = self._integer_series(cluster[head_col], f"{cluster_path}:head")
                grouped = pd.DataFrame({"psap": psaps, "head": heads}).groupby("psap", sort=False)
                heads_by_psap = {
                    int(psap): tuple(dict.fromkeys(int(value) for value in group["head"]))
                    for psap, group in grouped
                }
                head_source.update({psap: "CLUSTER_MAPPING_HEAD" for psap in heads_by_psap})

        if multi_path is not None:
            multi = self._read_csv_case_insensitive(multi_path)
            multi_rows = len(multi)
            member_col = self._required_column(multi, "member_psap_id", multi_path)
            head_col = self._required_column(multi, "cluster_head_psap_id", multi_path)
            cluster_col = self._optional_column(multi, "cluster", "cluster_id")
            rank_col = self._optional_column(multi, "cluster_head_rank", "head_rank")
            members = self._integer_series(multi[member_col], f"{multi_path}:member")
            heads = self._integer_series(multi[head_col], f"{multi_path}:head")
            normalized = pd.DataFrame({"member": members, "head": heads})
            if rank_col:
                normalized["rank"] = pd.to_numeric(multi[rank_col], errors="coerce")
            else:
                normalized["rank"] = np.arange(len(normalized), dtype="int64")
            normalized = normalized.sort_values(["member", "rank", "head"], kind="stable")
            for member, group in normalized.groupby("member", sort=False):
                heads_by_psap[int(member)] = tuple(
                    dict.fromkeys(int(value) for value in group["head"])
                )
                head_source[int(member)] = "MULTI_HEAD_MAPPING"
            if cluster_col:
                clusters = self._integer_series(multi[cluster_col], f"{multi_path}:cluster")
                multi_cluster = self._consistent_scalar_map(members, clusters, "cluster")
                for psap, cluster_id in multi_cluster.items():
                    prior = cluster_by_psap.get(psap)
                    if prior is not None and prior != cluster_id:
                        raise ArtifactValidationError(
                            f"PSAP {psap} has cluster {prior} in cluster mapping but "
                            f"{cluster_id} in multi-head mapping"
                        )
                    cluster_by_psap[psap] = cluster_id

        discovery = {
            "cluster_mapping_candidates": [str(path) for path in cluster_candidates],
            "multi_head_mapping_candidates": [str(path) for path in multi_candidates],
            "cluster_coverage_psaps": len(cluster_by_psap),
            "head_mapping_coverage_psaps": len(heads_by_psap),
        }
        return _MappingBundle(
            cluster_by_psap=cluster_by_psap,
            heads_by_psap=heads_by_psap,
            head_mapping_source=head_source,
            cluster_path=cluster_path,
            multi_head_path=multi_path,
            cluster_rows=cluster_rows,
            multi_head_rows=multi_rows,
            discovery=discovery,
        )

    def _resolve_mapping_path(
        self,
        explicit: str | Path | None,
        candidates: list[Path],
        *,
        kind: str,
    ) -> Path | None:
        if explicit is not None:
            path = Path(explicit).expanduser().resolve()
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"Explicit {kind} mapping does not exist: {path}")
            return path
        if not candidates:
            return None

        def rank(path: Path) -> tuple[int, int, str]:
            stem = path.stem.lower()
            if kind == "cluster" and stem == "training_cluster_mapping":
                priority = 0
            elif "k15" in stem or "k_15" in stem:
                priority = 1
            elif kind == "multi_head" and stem == "cluster_multi_head_member_map":
                priority = 0
            else:
                priority = 2
            return priority, len(path.relative_to(self.run_dir).parts), str(path)

        return min(candidates, key=rank).resolve()

    @staticmethod
    def _read_csv_case_insensitive(path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        if frame.empty:
            raise ArtifactValidationError(f"Mapping CSV is empty: {path}")
        lowered = [str(column).strip().lower() for column in frame.columns]
        if len(lowered) != len(set(lowered)):
            raise ArtifactValidationError(f"Mapping has case-insensitive duplicate columns: {path}")
        frame.columns = lowered
        return frame

    @staticmethod
    def _required_column(frame: pd.DataFrame, name: str, path: Path) -> str:
        if name not in frame.columns:
            raise ArtifactValidationError(f"{path} is missing required column {name!r}")
        return name

    @staticmethod
    def _optional_column(frame: pd.DataFrame, *names: str) -> str | None:
        return next((name for name in names if name in frame.columns), None)

    @staticmethod
    def _integer_series(values: pd.Series, label: str) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        array = numeric.to_numpy(dtype="float64")
        if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
            raise ArtifactValidationError(f"{label} must contain finite integer values")
        return numeric.astype("int64")

    @staticmethod
    def _consistent_scalar_map(
        keys: pd.Series, values: pd.Series, value_name: str
    ) -> dict[int, int]:
        frame = pd.DataFrame({"key": keys, "value": values}).drop_duplicates()
        conflicts = frame.groupby("key")["value"].nunique()
        conflicts = conflicts[conflicts > 1]
        if not conflicts.empty:
            raise ArtifactValidationError(
                f"Conflicting {value_name} assignments for IDs {conflicts.index.tolist()[:10]}"
            )
        return {
            int(key): int(group["value"].iloc[0])
            for key, group in frame.groupby("key", sort=False)
        }

    @staticmethod
    def _file_hash(path: Path | None) -> str | None:
        if path is None:
            return None
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


__all__ = [
    "ArtifactValidationError",
    "DetectorProtocol",
    "ProductionScorer",
    "ScoreResult",
]
