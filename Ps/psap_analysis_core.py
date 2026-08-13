"""Boundary labels, signaling/KPI association tests, and ML-readiness screening.

This module intentionally does not call a boundary comparison an ML model.
The geometry stage creates conservative ground-truth labels.  Those labels are
then used to test whether pre-routing/network features contain useful signal.
"""

from __future__ import annotations

import csv
import json
import math
import re
import warnings
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyproj import Geod
from scipy.stats import chi2_contingency, mannwhitneyu, pointbiserialr
from shapely.geometry import Point, Polygon, shape
from shapely.ops import nearest_points, unary_union
from shapely.strtree import STRtree


GEOD = Geod(ellps="WGS84")
STRONG_POSITIVE = "DEFINITE_MISROUTE"
STRONG_NEGATIVE = "CORRECT_UNAMBIGUOUS"


def _canon(name: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(name).strip().upper()).strip("_")


def normalize_id(value: object) -> Optional[str]:
    """Normalize Oracle/CSV numeric identifiers without losing text IDs."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().upper()
    if not text or text in {"NAN", "NONE", "NULL", "<NULL>"}:
        return None
    if re.fullmatch(r"[+-]?\d+(?:\.0+)?", text):
        text = text.split(".")[0]
        sign = "-" if text.startswith("-") else ""
        digits = text.lstrip("+-").lstrip("0") or "0"
        return sign + digits
    return text


def normalize_hex(value: object) -> Optional[str]:
    value = normalize_id(value)
    if value is None:
        return None
    return value.replace("0X", "").replace(" ", "")


def _looks_like_header(line: str, expected_tokens: Iterable[str]) -> bool:
    upper = line.upper()
    return "," in line and sum(token.upper() in upper for token in expected_tokens) >= 2


def read_sqlplus_csv(path: Path, expected_tokens: Iterable[str]) -> pd.DataFrame:
    """Read normal SQL*Plus CSV and ref-cursor CSV with harmless preamble lines."""
    if not path.exists():
        return pd.DataFrame()
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if _looks_like_header(line, expected_tokens)),
        None,
    )
    if start is None:
        raise ValueError(f"Could not locate a CSV header in {path}")
    useful = "\n".join(lines[start:])
    frame = pd.read_csv(StringIO(useful), dtype=str, low_memory=False, on_bad_lines="warn")
    frame.columns = [_canon(c) for c in frame.columns]
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    frame = frame.dropna(how="all")
    return frame


def _parse_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)", expand=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_timestamp(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _fix_geometry(geom):
    if geom is None or geom.is_empty:
        return None
    if not geom.is_valid:
        try:
            from shapely.validation import make_valid

            geom = make_valid(geom)
        except Exception:
            geom = geom.buffer(0)
    if geom.geom_type not in {"Polygon", "MultiPolygon"}:
        polygons = [g for g in getattr(geom, "geoms", []) if g.geom_type in {"Polygon", "MultiPolygon"}]
        geom = unary_union(polygons) if polygons else None
    return None if geom is None or geom.is_empty else geom


def _infer_geometry(payload: dict):
    candidate = payload.get("geometry", payload)
    if candidate.get("type"):
        return shape(candidate)
    coordinates = candidate.get("coordinates")
    if coordinates is None:
        raise ValueError("No geometry.coordinates in boundary JSON")
    # Polygon: [ring][point][xy]; MultiPolygon: [polygon][ring][point][xy].
    depth = 0
    probe = coordinates
    while isinstance(probe, list) and probe:
        depth += 1
        probe = probe[0]
    geom_type = "Polygon" if depth == 3 else "MultiPolygon"
    return shape({"type": geom_type, "coordinates": coordinates})


@dataclass
class BoundaryIndex:
    psap_ids: list[str]
    geometries: list
    by_psap: dict[str, object]
    tree: STRtree
    metadata: pd.DataFrame


def build_boundary_index(chunks: pd.DataFrame) -> BoundaryIndex:
    required = {"FCC_PSAP_ID", "CHUNK_NO", "PROPERTIES_CHUNK"}
    missing = required - set(chunks.columns)
    if missing:
        raise ValueError(f"Boundary export is missing columns: {sorted(missing)}")
    work = chunks.copy()
    work["FCC_PSAP_ID_NORM"] = work["FCC_PSAP_ID"].map(normalize_id)
    work["CHUNK_NO"] = pd.to_numeric(work["CHUNK_NO"], errors="coerce")
    work = work.dropna(subset=["FCC_PSAP_ID_NORM", "CHUNK_NO"])
    group_cols = [c for c in ["NENA_ID", "FCC_PSAP_ID_NORM"] if c in work.columns]
    rows = []
    failures = []
    for keys, group in work.sort_values("CHUNK_NO").groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = dict(zip(group_cols, keys))
        raw = "".join(group["PROPERTIES_CHUNK"].fillna("").astype(str))
        try:
            geom = _fix_geometry(_infer_geometry(json.loads(raw)))
            if geom is None:
                raise ValueError("empty/unsupported geometry")
            rows.append({**values, "GEOMETRY": geom, "VALID": True, "ERROR": None})
        except Exception as exc:
            failures.append({**values, "VALID": False, "ERROR": str(exc)[:300]})

    if not rows:
        sample = failures[:3]
        raise ValueError(f"No usable PSAP polygons were parsed. Examples: {sample}")
    parsed = pd.DataFrame(rows)
    dissolved = []
    for psap_id, group in parsed.groupby("FCC_PSAP_ID_NORM"):
        geom = _fix_geometry(unary_union(group["GEOMETRY"].tolist()))
        if geom is not None:
            dissolved.append((psap_id, geom))
    psap_ids = [item[0] for item in dissolved]
    geometries = [item[1] for item in dissolved]
    metadata = pd.concat(
        [parsed.drop(columns="GEOMETRY"), pd.DataFrame(failures)], ignore_index=True
    )
    return BoundaryIndex(
        psap_ids=psap_ids,
        geometries=geometries,
        by_psap=dict(dissolved),
        tree=STRtree(geometries),
        metadata=metadata,
    )


def geodesic_uncertainty_polygon(lon: float, lat: float, radius_m: float, vertices: int = 48):
    radius_m = max(float(radius_m), 1.0)
    azimuths = np.linspace(0.0, 360.0, vertices, endpoint=False)
    lons, lats, _ = GEOD.fwd(
        np.full(vertices, lon), np.full(vertices, lat), azimuths, np.full(vertices, radius_m)
    )
    return Polygon(zip(lons, lats))


def _geodesic_distance_to_geometry(point: Point, geom) -> float:
    if geom is None:
        return np.nan
    if geom.covers(point):
        return 0.0
    near = nearest_points(point, geom)[1]
    _, _, distance = GEOD.inv(point.x, point.y, near.x, near.y)
    return float(distance)


def label_routes(calls: pd.DataFrame, boundaries: BoundaryIndex) -> pd.DataFrame:
    required = {"LATITUDE", "LONGITUDE", "UNCERT_METERS", "FCC_PSAP_ID"}
    missing = required - set(calls.columns)
    if missing:
        raise ValueError(f"PSAPSIM call export is missing: {sorted(missing)}")
    out = calls.copy()
    out["LATITUDE_NUM"] = _parse_numeric(out["LATITUDE"])
    out["LONGITUDE_NUM"] = _parse_numeric(out["LONGITUDE"])
    out["UNCERT_METERS_NUM"] = _parse_numeric(out["UNCERT_METERS"])
    out["ROUTED_FCC_PSAP_ID"] = out["FCC_PSAP_ID"].map(normalize_id)

    results = []
    for row in out[["LATITUDE_NUM", "LONGITUDE_NUM", "UNCERT_METERS_NUM", "ROUTED_FCC_PSAP_ID"]].itertuples(index=False):
        lat, lon, uncertainty, routed = row
        base = {
            "EXPECTED_FCC_PSAP_ID": None,
            "CENTER_PSAP_IDS": None,
            "UNCERTAINTY_OVERLAP_PSAP_IDS": None,
            "UNCERTAINTY_OVERLAP_PSAP_COUNT": 0,
            "ROUTED_BOUNDARY_AVAILABLE": routed in boundaries.by_psap if routed else False,
            "ROUTED_CONTAINS_CENTER": False,
            "ROUTED_INTERSECTS_UNCERTAINTY": False,
            "DISTANCE_TO_ROUTED_BOUNDARY_M": np.nan,
            "ROUTE_INTEGRITY_STATUS": "INVALID_LOCATION",
            "MISROUTE_LABEL": np.nan,
            "LABEL_ELIGIBLE": False,
        }
        if pd.isna(lat) or pd.isna(lon) or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            results.append(base)
            continue
        uncertainty = 0.0 if pd.isna(uncertainty) or uncertainty < 0 else float(uncertainty)
        point = Point(float(lon), float(lat))
        area = geodesic_uncertainty_polygon(float(lon), float(lat), uncertainty)

        point_candidates = boundaries.tree.query(point, predicate="intersects")
        center_ids = sorted(
            {
                boundaries.psap_ids[int(i)]
                for i in point_candidates
                if boundaries.geometries[int(i)].covers(point)
            }
        )
        area_candidates = boundaries.tree.query(area, predicate="intersects")
        overlap_ids = sorted({boundaries.psap_ids[int(i)] for i in area_candidates})
        fully_covering = sorted(
            {
                boundaries.psap_ids[int(i)]
                for i in area_candidates
                if boundaries.geometries[int(i)].covers(area)
            }
        )
        routed_geom = boundaries.by_psap.get(routed)
        routed_center = bool(routed_geom is not None and routed_geom.covers(point))
        routed_overlap = bool(routed_geom is not None and routed_geom.intersects(area))
        distance = _geodesic_distance_to_geometry(point, routed_geom)

        base.update(
            {
                "CENTER_PSAP_IDS": "|".join(center_ids) or None,
                "UNCERTAINTY_OVERLAP_PSAP_IDS": "|".join(overlap_ids) or None,
                "UNCERTAINTY_OVERLAP_PSAP_COUNT": len(overlap_ids),
                "ROUTED_CONTAINS_CENTER": routed_center,
                "ROUTED_INTERSECTS_UNCERTAINTY": routed_overlap,
                "DISTANCE_TO_ROUTED_BOUNDARY_M": distance,
            }
        )

        if routed is None:
            status = "MISSING_ROUTED_PSAP"
        elif routed_geom is None:
            status = "ROUTED_PSAP_BOUNDARY_MISSING"
        elif not center_ids:
            status = "NO_BOUNDARY_AT_REPORTED_CENTER"
        elif len(center_ids) > 1:
            status = "AMBIGUOUS_OVERLAPPING_BOUNDARIES"
        elif routed_geom.covers(area) and fully_covering == [routed]:
            status = STRONG_NEGATIVE
            base["EXPECTED_FCC_PSAP_ID"] = routed
            base["MISROUTE_LABEL"] = 0
            base["LABEL_ELIGIBLE"] = True
        else:
            alternative_full = [pid for pid in fully_covering if pid != routed]
            if not routed_overlap and len(alternative_full) == 1:
                status = STRONG_POSITIVE
                base["EXPECTED_FCC_PSAP_ID"] = alternative_full[0]
                base["MISROUTE_LABEL"] = 1
                base["LABEL_ELIGIBLE"] = True
            elif center_ids[0] == routed:
                status = "ROUTED_PSAP_CONSISTENT_BUT_BOUNDARY_AMBIGUOUS"
                base["EXPECTED_FCC_PSAP_ID"] = routed
            else:
                status = "LIKELY_MISROUTE_BOUNDARY_AMBIGUOUS"
                base["EXPECTED_FCC_PSAP_ID"] = center_ids[0]
        base["ROUTE_INTEGRITY_STATUS"] = status
        results.append(base)

    return pd.concat([out.reset_index(drop=True), pd.DataFrame(results)], axis=1)


def _coalesced(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    present = [name for name in names if name in frame.columns]
    if not present:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    result = frame[present[0]].astype("string")
    for name in present[1:]:
        result = result.fillna(frame[name].astype("string"))
    return result


def _prefix_feature_columns(features: pd.DataFrame, prefix: str, keys: set[str]) -> pd.DataFrame:
    rename = {col: f"{prefix}{col}" for col in features.columns if col not in keys}
    return features.rename(columns=rename)


def attach_feature_table(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    prefix: str,
    max_time_gap_seconds: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Attach a feature table using the best validated exact ID, then reciprocal nearest time."""
    if features.empty:
        return labels.copy(), {"source": prefix.rstrip("_"), "method": "NO_DATA", "coverage": 0.0}
    left = labels.copy()
    right = features.copy().reset_index(drop=True)
    right["_FEATURE_ROW_ID"] = np.arange(len(right))
    candidates = [
        ("PLRF_CID", "HDR_TRID"),
        ("PLRF_CID", "CTID"),
        ("PLRF_CID", "UNIQ911_CID"),
    ]
    scores = []
    for left_key, right_key in candidates:
        if left_key not in left or right_key not in right:
            continue
        lk = left[left_key].map(normalize_id)
        rk = right[right_key].map(normalize_id)
        counts = rk.value_counts(dropna=True)
        unique_keys = set(counts[counts == 1].index)
        coverage = lk.isin(unique_keys).mean() if len(left) else 0.0
        scores.append((coverage, left_key, right_key, lk, rk, unique_keys))
    if scores:
        coverage, left_key, right_key, lk, rk, unique_keys = max(scores, key=lambda x: x[0])
        if coverage >= 0.50:
            left["_JOIN_KEY"] = lk
            usable = right.assign(_JOIN_KEY=rk)
            usable = usable[usable["_JOIN_KEY"].isin(unique_keys)].copy()
            usable = _prefix_feature_columns(usable, prefix, {"_JOIN_KEY"})
            merged = left.merge(usable, on="_JOIN_KEY", how="left", validate="many_to_one")
            merged = merged.drop(columns="_JOIN_KEY")
            return merged, {
                "source": prefix.rstrip("_"),
                "method": f"EXACT:{left_key}={right_key}",
                "coverage": float(coverage),
                "rows": int(len(right)),
            }

    # Strict fallback: same setup ECGI + ESRK and reciprocal nearest timestamp.
    left_time_name = next((c for c in ["CALL_BEGIN_TIME_UTC", "CALL_BEGIN_TIME"] if c in left), None)
    right_time_name = next((c for c in ["CALL_BEGIN_TIME_UTC", "CALL_BEGIN_TIME"] if c in right), None)
    if left_time_name and right_time_name and "SETUP_ECGI_HEX" in left and "SETUP_ECGI_HEX" in right:
        left = left.reset_index(drop=True)
        left["_LABEL_ROW_ID"] = np.arange(len(left))
        left["_JOIN_TIME"] = _parse_timestamp(left[left_time_name])
        right["_JOIN_TIME"] = _parse_timestamp(right[right_time_name])
        left["_JOIN_ECGI"] = left["SETUP_ECGI_HEX"].map(normalize_hex)
        right["_JOIN_ECGI"] = right["SETUP_ECGI_HEX"].map(normalize_hex)
        left["_JOIN_ESRK"] = _coalesced(left, ["GMLC_ESRK", "ESRK"]).map(normalize_id)
        right["_JOIN_ESRK"] = _coalesced(right, ["GMLC_ESRK", "ESRK"]).map(normalize_id)
        by = ["_JOIN_ECGI", "_JOIN_ESRK"]
        lvalid = left.dropna(subset=["_JOIN_TIME", *by]).sort_values("_JOIN_TIME")
        rvalid = right.dropna(subset=["_JOIN_TIME", *by]).sort_values("_JOIN_TIME")
        if len(lvalid) and len(rvalid):
            tolerance = pd.Timedelta(seconds=max_time_gap_seconds)
            forward = pd.merge_asof(
                lvalid[["_LABEL_ROW_ID", "_JOIN_TIME", *by]],
                rvalid[["_FEATURE_ROW_ID", "_JOIN_TIME", *by]],
                on="_JOIN_TIME", by=by, direction="nearest", tolerance=tolerance,
            )
            reverse = pd.merge_asof(
                rvalid[["_FEATURE_ROW_ID", "_JOIN_TIME", *by]],
                lvalid[["_LABEL_ROW_ID", "_JOIN_TIME", *by]],
                on="_JOIN_TIME", by=by, direction="nearest", tolerance=tolerance,
            )
            reciprocal = forward.merge(
                reverse[["_FEATURE_ROW_ID", "_LABEL_ROW_ID"]],
                on=["_FEATURE_ROW_ID", "_LABEL_ROW_ID"], how="inner",
            )[["_LABEL_ROW_ID", "_FEATURE_ROW_ID"]]
            attach = left[["_LABEL_ROW_ID"]].merge(reciprocal, on="_LABEL_ROW_ID", how="left")
            right_prefixed = _prefix_feature_columns(right, prefix, {"_FEATURE_ROW_ID"})
            attach = attach.merge(right_prefixed, on="_FEATURE_ROW_ID", how="left", validate="many_to_one")
            merged = left.merge(attach, on="_LABEL_ROW_ID", how="left", validate="one_to_one")
            merged = merged.drop(columns=[c for c in ["_LABEL_ROW_ID", "_JOIN_TIME", "_JOIN_ECGI", "_JOIN_ESRK"] if c in merged])
            coverage = reciprocal["_LABEL_ROW_ID"].nunique() / max(len(left), 1)
            return merged, {
                "source": prefix.rstrip("_"),
                "method": f"RECIPROCAL_NEAREST:ECGI+ESRK+/-{max_time_gap_seconds}s",
                "coverage": float(coverage),
                "rows": int(len(right)),
            }

    return labels.copy(), {
        "source": prefix.rstrip("_"),
        "method": "NO_SAFE_CALL_LEVEL_JOIN",
        "coverage": 0.0,
        "rows": int(len(right)),
    }


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    p = pd.to_numeric(pvalues, errors="coerce")
    valid = p.dropna().sort_values()
    if valid.empty:
        return pd.Series(np.nan, index=p.index)
    ranked = valid * len(valid) / np.arange(1, len(valid) + 1)
    adjusted = ranked.iloc[::-1].cummin().iloc[::-1].clip(upper=1.0)
    result = pd.Series(np.nan, index=p.index)
    result.loc[adjusted.index] = adjusted
    return result


LEAKAGE_TOKENS = {
    "LATITUDE", "LONGITUDE", "ROUTED_FCC_PSAP_ID", "EXPECTED_FCC_PSAP_ID",
    "CENTER_PSAP_IDS", "UNCERTAINTY_OVERLAP_PSAP_IDS", "ROUTE_INTEGRITY_STATUS",
    "MISROUTE_LABEL", "LABEL_ELIGIBLE", "ROUTED_CONTAINS_CENTER",
    "ROUTED_INTERSECTS_UNCERTAINTY", "DISTANCE_TO_ROUTED_BOUNDARY_M",
    "FCC_PSAP_ID", "PSAP_ID", "PSAP_NAME",
}
IDENTIFIER_TOKENS = {"PLRF_CID", "HDR_TRID", "CTID", "UNIQ911_CID", "SCHEDULE_ID", "TESTSIM_ID"}
POST_OUTCOME_TOKENS = {
    "COMPLETE_CALL", "CALL_DURATION_SEC", "ESRK_MATCH_YN", "VALIDATION_PASSED",
    "EMAIL_YN", "TEXTMSG_YN", "SUMMARY_YN", "CLASS_OF_SERVICE_YN",
    "ROUTE_STATUS_GMLC", "LRR_INVITE_STATUS", "SIP_STATUS",
    "DEFAULT_ROUTED_CALL", "PSAP_SIM_CALL", "EXCEPTION_FLAG",
    "DR_CNT", "DR_CALL_CNT", "Z_RELAYED2PSAP",
}


def feature_role(column: str) -> str:
    base = column.removeprefix("GMLC_").removeprefix("LSR_")
    if (
        base in LEAKAGE_TOKENS
        or column in LEAKAGE_TOKENS
        or any(token in base for token in (
            "LATITUDE", "LONGITUDE", "EXPECTED_FCC", "ROUTED_FCC",
            "ROUTE_INTEGRITY", "MISROUTE", "LABEL_ELIGIBLE",
            "BOUNDARY_AVAILABLE", "CONTAINS_CENTER", "INTERSECTS_UNCERTAINTY",
            "UNCERTAINTY_OVERLAP", "DISTANCE_TO_ROUTED_BOUNDARY",
        ))
    ):
        return "LABEL_OR_GEOMETRY_LEAKAGE"
    if base in IDENTIFIER_TOKENS or "FEATURE_ROW_ID" in base:
        return "TECHNICAL_JOIN_KEY"
    if base in {"LATITUDE_NUM", "LONGITUDE_NUM", "UNCERT_METERS_NUM"}:
        return "DERIVED_DUPLICATE"
    if base in {
        "CALL_DATE", "CALL_BEGIN_TIME", "CALL_END_TIME", "CALL_BEGIN_TIME_UTC",
        "CALL_END_TIME_UTC", "LOCATE_BEGIN_TIME", "LOCATE_BEGIN_TIME_UTC",
        "DATETIME_INS", "EVENT_TIME_UTC",
    }:
        return "RAW_TIME_KEY"
    if base in POST_OUTCOME_TOKENS:
        return "POST_ROUTING_CONSEQUENCE"
    return "PREDICTOR_CANDIDATE"


def infer_feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    numeric_tokens = (
        "UNCERT", "CONFIDENCE", "LOCATE_TIME", "COUNT", "_CNT", "ATTEMPTS",
        "DURATION", "P2_", "FAIL_LOCATE", "P2BEYOND", "DR_", "E2VALIDATED",
        "RELAYED", "RANGE",
    )
    for col in frame.columns:
        if feature_role(col) not in {"PREDICTOR_CANDIDATE", "POST_ROUTING_CONSEQUENCE"}:
            continue
        converted = _parse_numeric(frame[col])
        numeric_fraction = converted.notna().mean()
        if any(token in col for token in numeric_tokens) and numeric_fraction >= 0.50:
            numeric.append(col)
        elif frame[col].nunique(dropna=True) >= 2:
            categorical.append(col)
    return numeric, categorical


def analyze_numeric(frame: pd.DataFrame, label: str = "MISROUTE_LABEL") -> pd.DataFrame:
    rows = []
    numeric, _ = infer_feature_types(frame)
    y = pd.to_numeric(frame[label], errors="coerce")
    for col in numeric:
        x = _parse_numeric(frame[col])
        valid = x.notna() & y.notna()
        x0, x1 = x[valid & (y == 0)], x[valid & (y == 1)]
        if len(x0) < 5 or len(x1) < 5 or x[valid].nunique() < 2:
            continue
        try:
            u, p = mannwhitneyu(x1, x0, alternative="two-sided")
            auc = u / (len(x1) * len(x0))
            cliff = 2 * auc - 1
        except Exception:
            p = auc = cliff = np.nan
        try:
            corr, corr_p = pointbiserialr(y[valid], x[valid])
        except Exception:
            corr = corr_p = np.nan
        rows.append(
            {
                "FEATURE": col,
                "ROLE": feature_role(col),
                "N_CORRECT": len(x0),
                "N_MISROUTE": len(x1),
                "MISSING_PCT": 100 * (1 - valid.mean()),
                "CORRECT_MEDIAN": x0.median(),
                "MISROUTE_MEDIAN": x1.median(),
                "MEDIAN_DIFFERENCE": x1.median() - x0.median(),
                "UNIVARIATE_AUC": auc,
                "CLIFFS_DELTA": cliff,
                "POINT_BISERIAL_R": corr,
                "P_VALUE": min(p, corr_p) if not pd.isna(corr_p) else p,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["FDR_Q_VALUE"] = benjamini_hochberg(result["P_VALUE"])
        result["ABS_EFFECT"] = result["CLIFFS_DELTA"].abs()
        result = result.sort_values(["ABS_EFFECT", "FDR_Q_VALUE"], ascending=[False, True])
    return result


def _cramers_v(table: pd.DataFrame) -> tuple[float, float]:
    if min(table.shape) < 2:
        return np.nan, np.nan
    try:
        chi2, p, _, _ = chi2_contingency(table, correction=False)
        n = table.to_numpy().sum()
        phi2 = chi2 / n
        r, k = table.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
        rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
        kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
        denom = min(kcorr - 1, rcorr - 1)
        return (math.sqrt(phi2corr / denom) if denom > 0 else np.nan), p
    except Exception:
        return np.nan, np.nan


def analyze_categorical(
    frame: pd.DataFrame,
    label: str = "MISROUTE_LABEL",
    min_level_calls: int = 5,
    max_levels: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, categorical = infer_feature_types(frame)
    summary_rows, level_rows = [], []
    y = pd.to_numeric(frame[label], errors="coerce")
    baseline = y.mean()
    for col in categorical:
        x = frame[col].astype("string").fillna("<MISSING>").str.slice(0, 200)
        valid = y.notna()
        x, yy = x[valid], y[valid].astype(int)
        if x.nunique() > max_levels:
            keep = set(x.value_counts().head(max_levels - 1).index)
            x = x.where(x.isin(keep), "<OTHER_RARE>")
        table = pd.crosstab(x, yy)
        if 0 not in table:
            table[0] = 0
        if 1 not in table:
            table[1] = 0
        v, p = _cramers_v(table[[0, 1]])
        summary_rows.append(
            {
                "FEATURE": col,
                "ROLE": feature_role(col),
                "LEVELS_TESTED": len(table),
                "MISSING_PCT": 100 * (frame[col].isna().mean()),
                "CRAMERS_V": v,
                "P_VALUE": p,
            }
        )
        for level, counts in table.iterrows():
            total = int(counts[0] + counts[1])
            if total < min_level_calls:
                continue
            rate = (counts[1] + 0.5) / (total + 1.0)
            level_rows.append(
                {
                    "FEATURE": col,
                    "LEVEL": level,
                    "CALLS": total,
                    "CORRECT": int(counts[0]),
                    "MISROUTES": int(counts[1]),
                    "MISROUTE_RATE": counts[1] / total,
                    "SMOOTHED_RATE_RATIO": rate / max(baseline, 1e-12),
                }
            )
    summary = pd.DataFrame(summary_rows)
    levels = pd.DataFrame(level_rows)
    if not summary.empty:
        summary["FDR_Q_VALUE"] = benjamini_hochberg(summary["P_VALUE"])
        summary = summary.sort_values(["CRAMERS_V", "FDR_Q_VALUE"], ascending=[False, True])
    if not levels.empty:
        levels = levels.sort_values(["SMOOTHED_RATE_RATIO", "MISROUTES"], ascending=[False, False])
    return summary, levels


def _wilson_lower(successes: pd.Series, totals: pd.Series, z: float = 1.96) -> pd.Series:
    p = successes / totals
    denominator = 1 + z**2 / totals
    centre = p + z**2 / (2 * totals)
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * totals)) / totals)
    return (centre - spread) / denominator


def entity_patterns(frame: pd.DataFrame, min_calls: int = 10) -> pd.DataFrame:
    candidates = [
        "SETUP_ECGI_HEX", "INVITE_ECGI_HEX", "LOCATE_ECGI_HEX", "USID", "ENBID",
        "GMLC_ESRK", "ESRK", "GMLC_GMLC_ESRK", "GMLC_ESRK", "GMLC_MME_NAME",
        "LSR_MME_NAME", "GMLC_GMLC_VENDOR", "GMLC_ORIG_AUD_HOST",
        "ROUTED_FCC_PSAP_ID", "EXPECTED_FCC_PSAP_ID",
    ]
    rows = []
    work = frame[frame["LABEL_ELIGIBLE"]].copy()
    for col in dict.fromkeys(candidates):
        if col not in work:
            continue
        grouped = (
            work.assign(_ENTITY=work[col].astype("string").fillna("<MISSING>"))
            .groupby("_ENTITY", dropna=False)
            .agg(
                CALLS=("MISROUTE_LABEL", "size"),
                MISROUTES=("MISROUTE_LABEL", "sum"),
                ACTIVE_DAYS=("CALL_DATE", "nunique") if "CALL_DATE" in work else ("MISROUTE_LABEL", "size"),
            )
            .reset_index()
        )
        grouped = grouped[grouped["CALLS"] >= min_calls]
        if grouped.empty:
            continue
        grouped["MISROUTE_RATE"] = grouped["MISROUTES"] / grouped["CALLS"]
        grouped["WILSON_LOWER_95"] = _wilson_lower(grouped["MISROUTES"], grouped["CALLS"])
        grouped.insert(0, "ENTITY_TYPE", col)
        grouped = grouped.rename(columns={"_ENTITY": "ENTITY"})
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(
        ["MISROUTES", "WILSON_LOWER_95", "CALLS"], ascending=[False, False, False]
    )


def screening_features(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric, categorical = infer_feature_types(frame)
    numeric = [c for c in numeric if feature_role(c) == "PREDICTOR_CANDIDATE"]
    categorical = [c for c in categorical if feature_role(c) == "PREDICTOR_CANDIDATE"]
    categorical = [
        c for c in categorical
        if 2 <= frame[c].nunique(dropna=True) <= min(500, max(20, len(frame) // 5))
    ]
    return numeric, categorical


def _score_at_fraction(y_true: np.ndarray, scores: np.ndarray, fraction: float) -> dict:
    n = max(1, int(math.ceil(len(scores) * fraction)))
    chosen = np.argsort(scores)[-n:]
    positives = y_true.sum()
    precision = y_true[chosen].mean() if n else np.nan
    recall = y_true[chosen].sum() / positives if positives else np.nan
    baseline = y_true.mean()
    return {
        f"precision_top_{int(fraction*100)}pct": float(precision),
        f"recall_top_{int(fraction*100)}pct": float(recall),
        f"lift_top_{int(fraction*100)}pct": float(precision / baseline) if baseline else np.nan,
    }


def run_screening_model(frame: pd.DataFrame, random_state: int = 42) -> tuple[dict, pd.DataFrame]:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    work = frame[frame["LABEL_ELIGIBLE"]].copy()
    y = work["MISROUTE_LABEL"].astype(int)
    positives, negatives = int(y.sum()), int((1 - y).sum())
    result = {
        "status": "SKIPPED",
        "reason": None,
        "rows": len(work),
        "positives": positives,
        "negatives": negatives,
        "positive_rate": float(y.mean()) if len(y) else np.nan,
    }
    if positives < 20 or negatives < 50:
        result["reason"] = "Need at least 20 strong misroutes and 50 strong correct calls for screening."
        return result, pd.DataFrame()

    numeric, categorical = screening_features(work)
    features = numeric + categorical
    if not features:
        result["reason"] = "No eligible pre-routing/network features were available."
        return result, pd.DataFrame()

    times = _parse_timestamp(_coalesced(work, ["CALL_BEGIN_TIME_UTC", "CALL_BEGIN_TIME", "CALL_DATE"]))
    order = np.argsort(times.fillna(pd.Timestamp("1970-01-01", tz="UTC")).to_numpy())
    cut = int(len(order) * 0.70)
    train_idx, test_idx = order[:cut], order[cut:]
    if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
        result["reason"] = "Chronological train/test split does not contain both labels in both periods."
        return result, pd.DataFrame()

    x = work[features].copy()
    for col in numeric:
        x[col] = _parse_numeric(x[col])
    for col in categorical:
        x[col] = x[col].astype("string")

    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median", add_indicator=True)),
         ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")),
         ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]
    )
    prep = ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)],
        remainder="drop",
    )
    models = {
        "logistic": LogisticRegression(
            max_iter=1500, class_weight="balanced", solver="liblinear", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=random_state,
        ),
    }
    evaluations = []
    fitted = {}
    for name, model in models.items():
        pipe = Pipeline([("prepare", prep), ("model", model)])
        pipe.fit(x.iloc[train_idx], y.iloc[train_idx])
        scores = pipe.predict_proba(x.iloc[test_idx])[:, 1]
        y_test = y.iloc[test_idx].to_numpy()
        metric = {
            "model": name,
            "roc_auc": float(roc_auc_score(y_test, scores)),
            "pr_auc": float(average_precision_score(y_test, scores)),
            "test_positive_rate": float(y_test.mean()),
        }
        metric.update(_score_at_fraction(y_test, scores, 0.05))
        metric.update(_score_at_fraction(y_test, scores, 0.10))
        evaluations.append(metric)
        fitted[name] = pipe

    best = max(evaluations, key=lambda item: item["pr_auc"])
    best_pipe = fitted[best["model"]]
    sample_idx = test_idx[-min(len(test_idx), 5000):]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        importance = permutation_importance(
            best_pipe, x.iloc[sample_idx], y.iloc[sample_idx], scoring="average_precision",
            n_repeats=3, random_state=random_state, n_jobs=-1,
        )
    importance_frame = pd.DataFrame(
        {
            "FEATURE": features,
            "PERMUTATION_IMPORTANCE_MEAN": importance.importances_mean,
            "PERMUTATION_IMPORTANCE_STD": importance.importances_std,
            "ROLE": [feature_role(c) for c in features],
        }
    ).sort_values("PERMUTATION_IMPORTANCE_MEAN", ascending=False)

    baseline = best["test_positive_rate"]
    evidence = (
        best["pr_auc"] >= baseline + 0.03
        and best["pr_auc"] / max(baseline, 1e-12) >= 2.0
        and best["lift_top_10pct"] >= 2.0
    )
    result.update(
        {
            "status": "COMPLETED",
            "features": features,
            "numeric_features": numeric,
            "categorical_features": categorical,
            "chronological_split_index": cut,
            "models": evaluations,
            "best_model": best["model"],
            "feature_signal_assessment": "PROMISING" if evidence else "NOT_YET_STRONG",
            "warning": "Screening only; this is not a final production model or unbiased performance claim.",
        }
    )
    return result, importance_frame


def targeted_column_candidates(inventory: pd.DataFrame, exported_columns: set[str]) -> pd.DataFrame:
    if inventory.empty or "COLUMN_NAME" not in inventory:
        return pd.DataFrame()
    keywords = re.compile(
        r"MME|SIP|DIAMETER|RESULT|STATUS|CAUSE|ERROR|FAIL|TIMEOUT|TIMER|"
        r"ESRK|ECGI|ROUTE|PSAP|LOCAT|UNCERT|HOST|NODE|TRANSFER|ANSWER|"
        r"P2|DEFAULT|VALIDAT|COMPLETE|CALL_ID|CTID|TRID",
        re.I,
    )
    work = inventory.copy()
    work["COLUMN_NAME"] = work["COLUMN_NAME"].map(_canon)
    work = work[work["COLUMN_NAME"].str.contains(keywords, na=False)]
    work = work[~work["COLUMN_NAME"].isin(exported_columns)]
    return work.sort_values(["TABLE_NAME", "COLUMN_ID"])


def _safe_json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def save_plots(labels: pd.DataFrame, cat_summary: pd.DataFrame, numeric_summary: pd.DataFrame, output_dir: Path):
    sns.set_theme(style="whitegrid")
    counts = labels["ROUTE_INTEGRITY_STATUS"].value_counts().head(12).sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot.barh(ax=ax, color="#3478bf")
    ax.set_title("Boundary route-integrity outcomes")
    ax.set_xlabel("Calls")
    fig.tight_layout()
    fig.savefig(output_dir / "01_boundary_outcomes.png", dpi=160)
    plt.close(fig)

    if not cat_summary.empty:
        view = cat_summary.head(15).sort_values("CRAMERS_V")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(view["FEATURE"], view["CRAMERS_V"], color="#e87722")
        ax.set_title("Strongest categorical relationships with definite misroute")
        ax.set_xlabel("Bias-corrected Cramer's V")
        fig.tight_layout()
        fig.savefig(output_dir / "02_categorical_associations.png", dpi=160)
        plt.close(fig)

    if not numeric_summary.empty:
        view = numeric_summary.head(15).sort_values("ABS_EFFECT")
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = np.where(view["CLIFFS_DELTA"] >= 0, "#cc3311", "#0077bb")
        ax.barh(view["FEATURE"], view["CLIFFS_DELTA"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Numeric feature effect: misroute versus correct")
        ax.set_xlabel("Cliff's delta")
        fig.tight_layout()
        fig.savefig(output_dir / "03_numeric_associations.png", dpi=160)
        plt.close(fig)


def build_recommendations(
    labels: pd.DataFrame,
    joins: list[dict],
    screening: dict,
    entity: pd.DataFrame,
) -> list[str]:
    recommendations = []
    status_counts = labels["ROUTE_INTEGRITY_STATUS"].value_counts()
    positives = int(status_counts.get(STRONG_POSITIVE, 0))
    negatives = int(status_counts.get(STRONG_NEGATIVE, 0))
    eligible_rate = (positives + negatives) / max(len(labels), 1)
    if eligible_rate < 0.80:
        recommendations.append(
            "Improve label coverage first: inspect invalid coordinates, missing routed-PSAP boundaries, and boundary-ambiguous calls."
        )
    if positives < 50:
        recommendations.append(
            "Export a longer date range; fewer than 50 definite misroutes is too small for dependable ML design."
        )
    for join in joins:
        if join.get("coverage", 0) < 0.80:
            recommendations.append(
                f"{join.get('source')} call-level join coverage is {join.get('coverage', 0):.1%}; "
                "validate a shared transaction ID before using its signaling fields."
            )
    if screening.get("status") == "COMPLETED":
        if screening.get("feature_signal_assessment") == "PROMISING":
            recommendations.append(
                "Current pre-routing/network features show promising chronological signal; next design an entity-hour future-window model and grouped time validation."
            )
        else:
            recommendations.append(
                "Current features do not yet separate future misroutes strongly; target MME/GMLC route decision, ESRK lookup/version, SIP route, and final PSAP-leg fields from the selected-table inventory."
            )
    else:
        recommendations.append(f"Screening model was skipped: {screening.get('reason')}")
    if not entity.empty:
        top = entity.iloc[0]
        if top["MISROUTES"] >= max(10, positives * 0.25):
            recommendations.append(
                "Misroutes are concentrated in a repeated network entity; test a deterministic mapping/configuration alarm before choosing ML."
            )
    return list(dict.fromkeys(recommendations))


def run_analysis(root_dir: str | Path, run_screening: bool = True, max_calls: Optional[int] = None) -> dict:
    root = Path(root_dir)
    data_dir = root / "data"
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    calls = read_sqlplus_csv(data_dir / "psapsim_calls.csv", ["PLRF_CID", "LATITUDE", "FCC_PSAP_ID"])
    chunks = read_sqlplus_csv(
        data_dir / "psap_boundaries_chunks.csv", ["FCC_PSAP_ID", "CHUNK_NO", "PROPERTIES_CHUNK"]
    )
    if max_calls:
        calls = calls.head(max_calls).copy()
    boundary_index = build_boundary_index(chunks)
    labels = label_routes(calls, boundary_index)
    labels["CALL_DATE"] = _parse_timestamp(
        _coalesced(labels, ["CALL_BEGIN_TIME_UTC", "CALL_BEGIN_TIME", "CALL_DATE"])
    ).dt.date.astype("string")
    event_time = _parse_timestamp(
        _coalesced(labels, ["CALL_BEGIN_TIME_UTC", "CALL_BEGIN_TIME", "DATETIME_INS"])
    )
    labels["EVENT_TIME_UTC"] = event_time
    labels["EVENT_HOUR_UTC"] = event_time.dt.hour.astype("Int64")
    labels["EVENT_DAY_OF_WEEK"] = event_time.dt.day_name().astype("string")
    labels["EVENT_IS_WEEKEND"] = event_time.dt.dayofweek.isin([5, 6]).astype("Int64")

    joins = []
    merged = labels
    gmlc = read_sqlplus_csv(
        data_dir / "gmlc_psapsim_features.csv", ["HDR_TRID", "CALL_BEGIN_TIME_UTC", "SETUP_ECGI_HEX"]
    ) if (data_dir / "gmlc_psapsim_features.csv").exists() else pd.DataFrame()
    merged, gmlc_join = attach_feature_table(merged, gmlc, "GMLC_")
    joins.append(gmlc_join)
    lsr = read_sqlplus_csv(
        data_dir / "lsr_csr_features.csv", ["CALL_DATE", "MME_NAME", "PSAP_ID"]
    ) if (data_dir / "lsr_csr_features.csv").exists() else pd.DataFrame()
    merged, lsr_join = attach_feature_table(merged, lsr, "LSR_")
    joins.append(lsr_join)

    strong = merged[merged["LABEL_ELIGIBLE"]].copy()
    numeric_summary = analyze_numeric(strong)
    categorical_summary, categorical_levels = analyze_categorical(strong)
    entities = entity_patterns(merged)
    screening, importance = (
        run_screening_model(merged) if run_screening else (
            {"status": "SKIPPED", "reason": "Disabled in notebook configuration"}, pd.DataFrame()
        )
    )

    inventory_path = data_dir / "selected_table_columns.csv"
    inventory = read_sqlplus_csv(
        inventory_path, ["TABLE_NAME", "COLUMN_NAME", "DATA_TYPE"]
    ) if inventory_path.exists() else pd.DataFrame()
    exported = set(merged.columns) | set(gmlc.columns) | set(lsr.columns)
    next_columns = targeted_column_candidates(inventory, exported)
    recommendations = build_recommendations(merged, joins, screening, entities)

    label_summary = (
        merged.groupby("ROUTE_INTEGRITY_STATUS", dropna=False)
        .agg(CALLS=("ROUTE_INTEGRITY_STATUS", "size"),
             DISTINCT_DAYS=("CALL_DATE", "nunique"),
             DISTINCT_ECGIS=("SETUP_ECGI_HEX", "nunique") if "SETUP_ECGI_HEX" in merged else ("ROUTE_INTEGRITY_STATUS", "size"))
        .reset_index()
        .sort_values("CALLS", ascending=False)
    )

    merged.to_csv(output_dir / "calls_with_boundary_labels_and_features.csv", index=False)
    strong.to_csv(output_dir / "strong_labeled_calls.csv", index=False)
    label_summary.to_csv(output_dir / "label_summary.csv", index=False)
    pd.DataFrame(joins).to_csv(output_dir / "join_quality.csv", index=False)
    numeric_summary.to_csv(output_dir / "numeric_associations.csv", index=False)
    categorical_summary.to_csv(output_dir / "categorical_associations.csv", index=False)
    categorical_levels.to_csv(output_dir / "categorical_level_patterns.csv", index=False)
    entities.to_csv(output_dir / "entity_patterns.csv", index=False)
    importance.to_csv(output_dir / "screening_feature_importance.csv", index=False)
    next_columns.to_csv(output_dir / "targeted_next_columns.csv", index=False)
    boundary_index.metadata.to_csv(output_dir / "boundary_parse_quality.csv", index=False)
    save_plots(merged, categorical_summary, numeric_summary, output_dir)

    report = {
        "input_rows": int(len(calls)),
        "usable_boundary_psaps": int(len(boundary_index.psap_ids)),
        "strong_correct": int((merged["ROUTE_INTEGRITY_STATUS"] == STRONG_NEGATIVE).sum()),
        "definite_misroutes": int((merged["ROUTE_INTEGRITY_STATUS"] == STRONG_POSITIVE).sum()),
        "strong_label_coverage": float(merged["LABEL_ELIGIBLE"].mean()),
        "join_quality": joins,
        "screening": screening,
        "recommendations": recommendations,
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(report, indent=2, default=_safe_json), encoding="utf-8"
    )
    markdown = [
        "# PSAP routing-integrity analysis summary",
        "",
        f"- Input calls: {report['input_rows']:,}",
        f"- Strong correct routes: {report['strong_correct']:,}",
        f"- Definite misroutes: {report['definite_misroutes']:,}",
        f"- Strong-label coverage: {report['strong_label_coverage']:.1%}",
        f"- Screening status: {screening.get('status')}",
        f"- Feature signal: {screening.get('feature_signal_assessment', 'NOT_ASSESSED')}",
        "",
        "## Recommended next actions",
        "",
        *[f"- {item}" for item in recommendations],
    ]
    (output_dir / "analysis_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    return {
        "report": report,
        "label_summary": label_summary,
        "numeric_associations": numeric_summary,
        "categorical_associations": categorical_summary,
        "categorical_levels": categorical_levels,
        "entity_patterns": entities,
        "feature_importance": importance,
        "targeted_next_columns": next_columns,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=r"C:\temp\psap_route_integrity_v1")
    parser.add_argument("--no-screening", action="store_true")
    parser.add_argument("--max-calls", type=int)
    args = parser.parse_args()
    result = run_analysis(args.root, run_screening=not args.no_screening, max_calls=args.max_calls)
    print(json.dumps(result["report"], indent=2, default=_safe_json))
