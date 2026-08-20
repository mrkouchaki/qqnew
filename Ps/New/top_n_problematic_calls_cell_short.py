# ============================================================
# TOP-N PROBLEMATIC PATTERNS — INDIVIDUAL DEFINITE-MISROUTE CALLS
#
# TOP_N = 1 selects the first row of misroute_top100_network_expert.csv
# (the row containing 8 problematic calls in the current report).
# This cell never loads the 92,674 correctly routed control calls.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display, Markdown


# Change only this value.
TOP_N = 1

OUTPUT_DIR = Path(
    globals().get(
        "OUTPUT_DIR",
        r"C:\temp\gmlc_v2\outputs_psap_rca_v4",
    )
)

TOP100_PATH = OUTPUT_DIR / "misroute_top100_network_expert.csv"
CALLS_PATH = OUTPUT_DIR / "definite_misroutes_v4.csv"


def clean_name(value):
    return str(value).strip().upper().replace(" ", "_")


def norm_id(series):
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.strip('"')
        .str.replace(r"\.0$", "", regex=True)
        .str.upper()
    )


def mode_text(series):
    values = series.astype("string").fillna("").str.strip()
    values = values[~values.isin(["", "<MISSING>", "NAN", "NONE"])]
    return "<MISSING>" if values.empty else values.value_counts().index[0]


if TOP_N < 1:
    raise ValueError("TOP_N must be at least 1.")
if not TOP100_PATH.exists():
    raise FileNotFoundError(f"Missing Top-100 file: {TOP100_PATH}")
if not CALLS_PATH.exists():
    raise FileNotFoundError(f"Missing call-level file: {CALLS_PATH}")


# ------------------------------------------------------------
# 1. Select the first TOP_N ranked problematic patterns
# ------------------------------------------------------------

top100 = pd.read_csv(TOP100_PATH, dtype="string", low_memory=False)
top100.columns = [clean_name(c) for c in top100.columns]

pattern_keys = [
    "FCC_PSAP_ID",
    "EXPECTED_FCC_PSAP_ID",
    "SETUP_ECGI_HEX",
    "USID",
    "ESRK",
    "CALL_POPULATION_TAG",
]

required_top = {"PRESENTATION_RANK", *pattern_keys}
missing = sorted(required_top - set(top100.columns))
if missing:
    raise ValueError(f"Top-100 file is missing columns: {missing}")

top100["PRESENTATION_RANK"] = pd.to_numeric(
    top100["PRESENTATION_RANK"], errors="coerce"
).astype("Int64")

patterns = (
    top100.sort_values("PRESENTATION_RANK", kind="stable")
    .head(TOP_N)
    .copy()
)

match_columns = []
for column in pattern_keys:
    match_column = f"__MATCH_{column}"
    patterns[match_column] = norm_id(patterns[column])
    match_columns.append(match_column)

top_metadata = {
    "PSAP_NAME": "ROUTED_PSAP_NAME",
    "EXPECTED_PSAP_NAME": "EXPECTED_PSAP_NAME",
    "MARKET": "MARKET",
    "MARKET_CLUSTER": "MARKET_CLUSTER",
    "REGION": "REGION",
    "STATE": "STATE",
    "COUNTY": "COUNTY",
    "MISROUTE_CALLS": "TOP100_MISROUTE_CALLS",
    "PROBABLE_CAUSE_CATEGORY": "TOP100_PROBABLE_CAUSE",
    "EVIDENCE_CONCLUSION": "TOP100_EVIDENCE_CONCLUSION",
    "EVIDENCE_SUMMARY": "TOP100_EVIDENCE_SUMMARY",
}

available_metadata = {
    source: target
    for source, target in top_metadata.items()
    if source in patterns.columns
}

pattern_lookup = patterns[
    ["PRESENTATION_RANK", *match_columns, *available_metadata]
].rename(columns=available_metadata)

if pattern_lookup.duplicated(match_columns).any():
    print(
        "WARNING: Selected ranks contain identical complete cohort keys; "
        "the first rank will be used for those calls."
    )
    pattern_lookup = pattern_lookup.drop_duplicates(match_columns, keep="first")


# ------------------------------------------------------------
# 2. Read only useful columns from definite_misroutes_v4.csv
# ------------------------------------------------------------

aliases = {
    "SIP_METHOD": ["SIP_SIP_METHOD", "SIP_METHOD"],
    "IMS_SIP_METHOD": ["IMS__SIP_METHOD", "IMS_SIP_METHOD"],
    "IMS_REGISTER_PCSCF_STATUS": [
        "IMS__REGISTER_PCSCF_STATUS",
        "IMS_REGISTER_PCSCF_STATUS",
    ],
    "IMS_REGISTER_PCSCF_REASONCODES": [
        "IMS__REGISTER_PCSCF_REASONCODES",
        "IMS_REGISTER_PCSCF_REASONCODES",
    ],
    "RAW_CORRELATION_STATUS": [
        "RAW__CORRELATION_STATUS",
        "RAW_CORRELATION_STATUS",
    ],
    "RAW_ECSCF_STATUS": ["RAW__ECSCF_STATUS", "RAW_ECSCF_STATUS"],
    "RAW_REGISTER_PCSCF_STATUS": [
        "RAW__REGISTER_PCSCF_STATUS",
        "RAW_REGISTER_PCSCF_STATUS",
    ],
    "RAW_NON_REGISTER_PCSCF_STATUS": [
        "RAW__NON_REGISTER_PCSCF_STATUS",
        "RAW_NON_REGISTER_PCSCF_STATUS",
    ],
}

direct_columns = {
    *pattern_keys,
    "CALL_KEY",
    "UNIQ911_CID",
    "GMLC_ROW_ID",
    "CALL_GRAIN_METHOD",
    "ROUTE_INTEGRITY_STATUS",
    "MISROUTE_LABEL",
    "CALL_BEGIN_TIME_UTC",
    "CALL_BEGIN_DATETIME",
    "CALL_DATETIME",
    "CALL_DATE_UTC",
    "LATITUDE",
    "LONGITUDE",
    "UNCERT_METERS",
    "ROUTED_INTERSECTS_UNCERT_AREA",
    "DISTANCE_TO_ROUTED_BOUNDARY_M",
    "DISTANCE_TO_EXPECTED_BOUNDARY_M",
    "ROUTE_STATUS_GMLC",
    "ROUTE_ESINET",
    "ROUTE_ESZ",
    "ROUTE_FALLBACK",
    "DEFAULT_ROUTED_CALL",
    "LBR_PSAP_DIFF",
    "SIP_STATUS",
    "FAILURE_SHORT_DR",
}

wanted = direct_columns | {
    candidate
    for candidate_list in aliases.values()
    for candidate in candidate_list
}

raw_header = pd.read_csv(
    CALLS_PATH,
    nrows=0,
    encoding_errors="replace",
).columns
header_map = {clean_name(c): c for c in raw_header}

missing_call_keys = sorted(set(pattern_keys) - set(header_map))
if missing_call_keys:
    raise ValueError(
        f"definite_misroutes_v4.csv is missing matching columns: "
        f"{missing_call_keys}"
    )

usecols = [
    original
    for cleaned, original in header_map.items()
    if cleaned in wanted
]

calls = pd.read_csv(
    CALLS_PATH,
    usecols=usecols,
    dtype="string",
    low_memory=False,
    encoding_errors="replace",
)
calls.columns = [clean_name(c) for c in calls.columns]

# Canonicalize prefixed IMS/RAW names without changing their values.
for output_column, candidates in aliases.items():
    if output_column in calls.columns:
        continue
    source_column = next((c for c in candidates if c in calls.columns), None)
    if source_column:
        calls[output_column] = calls[source_column]

# Keep only explicit definite misroutes if the label fields are available.
if "ROUTE_INTEGRITY_STATUS" in calls.columns:
    calls = calls[
        norm_id(calls["ROUTE_INTEGRITY_STATUS"]).eq("DEFINITE_MISROUTE")
    ].copy()
elif "MISROUTE_LABEL" in calls.columns:
    calls = calls[
        pd.to_numeric(calls["MISROUTE_LABEL"], errors="coerce").eq(1)
    ].copy()

for source_column, match_column in zip(pattern_keys, match_columns):
    calls[match_column] = norm_id(calls[source_column])

problem_calls = calls.merge(
    pattern_lookup,
    on=match_columns,
    how="inner",
    validate="many_to_one",
)

if problem_calls.empty:
    raise ValueError(
        "No definite-misroute calls matched the selected Top-N cohort keys."
    )


# ------------------------------------------------------------
# 3. Add time and geometry fields needed for analysis/map later
# ------------------------------------------------------------

time_source = next(
    (
        c
        for c in [
            "CALL_BEGIN_TIME_UTC",
            "CALL_BEGIN_DATETIME",
            "CALL_DATETIME",
            "CALL_DATE_UTC",
        ]
        if c in problem_calls.columns
    ),
    None,
)

problem_calls["CALL_TIME_UTC"] = (
    pd.to_datetime(problem_calls[time_source], errors="coerce", utc=True)
    if time_source
    else pd.NaT
)

numeric_columns = [
    "LATITUDE",
    "LONGITUDE",
    "UNCERT_METERS",
    "DISTANCE_TO_ROUTED_BOUNDARY_M",
    "DISTANCE_TO_EXPECTED_BOUNDARY_M",
]
for column in numeric_columns:
    if column in problem_calls.columns:
        problem_calls[column] = pd.to_numeric(
            problem_calls[column], errors="coerce"
        )

routed_distance = problem_calls.get(
    "DISTANCE_TO_ROUTED_BOUNDARY_M",
    pd.Series(np.nan, index=problem_calls.index),
)
expected_distance = problem_calls.get(
    "DISTANCE_TO_EXPECTED_BOUNDARY_M",
    pd.Series(np.nan, index=problem_calls.index),
)
uncertainty = problem_calls.get(
    "UNCERT_METERS",
    pd.Series(np.nan, index=problem_calls.index),
)

# Sign is added only for interpretation:
# negative = call is outside routed PSAP; positive = inside expected PSAP.
problem_calls["SIGNED_DISTANCE_TO_ROUTED_PSAP_M"] = -routed_distance
problem_calls["SIGNED_DISTANCE_TO_EXPECTED_PSAP_M"] = expected_distance

# Approximate margin remaining after accounting for the uncertainty radius.
problem_calls["ROUTED_CLEARANCE_AFTER_UNCERTAINTY_M"] = (
    routed_distance - uncertainty
)
problem_calls["EXPECTED_MARGIN_AFTER_UNCERTAINTY_M"] = (
    expected_distance - uncertainty
)

problem_calls["GEOMETRY_RESULT"] = (
    "Uncertainty area is wholly inside expected PSAP and does not "
    "intersect routed PSAP"
)


# ------------------------------------------------------------
# 4. Individual-call report: no correctly routed controls
# ------------------------------------------------------------

detail_columns = [
    "PRESENTATION_RANK",
    "CALL_TIME_UTC",
    "CALL_KEY",
    "UNIQ911_CID",
    "GMLC_ROW_ID",
    "CALL_GRAIN_METHOD",
    "ROUTE_INTEGRITY_STATUS",
    "MISROUTE_LABEL",
    "FCC_PSAP_ID",
    "ROUTED_PSAP_NAME",
    "EXPECTED_FCC_PSAP_ID",
    "EXPECTED_PSAP_NAME",
    "MARKET",
    "MARKET_CLUSTER",
    "REGION",
    "STATE",
    "COUNTY",
    "LATITUDE",
    "LONGITUDE",
    "UNCERT_METERS",
    "DISTANCE_TO_ROUTED_BOUNDARY_M",
    "SIGNED_DISTANCE_TO_ROUTED_PSAP_M",
    "ROUTED_CLEARANCE_AFTER_UNCERTAINTY_M",
    "DISTANCE_TO_EXPECTED_BOUNDARY_M",
    "SIGNED_DISTANCE_TO_EXPECTED_PSAP_M",
    "EXPECTED_MARGIN_AFTER_UNCERTAINTY_M",
    "ROUTED_INTERSECTS_UNCERT_AREA",
    "GEOMETRY_RESULT",
    "SETUP_ECGI_HEX",
    "USID",
    "ESRK",
    "CALL_POPULATION_TAG",
    "ROUTE_STATUS_GMLC",
    "ROUTE_ESINET",
    "ROUTE_ESZ",
    "ROUTE_FALLBACK",
    "DEFAULT_ROUTED_CALL",
    "LBR_PSAP_DIFF",
    "SIP_STATUS",
    "SIP_METHOD",
    "FAILURE_SHORT_DR",
    "IMS_SIP_METHOD",
    "IMS_REGISTER_PCSCF_STATUS",
    "IMS_REGISTER_PCSCF_REASONCODES",
    "RAW_CORRELATION_STATUS",
    "RAW_ECSCF_STATUS",
    "RAW_REGISTER_PCSCF_STATUS",
    "RAW_NON_REGISTER_PCSCF_STATUS",
    "TOP100_MISROUTE_CALLS",
    "TOP100_PROBABLE_CAUSE",
    "TOP100_EVIDENCE_CONCLUSION",
    "TOP100_EVIDENCE_SUMMARY",
]

detail_columns = [c for c in detail_columns if c in problem_calls.columns]
problem_call_detail = (
    problem_calls[detail_columns]
    .sort_values(
        ["PRESENTATION_RANK", "CALL_TIME_UTC"],
        kind="stable",
        na_position="last",
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 5. One compact row per selected Top-100 pattern
# ------------------------------------------------------------

summary_rows = []
for rank, group in problem_call_detail.groupby(
    "PRESENTATION_RANK", sort=True, dropna=False
):
    def first_value(column):
        if column not in group.columns:
            return pd.NA
        values = group[column].dropna()
        return pd.NA if values.empty else values.iloc[0]

    def number(column, operation):
        if column not in group.columns:
            return np.nan
        values = pd.to_numeric(group[column], errors="coerce")
        if values.notna().sum() == 0:
            return np.nan
        return getattr(values, operation)()

    def mode(column):
        return (
            "<MISSING>"
            if column not in group.columns
            else mode_text(group[column])
        )

    times = (
        group["CALL_TIME_UTC"].dropna()
        if "CALL_TIME_UTC" in group.columns
        else pd.Series(dtype="datetime64[ns, UTC]")
    )

    summary_rows.append(
        {
            "PRESENTATION_RANK": rank,
            "MATCHED_PROBLEM_CALLS": len(group),
            "TOP100_REPORTED_MISROUTE_CALLS": first_value(
                "TOP100_MISROUTE_CALLS"
            ),
            "FIRST_CALL_UTC": pd.NaT if times.empty else times.min(),
            "LAST_CALL_UTC": pd.NaT if times.empty else times.max(),
            "FCC_PSAP_ID": first_value("FCC_PSAP_ID"),
            "ROUTED_PSAP_NAME": first_value("ROUTED_PSAP_NAME"),
            "EXPECTED_FCC_PSAP_ID": first_value("EXPECTED_FCC_PSAP_ID"),
            "EXPECTED_PSAP_NAME": first_value("EXPECTED_PSAP_NAME"),
            "USID_MODE": mode("USID"),
            "ESRK_MODE": mode("ESRK"),
            "AVG_UNCERTAINTY_M": number("UNCERT_METERS", "mean"),
            "MIN_DISTANCE_TO_ROUTED_PSAP_M": number(
                "DISTANCE_TO_ROUTED_BOUNDARY_M", "min"
            ),
            "AVG_DISTANCE_TO_ROUTED_PSAP_M": number(
                "DISTANCE_TO_ROUTED_BOUNDARY_M", "mean"
            ),
            "MAX_DISTANCE_TO_ROUTED_PSAP_M": number(
                "DISTANCE_TO_ROUTED_BOUNDARY_M", "max"
            ),
            "MIN_DISTANCE_TO_EXPECTED_PSAP_M": number(
                "DISTANCE_TO_EXPECTED_BOUNDARY_M", "min"
            ),
            "AVG_DISTANCE_TO_EXPECTED_PSAP_M": number(
                "DISTANCE_TO_EXPECTED_BOUNDARY_M", "mean"
            ),
            "MAX_DISTANCE_TO_EXPECTED_PSAP_M": number(
                "DISTANCE_TO_EXPECTED_BOUNDARY_M", "max"
            ),
            "ROUTE_STATUS_GMLC_MODE": mode("ROUTE_STATUS_GMLC"),
            "ROUTE_ESINET_MODE": mode("ROUTE_ESINET"),
            "ROUTE_ESZ_MODE": mode("ROUTE_ESZ"),
            "ROUTE_FALLBACK_MODE": mode("ROUTE_FALLBACK"),
            "SIP_STATUS_MODE": mode("SIP_STATUS"),
            "SIP_METHOD_MODE": mode("SIP_METHOD"),
            "IMS_SIP_METHOD_MODE": mode("IMS_SIP_METHOD"),
            "IMS_REGISTER_PCSCF_STATUS_MODE": mode(
                "IMS_REGISTER_PCSCF_STATUS"
            ),
            "RAW_CORRELATION_STATUS_MODE": mode(
                "RAW_CORRELATION_STATUS"
            ),
            "RAW_ECSCF_STATUS_MODE": mode("RAW_ECSCF_STATUS"),
            "PROBABLE_CAUSE": first_value("TOP100_PROBABLE_CAUSE"),
            "EVIDENCE_CONCLUSION": first_value(
                "TOP100_EVIDENCE_CONCLUSION"
            ),
        }
    )

problem_pattern_summary = pd.DataFrame(summary_rows)


# ------------------------------------------------------------
# 6. Save files for review and for the next map cell
# ------------------------------------------------------------

detail_path = OUTPUT_DIR / f"top_{TOP_N}_problematic_calls_detail.csv"
summary_path = OUTPUT_DIR / f"top_{TOP_N}_problematic_patterns_summary.csv"

problem_call_detail.to_csv(detail_path, index=False)
problem_pattern_summary.to_csv(summary_path, index=False)

display(Markdown(f"## Top {TOP_N} problematic pattern(s): compact summary"))
display(problem_pattern_summary)

display(Markdown("## Individual definite-misroute calls only"))
with pd.option_context(
    "display.max_rows", 500,
    "display.max_columns", None,
    "display.max_colwidth", 120,
):
    display(problem_call_detail)

print(f"Problematic patterns requested: {TOP_N}")
print(f"Individual definite-misroute calls returned: {len(problem_call_detail):,}")
print(f"Saved detail:  {detail_path}")
print(f"Saved summary: {summary_path}")
print("Latitude/longitude and both PSAP distances are preserved for the map cell.")
