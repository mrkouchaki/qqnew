# ============================================================
# STRONG DEFINITE MISROUTES V2.1 — BOUNDED FULL RESULTS
#
# V2.1 additions:
#   1. Call latitude/longitude in the call report.
#   2. Representative interior latitude/longitude for routed and expected
#      PSAP service polygons in both reports.
#   3. A small set of meaningful routing/signaling evidence flags.
#
# Important: PSAP representative coordinates are points inside the service
# polygon. They are NOT the physical PSAP building coordinates.
# ============================================================

from pathlib import Path
import re

import numpy as np
import pandas as pd
from IPython.display import display, Markdown


# Change only these settings if needed.
MIN_ROUTED_DISTANCE_M = 1000
MAX_ROUTED_DISTANCE_M = 30000  # Set to None for no upper limit.
MIN_CALLS_PER_PAIR = 1
MAX_PAIRS_TO_SHOW = 100

OUTPUT_DIR = Path(globals().get(
    "OUTPUT_DIR", r"C:\temp\gmlc_v2\outputs_psap_rca_v4"
))
SOURCE = OUTPUT_DIR / "definite_misroutes_v4.csv"
TOP100 = OUTPUT_DIR / "misroute_top100_network_expert.csv"

if not SOURCE.exists():
    raise FileNotFoundError(f"Missing: {SOURCE}")


def clean(column):
    return str(column).strip().upper().replace(" ", "_")


def norm(series):
    return (
        series.astype("string").fillna("").str.strip().str.strip('"')
        .str.replace(r"\.0$", "", regex=True).str.upper()
    )


def norm_value(value):
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\.0$", "", str(value).strip().strip('"').upper())


def mode_text(series):
    values = series.astype("string").fillna("").str.strip()
    values = values[
        ~values.str.upper().isin(["", "<MISSING>", "NAN", "NONE", "NULL"])
    ]
    return "<MISSING>" if values.empty else values.value_counts().index[0]


def join_unique(series, limit=15):
    values = series.astype("string").fillna("").str.strip()
    values = sorted(set(values[
        ~values.str.upper().isin(["", "<MISSING>", "NAN", "NONE", "NULL"])
    ]))
    shown = " | ".join(values[:limit])
    if len(values) > limit:
        shown += f" | +{len(values) - limit} more"
    return shown or "<MISSING>"


def text_problem_flag(series, pattern):
    return (
        series.astype("string").fillna("").str.upper()
        .str.contains(pattern, regex=True, na=False)
        .astype("Int8")
    )


def abnormal_sip_or_failure(value):
    """401/407 challenges are treated as normal, not signaling failures."""
    text = "" if value is None or pd.isna(value) else str(value).upper()
    if re.search(r"TIMEOUT|FAIL|ERROR|REJECT|UNAVAILABLE|ABORT", text):
        return 1
    codes = [int(code) for code in re.findall(r"(?<!\d)([456]\d{2})(?!\d)", text)]
    return int(any(code not in {401, 407} for code in codes))


def parse_utc(series):
    try:
        return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce", utc=True)


# ------------------------------------------------------------
# 1. Obtain PSAP geometries already reconstructed by Section 2
# ------------------------------------------------------------

psap_geometry_by_fcc = {}

if "boundary_by_fcc" in globals() and isinstance(boundary_by_fcc, dict):
    psap_geometry_by_fcc = {
        norm_value(fcc): geometry
        for fcc, geometry in boundary_by_fcc.items()
        if geometry is not None
    }
elif "boundaries" in globals() and isinstance(boundaries, pd.DataFrame):
    if {"FCC_PSAP_ID", "GEOMETRY"}.issubset(boundaries.columns):
        psap_geometry_by_fcc = {
            norm_value(fcc): geometry
            for fcc, geometry in zip(
                boundaries["FCC_PSAP_ID"], boundaries["GEOMETRY"]
            )
            if geometry is not None
        }
elif (
    "reconstruct_boundaries" in globals()
    and "paths" in globals()
    and "BOUNDARIES" in paths
):
    boundary_frame, _ = reconstruct_boundaries(paths["BOUNDARIES"])
    psap_geometry_by_fcc = {
        norm_value(fcc): geometry
        for fcc, geometry in zip(
            boundary_frame["FCC_PSAP_ID"], boundary_frame["GEOMETRY"]
        )
        if geometry is not None
    }
else:
    raise RuntimeError(
        "PSAP boundary geometries are not in memory. Run notebook Section 2 "
        "(Reconstruct and validate PSAP boundaries), then run this V2 cell."
    )


def representative_coordinates(fcc_psap_id):
    geometry = psap_geometry_by_fcc.get(norm_value(fcc_psap_id))
    if geometry is None or geometry.is_empty:
        return np.nan, np.nan
    try:
        point = geometry.representative_point()
        return float(point.y), float(point.x)
    except Exception:
        return np.nan, np.nan


all_psap_ids = set(psap_geometry_by_fcc)
psap_coordinate_lookup = {
    fcc: representative_coordinates(fcc)
    for fcc in all_psap_ids
}


# ------------------------------------------------------------
# 2. Read all definite-misroute calls from the existing output
# ------------------------------------------------------------

calls = pd.read_csv(
    SOURCE,
    dtype="string",
    low_memory=False,
    encoding_errors="replace",
)
calls.columns = [clean(c) for c in calls.columns]

# Canonicalize source variants.
aliases = {
    "SIP_METHOD": ["SIP_SIP_METHOD"],
    "IMS_SIP_METHOD": ["IMS__SIP_METHOD"],
    "IMS_REGISTER_PCSCF_STATUS": ["IMS__REGISTER_PCSCF_STATUS"],
    "IMS_REGISTER_PCSCF_REASONCODES": ["IMS__REGISTER_PCSCF_REASONCODES"],
    "RAW_CORRELATION_STATUS": ["RAW__CORRELATION_STATUS"],
    "RAW_ECSCF_STATUS": ["RAW__ECSCF_STATUS"],
    "RAW_REGISTER_PCSCF_STATUS": ["RAW__REGISTER_PCSCF_STATUS"],
    "RAW_NON_REGISTER_PCSCF_STATUS": ["RAW__NON_REGISTER_PCSCF_STATUS"],
}
for target, candidates in aliases.items():
    if target not in calls.columns:
        source_column = next((c for c in candidates if c in calls.columns), None)
        if source_column:
            calls[target] = calls[source_column]

required = {
    "FCC_PSAP_ID",
    "EXPECTED_FCC_PSAP_ID",
    "LATITUDE",
    "LONGITUDE",
    "DISTANCE_TO_ROUTED_BOUNDARY_M",
    "DISTANCE_TO_EXPECTED_BOUNDARY_M",
}
missing = sorted(required - set(calls.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Safety: retain only explicit definite misroutes.
if "ROUTE_INTEGRITY_STATUS" in calls.columns:
    calls = calls[
        norm(calls["ROUTE_INTEGRITY_STATUS"]).eq("DEFINITE_MISROUTE")
    ].copy()
elif "MISROUTE_LABEL" in calls.columns:
    calls = calls[
        pd.to_numeric(calls["MISROUTE_LABEL"], errors="coerce").eq(1)
    ].copy()

for column in [
    "FCC_PSAP_ID", "EXPECTED_FCC_PSAP_ID",
    "USID", "ESRK", "SETUP_ECGI_HEX",
]:
    if column in calls.columns:
        calls[column] = norm(calls[column])

for column in [
    "LATITUDE", "LONGITUDE", "UNCERT_METERS",
    "DISTANCE_TO_ROUTED_BOUNDARY_M",
    "DISTANCE_TO_EXPECTED_BOUNDARY_M",
]:
    if column in calls.columns:
        calls[column] = pd.to_numeric(calls[column], errors="coerce")

# Use the first successfully parsed time field.
calls["CALL_TIME_UTC"] = pd.Series(
    pd.NaT, index=calls.index, dtype="datetime64[ns, UTC]"
)
for column in [
    "CALL_BEGIN_TIME_UTC", "CALL_BEGIN_DATETIME", "CALL_DATETIME",
    "DATETIME_UTC", "CALL_DATE_UTC", "CALL_DATE",
]:
    if column in calls.columns:
        calls["CALL_TIME_UTC"] = calls["CALL_TIME_UTC"].fillna(
            parse_utc(calls[column])
        )


# ------------------------------------------------------------
# 3. Add uncertainty-clearance fields and PSAP coordinates
# ------------------------------------------------------------

uncertainty = calls.get(
    "UNCERT_METERS", pd.Series(0.0, index=calls.index)
).fillna(0.0)

# Approximate distance from the outer edge of the uncertainty circle to the
# corresponding PSAP boundary. The exact label still comes from polygon tests.
calls["ROUTED_CLEARANCE_AFTER_UNCERTAINTY_M"] = (
    calls["DISTANCE_TO_ROUTED_BOUNDARY_M"] - uncertainty
)
calls["EXPECTED_MARGIN_AFTER_UNCERTAINTY_M"] = (
    calls["DISTANCE_TO_EXPECTED_BOUNDARY_M"] - uncertainty
)

routed_points = calls["FCC_PSAP_ID"].map(psap_coordinate_lookup)
expected_points = calls["EXPECTED_FCC_PSAP_ID"].map(psap_coordinate_lookup)

calls["ROUTED_PSAP_REP_LATITUDE"] = routed_points.map(
    lambda value: value[0] if isinstance(value, tuple) else np.nan
)
calls["ROUTED_PSAP_REP_LONGITUDE"] = routed_points.map(
    lambda value: value[1] if isinstance(value, tuple) else np.nan
)
calls["EXPECTED_PSAP_REP_LATITUDE"] = expected_points.map(
    lambda value: value[0] if isinstance(value, tuple) else np.nan
)
calls["EXPECTED_PSAP_REP_LONGITUDE"] = expected_points.map(
    lambda value: value[1] if isinstance(value, tuple) else np.nan
)

calls["CALL_COORDINATES_VALID"] = (
    calls["LATITUDE"].between(-90, 90)
    & calls["LONGITUDE"].between(-180, 180)
).astype("Int8")


# ------------------------------------------------------------
# 4. Add PSAP names using existing outputs
# ------------------------------------------------------------

calls["ROUTED_PSAP_NAME"] = calls.get(
    "PSAP_NAME", pd.Series(pd.NA, index=calls.index, dtype="string")
)
name_map = {}

if "PSAP_NAME" in calls.columns:
    named = calls[
        calls["FCC_PSAP_ID"].ne("") & calls["PSAP_NAME"].notna()
    ]
    name_map.update(
        named.groupby("FCC_PSAP_ID")["PSAP_NAME"].agg(mode_text).to_dict()
    )

if TOP100.exists():
    names = pd.read_csv(TOP100, dtype="string", low_memory=False)
    names.columns = [clean(c) for c in names.columns]
    for id_column, name_column in [
        ("FCC_PSAP_ID", "PSAP_NAME"),
        ("EXPECTED_FCC_PSAP_ID", "EXPECTED_PSAP_NAME"),
    ]:
        if id_column in names.columns and name_column in names.columns:
            names[id_column] = norm(names[id_column])
            for psap_id, psap_name in zip(
                names[id_column], names[name_column]
            ):
                if psap_id and pd.notna(psap_name):
                    name_map.setdefault(psap_id, str(psap_name))

if "EXPECTED_PSAP_NAME" not in calls.columns:
    calls["EXPECTED_PSAP_NAME"] = pd.NA

calls["ROUTED_PSAP_NAME"] = calls["ROUTED_PSAP_NAME"].fillna(
    calls["FCC_PSAP_ID"].map(name_map)
).fillna("<NAME NOT AVAILABLE>")
calls["EXPECTED_PSAP_NAME"] = calls["EXPECTED_PSAP_NAME"].fillna(
    calls["EXPECTED_FCC_PSAP_ID"].map(name_map)
).fillna("<NAME NOT AVAILABLE>")


# ------------------------------------------------------------
# 5. Add only meaningful network/signaling evidence flags
# ------------------------------------------------------------

route_status = calls.get(
    "ROUTE_STATUS_GMLC", pd.Series("", index=calls.index, dtype="string")
)
route_fallback = calls.get(
    "ROUTE_FALLBACK", pd.Series("", index=calls.index, dtype="string")
)

calls["GMLC_ROUTE_PROBLEM_FLAG"] = text_problem_flag(
    route_status,
    r"MISMATCH|MISSING|NO INVITE|REDIRECT|ERROR|FAIL",
)
calls["FALLBACK_WDLS_PROBLEM_FLAG"] = text_problem_flag(
    route_fallback,
    r"WDLS|SUSPICIOUS|ERROR|FAIL|NO ESRK",
)

if "LBR_PSAP_DIFF" in calls.columns:
    lbr_numeric = pd.to_numeric(calls["LBR_PSAP_DIFF"], errors="coerce")
    calls["LBR_PSAP_DIFF_FLAG"] = (
        lbr_numeric.fillna(0).gt(0)
        | norm(calls["LBR_PSAP_DIFF"]).isin(["Y", "YES", "TRUE"])
    ).astype("Int8")
else:
    calls["LBR_PSAP_DIFF_FLAG"] = 0

ims_raw_columns = [
    "FAILURE_SHORT_DR",
    "IMS_REGISTER_PCSCF_STATUS",
    "IMS_REGISTER_PCSCF_REASONCODES",
    "RAW_CORRELATION_STATUS",
    "RAW_ECSCF_STATUS",
    "RAW_REGISTER_PCSCF_STATUS",
    "RAW_NON_REGISTER_PCSCF_STATUS",
]
available_ims_raw = [c for c in ims_raw_columns if c in calls.columns]

if available_ims_raw:
    combined_ims_raw = calls[available_ims_raw].fillna("").astype("string").agg(
        " | ".join, axis=1
    )
    combined_upper = combined_ims_raw.str.upper()
    failure_word = combined_upper.str.contains(
        r"FAIL|ERROR|TIMEOUT|REJECT|INVALID|UNAVAILABLE|DROP",
        regex=True,
        na=False,
    )
    # Exclude ordinary SIP authentication challenges 401 and 407.
    abnormal_response = combined_upper.str.contains(
        r"(?<!\d)(?:4(?!01|07)\d{2}|5\d{2}|6\d{2})(?!\d)",
        regex=True,
        na=False,
    )
    calls["IMS_RAW_SIGNALING_PROBLEM_FLAG"] = (
        failure_word | abnormal_response
    ).astype("Int8")
else:
    calls["IMS_RAW_SIGNALING_PROBLEM_FLAG"] = 0

evidence = pd.Series("", index=calls.index, dtype="string")

def append_evidence(mask, piece):
    active = pd.Series(mask, index=calls.index).fillna(False).astype(bool)
    if not active.any():
        return
    if not isinstance(piece, pd.Series):
        piece = pd.Series(str(piece), index=calls.index, dtype="string")
    else:
        piece = piece.astype("string")
    existing = evidence.loc[active]
    addition = piece.loc[active]
    evidence.loc[active] = np.where(
        existing.eq(""), addition, existing + "; " + addition
    )

append_evidence(
    calls["GMLC_ROUTE_PROBLEM_FLAG"].eq(1),
    "GMLC route status: " + calls.get(
        "ROUTE_STATUS_GMLC", pd.Series("", index=calls.index)
    ).fillna("").astype("string"),
)
append_evidence(
    calls["FALLBACK_WDLS_PROBLEM_FLAG"].eq(1),
    "Fallback/WDLS: " + calls.get(
        "ROUTE_FALLBACK", pd.Series("", index=calls.index)
    ).fillna("").astype("string"),
)
append_evidence(calls["LBR_PSAP_DIFF_FLAG"].eq(1), "LBR PSAP differs")
append_evidence(
    calls["IMS_RAW_SIGNALING_PROBLEM_FLAG"].eq(1),
    "IMS/RAW signaling error or abnormal response",
)
calls["ADDITIONAL_NETWORK_EVIDENCE"] = evidence.mask(
    evidence.eq(""), "No additional network fault; geometry is primary evidence"
)

calls["ANY_ADDITIONAL_NETWORK_PROBLEM_FLAG"] = (
    calls[[
        "GMLC_ROUTE_PROBLEM_FLAG",
        "FALLBACK_WDLS_PROBLEM_FLAG",
        "LBR_PSAP_DIFF_FLAG",
        "IMS_RAW_SIGNALING_PROBLEM_FLAG",
    ]].max(axis=1).astype("Int8")
)


# ------------------------------------------------------------
# 6. Filter strongest calls: >= 1 km outside routed PSAP
# ------------------------------------------------------------

distance_mask = calls["DISTANCE_TO_ROUTED_BOUNDARY_M"].ge(
    MIN_ROUTED_DISTANCE_M
)
if MAX_ROUTED_DISTANCE_M is not None:
    distance_mask &= calls["DISTANCE_TO_ROUTED_BOUNDARY_M"].le(
        MAX_ROUTED_DISTANCE_M
    )

strong_misroute_calls_v2 = calls[distance_mask].copy()

if "ROUTED_INTERSECTS_UNCERT_AREA" in strong_misroute_calls_v2.columns:
    intersects = norm(
        strong_misroute_calls_v2["ROUTED_INTERSECTS_UNCERT_AREA"]
    )
    strong_misroute_calls_v2 = strong_misroute_calls_v2[
        intersects.isin(["0", "FALSE", "NO"])
    ].copy()

if strong_misroute_calls_v2.empty:
    print(
        f"No definite misroutes are between {MIN_ROUTED_DISTANCE_M:,.0f} m "
        f"and {MAX_ROUTED_DISTANCE_M:,.0f} m outside the routed PSAP boundary."
        if MAX_ROUTED_DISTANCE_M is not None else
        f"No definite misroutes are >= {MIN_ROUTED_DISTANCE_M:,.0f} m "
        "outside the routed PSAP boundary."
    )
else:
    pair_keys = [
        "FCC_PSAP_ID",
        "ROUTED_PSAP_NAME",
        "ROUTED_PSAP_REP_LATITUDE",
        "ROUTED_PSAP_REP_LONGITUDE",
        "EXPECTED_FCC_PSAP_ID",
        "EXPECTED_PSAP_NAME",
        "EXPECTED_PSAP_REP_LATITUDE",
        "EXPECTED_PSAP_REP_LONGITUDE",
    ]
    grouped = strong_misroute_calls_v2.groupby(
        pair_keys, dropna=False, sort=False
    )

    strong_misroute_pairs_v2 = grouped.agg(
        CALL_COUNT=("FCC_PSAP_ID", "size"),
        FIRST_CALL_UTC=("CALL_TIME_UTC", "min"),
        LAST_CALL_UTC=("CALL_TIME_UTC", "max"),
        MIN_ROUTED_DISTANCE_M=("DISTANCE_TO_ROUTED_BOUNDARY_M", "min"),
        AVG_ROUTED_DISTANCE_M=("DISTANCE_TO_ROUTED_BOUNDARY_M", "mean"),
        MAX_ROUTED_DISTANCE_M=("DISTANCE_TO_ROUTED_BOUNDARY_M", "max"),
        MIN_ROUTED_CLEARANCE_AFTER_UNCERTAINTY_M=(
            "ROUTED_CLEARANCE_AFTER_UNCERTAINTY_M", "min"
        ),
        AVG_ROUTED_CLEARANCE_AFTER_UNCERTAINTY_M=(
            "ROUTED_CLEARANCE_AFTER_UNCERTAINTY_M", "mean"
        ),
        MIN_EXPECTED_DISTANCE_M=("DISTANCE_TO_EXPECTED_BOUNDARY_M", "min"),
        AVG_EXPECTED_DISTANCE_M=("DISTANCE_TO_EXPECTED_BOUNDARY_M", "mean"),
        MAX_EXPECTED_DISTANCE_M=("DISTANCE_TO_EXPECTED_BOUNDARY_M", "max"),
        GMLC_ROUTE_PROBLEM_CALLS=("GMLC_ROUTE_PROBLEM_FLAG", "sum"),
        FALLBACK_WDLS_PROBLEM_CALLS=("FALLBACK_WDLS_PROBLEM_FLAG", "sum"),
        LBR_PSAP_DIFF_CALLS=("LBR_PSAP_DIFF_FLAG", "sum"),
        IMS_RAW_SIGNALING_PROBLEM_CALLS=(
            "IMS_RAW_SIGNALING_PROBLEM_FLAG", "sum"
        ),
        ANY_ADDITIONAL_NETWORK_PROBLEM_CALLS=(
            "ANY_ADDITIONAL_NETWORK_PROBLEM_FLAG", "sum"
        ),
    ).reset_index()

    for column, output_column in [("USID", "USIDS"), ("ESRK", "ESRKS")]:
        if column in strong_misroute_calls_v2.columns:
            strong_misroute_pairs_v2[output_column] = (
                grouped[column].agg(join_unique).values
            )

    for column in [
        "ROUTE_STATUS_GMLC",
        "ROUTE_FALLBACK",
        "LBR_PSAP_DIFF",
        "ESPOSREQ_INITIAL_LOCATION_SOURCE",
        "ESPOSREQ_LASTKNOWN_LOCATION_SOURCE",
        "IMS_REGISTER_PCSCF_REASONCODES",
        "ADDITIONAL_NETWORK_EVIDENCE",
    ]:
        if column in strong_misroute_calls_v2.columns:
            strong_misroute_pairs_v2[f"{column}_MODE"] = (
                grouped[column].agg(mode_text).values
            )

    strong_misroute_pairs_v2["PSAP_COORDINATE_DEFINITION"] = (
        "Representative interior point of PSAP service polygon; "
        "not physical PSAP building"
    )
    strong_misroute_pairs_v2["GEOMETRY_EVIDENCE"] = (
        (
            f"Every call is a definite misroute between "
            f"{MIN_ROUTED_DISTANCE_M:,.0f} and "
            f"{MAX_ROUTED_DISTANCE_M:,.0f} m outside routed PSAP boundary"
            if MAX_ROUTED_DISTANCE_M is not None else
            f"Every call is a definite misroute >= "
            f"{MIN_ROUTED_DISTANCE_M:,.0f} m outside routed PSAP boundary"
        )
    )

    strong_misroute_pairs_v2 = (
        strong_misroute_pairs_v2[
            strong_misroute_pairs_v2["CALL_COUNT"].ge(MIN_CALLS_PER_PAIR)
        ]
        .sort_values(
            ["CALL_COUNT", "MAX_ROUTED_DISTANCE_M"],
            ascending=[False, False],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    strong_misroute_pairs_v2.insert(
        0, "PROBLEM_RANK", range(1, len(strong_misroute_pairs_v2) + 1)
    )

    essential_context = [
        "ROUTE_STATUS_GMLC",
        "ROUTE_ESINET",
        "ROUTE_ESZ",
        "ROUTE_FALLBACK",
        "LBR_SUCCESS",
        "LBR_PSAP_DIFF",
        "ESPOSREQ_INITIAL_LOCATION_SOURCE",
        "ESPOSREQ_LASTKNOWN_LOCATION_SOURCE",
        "SIP_STATUS",
        "SIP_METHOD",
        "FAILURE_SHORT_DR",
        "IMS_SIP_METHOD",
        "IMS_REGISTER_PCSCF_STATUS",
        "IMS_REGISTER_PCSCF_REASONCODES",
        "RAW_CORRELATION_STATUS",
        "RAW_ECSCF_STATUS",
    ]

    call_columns = [
        "CALL_TIME_UTC",
        "CALL_KEY",
        "UNIQ911_CID",
        "GMLC_ROW_ID",
        "FCC_PSAP_ID",
        "ROUTED_PSAP_NAME",
        "EXPECTED_FCC_PSAP_ID",
        "EXPECTED_PSAP_NAME",
        "LATITUDE",
        "LONGITUDE",
        "CALL_COORDINATES_VALID",
        "ROUTED_PSAP_REP_LATITUDE",
        "ROUTED_PSAP_REP_LONGITUDE",
        "EXPECTED_PSAP_REP_LATITUDE",
        "EXPECTED_PSAP_REP_LONGITUDE",
        "UNCERT_METERS",
        "DISTANCE_TO_ROUTED_BOUNDARY_M",
        "ROUTED_CLEARANCE_AFTER_UNCERTAINTY_M",
        "DISTANCE_TO_EXPECTED_BOUNDARY_M",
        "EXPECTED_MARGIN_AFTER_UNCERTAINTY_M",
        "SETUP_ECGI_HEX",
        "USID",
        "ESRK",
        "CALL_POPULATION_TAG",
        *essential_context,
        "GMLC_ROUTE_PROBLEM_FLAG",
        "FALLBACK_WDLS_PROBLEM_FLAG",
        "LBR_PSAP_DIFF_FLAG",
        "IMS_RAW_SIGNALING_PROBLEM_FLAG",
        "ANY_ADDITIONAL_NETWORK_PROBLEM_FLAG",
        "ADDITIONAL_NETWORK_EVIDENCE",
    ]
    call_columns = [
        column for column in call_columns
        if column in strong_misroute_calls_v2.columns
    ]
    strong_misroute_calls_v2 = (
        strong_misroute_calls_v2[call_columns]
        .sort_values(
            "DISTANCE_TO_ROUTED_BOUNDARY_M",
            ascending=False,
            kind="stable",
        )
        .reset_index(drop=True)
    )

    threshold_tag = (
        f"{int(MIN_ROUTED_DISTANCE_M)}m_to_{int(MAX_ROUTED_DISTANCE_M)}m"
        if MAX_ROUTED_DISTANCE_M is not None else
        f"over_{int(MIN_ROUTED_DISTANCE_M)}m"
    )
    pair_path = OUTPUT_DIR / f"strong_misroute_pairs_{threshold_tag}_v2_1.csv"
    call_path = OUTPUT_DIR / f"strong_misroute_calls_{threshold_tag}_v2_1.csv"

    strong_misroute_pairs_v2.to_csv(pair_path, index=False)
    strong_misroute_calls_v2.to_csv(call_path, index=False)

    display(Markdown(
        (
            f"## V2.1 strong PSAP pairs: routed distance "
            f"{MIN_ROUTED_DISTANCE_M:,.0f}–"
            f"{MAX_ROUTED_DISTANCE_M:,.0f} m"
            if MAX_ROUTED_DISTANCE_M is not None else
            f"## V2.1 strong PSAP pairs: routed distance >= "
            f"{MIN_ROUTED_DISTANCE_M:,.0f} m"
        )
    ))
    with pd.option_context(
        "display.max_rows", MAX_PAIRS_TO_SHOW,
        "display.max_columns", None,
        "display.max_colwidth", 100,
    ):
        display(strong_misroute_pairs_v2.head(MAX_PAIRS_TO_SHOW))

    display(Markdown("## V2.1 individual calls with call and PSAP coordinates"))
    with pd.option_context(
        "display.max_rows", 500,
        "display.max_columns", None,
        "display.max_colwidth", 100,
    ):
        display(strong_misroute_calls_v2)

    print(f"All definite misroutes scanned: {len(calls):,}")
    print(f"Calls over threshold: {len(strong_misroute_calls_v2):,}")
    print(f"PSAP pairs: {len(strong_misroute_pairs_v2):,}")
    print(f"Saved V2.1 pair report: {pair_path}")
    print(f"Saved V2.1 call report: {call_path}")
