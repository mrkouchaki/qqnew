# ============================================================
# NEXT CELL — STRONG DEFINITE MISROUTES FROM THE FULL RESULTS
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display, Markdown

# Only change these values if needed.
MIN_ROUTED_DISTANCE_M = 1000
MIN_CALLS_PER_PAIR = 1
MAX_PAIRS_TO_SHOW = 100

OUTPUT_DIR = Path(globals().get(
    "OUTPUT_DIR", r"C:\temp\gmlc_v2\outputs_psap_rca_v4"
))
SOURCE = OUTPUT_DIR / "definite_misroutes_v4.csv"
TOP100 = OUTPUT_DIR / "misroute_top100_network_expert.csv"

if not SOURCE.exists():
    raise FileNotFoundError(f"Missing: {SOURCE}")

def clean(c):
    return str(c).strip().upper().replace(" ", "_")

def norm(s):
    return (s.astype("string").fillna("").str.strip().str.strip('"')
            .str.replace(r"\.0$", "", regex=True).str.upper())

def mode_text(s):
    x = s.astype("string").fillna("").str.strip()
    x = x[~x.str.upper().isin(["", "<MISSING>", "NAN", "NONE"])]
    return "<MISSING>" if x.empty else x.value_counts().index[0]

def join_unique(s, limit=15):
    x = s.astype("string").fillna("").str.strip()
    x = sorted(set(x[~x.str.upper().isin(["", "<MISSING>", "NAN", "NONE"])]))
    return (" | ".join(x[:limit]) + (f" | +{len(x)-limit} more" if len(x) > limit else "")) or "<MISSING>"

# Read the complete definite-misroute output (not the previous Top-N dataframe).
calls = pd.read_csv(SOURCE, dtype="string", low_memory=False,
                    encoding_errors="replace")
calls.columns = [clean(c) for c in calls.columns]

# Normalize source column variants.
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
        source_col = next((c for c in candidates if c in calls.columns), None)
        if source_col:
            calls[target] = calls[source_col]

required = {
    "FCC_PSAP_ID", "EXPECTED_FCC_PSAP_ID",
    "DISTANCE_TO_ROUTED_BOUNDARY_M",
    "DISTANCE_TO_EXPECTED_BOUNDARY_M",
}
missing = sorted(required - set(calls.columns))
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Safety: keep only definite misroutes.
if "ROUTE_INTEGRITY_STATUS" in calls.columns:
    calls = calls[norm(calls["ROUTE_INTEGRITY_STATUS"]).eq("DEFINITE_MISROUTE")].copy()
elif "MISROUTE_LABEL" in calls.columns:
    calls = calls[pd.to_numeric(calls["MISROUTE_LABEL"], errors="coerce").eq(1)].copy()

for c in ["FCC_PSAP_ID", "EXPECTED_FCC_PSAP_ID", "USID", "ESRK", "SETUP_ECGI_HEX"]:
    if c in calls.columns:
        calls[c] = norm(calls[c])

for c in ["LATITUDE", "LONGITUDE", "UNCERT_METERS",
          "DISTANCE_TO_ROUTED_BOUNDARY_M",
          "DISTANCE_TO_EXPECTED_BOUNDARY_M"]:
    if c in calls.columns:
        calls[c] = pd.to_numeric(calls[c], errors="coerce")

# Use the first available timestamp per call.
calls["CALL_TIME_UTC"] = pd.Series(
    pd.NaT, index=calls.index, dtype="datetime64[ns, UTC]"
)
for c in ["CALL_BEGIN_TIME_UTC", "CALL_BEGIN_DATETIME", "CALL_DATETIME",
          "DATETIME_UTC", "CALL_DATE_UTC", "CALL_DATE"]:
    if c in calls.columns:
        try:
            parsed = pd.to_datetime(calls[c], errors="coerce", utc=True, format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(calls[c], errors="coerce", utc=True)
        calls["CALL_TIME_UTC"] = calls["CALL_TIME_UTC"].fillna(parsed)

uncert = calls.get("UNCERT_METERS", pd.Series(0.0, index=calls.index)).fillna(0)
calls["ROUTED_CLEARANCE_AFTER_UNCERTAINTY_M"] = (
    calls["DISTANCE_TO_ROUTED_BOUNDARY_M"] - uncert
)
calls["EXPECTED_MARGIN_AFTER_UNCERTAINTY_M"] = (
    calls["DISTANCE_TO_EXPECTED_BOUNDARY_M"] - uncert
)

# Build PSAP ID -> name lookup from existing outputs.
calls["ROUTED_PSAP_NAME"] = calls.get(
    "PSAP_NAME", pd.Series(pd.NA, index=calls.index, dtype="string")
)
name_map = {}
if "PSAP_NAME" in calls.columns:
    named = calls[calls["FCC_PSAP_ID"].ne("") & calls["PSAP_NAME"].notna()]
    name_map.update(named.groupby("FCC_PSAP_ID")["PSAP_NAME"].agg(mode_text).to_dict())

if TOP100.exists():
    names = pd.read_csv(TOP100, dtype="string", low_memory=False)
    names.columns = [clean(c) for c in names.columns]
    for id_col, name_col in [
        ("FCC_PSAP_ID", "PSAP_NAME"),
        ("EXPECTED_FCC_PSAP_ID", "EXPECTED_PSAP_NAME"),
    ]:
        if id_col in names.columns and name_col in names.columns:
            names[id_col] = norm(names[id_col])
            for psap_id, psap_name in zip(names[id_col], names[name_col]):
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

# Main filter: at least 1 km outside the PSAP that actually received the call.
strong_misroute_calls = calls[
    calls["DISTANCE_TO_ROUTED_BOUNDARY_M"].ge(MIN_ROUTED_DISTANCE_M)
].copy()

if "ROUTED_INTERSECTS_UNCERT_AREA" in strong_misroute_calls.columns:
    intersects = norm(strong_misroute_calls["ROUTED_INTERSECTS_UNCERT_AREA"])
    strong_misroute_calls = strong_misroute_calls[
        intersects.isin(["0", "FALSE", "NO"])
    ].copy()

if strong_misroute_calls.empty:
    print(f"No definite misroutes are >= {MIN_ROUTED_DISTANCE_M:,.0f} m outside the routed PSAP boundary.")
else:
    pair_keys = ["FCC_PSAP_ID", "ROUTED_PSAP_NAME",
                 "EXPECTED_FCC_PSAP_ID", "EXPECTED_PSAP_NAME"]
    g = strong_misroute_calls.groupby(pair_keys, dropna=False, sort=False)

    strong_misroute_pairs = g.agg(
        CALL_COUNT=("FCC_PSAP_ID", "size"),
        FIRST_CALL_UTC=("CALL_TIME_UTC", "min"),
        LAST_CALL_UTC=("CALL_TIME_UTC", "max"),
        MIN_ROUTED_DISTANCE_M=("DISTANCE_TO_ROUTED_BOUNDARY_M", "min"),
        AVG_ROUTED_DISTANCE_M=("DISTANCE_TO_ROUTED_BOUNDARY_M", "mean"),
        MAX_ROUTED_DISTANCE_M=("DISTANCE_TO_ROUTED_BOUNDARY_M", "max"),
        MIN_ROUTED_CLEARANCE_AFTER_UNCERTAINTY_M=("ROUTED_CLEARANCE_AFTER_UNCERTAINTY_M", "min"),
        AVG_ROUTED_CLEARANCE_AFTER_UNCERTAINTY_M=("ROUTED_CLEARANCE_AFTER_UNCERTAINTY_M", "mean"),
        MIN_EXPECTED_DISTANCE_M=("DISTANCE_TO_EXPECTED_BOUNDARY_M", "min"),
        AVG_EXPECTED_DISTANCE_M=("DISTANCE_TO_EXPECTED_BOUNDARY_M", "mean"),
        MAX_EXPECTED_DISTANCE_M=("DISTANCE_TO_EXPECTED_BOUNDARY_M", "max"),
    ).reset_index()

    # Add IDs and the network/signaling values that dominate each PSAP pair.
    for c, out in [("USID", "USIDS"), ("ESRK", "ESRKS")]:
        if c in strong_misroute_calls.columns:
            strong_misroute_pairs[out] = g[c].agg(join_unique).values

    mode_features = [
        "ROUTE_STATUS_GMLC", "ROUTE_ESINET", "ROUTE_ESZ",
        "ROUTE_FALLBACK", "DEFAULT_ROUTED_CALL", "LBR_PSAP_DIFF",
        "SIP_STATUS", "SIP_METHOD", "FAILURE_SHORT_DR",
        "IMS_SIP_METHOD", "IMS_REGISTER_PCSCF_STATUS",
        "IMS_REGISTER_PCSCF_REASONCODES", "RAW_CORRELATION_STATUS",
        "RAW_ECSCF_STATUS", "RAW_REGISTER_PCSCF_STATUS",
        "RAW_NON_REGISTER_PCSCF_STATUS",
    ]
    for c in mode_features:
        if c in strong_misroute_calls.columns:
            strong_misroute_pairs[f"{c}_MODE"] = g[c].agg(mode_text).values

    strong_misroute_pairs["GEOMETRY_EVIDENCE"] = (
        f"Every call is a definite misroute and >= {MIN_ROUTED_DISTANCE_M:,.0f} m "
        "outside the routed PSAP boundary"
    )
    strong_misroute_pairs = (
        strong_misroute_pairs[
            strong_misroute_pairs["CALL_COUNT"].ge(MIN_CALLS_PER_PAIR)
        ]
        .sort_values(["CALL_COUNT", "MAX_ROUTED_DISTANCE_M"],
                     ascending=[False, False], kind="stable")
        .reset_index(drop=True)
    )
    strong_misroute_pairs.insert(0, "PROBLEM_RANK",
                                 range(1, len(strong_misroute_pairs) + 1))

    call_columns = [
        "CALL_TIME_UTC", "CALL_KEY", "UNIQ911_CID", "GMLC_ROW_ID",
        *pair_keys, "LATITUDE", "LONGITUDE", "UNCERT_METERS",
        "DISTANCE_TO_ROUTED_BOUNDARY_M",
        "ROUTED_CLEARANCE_AFTER_UNCERTAINTY_M",
        "DISTANCE_TO_EXPECTED_BOUNDARY_M",
        "EXPECTED_MARGIN_AFTER_UNCERTAINTY_M",
        "SETUP_ECGI_HEX", "USID", "ESRK", "CALL_POPULATION_TAG",
        *mode_features,
    ]
    call_columns = [c for c in call_columns if c in strong_misroute_calls.columns]
    strong_misroute_calls = strong_misroute_calls[call_columns].sort_values(
        "DISTANCE_TO_ROUTED_BOUNDARY_M", ascending=False
    ).reset_index(drop=True)

    tag = f"over_{int(MIN_ROUTED_DISTANCE_M)}m"
    pair_path = OUTPUT_DIR / f"strong_misroute_pairs_{tag}.csv"
    call_path = OUTPUT_DIR / f"strong_misroute_calls_{tag}.csv"
    strong_misroute_pairs.to_csv(pair_path, index=False)
    strong_misroute_calls.to_csv(call_path, index=False)

    display(Markdown(f"## Strong PSAP pairs: routed distance ≥ {MIN_ROUTED_DISTANCE_M:,.0f} m"))
    with pd.option_context("display.max_rows", MAX_PAIRS_TO_SHOW,
                           "display.max_columns", None,
                           "display.max_colwidth", 100):
        display(strong_misroute_pairs.head(MAX_PAIRS_TO_SHOW))

    display(Markdown("## Individual calls behind these PSAP-pair problems"))
    with pd.option_context("display.max_rows", 500,
                           "display.max_columns", None,
                           "display.max_colwidth", 100):
        display(strong_misroute_calls)

    print(f"All definite misroutes scanned: {len(calls):,}")
    print(f"Calls over threshold: {len(strong_misroute_calls):,}")
    print(f"PSAP pairs: {len(strong_misroute_pairs):,}")
    print(f"Saved: {pair_path}")
    print(f"Saved: {call_path}")
