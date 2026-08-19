"""
Paste this entire file into ONE new notebook cell immediately after the v4
"Results" cell and before "Output reconciliation".

It reuses the completed v4 SQLite staging database. It does NOT rebuild PSAP
boundaries, relabel calls, or reread the multi-GB Oracle CSV exports.

Outputs:
  - misroute_investigation_v5.csv
  - psap_2160_investigation_v5.csv
  - misroute_evidence_long_v5.csv
  - misroute_investigation_v5_data_dictionary.csv
"""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import math
import re
import sqlite3

import numpy as np
import pandas as pd
from IPython.display import display, Markdown


# ---------------------------------------------------------------------------
# 0. Reopen the completed v4 staging database if Section 9 closed it.
# ---------------------------------------------------------------------------
if "OUTPUT_DIR" not in globals():
    OUTPUT_DIR = Path(r"C:\temp\gmlc_v2\outputs_psap_rca_v4")
else:
    OUTPUT_DIR = Path(OUTPUT_DIR)

if "DB_PATH" not in globals():
    DB_PATH = OUTPUT_DIR / "psap_rca_stage_v4.sqlite"
else:
    DB_PATH = Path(DB_PATH)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not DB_PATH.exists():
    raise FileNotFoundError(
        f"Completed v4 staging database was not found: {DB_PATH}\n"
        "Run v4 through the enriched-output/join-audit cell first."
    )

try:
    con.execute("SELECT 1").fetchone()
except Exception:
    con = sqlite3.connect(DB_PATH)

con.execute("PRAGMA temp_store=FILE")
con.execute("PRAGMA cache_size=-250000")       # about 250 MB
con.execute("PRAGMA busy_timeout=60000")


# ---------------------------------------------------------------------------
# 1. Small helpers and transparent evidence rules.
# ---------------------------------------------------------------------------
def v5_qi(name):
    return '"' + str(name).replace('"', '""') + '"'


def v5_columns(table_name):
    return [r[1] for r in con.execute(f"PRAGMA table_info({v5_qi(table_name)})")]


def v5_norm_text(value):
    if value is None:
        return "<MISSING>"
    s = str(value).strip().strip('"').upper()
    return s if s and s not in {"NAN", "NONE", "NULL", "<NA>"} else "<MISSING>"


def v5_norm_id(value):
    s = v5_norm_text(value)
    if s == "<MISSING>":
        return s
    return re.sub(r"\.0+$", "", s)


def v5_to_real(value):
    if value is None:
        return None
    try:
        x = float(str(value).strip().replace(",", ""))
        return x if math.isfinite(x) else None
    except Exception:
        return None


def v5_truthy(value):
    if value is None:
        return None
    s = v5_norm_text(value)
    if s == "<MISSING>":
        return None
    if s in {"N", "NO", "FALSE", "F", "0", "OFF", "NONE", "NOT USED"}:
        return 0
    if s in {"Y", "YES", "TRUE", "T", "1", "ON", "USED"}:
        return 1
    # A populated fallback/default value is evidence that a path was specified.
    return 1


def v5_positive(value):
    x = v5_to_real(value)
    if x is not None:
        return int(x > 0)
    return v5_truthy(value)


ISSUE_RE = re.compile(
    r"FAIL|ERROR|REJECT|TIMEOUT|MISMATCH|INVALID|EXCEPTION|NOT[ _-]?FOUND|"
    r"DENIED|UNAVAILABLE|DROP|^4\d\d$|^5\d\d$|^6\d\d$",
    re.IGNORECASE,
)


def v5_issue_flag(value):
    if value is None:
        return None
    s = v5_norm_text(value)
    if s == "<MISSING>":
        return None
    if ISSUE_RE.search(s):
        return 1
    if any(token in s for token in ("SUCCESS", "PASSED", "REGISTERED", "COMPLETE", " OK")):
        return 0
    if s in {"OK", "200", "0", "Y", "YES", "TRUE"}:
        return 0
    return 0


ORACLE_TIME_FORMATS = (
    "%d-%b-%y %I.%M.%S.%f %p UTC",
    "%d-%b-%y %I.%M.%S %p UTC",
    "%d-%b-%Y %I.%M.%S.%f %p UTC",
    "%d-%b-%Y %I.%M.%S %p UTC",
    "%d-%b-%y %H:%M:%S.%f UTC",
    "%d-%b-%y %H:%M:%S UTC",
)


def v5_utc_epoch(value):
    """Parse Oracle/ISO timestamps before MIN/MAX; never compare raw strings."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    upper = re.sub(r"\s+", " ", s.upper())
    for fmt in ORACLE_TIME_FORMATS:
        try:
            return datetime.strptime(upper, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    try:
        ts = pd.to_datetime(s, errors="coerce", utc=True)
        return None if pd.isna(ts) else float(ts.timestamp())
    except Exception:
        return None


def v5_epoch_iso(value):
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except Exception:
        return None


def v5_key_join(*values):
    return json.dumps([v5_norm_text(v) for v in values], separators=(",", ":"))


class V5ModePack:
    """Deterministic mode, its support, and its share within one group."""
    def __init__(self):
        self.counts = Counter()

    def step(self, value):
        self.counts[v5_norm_text(value)] += 1

    def finalize(self):
        if not self.counts:
            return json.dumps({"value": "<MISSING>", "count": 0, "total": 0, "share": None})
        value, count = sorted(self.counts.items(), key=lambda x: (-x[1], x[0]))[0]
        total = sum(self.counts.values())
        return json.dumps(
            {"value": value, "count": int(count), "total": int(total), "share": count / total},
            separators=(",", ":"),
        )


def v5_pack_value(pack):
    try:
        return json.loads(pack)["value"]
    except Exception:
        return "<MISSING>"


def v5_pack_share(pack):
    try:
        return float(json.loads(pack)["share"])
    except Exception:
        return None


def v5_pos(x):
    try:
        return max(float(x), 0.0)
    except Exception:
        return 0.0


def v5_mode_signal(case_value, control_value, case_share):
    if case_value in (None, "<MISSING>"):
        return 0
    try:
        return int(str(case_value) != str(control_value) and float(case_share or 0) >= 0.50)
    except Exception:
        return 0


def v5_evidence_score(
    route_delta, fallback_delta, lbr_delta, sip_delta,
    ims_delta, raw_reg_delta, raw_ecscf_delta,
    route_case, route_control, route_case_share,
    esinet_case, esinet_control, esinet_case_share,
    esz_case, esz_control, esz_case_share,
):
    score = 0
    score += 35 if v5_pos(route_delta) >= 0.20 else 20 if v5_pos(route_delta) >= 0.10 else 0
    score += 15 if v5_pos(fallback_delta) >= 0.15 else 0
    score += 15 if v5_pos(lbr_delta) >= 0.20 else 0
    score += 10 if v5_pos(sip_delta) >= 0.15 else 0
    score += 10 if v5_pos(ims_delta) >= 0.15 else 0
    score += 10 if v5_pos(raw_reg_delta) >= 0.15 else 0
    score += 5 if v5_pos(raw_ecscf_delta) >= 0.15 else 0
    score += 10 * v5_mode_signal(route_case, route_control, route_case_share)
    score += 5 * v5_mode_signal(esinet_case, esinet_control, esinet_case_share)
    score += 5 * v5_mode_signal(esz_case, esz_control, esz_case_share)
    return int(min(score, 100))


def v5_conclusion(score, control_calls, ims_attach, raw_attach):
    score = int(score or 0)
    control_calls = int(control_calls or 0)
    if control_calls < 20:
        return "INSUFFICIENT_MATCHED_CORRECT_CONTROL"
    if score >= 50:
        return "STRONG_NETWORK_CORROBORATION"
    if score >= 25:
        return "MODERATE_NETWORK_CORROBORATION"
    if score > 0:
        return "WEAK_OR_MIXED_NETWORK_EVIDENCE"
    if float(ims_attach or 0) < 0.25 and float(raw_attach or 0) < 0.25:
        return "INSUFFICIENT_IMS_CCDR_ENRICHMENT"
    return "GEOMETRY_ONLY_NO_NETWORK_DIFFERENTIAL"


def v5_probable_cause(
    route_delta, fallback_delta, lbr_delta, sip_delta,
    ims_delta, raw_reg_delta, raw_ecscf_delta,
    route_case, route_control, route_case_share,
    control_calls, ims_attach, raw_attach,
):
    if int(control_calls or 0) < 20:
        return "INSUFFICIENT_CONTROL"
    if v5_pos(route_delta) >= 0.15 or v5_mode_signal(route_case, route_control, route_case_share):
        return "ROUTING_MAPPING_OR_CROSSWALK"
    if v5_pos(fallback_delta) >= 0.15:
        return "DEFAULT_OR_FALLBACK_ROUTING"
    if v5_pos(lbr_delta) >= 0.20:
        return "LBR_EXPECTED_PSAP_DIVERGENCE"
    if v5_pos(ims_delta) >= 0.15 and max(v5_pos(raw_reg_delta), v5_pos(raw_ecscf_delta)) >= 0.15:
        return "MIXED_IMS_CCDR_ASSOCIATION"
    if v5_pos(ims_delta) >= 0.15 or v5_pos(sip_delta) >= 0.15:
        return "IMS_OR_SIP_SIGNALING_ASSOCIATION"
    if max(v5_pos(raw_reg_delta), v5_pos(raw_ecscf_delta)) >= 0.15:
        return "CCDR_REGISTRATION_OR_ECSCF_ASSOCIATION"
    if float(ims_attach or 0) < 0.25 and float(raw_attach or 0) < 0.25:
        return "INSUFFICIENT_ENRICHMENT"
    return "GEOMETRY_ONLY_NETWORK_NORMAL_OR_MIXED"


def v5_summary(misroutes, controls, method, route_d, fallback_d, lbr_d, ims_d, raw_d):
    def pp(v):
        try:
            return f"{100.0 * float(v):+.1f} pp"
        except Exception:
            return "n/a"
    return (
        f"{int(misroutes or 0):,} geometry-defined misroutes; "
        f"{int(controls or 0):,} correct controls ({method}). "
        f"Misroute-minus-control indicators: route mismatch {pp(route_d)}, "
        f"fallback {pp(fallback_d)}, LBR PSAP difference {pp(lbr_d)}, "
        f"IMS registration/SIP {pp(ims_d)}, CCDR registration {pp(raw_d)}."
    )


con.create_function("V5_NORM_TEXT", 1, v5_norm_text)
con.create_function("V5_NORM_ID", 1, v5_norm_id)
con.create_function("V5_TO_REAL", 1, v5_to_real)
con.create_function("V5_TRUTHY", 1, v5_truthy)
con.create_function("V5_POSITIVE", 1, v5_positive)
con.create_function("V5_ISSUE_FLAG", 1, v5_issue_flag)
con.create_function("V5_UTC_EPOCH", 1, v5_utc_epoch)
con.create_function("V5_EPOCH_ISO", 1, v5_epoch_iso)
con.create_function("V5_KEY_JOIN", -1, v5_key_join)
con.create_function("V5_PACK_VALUE", 1, v5_pack_value)
con.create_function("V5_PACK_SHARE", 1, v5_pack_share)
con.create_function("V5_EVIDENCE_SCORE", 16, v5_evidence_score)
con.create_function("V5_CONCLUSION", 4, v5_conclusion)
con.create_function("V5_PROBABLE_CAUSE", 14, v5_probable_cause)
con.create_function("V5_SUMMARY", 8, v5_summary)
con.create_aggregate("V5_MODE_PACK", 1, V5ModePack)


# ---------------------------------------------------------------------------
# 2. Verify v4 contracts and materialize a narrow, one-row-per-call view.
# ---------------------------------------------------------------------------
required_tables = {"gmlc_calls_v4", "ims_agg_v4", "raw_call_agg_v4"}
actual_tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
missing_tables = required_tables - actual_tables
if missing_tables:
    raise RuntimeError(f"Missing completed v4 table(s): {sorted(missing_tables)}")

gmlc_cols = set(v5_columns("gmlc_calls_v4"))
ims_cols = set(v5_columns("ims_agg_v4"))
raw_cols = set(v5_columns("raw_call_agg_v4"))

for required in ("FCC_PSAP_ID", "EXPECTED_FCC_PSAP_ID", "MISROUTE_LABEL", "CALL_KEY", "GMLC_JOIN_KEY"):
    if required not in gmlc_cols:
        raise RuntimeError(f"v4 table gmlc_calls_v4 is missing required column: {required}")


def gmlc_expr(name):
    return f"c.{v5_qi(name)}" if name in gmlc_cols else "NULL"


def ims_expr(name):
    return f"i.{v5_qi(name)}" if name in ims_cols else "NULL"


def raw_expr(name):
    return f"r.{v5_qi(name)}" if name in raw_cols else "NULL"


def coalesce_existing(expressions):
    usable = [x for x in expressions if not x.startswith("NULL")]
    return "COALESCE(" + ",".join(usable) + ")" if usable else "NULL"


dimension_exprs = {
    "PSAP_NAME": coalesce_existing([gmlc_expr("PSAP_NAME"), gmlc_expr("ROUTE_PSAP_NAME"), gmlc_expr("PSAP_NAME_ESRK")]),
    "ROUTE_PSAP_NAME": gmlc_expr("ROUTE_PSAP_NAME"),
    "MARKET": coalesce_existing([gmlc_expr("MARKET"), ims_expr("IMS__RAN_MARKET")]),
    "MARKET_CLUSTER": coalesce_existing([gmlc_expr("MARKET_CLUSTER"), ims_expr("IMS__RAN_MARKET_CLUSTER")]),
    "REGION": coalesce_existing([gmlc_expr("REGION"), ims_expr("IMS__RAN_REGION")]),
    "STATE": gmlc_expr("STATE"),
    "COUNTY": gmlc_expr("COUNTY"),
    "SETUP_ECGI_HEX": gmlc_expr("SETUP_ECGI_HEX"),
    "USID": gmlc_expr("USID"),
    "ESRK": gmlc_expr("ESRK"),
}

CAT_CANDIDATES = [
    ("ROUTE_STATUS_GMLC", gmlc_expr("ROUTE_STATUS_GMLC")),
    ("ROUTE_ESZ", gmlc_expr("ROUTE_ESZ")),
    ("ROUTE_ESINET", gmlc_expr("ROUTE_ESINET")),
    ("ROUTE_FALLBACK", gmlc_expr("ROUTE_FALLBACK")),
    ("DEFAULT_ROUTED_CALL", gmlc_expr("DEFAULT_ROUTED_CALL")),
    ("SIP_STATUS", gmlc_expr("SIP_STATUS")),
    ("SIP_SIP_METHOD", gmlc_expr("SIP_SIP_METHOD")),
    ("FAILURE_SHORT_DR", gmlc_expr("FAILURE_SHORT_DR")),
    ("IMS__SIP_METHOD", ims_expr("IMS__SIP_METHOD")),
    ("IMS__REGISTER_PCSCF_REASONCODES", ims_expr("IMS__REGISTER_PCSCF_REASONCODES")),
    ("IMS__REGISTER_PCSCF_STATUS", ims_expr("IMS__REGISTER_PCSCF_STATUS")),
    ("RAW__CORRELATION_STATUS", raw_expr("RAW__CORRELATION_STATUS")),
    ("RAW__ECSCF_STATUS", raw_expr("RAW__ECSCF_STATUS")),
    ("RAW__NON_REGISTER_PCSCF_STATUS", raw_expr("RAW__NON_REGISTER_PCSCF_STATUS")),
    ("RAW__REGISTER_PCSCF_STATUS", raw_expr("RAW__REGISTER_PCSCF_STATUS")),
]
CAT_CANDIDATES = [(n, e) for n, e in CAT_CANDIDATES if e != "NULL"]

NUM_CANDIDATES = [
    ("LBR_PSAP_DIFF", gmlc_expr("LBR_PSAP_DIFF")),
    ("LBR_SUCCESS", gmlc_expr("LBR_SUCCESS")),
    ("LBR_ATTEMPTED", gmlc_expr("LBR_ATTEMPTED")),
    ("INVITE_CNT", gmlc_expr("INVITE_CNT")),
    ("LOCATE_CNT", gmlc_expr("LOCATE_CNT")),
    ("IMS__DURATION__MEAN", ims_expr("IMS__DURATION__MEAN")),
    ("IMS__COUNT_OF_REGISTER_PCSCF__MEAN", ims_expr("IMS__COUNT_OF_REGISTER_PCSCF__MEAN")),
    ("RAW__COUNT_OF_ECSCF__MEAN", raw_expr("RAW__COUNT_OF_ECSCF__MEAN")),
    ("RAW__COUNT_OF_NON_REGISTER_PCSCF__MEAN", raw_expr("RAW__COUNT_OF_NON_REGISTER_PCSCF__MEAN")),
    ("RAW__COUNT_OF_REGISTER_PCSCF__MEAN", raw_expr("RAW__COUNT_OF_REGISTER_PCSCF__MEAN")),
]
NUM_CANDIDATES = [(n, e) for n, e in NUM_CANDIDATES if e != "NULL"]

for table in (
    "evidence_base_v5", "case_profiles_v5", "control_profiles_v5",
    "misroute_case_mapped_v5", "strong_cohort_v5", "psap_lookup_v5",
    "misroute_investigation_wide_v5", "misroute_investigation_v5",
    "misroute_evidence_long_v5",
):
    con.execute(f"DROP TABLE IF EXISTS temp.{v5_qi(table)}")

source_select = [
    f"V5_NORM_ID(c.{v5_qi('FCC_PSAP_ID')}) AS FCC_PSAP_ID",
    f"V5_NORM_ID(c.{v5_qi('EXPECTED_FCC_PSAP_ID')}) AS EXPECTED_FCC_PSAP_ID",
    f"CAST(c.{v5_qi('MISROUTE_LABEL')} AS INTEGER) AS MISROUTE_LABEL",
    f"V5_NORM_TEXT({gmlc_expr('CALL_POPULATION_TAG')}) AS CALL_POPULATION_TAG",
    f"{gmlc_expr('CALL_BEGIN_TIME_UTC')} AS CALL_TIME_RAW",
    f"c.{v5_qi('CALL_KEY')} AS CALL_KEY",
    f"c.{v5_qi('GMLC_JOIN_KEY')} AS GMLC_JOIN_KEY",
]
source_select += [f"{expr} AS {v5_qi(name)}" for name, expr in dimension_exprs.items()]
source_select += [f"{expr} AS {v5_qi(name)}" for name, expr in CAT_CANDIDATES]
source_select += [f"{expr} AS {v5_qi(name)}" for name, expr in NUM_CANDIDATES]
source_select += [
    "CASE WHEN i.__KEY IS NOT NULL THEN 1 ELSE 0 END AS IMS_ATTACHED",
    "CASE WHEN r.__KEY IS NOT NULL THEN 1 ELSE 0 END AS RAW_ATTACHED",
]

route_status_ref = v5_qi("ROUTE_STATUS_GMLC") if any(n == "ROUTE_STATUS_GMLC" for n, _ in CAT_CANDIDATES) else "NULL"
route_fallback_ref = v5_qi("ROUTE_FALLBACK") if any(n == "ROUTE_FALLBACK" for n, _ in CAT_CANDIDATES) else "NULL"
default_ref = v5_qi("DEFAULT_ROUTED_CALL") if any(n == "DEFAULT_ROUTED_CALL" for n, _ in CAT_CANDIDATES) else "NULL"
sip_status_ref = v5_qi("SIP_STATUS") if any(n == "SIP_STATUS" for n, _ in CAT_CANDIDATES) else "NULL"
lbr_diff_ref = v5_qi("LBR_PSAP_DIFF") if any(n == "LBR_PSAP_DIFF" for n, _ in NUM_CANDIDATES) else "NULL"
ims_reg_reason_ref = v5_qi("IMS__REGISTER_PCSCF_REASONCODES") if any(n == "IMS__REGISTER_PCSCF_REASONCODES" for n, _ in CAT_CANDIDATES) else "NULL"
ims_reg_status_ref = v5_qi("IMS__REGISTER_PCSCF_STATUS") if any(n == "IMS__REGISTER_PCSCF_STATUS" for n, _ in CAT_CANDIDATES) else "NULL"
raw_reg_ref = v5_qi("RAW__REGISTER_PCSCF_STATUS") if any(n == "RAW__REGISTER_PCSCF_STATUS" for n, _ in CAT_CANDIDATES) else "NULL"
raw_nonreg_ref = v5_qi("RAW__NON_REGISTER_PCSCF_STATUS") if any(n == "RAW__NON_REGISTER_PCSCF_STATUS" for n, _ in CAT_CANDIDATES) else "NULL"
raw_ecscf_ref = v5_qi("RAW__ECSCF_STATUS") if any(n == "RAW__ECSCF_STATUS" for n, _ in CAT_CANDIDATES) else "NULL"

con.execute(
    f"""
    CREATE TEMP TABLE evidence_base_v5 AS
    WITH src AS (
        SELECT {', '.join(source_select)}
        FROM gmlc_calls_v4 c
        LEFT JOIN ims_agg_v4 i ON i.__KEY = c.GMLC_JOIN_KEY
        LEFT JOIN raw_call_agg_v4 r ON r.__KEY = c.CALL_KEY
        WHERE c.MISROUTE_LABEL IN (0, 1)
    )
    SELECT src.*,
           V5_NORM_TEXT(SETUP_ECGI_HEX) AS SETUP_ECGI_HEX_N,
           V5_NORM_ID(USID) AS USID_N,
           V5_NORM_ID(ESRK) AS ESRK_N,
           V5_NORM_TEXT(MARKET) AS MARKET_N,
           CASE WHEN UPPER(COALESCE(CAST({route_status_ref} AS TEXT), '')) LIKE '%MISMATCH%'
                THEN 1 ELSE 0 END AS FLAG_ROUTE_MISMATCH,
           CASE WHEN COALESCE(V5_TRUTHY({route_fallback_ref}), 0) = 1
                     OR COALESCE(V5_TRUTHY({default_ref}), 0) = 1
                THEN 1 ELSE 0 END AS FLAG_DEFAULT_OR_FALLBACK,
           COALESCE(V5_POSITIVE({lbr_diff_ref}), 0) AS FLAG_LBR_PSAP_DIFF,
           COALESCE(V5_ISSUE_FLAG({sip_status_ref}), 0) AS FLAG_SIP_ISSUE,
           CASE WHEN COALESCE(V5_ISSUE_FLAG({ims_reg_reason_ref}), 0) = 1
                     OR COALESCE(V5_ISSUE_FLAG({ims_reg_status_ref}), 0) = 1
                THEN 1 ELSE 0 END AS FLAG_IMS_REG_ISSUE,
           CASE WHEN COALESCE(V5_ISSUE_FLAG({raw_reg_ref}), 0) = 1
                     OR COALESCE(V5_ISSUE_FLAG({raw_nonreg_ref}), 0) = 1
                THEN 1 ELSE 0 END AS FLAG_RAW_REG_ISSUE,
           COALESCE(V5_ISSUE_FLAG({raw_ecscf_ref}), 0) AS FLAG_RAW_ECSCF_ISSUE
    FROM src
    """
)
con.execute("CREATE INDEX temp.idx_evidence_base_v5_label ON evidence_base_v5(MISROUTE_LABEL)")
con.execute("CREATE INDEX temp.idx_evidence_base_v5_exact ON evidence_base_v5(SETUP_ECGI_HEX_N, USID_N, ESRK_N, CALL_POPULATION_TAG)")

base_count = con.execute("SELECT COUNT(*) FROM evidence_base_v5").fetchone()[0]
misroute_count = con.execute("SELECT COUNT(*) FROM evidence_base_v5 WHERE MISROUTE_LABEL=1").fetchone()[0]
print(f"Prepared narrow v5 evidence base: {base_count:,} strong calls; {misroute_count:,} definite misroutes")


# ---------------------------------------------------------------------------
# 3. Build case profiles and matched-correct control profiles.
# ---------------------------------------------------------------------------
def token(name):
    return re.sub(r"[^A-Z0-9]+", "_", str(name).upper()).strip("_")


DIM_CAT_COLS = ["PSAP_NAME", "ROUTE_PSAP_NAME", "MARKET", "MARKET_CLUSTER", "REGION", "STATE", "COUNTY"]
EVIDENCE_CAT_COLS = [name for name, _ in CAT_CANDIDATES]
PROFILE_CAT_COLS = list(dict.fromkeys(DIM_CAT_COLS + EVIDENCE_CAT_COLS))

FLAG_COLS = [
    "FLAG_ROUTE_MISMATCH", "FLAG_DEFAULT_OR_FALLBACK", "FLAG_LBR_PSAP_DIFF",
    "FLAG_SIP_ISSUE", "FLAG_IMS_REG_ISSUE", "FLAG_RAW_REG_ISSUE",
    "FLAG_RAW_ECSCF_ISSUE", "IMS_ATTACHED", "RAW_ATTACHED",
]
PROFILE_NUM_COLS = [name for name, _ in NUM_CANDIDATES] + FLAG_COLS


def profile_select(prefix=""):
    parts = []
    for col in PROFILE_CAT_COLS:
        parts.append(f"V5_MODE_PACK({v5_qi(col)}) AS {v5_qi('CAT__' + token(col))}")
    for col in PROFILE_NUM_COLS:
        parts.append(f"AVG(V5_TO_REAL({v5_qi(col)})) AS {v5_qi('NUM__' + token(col))}")
    return parts


case_group_cols = [
    "FCC_PSAP_ID", "EXPECTED_FCC_PSAP_ID", "SETUP_ECGI_HEX_N",
    "USID_N", "ESRK_N", "CALL_POPULATION_TAG",
]
case_select = [v5_qi(c) for c in case_group_cols] + [
    "COUNT(*) AS MISROUTE_CALLS",
    "MIN(V5_UTC_EPOCH(CALL_TIME_RAW)) AS FIRST_EPOCH",
    "MAX(V5_UTC_EPOCH(CALL_TIME_RAW)) AS LAST_EPOCH",
    "V5_KEY_JOIN(SETUP_ECGI_HEX_N, USID_N, ESRK_N, CALL_POPULATION_TAG) AS EXACT_COHORT_KEY",
    "V5_KEY_JOIN(SETUP_ECGI_HEX_N, USID_N, CALL_POPULATION_TAG) AS ECGI_USID_KEY",
    "V5_KEY_JOIN(SETUP_ECGI_HEX_N, CALL_POPULATION_TAG) AS ECGI_KEY",
    "V5_KEY_JOIN(MARKET_N, CALL_POPULATION_TAG) AS MARKET_KEY",
    "V5_KEY_JOIN(CALL_POPULATION_TAG) AS POPULATION_KEY",
] + profile_select()

con.execute(
    f"""
    CREATE TEMP TABLE case_profiles_v5 AS
    SELECT {', '.join(case_select)}
    FROM evidence_base_v5
    WHERE MISROUTE_LABEL=1
    GROUP BY {', '.join(v5_qi(c) for c in case_group_cols)}
    """
)


def build_control_profile(method, key_cols):
    select_parts = [
        f"'{method}' AS CONTROL_MATCH_LEVEL",
        f"V5_KEY_JOIN({', '.join(v5_qi(c) for c in key_cols)}) AS CONTROL_KEY",
        "COUNT(*) AS CONTROL_CALLS",
    ] + profile_select()
    return f"""
        SELECT {', '.join(select_parts)}
        FROM evidence_base_v5
        WHERE MISROUTE_LABEL=0
        GROUP BY {', '.join(v5_qi(c) for c in key_cols)}
    """


control_sqls = [
    build_control_profile("ECGI_USID_ESRK_POP", ["SETUP_ECGI_HEX_N", "USID_N", "ESRK_N", "CALL_POPULATION_TAG"]),
    build_control_profile("ECGI_USID_POP", ["SETUP_ECGI_HEX_N", "USID_N", "CALL_POPULATION_TAG"]),
    build_control_profile("ECGI_POP", ["SETUP_ECGI_HEX_N", "CALL_POPULATION_TAG"]),
    build_control_profile("MARKET_POP", ["MARKET_N", "CALL_POPULATION_TAG"]),
    build_control_profile("POPULATION_BASELINE", ["CALL_POPULATION_TAG"]),
]
con.execute("CREATE TEMP TABLE control_profiles_v5 AS " + " UNION ALL ".join(control_sqls))
con.execute("CREATE INDEX temp.idx_control_profiles_v5 ON control_profiles_v5(CONTROL_MATCH_LEVEL, CONTROL_KEY)")

# Cohort denominator: all strong calls sharing the exact cell/subscriber-routing context.
con.execute(
    """
    CREATE TEMP TABLE strong_cohort_v5 AS
    SELECT V5_KEY_JOIN(SETUP_ECGI_HEX_N, USID_N, ESRK_N, CALL_POPULATION_TAG) AS EXACT_COHORT_KEY,
           COUNT(*) AS STRONG_CALLS,
           SUM(CASE WHEN MISROUTE_LABEL=1 THEN 1 ELSE 0 END) AS COHORT_MISROUTES,
           SUM(CASE WHEN MISROUTE_LABEL=0 THEN 1 ELSE 0 END) AS COHORT_CORRECT_CALLS
    FROM evidence_base_v5
    GROUP BY SETUP_ECGI_HEX_N, USID_N, ESRK_N, CALL_POPULATION_TAG
    """
)
con.execute("CREATE INDEX temp.idx_strong_cohort_v5 ON strong_cohort_v5(EXACT_COHORT_KEY)")

# PSAP ID -> most frequently observed name, for routed and geometry-expected IDs.
con.execute(
    """
    CREATE TEMP TABLE psap_lookup_v5 AS
    SELECT FCC_PSAP_ID, V5_MODE_PACK(PSAP_NAME) AS PSAP_NAME_PACK
    FROM evidence_base_v5
    WHERE FCC_PSAP_ID <> '<MISSING>'
    GROUP BY FCC_PSAP_ID
    """
)
con.execute("CREATE INDEX temp.idx_psap_lookup_v5 ON psap_lookup_v5(FCC_PSAP_ID)")

# Pick the narrowest correct-call control cohort with enough support.
con.execute(
    """
    CREATE TEMP TABLE misroute_case_mapped_v5 AS
    WITH joined AS (
      SELECT p.*,
             ce.CONTROL_CALLS AS EXACT_CONTROL_CALLS,
             cu.CONTROL_CALLS AS ECGI_USID_CONTROL_CALLS,
             cg.CONTROL_CALLS AS ECGI_CONTROL_CALLS,
             cm.CONTROL_CALLS AS MARKET_CONTROL_CALLS,
             cp.CONTROL_CALLS AS POP_CONTROL_CALLS
      FROM case_profiles_v5 p
      LEFT JOIN control_profiles_v5 ce
        ON ce.CONTROL_MATCH_LEVEL='ECGI_USID_ESRK_POP' AND ce.CONTROL_KEY=p.EXACT_COHORT_KEY
      LEFT JOIN control_profiles_v5 cu
        ON cu.CONTROL_MATCH_LEVEL='ECGI_USID_POP' AND cu.CONTROL_KEY=p.ECGI_USID_KEY
      LEFT JOIN control_profiles_v5 cg
        ON cg.CONTROL_MATCH_LEVEL='ECGI_POP' AND cg.CONTROL_KEY=p.ECGI_KEY
      LEFT JOIN control_profiles_v5 cm
        ON cm.CONTROL_MATCH_LEVEL='MARKET_POP' AND cm.CONTROL_KEY=p.MARKET_KEY
      LEFT JOIN control_profiles_v5 cp
        ON cp.CONTROL_MATCH_LEVEL='POPULATION_BASELINE' AND cp.CONTROL_KEY=p.POPULATION_KEY
    )
    SELECT joined.*,
           CASE
             WHEN SETUP_ECGI_HEX_N <> '<MISSING>' AND USID_N NOT IN ('<MISSING>','-1')
                  AND ESRK_N NOT IN ('<MISSING>','-1') AND COALESCE(EXACT_CONTROL_CALLS,0)>=20
               THEN 'ECGI_USID_ESRK_POP'
             WHEN SETUP_ECGI_HEX_N <> '<MISSING>' AND USID_N NOT IN ('<MISSING>','-1')
                  AND COALESCE(ECGI_USID_CONTROL_CALLS,0)>=20
               THEN 'ECGI_USID_POP'
             WHEN SETUP_ECGI_HEX_N <> '<MISSING>' AND COALESCE(ECGI_CONTROL_CALLS,0)>=50
               THEN 'ECGI_POP'
             WHEN COALESCE(MARKET_CONTROL_CALLS,0)>=200
               THEN 'MARKET_POP'
             ELSE 'POPULATION_BASELINE'
           END AS CONTROL_MATCH_LEVEL,
           CASE
             WHEN SETUP_ECGI_HEX_N <> '<MISSING>' AND USID_N NOT IN ('<MISSING>','-1')
                  AND ESRK_N NOT IN ('<MISSING>','-1') AND COALESCE(EXACT_CONTROL_CALLS,0)>=20
               THEN EXACT_COHORT_KEY
             WHEN SETUP_ECGI_HEX_N <> '<MISSING>' AND USID_N NOT IN ('<MISSING>','-1')
                  AND COALESCE(ECGI_USID_CONTROL_CALLS,0)>=20
               THEN ECGI_USID_KEY
             WHEN SETUP_ECGI_HEX_N <> '<MISSING>' AND COALESCE(ECGI_CONTROL_CALLS,0)>=50
               THEN ECGI_KEY
             WHEN COALESCE(MARKET_CONTROL_CALLS,0)>=200
               THEN MARKET_KEY
             ELSE POPULATION_KEY
           END AS CONTROL_KEY
    FROM joined
    """
)


# ---------------------------------------------------------------------------
# 4. Produce the requested one-row-per-misroute-pattern evidence table.
# ---------------------------------------------------------------------------
base_out = [
    "m.FCC_PSAP_ID",
    "m.EXPECTED_FCC_PSAP_ID",
    "V5_PACK_VALUE(m.CAT__PSAP_NAME) AS PSAP_NAME",
    "V5_PACK_VALUE(x.PSAP_NAME_PACK) AS EXPECTED_PSAP_NAME",
    "V5_PACK_VALUE(m.CAT__MARKET) AS MARKET",
    "V5_PACK_VALUE(m.CAT__MARKET_CLUSTER) AS MARKET_CLUSTER",
    "V5_PACK_VALUE(m.CAT__REGION) AS REGION",
    "V5_PACK_VALUE(m.CAT__STATE) AS STATE",
    "V5_PACK_VALUE(m.CAT__COUNTY) AS COUNTY",
    "m.SETUP_ECGI_HEX_N AS SETUP_ECGI_HEX",
    "m.USID_N AS USID",
    "m.ESRK_N AS ESRK",
    "m.CALL_POPULATION_TAG",
    "COALESCE(sc.STRONG_CALLS, m.MISROUTE_CALLS) AS TOTAL_STRONG_CALLS",
    "m.MISROUTE_CALLS",
    "COALESCE(sc.COHORT_MISROUTES, m.MISROUTE_CALLS) AS COHORT_MISROUTES_ALL_EXPECTED_PSAPS",
    "COALESCE(sc.COHORT_CORRECT_CALLS,0) AS EXACT_CONTEXT_CORRECT_CALLS",
    "ROUND(100.0*m.MISROUTE_CALLS/NULLIF(COALESCE(sc.STRONG_CALLS,m.MISROUTE_CALLS),0),6) AS PATTERN_MISROUTE_RATE_PCT",
    "ROUND(100.0*COALESCE(sc.COHORT_MISROUTES,m.MISROUTE_CALLS)/NULLIF(COALESCE(sc.STRONG_CALLS,m.MISROUTE_CALLS),0),6) AS COHORT_MISROUTE_RATE_PCT",
    "ROUND(100.0*m.MISROUTE_CALLS/NULLIF(COALESCE(sc.COHORT_MISROUTES,m.MISROUTE_CALLS),0),6) AS SHARE_OF_CONTEXT_MISROUTES_PCT",
    "V5_EPOCH_ISO(m.FIRST_EPOCH) AS FIRST_CALL_UTC",
    "V5_EPOCH_ISO(m.LAST_EPOCH) AS LAST_MISROUTE_UTC",
    "m.CONTROL_MATCH_LEVEL",
    "COALESCE(ctrl.CONTROL_CALLS,0) AS MATCHED_CORRECT_CONTROL_CALLS",
]

cat_out = []
for col in EVIDENCE_CAT_COLS:
    t = token(col)
    pack = v5_qi("CAT__" + t)
    cat_out += [
        f"V5_PACK_VALUE(m.{pack}) AS {v5_qi(t + '_MISROUTE_MODE')}",
        f"ROUND(V5_PACK_SHARE(m.{pack}),6) AS {v5_qi(t + '_MISROUTE_MODE_SHARE')}",
        f"V5_PACK_VALUE(ctrl.{pack}) AS {v5_qi(t + '_CONTROL_MODE')}",
        f"ROUND(V5_PACK_SHARE(ctrl.{pack}),6) AS {v5_qi(t + '_CONTROL_MODE_SHARE')}",
        f"CASE WHEN V5_PACK_VALUE(m.{pack})=V5_PACK_VALUE(ctrl.{pack}) THEN 1 ELSE 0 END AS {v5_qi(t + '_MODE_AGREEMENT')}",
    ]

num_out = []
for col in [name for name, _ in NUM_CANDIDATES]:
    t = token(col)
    ncol = v5_qi("NUM__" + t)
    num_out += [
        f"ROUND(m.{ncol},6) AS {v5_qi(t + '_MISROUTE_MEAN')}",
        f"ROUND(ctrl.{ncol},6) AS {v5_qi(t + '_CONTROL_MEAN')}",
        f"ROUND(m.{ncol}-ctrl.{ncol},6) AS {v5_qi(t + '_DIFFERENCE')}",
    ]

friendly_flags = {
    "FLAG_ROUTE_MISMATCH": "ROUTE_MISMATCH_RATE",
    "FLAG_DEFAULT_OR_FALLBACK": "DEFAULT_OR_FALLBACK_RATE",
    "FLAG_LBR_PSAP_DIFF": "LBR_PSAP_DIFFERENCE_RATE",
    "FLAG_SIP_ISSUE": "SIP_ISSUE_INDICATOR_RATE",
    "FLAG_IMS_REG_ISSUE": "IMS_REGISTRATION_ISSUE_INDICATOR_RATE",
    "FLAG_RAW_REG_ISSUE": "CCDR_REGISTRATION_ISSUE_INDICATOR_RATE",
    "FLAG_RAW_ECSCF_ISSUE": "CCDR_ECSCF_ISSUE_INDICATOR_RATE",
    "IMS_ATTACHED": "IMS_ATTACHMENT_RATE",
    "RAW_ATTACHED": "CCDR_ATTACHMENT_RATE",
}
flag_out = []
for source, friendly in friendly_flags.items():
    ncol = v5_qi("NUM__" + token(source))
    flag_out += [
        f"ROUND(m.{ncol},6) AS {v5_qi(friendly + '_MISROUTE')}",
        f"ROUND(ctrl.{ncol},6) AS {v5_qi(friendly + '_CONTROL')}",
        f"ROUND(100.0*(m.{ncol}-ctrl.{ncol}),3) AS {v5_qi(friendly + '_DELTA_PP')}",
    ]

con.execute(
    f"""
    CREATE TEMP TABLE misroute_investigation_wide_v5 AS
    SELECT {', '.join(base_out + cat_out + num_out + flag_out)}
    FROM misroute_case_mapped_v5 m
    LEFT JOIN control_profiles_v5 ctrl
      ON ctrl.CONTROL_MATCH_LEVEL=m.CONTROL_MATCH_LEVEL AND ctrl.CONTROL_KEY=m.CONTROL_KEY
    LEFT JOIN strong_cohort_v5 sc ON sc.EXACT_COHORT_KEY=m.EXACT_COHORT_KEY
    LEFT JOIN psap_lookup_v5 x ON x.FCC_PSAP_ID=m.EXPECTED_FCC_PSAP_ID
    """
)


def wide_col(name, default="NULL"):
    return v5_qi(name) if name in set(v5_columns("misroute_investigation_wide_v5")) else default


route_delta = f"({wide_col('ROUTE_MISMATCH_RATE_DELTA_PP','0')} / 100.0)"
fallback_delta = f"({wide_col('DEFAULT_OR_FALLBACK_RATE_DELTA_PP','0')} / 100.0)"
lbr_delta = f"({wide_col('LBR_PSAP_DIFFERENCE_RATE_DELTA_PP','0')} / 100.0)"
sip_delta = f"({wide_col('SIP_ISSUE_INDICATOR_RATE_DELTA_PP','0')} / 100.0)"
ims_delta = f"({wide_col('IMS_REGISTRATION_ISSUE_INDICATOR_RATE_DELTA_PP','0')} / 100.0)"
raw_reg_delta = f"({wide_col('CCDR_REGISTRATION_ISSUE_INDICATOR_RATE_DELTA_PP','0')} / 100.0)"
raw_ecscf_delta = f"({wide_col('CCDR_ECSCF_ISSUE_INDICATOR_RATE_DELTA_PP','0')} / 100.0)"

route_case = wide_col("ROUTE_STATUS_GMLC_MISROUTE_MODE")
route_ctrl = wide_col("ROUTE_STATUS_GMLC_CONTROL_MODE")
route_share = wide_col("ROUTE_STATUS_GMLC_MISROUTE_MODE_SHARE", "0")
esinet_case = wide_col("ROUTE_ESINET_MISROUTE_MODE")
esinet_ctrl = wide_col("ROUTE_ESINET_CONTROL_MODE")
esinet_share = wide_col("ROUTE_ESINET_MISROUTE_MODE_SHARE", "0")
esz_case = wide_col("ROUTE_ESZ_MISROUTE_MODE")
esz_ctrl = wide_col("ROUTE_ESZ_CONTROL_MODE")
esz_share = wide_col("ROUTE_ESZ_MISROUTE_MODE_SHARE", "0")
ims_attach = wide_col("IMS_ATTACHMENT_RATE_MISROUTE", "0")
raw_attach = wide_col("CCDR_ATTACHMENT_RATE_MISROUTE", "0")

score_sql = (
    f"V5_EVIDENCE_SCORE({route_delta},{fallback_delta},{lbr_delta},{sip_delta},"
    f"{ims_delta},{raw_reg_delta},{raw_ecscf_delta},"
    f"{route_case},{route_ctrl},{route_share},"
    f"{esinet_case},{esinet_ctrl},{esinet_share},"
    f"{esz_case},{esz_ctrl},{esz_share})"
)

cause_sql = (
    f"V5_PROBABLE_CAUSE({route_delta},{fallback_delta},{lbr_delta},{sip_delta},"
    f"{ims_delta},{raw_reg_delta},{raw_ecscf_delta},"
    f"{route_case},{route_ctrl},{route_share},"
    f"MATCHED_CORRECT_CONTROL_CALLS,{ims_attach},{raw_attach})"
)

con.execute(
    f"""
    CREATE TEMP TABLE misroute_investigation_v5 AS
    WITH scored AS (
      SELECT w.*, {score_sql} AS CORROBORATION_SCORE
      FROM misroute_investigation_wide_v5 w
    )
    SELECT scored.*,
           V5_CONCLUSION(CORROBORATION_SCORE, MATCHED_CORRECT_CONTROL_CALLS,
                         {ims_attach}, {raw_attach}) AS EVIDENCE_CONCLUSION,
           {cause_sql} AS PROBABLE_CAUSE_CATEGORY,
           V5_SUMMARY(MISROUTE_CALLS, MATCHED_CORRECT_CONTROL_CALLS,
                      CONTROL_MATCH_LEVEL, {route_delta}, {fallback_delta},
                      {lbr_delta}, {ims_delta}, {raw_reg_delta}) AS EVIDENCE_SUMMARY,
           'Associations corroborate or contextualize the geometry label; they do not prove causality and do not redefine the label.' AS CAUSALITY_NOTE
    FROM scored
    """
)


# Long, analysis-friendly companion table.
long_selects = []
key_cols = [
    "FCC_PSAP_ID", "EXPECTED_FCC_PSAP_ID", "PSAP_NAME", "EXPECTED_PSAP_NAME",
    "MARKET", "SETUP_ECGI_HEX", "USID", "ESRK", "CALL_POPULATION_TAG",
    "MISROUTE_CALLS", "FIRST_CALL_UTC", "LAST_MISROUTE_UTC",
    "CONTROL_MATCH_LEVEL", "MATCHED_CORRECT_CONTROL_CALLS",
    "CORROBORATION_SCORE", "EVIDENCE_CONCLUSION", "PROBABLE_CAUSE_CATEGORY",
]
key_sql = ", ".join(v5_qi(c) for c in key_cols)
final_cols = set(v5_columns("misroute_investigation_v5"))

for col in EVIDENCE_CAT_COLS:
    t = token(col)
    required = {t + "_MISROUTE_MODE", t + "_MISROUTE_MODE_SHARE", t + "_CONTROL_MODE", t + "_CONTROL_MODE_SHARE"}
    if required <= final_cols:
        long_selects.append(
            f"SELECT {key_sql}, 'CATEGORICAL_MODE' AS EVIDENCE_TYPE, '{col}' AS FEATURE, "
            f"CAST({v5_qi(t + '_MISROUTE_MODE')} AS TEXT) AS MISROUTE_VALUE, "
            f"{v5_qi(t + '_MISROUTE_MODE_SHARE')} AS MISROUTE_METRIC, "
            f"CAST({v5_qi(t + '_CONTROL_MODE')} AS TEXT) AS CONTROL_VALUE, "
            f"{v5_qi(t + '_CONTROL_MODE_SHARE')} AS CONTROL_METRIC, NULL AS DIFFERENCE "
            "FROM misroute_investigation_v5"
        )

for col in [name for name, _ in NUM_CANDIDATES]:
    t = token(col)
    required = {t + "_MISROUTE_MEAN", t + "_CONTROL_MEAN", t + "_DIFFERENCE"}
    if required <= final_cols:
        long_selects.append(
            f"SELECT {key_sql}, 'NUMERIC_MEAN' AS EVIDENCE_TYPE, '{col}' AS FEATURE, "
            f"CAST({v5_qi(t + '_MISROUTE_MEAN')} AS TEXT) AS MISROUTE_VALUE, "
            f"{v5_qi(t + '_MISROUTE_MEAN')} AS MISROUTE_METRIC, "
            f"CAST({v5_qi(t + '_CONTROL_MEAN')} AS TEXT) AS CONTROL_VALUE, "
            f"{v5_qi(t + '_CONTROL_MEAN')} AS CONTROL_METRIC, "
            f"{v5_qi(t + '_DIFFERENCE')} AS DIFFERENCE "
            "FROM misroute_investigation_v5"
        )

for _, friendly in friendly_flags.items():
    required = {friendly + "_MISROUTE", friendly + "_CONTROL", friendly + "_DELTA_PP"}
    if required <= final_cols:
        long_selects.append(
            f"SELECT {key_sql}, 'RATE' AS EVIDENCE_TYPE, '{friendly}' AS FEATURE, "
            f"CAST({v5_qi(friendly + '_MISROUTE')} AS TEXT) AS MISROUTE_VALUE, "
            f"{v5_qi(friendly + '_MISROUTE')} AS MISROUTE_METRIC, "
            f"CAST({v5_qi(friendly + '_CONTROL')} AS TEXT) AS CONTROL_VALUE, "
            f"{v5_qi(friendly + '_CONTROL')} AS CONTROL_METRIC, "
            f"{v5_qi(friendly + '_DELTA_PP')} AS DIFFERENCE "
            "FROM misroute_investigation_v5"
        )

if long_selects:
    con.execute("CREATE TEMP TABLE misroute_evidence_long_v5 AS " + " UNION ALL ".join(long_selects))
else:
    con.execute(
        "CREATE TEMP TABLE misroute_evidence_long_v5 AS "
        "SELECT NULL AS FCC_PSAP_ID, NULL AS FEATURE WHERE 0"
    )


# ---------------------------------------------------------------------------
# 5. Chunked exports (large-result safe), plus the requested 2160 view.
# ---------------------------------------------------------------------------
def v5_export(query, path, chunksize=50000):
    path = Path(path)
    wrote = False
    rows = 0
    for chunk in pd.read_sql_query(query, con, chunksize=chunksize):
        chunk.to_csv(path, mode="w" if not wrote else "a", header=not wrote, index=False)
        wrote = True
        rows += len(chunk)
    if not wrote:
        pd.read_sql_query(query + " LIMIT 0", con).to_csv(path, index=False)
    return rows


main_path = OUTPUT_DIR / "misroute_investigation_v5.csv"
psap_2160_path = OUTPUT_DIR / "psap_2160_investigation_v5.csv"
long_path = OUTPUT_DIR / "misroute_evidence_long_v5.csv"
dictionary_path = OUTPUT_DIR / "misroute_investigation_v5_data_dictionary.csv"

main_rows = v5_export(
    "SELECT * FROM misroute_investigation_v5 "
    "ORDER BY CORROBORATION_SCORE DESC, MISROUTE_CALLS DESC",
    main_path,
)
psap_2160_rows = v5_export(
    "SELECT * FROM misroute_investigation_v5 WHERE FCC_PSAP_ID='2160' "
    "ORDER BY CORROBORATION_SCORE DESC, MISROUTE_CALLS DESC",
    psap_2160_path,
)
long_rows = v5_export(
    "SELECT * FROM misroute_evidence_long_v5 "
    "ORDER BY FCC_PSAP_ID, EXPECTED_FCC_PSAP_ID, MISROUTE_CALLS DESC, FEATURE",
    long_path,
)

data_dictionary = pd.DataFrame([
    ("DEFINITE_MISROUTE input", "Geometry label only: uncertainty area misses routed PSAP and is wholly covered by one different PSAP."),
    ("TOTAL_STRONG_CALLS", "All correct+misroute strong-label calls in the same ECGI/USID/ESRK/population context."),
    ("MISROUTE_CALLS", "Calls in this exact routed PSAP -> expected PSAP pattern and context."),
    ("FIRST_CALL_UTC / LAST_MISROUTE_UTC", "Chronological limits after parsing Oracle/ISO timestamps; not raw-string MIN/MAX."),
    ("CONTROL_MATCH_LEVEL", "Narrowest sufficiently supported correctly routed comparison cohort."),
    ("*_MISROUTE_MODE / *_CONTROL_MODE", "Dominant value among misroutes versus matched correct calls."),
    ("*_MODE_SHARE", "Fraction of the respective cohort having its displayed dominant value."),
    ("*_RATE_DELTA_PP", "Misroute rate minus matched-correct rate in percentage points."),
    ("IMS_ATTACHMENT_RATE", "Fraction linked by exact GMLC.UNIQ911_CID = IMS.UNIQ911_CID."),
    ("CCDR_ATTACHMENT_RATE", "Fraction reaching RAW CCDR through safe IMSCHARGINGID/ICID or SESSIONID/SESSION_ID bridges."),
    ("CORROBORATION_SCORE", "Transparent 0-100 triage score from route/LBR/SIP/IMS/CCDR differences; not model probability."),
    ("EVIDENCE_CONCLUSION", "Whether network evidence strongly/moderately/weakly corroborates the geometry finding, or does not differ."),
    ("PROBABLE_CAUSE_CATEGORY", "Rule-based investigation direction, not proven root cause."),
])
data_dictionary.columns = ["FIELD", "MEANING"]
data_dictionary.to_csv(dictionary_path, index=False)


# ---------------------------------------------------------------------------
# 6. Compact notebook result.
# ---------------------------------------------------------------------------
display(Markdown("## V5 combined misroute investigation"))
print(f"Investigation patterns: {main_rows:,}")
print(f"PSAP 2160 patterns:     {psap_2160_rows:,}")
print(f"Long evidence rows:     {long_rows:,}")

summary = pd.read_sql_query(
    """
    SELECT EVIDENCE_CONCLUSION, PROBABLE_CAUSE_CATEGORY,
           COUNT(*) AS PATTERNS, SUM(MISROUTE_CALLS) AS MISROUTE_CALLS
    FROM misroute_investigation_v5
    GROUP BY EVIDENCE_CONCLUSION, PROBABLE_CAUSE_CATEGORY
    ORDER BY MISROUTE_CALLS DESC
    """,
    con,
)
display(summary)

preview_where = "WHERE FCC_PSAP_ID='2160'" if psap_2160_rows else ""
preview = pd.read_sql_query(
    f"""
    SELECT FCC_PSAP_ID, EXPECTED_FCC_PSAP_ID, PSAP_NAME, EXPECTED_PSAP_NAME,
           MARKET, SETUP_ECGI_HEX, USID, ESRK, CALL_POPULATION_TAG,
           TOTAL_STRONG_CALLS, MISROUTE_CALLS, FIRST_CALL_UTC, LAST_MISROUTE_UTC,
           CONTROL_MATCH_LEVEL, MATCHED_CORRECT_CONTROL_CALLS,
           ROUTE_STATUS_GMLC_MISROUTE_MODE,
           ROUTE_STATUS_GMLC_CONTROL_MODE,
           ROUTE_MISMATCH_RATE_DELTA_PP,
           LBR_PSAP_DIFFERENCE_RATE_DELTA_PP,
           IMS_REGISTRATION_ISSUE_INDICATOR_RATE_DELTA_PP,
           CCDR_REGISTRATION_ISSUE_INDICATOR_RATE_DELTA_PP,
           CORROBORATION_SCORE, EVIDENCE_CONCLUSION,
           PROBABLE_CAUSE_CATEGORY, EVIDENCE_SUMMARY
    FROM misroute_investigation_v5
    {preview_where}
    ORDER BY CORROBORATION_SCORE DESC, MISROUTE_CALLS DESC
    LIMIT 50
    """,
    con,
)
display(preview)

print("Saved:")
for p in (main_path, psap_2160_path, long_path, dictionary_path):
    print(" -", p)
print("V5 evidence step completed. No geometry labels were changed.")

