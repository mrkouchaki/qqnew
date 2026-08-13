-- PSAP routing-integrity export runner
--
-- Recommended Windows PowerShell commands:
--   New-Item -ItemType Directory -Force C:\temp | Out-Null
--   Expand-Archive .\psap_route_integrity_v1.zip C:\temp -Force
--   cd C:\temp\psap_route_integrity_v1\sql
--   sqlplus YOUR_ORACLE_USER@YOUR_TNS_ALIAS @00_run_all_exports.sql
--
-- SQL Developer alternative: open this file and use Run Script (F5).
-- Edit only START_UTC and END_UTC below. END_UTC is exclusive.

whenever sqlerror exit sql.sqlcode
set verify off feedback on echo on

define ROOT_DIR = "C:\temp\psap_route_integrity_v1"
define START_UTC = "2026-07-01 00:00:00"
define END_UTC   = "2026-08-01 00:00:00"

host if not exist "&&ROOT_DIR\data" mkdir "&&ROOT_DIR\data"
host if not exist "&&ROOT_DIR\outputs" mkdir "&&ROOT_DIR\outputs"

prompt Export 1/5: PSAPSIM calls and location/routing fields
@@01_export_psapsim_calls.sql "&&START_UTC" "&&END_UTC" "&&ROOT_DIR"

prompt Export 2/5: PSAP jurisdiction boundaries
@@02_export_psap_boundaries.sql "&&ROOT_DIR"

prompt Export 3/5: GMLC/signaling features already correlated to PSAPSIM
@@03_export_gmlc_psapsim_features.sql "&&START_UTC" "&&END_UTC" "&&ROOT_DIR"

prompt Export 4/5: LSR/CSR signaling and KPI features
@@04_export_lsr_csr_features.sql "&&START_UTC" "&&END_UTC" "&&ROOT_DIR"

prompt Export 5/5: exact source-column inventory for targeted follow-up
@@05_export_column_inventory.sql "&&ROOT_DIR"

prompt ================================================================
prompt Exports finished under &&ROOT_DIR\data
prompt Next: open ..\psap_misroute_feature_analysis.ipynb and Run All.
prompt ================================================================
exit
