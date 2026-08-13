-- Dataset 5: exact available columns for targeted next-step investigation.
-- This is not a general database search; it inventories only the four selected
-- tables used by this project.
-- Direct run:
--   sqlplus YOUR_USER@YOUR_TNS_ALIAS @05_export_column_inventory.sql ^
--     "C:\temp\psap_route_integrity_v1"

whenever sqlerror exit sql.sqlcode
set verify off feedback off echo off heading on pagesize 50000 linesize 32767
set trimspool on tab off termout on
set markup csv on delimiter , quote on

define ROOT_DIR = '&1'

spool "&&ROOT_DIR\data\selected_table_columns.csv"
select owner, table_name, column_id, column_name, data_type,
       data_length, data_precision, data_scale, nullable
from all_tab_columns
where (owner, table_name) in (
    ('E911','E911_PSAPSIM_CALL_DETAILS_LTE'),
    ('E911','E911_PSAP_BOUNDARIES'),
    ('E911','E911_LTE_GMLC_PSAPSIM_CALLS'),
    ('E911','E911_LTE_LSR_CSR_CD')
)
order by owner, table_name, column_id;
spool off
set markup csv off
prompt Wrote &&ROOT_DIR\data\selected_table_columns.csv

