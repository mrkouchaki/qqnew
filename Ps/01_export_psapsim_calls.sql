-- Dataset 1: strong-label source.
-- Coordinates + uncertainty describe the reported location; FCC_PSAP_ID is
-- the routed PSAP that will be checked against the boundary polygons.
-- Direct run:
--   sqlplus YOUR_USER@YOUR_TNS_ALIAS @01_export_psapsim_calls.sql ^
--     "2026-07-01 00:00:00" "2026-08-01 00:00:00" ^
--     "C:\temp\psap_route_integrity_v1"

whenever sqlerror exit sql.sqlcode
set verify off feedback off echo off heading on pagesize 50000 linesize 32767
set trimspool on tab off termout on
set markup csv on delimiter , quote on
alter session set nls_timestamp_tz_format='YYYY-MM-DD"T"HH24:MI:SS.FF6 TZH:TZM';
alter session set nls_timestamp_format='YYYY-MM-DD"T"HH24:MI:SS.FF6';

define START_UTC = '&1'
define END_UTC   = '&2'
define ROOT_DIR  = '&3'

spool "&&ROOT_DIR\data\psapsim_calls.csv"
select
    plrf_cid,
    gmlc_vendor,
    call_date,
    complete_call,
    call_begin_time,
    call_end_time,
    region,
    market_cluster,
    market,
    county,
    state,
    esrk,
    locate_begin_time,
    pos_method_used,
    latitude,
    longitude,
    uncert_meters,
    confidence,
    shape_type,
    test_call,
    datetime_ins,
    setup_ecgi_hex,
    call_begin_time_utc,
    call_duration_sec,
    invite_ecgi_hex,
    tac_hex,
    locate_begin_time_utc,
    psap_name,
    psap_id,
    fcc_psap_id,
    locate_ecgi_hex,
    email_yn,
    textmsg_yn,
    summary_yn,
    gmlc_esrk,
    cell_time_zone,
    tac_dec,
    locate_ecgi_dec,
    setup_ecgi_dec,
    invite_ecgi_dec,
    schedule_id,
    testsim_id,
    gmlc_site_address,
    esrk_match_yn,
    class_of_service_yn,
    validation_passed,
    usid,
    enbid,
    useid,
    psapsim_range
from e911.e911_psapsim_call_details_lte
where call_begin_time_utc >= from_tz(
          to_timestamp('&&START_UTC','YYYY-MM-DD HH24:MI:SS'),'UTC')
  and call_begin_time_utc <  from_tz(
          to_timestamp('&&END_UTC','YYYY-MM-DD HH24:MI:SS'),'UTC')
order by call_begin_time_utc;
spool off
set markup csv off
prompt Wrote &&ROOT_DIR\data\psapsim_calls.csv

