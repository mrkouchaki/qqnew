-- Dataset 3: extended GMLC/signaling features from the already correlated
-- E911_LTE_GMLC_PSAPSIM_CALLS table. The PL/SQL block selects only candidate
-- columns that actually exist, so optional schema-version fields do not break
-- the export. It deliberately excludes MSISDN/IMSI/PANI/PIDF.
-- Direct run:
--   sqlplus YOUR_USER@YOUR_TNS_ALIAS @03_export_gmlc_psapsim_features.sql ^
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

variable feature_rc refcursor
declare
    v_cols varchar2(32767);
    v_sql  varchar2(32767);
begin
    select listagg('t."' || c.column_name || '"', ',')
               within group (order by wanted.ord)
      into v_cols
      from (
          select column_value as column_name, rownum as ord
          from table(sys.odcivarchar2list(
              'HDR_TRID','CTID','UNIQ911_CID','GMLC_VENDOR',
              'CALL_DATE','CALL_BEGIN_TIME','CALL_END_TIME',
              'CALL_BEGIN_TIME_UTC','CALL_END_TIME_UTC','CALL_DURATION_SEC',
              'REGION','MARKET_CLUSTER','MARKET','COUNTY','STATE',
              'ESRK','GMLC_ESRK','POS_METHOD_USED','UNCERT_METERS','CONFIDENCE',
              'SETUP_ECGI_HEX','INVITE_ECGI_HEX','LOCATE_ECGI_HEX','HDV_ECGI',
              'TAC_HEX','TAC_DEC','USID','ENBID','USEID',
              'PSAP_ID','FCC_PSAP_ID','PSAP_NAME','PSAP_SIM_CALL',
              'P2_SUCCESS','P2_FAILURE_SHORT','P2_LOCATE','LOCATE_TIME_SEC',
              'DEFAULT_ROUTED_CALL','MLI_STATUS_CODE','PL_RESULT_CODE',
              'REL_RESULT_CODE','ALLOCATES_RESULT_CODE','ORIG_RESULT_CODE',
              'HDV_RESULT_CODE','ROUTE_STATUS_GMLC','LRR_INVITE_STATUS',
              'SIP_STATUS','SIP_SYSTEM_ID','SIP_ORIG_NODE','ORIG_AUD_HOST',
              'PL_AUD_HOST','HDV_AUD_HOST','MME_VENDOR','MME_POOL_ID',
              'MME_POOL_NAME','MME_NAME','INVITE_CN_MME_ID','NETWORK',
              'TEST_CALL','COMPLETE_CALL','EXCEPTION_FLAG','CELL_TIME_ZONE'
          ))
      ) wanted
      join all_tab_columns c
        on c.owner = 'E911'
       and c.table_name = 'E911_LTE_GMLC_PSAPSIM_CALLS'
       and c.column_name = wanted.column_name;

    if v_cols is null then
        raise_application_error(-20001,
            'No requested columns found in E911.E911_LTE_GMLC_PSAPSIM_CALLS');
    end if;

    v_sql := 'select ' || v_cols ||
             ' from e911.e911_lte_gmlc_psapsim_calls t' ||
             ' where t.call_begin_time_utc >= from_tz(to_timestamp(''' ||
             '&&START_UTC' || ''',''YYYY-MM-DD HH24:MI:SS''),''UTC'')' ||
             ' and t.call_begin_time_utc < from_tz(to_timestamp(''' ||
             '&&END_UTC' || ''',''YYYY-MM-DD HH24:MI:SS''),''UTC'')' ||
             ' order by t.call_begin_time_utc';
    open :feature_rc for v_sql;
end;
/

spool "&&ROOT_DIR\data\gmlc_psapsim_features.csv"
print feature_rc
spool off
set markup csv off
prompt Wrote &&ROOT_DIR\data\gmlc_psapsim_features.csv

