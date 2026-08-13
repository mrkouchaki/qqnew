-- Dataset 4: LSR/CSR signaling/KPI candidates. Only present columns are
-- exported. The script prefers CALL_BEGIN_TIME_UTC; otherwise it uses CALL_DATE.
-- Direct run:
--   sqlplus YOUR_USER@YOUR_TNS_ALIAS @04_export_lsr_csr_features.sql ^
--     "2026-07-01 00:00:00" "2026-08-01 00:00:00" ^
--     "C:\temp\psap_route_integrity_v1"

whenever sqlerror exit sql.sqlcode
set verify off feedback off echo off heading on pagesize 50000 linesize 32767
set trimspool on tab off termout on
set markup csv on delimiter , quote on
alter session set nls_timestamp_tz_format='YYYY-MM-DD"T"HH24:MI:SS.FF6 TZH:TZM';
alter session set nls_timestamp_format='YYYY-MM-DD"T"HH24:MI:SS.FF6';
alter session set nls_date_format='YYYY-MM-DD HH24:MI:SS';

define START_UTC = '&1'
define END_UTC   = '&2'
define ROOT_DIR  = '&3'

variable lsr_rc refcursor
declare
    v_cols      varchar2(32767);
    v_time_col  varchar2(128);
    v_sql       varchar2(32767);
begin
    select listagg('t."' || c.column_name || '"', ',')
               within group (order by wanted.ord)
      into v_cols
      from (
          select column_value as column_name, rownum as ord
          from table(sys.odcivarchar2list(
              'HDR_TRID','CTID','UNIQ911_CID','CALL_DATE','CALL_BEGIN_HR',
              'CALL_BEGIN_TIME_UTC','LOCATE_BEGIN_TIME_UTC',
              'GMLC_VENDOR','REGION','MARKET_CLUSTER','MARKET','COUNTY','STATE',
              'ESRK','GMLC_ESRK','SETUP_ECGI_HEX','INVITE_ECGI_HEX',
              'LOCATE_ECGI_HEX','TAC','TAC_HEX','USID','ENBID','USEID',
              'PSAP_ID','FCC_PSAP_ID','PSAP_NAME','ROUTE_PSAPID',
              'ROUTE_FCC_PSAPID','ROUTE_PSAPNAME','ROUTE_ESZ',
              'PL_RESULT_CODE','POS_METHOD_USED','FAIL_LOCATE','P2BEYOND1',
              'P2_SUCCESS_COUNT','LSR_CALL_CNT','VOLTE_ATTEMPTS','DR_CNT',
              'DR_CALL_CNT','E2VALIDATED','Z_RELAYED2PSAP',
              'MME_VENDOR','MME_POOL_ID','MME_POOL_NAME','MME_NAME',
              'INVITE_CN_MME_ID','HANDSET','NETWORK','USER_TYPE',
              'CELL_TIME_ZONE','TEST_CALL','COMPLETE_CALL'
          ))
      ) wanted
      join all_tab_columns c
        on c.owner = 'E911'
       and c.table_name = 'E911_LTE_LSR_CSR_CD'
       and c.column_name = wanted.column_name;

    select case
             when exists (
                 select 1 from all_tab_columns
                 where owner='E911' and table_name='E911_LTE_LSR_CSR_CD'
                   and column_name='CALL_BEGIN_TIME_UTC'
             ) then 'CALL_BEGIN_TIME_UTC'
             when exists (
                 select 1 from all_tab_columns
                 where owner='E911' and table_name='E911_LTE_LSR_CSR_CD'
                   and column_name='CALL_DATE'
             ) then 'CALL_DATE'
           end
      into v_time_col
      from dual;

    if v_cols is null or v_time_col is null then
        raise_application_error(-20002,
            'Required columns not found in E911.E911_LTE_LSR_CSR_CD');
    end if;

    v_sql := 'select ' || v_cols ||
             ' from e911.e911_lte_lsr_csr_cd t where ';
    if v_time_col = 'CALL_BEGIN_TIME_UTC' then
        v_sql := v_sql ||
            't.call_begin_time_utc >= from_tz(to_timestamp(''' ||
            '&&START_UTC' || ''',''YYYY-MM-DD HH24:MI:SS''),''UTC'')' ||
            ' and t.call_begin_time_utc < from_tz(to_timestamp(''' ||
            '&&END_UTC' || ''',''YYYY-MM-DD HH24:MI:SS''),''UTC'')';
    else
        v_sql := v_sql ||
            't.call_date >= to_date(''' || substr('&&START_UTC',1,10) ||
            ''',''YYYY-MM-DD'') and t.call_date < to_date(''' ||
            substr('&&END_UTC',1,10) || ''',''YYYY-MM-DD'')';
    end if;
    v_sql := v_sql || ' order by t.' || v_time_col;
    open :lsr_rc for v_sql;
end;
/

spool "&&ROOT_DIR\data\lsr_csr_features.csv"
print lsr_rc
spool off
set markup csv off
prompt Wrote &&ROOT_DIR\data\lsr_csr_features.csv

