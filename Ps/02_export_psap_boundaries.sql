-- Dataset 2: PSAP jurisdiction polygons.
-- PROPERTIES can exceed the SQL*Plus line limit, so it is exported in ordered
-- 30,000-character chunks. The notebook reassembles each boundary exactly.
-- Direct run:
--   sqlplus YOUR_USER@YOUR_TNS_ALIAS @02_export_psap_boundaries.sql ^
--     "C:\temp\psap_route_integrity_v1"

whenever sqlerror exit sql.sqlcode
set verify off feedback off echo off heading on pagesize 50000 linesize 32767
set trimspool on tab off termout on long 10000000 longchunksize 32767
set markup csv on delimiter , quote on

define ROOT_DIR = '&1'

spool "&&ROOT_DIR\data\psap_boundaries_chunks.csv"
with boundaries as (
    select
        nena_id,
        fcc_psap_id,
        to_clob(properties) as properties_clob
    from e911.e911_psap_boundaries
    where properties is not null
), chunks as (
    select
        b.nena_id,
        b.fcc_psap_id,
        x.chunk_no,
        dbms_lob.substr(
            b.properties_clob,
            30000,
            ((x.chunk_no - 1) * 30000) + 1
        ) as properties_chunk
    from boundaries b
    cross apply (
        select level as chunk_no
        from dual
        connect by level <= ceil(dbms_lob.getlength(b.properties_clob) / 30000)
    ) x
)
select nena_id, fcc_psap_id, chunk_no, properties_chunk
from chunks
order by fcc_psap_id, nena_id, chunk_no;
spool off
set markup csv off
prompt Wrote &&ROOT_DIR\data\psap_boundaries_chunks.csv

