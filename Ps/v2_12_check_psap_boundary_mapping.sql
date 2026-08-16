-- ============================================================================
-- v2_12_check_psap_boundary_mapping.sql
--
-- Purpose
--   Diagnose which PSAP identifier from E911_PSAPSIM_CALL_DETAILS_LTE maps to
--   which authoritative boundary identifier before running Python geometry.
--
-- Run from SQL*Plus / SQLcl:
--   sqlplus -L "YOUR_USER@YOUR_TNS_ALIAS" @v2_12_check_psap_boundary_mapping.sql
--
-- Run from Oracle SQL Developer:
--   Open this file and use Run Script (F5), not Run Statement (Ctrl+Enter).
--
-- Output directory must already exist:
--   C:\temp\psap_route_integrity_v2\audit
-- ============================================================================

WHENEVER SQLERROR EXIT SQL.SQLCODE
WHENEVER OSERROR EXIT FAILURE

DEFINE ROOT_DIR    = C:\temp\psap_route_integrity_v2
DEFINE SAMPLE_ROWS = 10000

SET ECHO OFF
SET VERIFY OFF
SET FEEDBACK OFF
SET HEADING ON
SET PAGESIZE 50000
SET LINESIZE 32767
SET TRIMSPOOL ON
SET TAB OFF
SET TERMOUT OFF
SET MARKUP CSV ON DELIMITER , QUOTE ON

-- --------------------------------------------------------------------------
-- 1. Source sizes and identifier coverage
-- --------------------------------------------------------------------------
SPOOL "&&ROOT_DIR.\audit\psap_boundary_source_counts.csv"

SELECT 'E911_PSAPSIM_CALL_DETAILS_LTE_SAMPLE' AS SOURCE_NAME,
       COUNT(*) AS ROWS_CHECKED,
       COUNT(PSAP_ID) AS NONNULL_PSAP_ID,
       COUNT(FCC_PSAP_ID) AS NONNULL_FCC_PSAP_ID,
       COUNT(PSAP_NAME) AS NONNULL_PSAP_NAME
FROM (
    SELECT PSAP_ID, FCC_PSAP_ID, PSAP_NAME
    FROM E911.E911_PSAPSIM_CALL_DETAILS_LTE
    WHERE PSAP_ID IS NOT NULL
       OR FCC_PSAP_ID IS NOT NULL
       OR PSAP_NAME IS NOT NULL
    FETCH FIRST &&SAMPLE_ROWS ROWS ONLY
);

SELECT 'E911_PSAP_BOUNDARIES' AS SOURCE_NAME,
       COUNT(*) AS SOURCE_ROWS,
       COUNT(DISTINCT NENA_ID) AS DISTINCT_NENA_ID,
       COUNT(DISTINCT FCC_PSAP_ID) AS DISTINCT_FCC_PSAP_ID
FROM E911.E911_PSAP_BOUNDARIES;

SELECT 'E911_PSAP_BOUNDARIES_FCC' AS SOURCE_NAME,
       COUNT(*) AS SOURCE_ROWS,
       COUNT(DISTINCT FCC_PSAP_ID) AS DISTINCT_FCC_PSAP_ID,
       COUNT(DISTINCT JSON_VALUE(
           PROPERTIES,
           '$.properties.FCC_ID' RETURNING VARCHAR2(100) NULL ON ERROR
       )) AS DISTINCT_JSON_FCC_ID,
       COUNT(DISTINCT JSON_VALUE(
           PROPERTIES,
           '$.properties.PSAP_NAME' RETURNING VARCHAR2(500) NULL ON ERROR
       )) AS DISTINCT_JSON_PSAP_NAME
FROM E911.E911_PSAP_BOUNDARIES_FCC;

SPOOL OFF

-- --------------------------------------------------------------------------
-- 2. Mapping coverage summary
--
-- Comparisons tested:
--   A. call FCC_PSAP_ID -> E911_PSAP_BOUNDARIES.FCC_PSAP_ID
--   B. call PSAP_ID     -> E911_PSAP_BOUNDARIES.NENA_ID
--   C. call FCC_PSAP_ID -> E911_PSAP_BOUNDARIES_FCC.FCC_PSAP_ID
--   D. call FCC_PSAP_ID -> FCC_ID inside BOUNDARIES_FCC.PROPERTIES
--   E. call PSAP_NAME   -> PSAP_NAME inside BOUNDARIES_FCC.PROPERTIES
-- --------------------------------------------------------------------------
SPOOL "&&ROOT_DIR.\audit\psap_id_mapping_summary.csv"

WITH
call_sample AS (
    SELECT
        TRIM(CAST(PSAP_ID AS VARCHAR2(100))) AS PSAP_ID_RAW,
        TRIM(CAST(FCC_PSAP_ID AS VARCHAR2(100))) AS FCC_PSAP_ID_RAW,
        TRIM(CAST(PSAP_NAME AS VARCHAR2(500))) AS PSAP_NAME_RAW
    FROM E911.E911_PSAPSIM_CALL_DETAILS_LTE
    WHERE PSAP_ID IS NOT NULL
       OR FCC_PSAP_ID IS NOT NULL
       OR PSAP_NAME IS NOT NULL
    FETCH FIRST &&SAMPLE_ROWS ROWS ONLY
),
calls AS (
    SELECT
        REGEXP_REPLACE(UPPER(PSAP_ID_RAW), '[.]0$', '') AS PSAP_ID_NORM,
        REGEXP_REPLACE(UPPER(FCC_PSAP_ID_RAW), '[.]0$', '') AS FCC_PSAP_ID_NORM,
        REGEXP_REPLACE(UPPER(PSAP_NAME_RAW), '[^A-Z0-9]', '') AS PSAP_NAME_NORM
    FROM call_sample
),
legacy_fcc AS (
    SELECT DISTINCT REGEXP_REPLACE(
        UPPER(TRIM(CAST(FCC_PSAP_ID AS VARCHAR2(100)))), '[.]0$', ''
    ) AS ID_VALUE
    FROM E911.E911_PSAP_BOUNDARIES
    WHERE FCC_PSAP_ID IS NOT NULL
),
legacy_nena AS (
    SELECT DISTINCT REGEXP_REPLACE(
        UPPER(TRIM(CAST(NENA_ID AS VARCHAR2(100)))), '[.]0$', ''
    ) AS ID_VALUE
    FROM E911.E911_PSAP_BOUNDARIES
    WHERE NENA_ID IS NOT NULL
),
current_fcc_column AS (
    SELECT DISTINCT REGEXP_REPLACE(
        UPPER(TRIM(CAST(FCC_PSAP_ID AS VARCHAR2(100)))), '[.]0$', ''
    ) AS ID_VALUE
    FROM E911.E911_PSAP_BOUNDARIES_FCC
    WHERE FCC_PSAP_ID IS NOT NULL
),
current_fcc_json AS (
    SELECT DISTINCT REGEXP_REPLACE(
        UPPER(TRIM(JSON_VALUE(
            PROPERTIES,
            '$.properties.FCC_ID' RETURNING VARCHAR2(100) NULL ON ERROR
        ))), '[.]0$', ''
    ) AS ID_VALUE
    FROM E911.E911_PSAP_BOUNDARIES_FCC
),
current_name_json AS (
    SELECT DISTINCT REGEXP_REPLACE(
        UPPER(TRIM(JSON_VALUE(
            PROPERTIES,
            '$.properties.PSAP_NAME' RETURNING VARCHAR2(500) NULL ON ERROR
        ))), '[^A-Z0-9]', ''
    ) AS ID_VALUE
    FROM E911.E911_PSAP_BOUNDARIES_FCC
),
comparisons AS (
    SELECT
        'CALL.FCC_PSAP_ID -> E911_PSAP_BOUNDARIES.FCC_PSAP_ID' AS COMPARISON,
        COUNT(*) AS SAMPLE_ROWS,
        COUNT(FCC_PSAP_ID_NORM) AS NONNULL_CALL_VALUES,
        COUNT(DISTINCT FCC_PSAP_ID_NORM) AS DISTINCT_CALL_VALUES,
        SUM(CASE WHEN EXISTS (
            SELECT 1 FROM legacy_fcc b WHERE b.ID_VALUE = c.FCC_PSAP_ID_NORM
        ) THEN 1 ELSE 0 END) AS MATCHED_ROWS
    FROM calls c

    UNION ALL

    SELECT
        'CALL.PSAP_ID -> E911_PSAP_BOUNDARIES.NENA_ID',
        COUNT(*),
        COUNT(PSAP_ID_NORM),
        COUNT(DISTINCT PSAP_ID_NORM),
        SUM(CASE WHEN EXISTS (
            SELECT 1 FROM legacy_nena b WHERE b.ID_VALUE = c.PSAP_ID_NORM
        ) THEN 1 ELSE 0 END)
    FROM calls c

    UNION ALL

    SELECT
        'CALL.FCC_PSAP_ID -> E911_PSAP_BOUNDARIES_FCC.FCC_PSAP_ID',
        COUNT(*),
        COUNT(FCC_PSAP_ID_NORM),
        COUNT(DISTINCT FCC_PSAP_ID_NORM),
        SUM(CASE WHEN EXISTS (
            SELECT 1 FROM current_fcc_column b WHERE b.ID_VALUE = c.FCC_PSAP_ID_NORM
        ) THEN 1 ELSE 0 END)
    FROM calls c

    UNION ALL

    SELECT
        'CALL.FCC_PSAP_ID -> BOUNDARIES_FCC.PROPERTIES.FCC_ID',
        COUNT(*),
        COUNT(FCC_PSAP_ID_NORM),
        COUNT(DISTINCT FCC_PSAP_ID_NORM),
        SUM(CASE WHEN EXISTS (
            SELECT 1 FROM current_fcc_json b WHERE b.ID_VALUE = c.FCC_PSAP_ID_NORM
        ) THEN 1 ELSE 0 END)
    FROM calls c

    UNION ALL

    SELECT
        'CALL.PSAP_NAME -> BOUNDARIES_FCC.PROPERTIES.PSAP_NAME',
        COUNT(*),
        COUNT(PSAP_NAME_NORM),
        COUNT(DISTINCT PSAP_NAME_NORM),
        SUM(CASE WHEN EXISTS (
            SELECT 1 FROM current_name_json b WHERE b.ID_VALUE = c.PSAP_NAME_NORM
        ) THEN 1 ELSE 0 END)
    FROM calls c
)
SELECT
    COMPARISON,
    SAMPLE_ROWS,
    NONNULL_CALL_VALUES,
    DISTINCT_CALL_VALUES,
    MATCHED_ROWS,
    NONNULL_CALL_VALUES - MATCHED_ROWS AS UNMATCHED_ROWS,
    ROUND(100 * MATCHED_ROWS / NULLIF(NONNULL_CALL_VALUES, 0), 2) AS MATCH_RATE_PCT
FROM comparisons
ORDER BY MATCH_RATE_PCT DESC NULLS LAST;

SPOOL OFF

-- --------------------------------------------------------------------------
-- 3. FCC_PSAP_ID values with flags for every FCC boundary source
-- --------------------------------------------------------------------------
SPOOL "&&ROOT_DIR.\audit\fcc_psap_id_value_profile.csv"

WITH
calls AS (
    SELECT
        TRIM(CAST(FCC_PSAP_ID AS VARCHAR2(100))) AS RAW_VALUE,
        REGEXP_REPLACE(
            UPPER(TRIM(CAST(FCC_PSAP_ID AS VARCHAR2(100)))), '[.]0$', ''
        ) AS NORM_VALUE
    FROM E911.E911_PSAPSIM_CALL_DETAILS_LTE
    WHERE FCC_PSAP_ID IS NOT NULL
    FETCH FIRST &&SAMPLE_ROWS ROWS ONLY
),
value_counts AS (
    SELECT NORM_VALUE, MIN(RAW_VALUE) AS EXAMPLE_RAW_VALUE, COUNT(*) AS CALL_ROWS
    FROM calls
    GROUP BY NORM_VALUE
),
legacy_fcc AS (
    SELECT DISTINCT REGEXP_REPLACE(
        UPPER(TRIM(CAST(FCC_PSAP_ID AS VARCHAR2(100)))), '[.]0$', ''
    ) AS ID_VALUE
    FROM E911.E911_PSAP_BOUNDARIES
    WHERE FCC_PSAP_ID IS NOT NULL
),
current_fcc_column AS (
    SELECT DISTINCT REGEXP_REPLACE(
        UPPER(TRIM(CAST(FCC_PSAP_ID AS VARCHAR2(100)))), '[.]0$', ''
    ) AS ID_VALUE
    FROM E911.E911_PSAP_BOUNDARIES_FCC
    WHERE FCC_PSAP_ID IS NOT NULL
),
current_fcc_json AS (
    SELECT DISTINCT REGEXP_REPLACE(
        UPPER(TRIM(JSON_VALUE(
            PROPERTIES,
            '$.properties.FCC_ID' RETURNING VARCHAR2(100) NULL ON ERROR
        ))), '[.]0$', ''
    ) AS ID_VALUE
    FROM E911.E911_PSAP_BOUNDARIES_FCC
)
SELECT
    v.EXAMPLE_RAW_VALUE AS CALL_FCC_PSAP_ID_RAW,
    v.NORM_VALUE AS CALL_FCC_PSAP_ID_NORM,
    v.CALL_ROWS,
    CASE WHEN EXISTS (
        SELECT 1 FROM legacy_fcc b WHERE b.ID_VALUE = v.NORM_VALUE
    ) THEN 'Y' ELSE 'N' END AS IN_E911_PSAP_BOUNDARIES,
    CASE WHEN EXISTS (
        SELECT 1 FROM current_fcc_column b WHERE b.ID_VALUE = v.NORM_VALUE
    ) THEN 'Y' ELSE 'N' END AS IN_E911_PSAP_BOUNDARIES_FCC_COLUMN,
    CASE WHEN EXISTS (
        SELECT 1 FROM current_fcc_json b WHERE b.ID_VALUE = v.NORM_VALUE
    ) THEN 'Y' ELSE 'N' END AS IN_E911_PSAP_BOUNDARIES_FCC_JSON
FROM value_counts v
ORDER BY v.CALL_ROWS DESC, v.NORM_VALUE
FETCH FIRST 200 ROWS ONLY;

SPOOL OFF

-- --------------------------------------------------------------------------
-- 4. PSAP_ID -> NENA_ID value profile
-- --------------------------------------------------------------------------
SPOOL "&&ROOT_DIR.\audit\psap_id_to_nena_value_profile.csv"

WITH
calls AS (
    SELECT
        TRIM(CAST(PSAP_ID AS VARCHAR2(100))) AS RAW_VALUE,
        REGEXP_REPLACE(
            UPPER(TRIM(CAST(PSAP_ID AS VARCHAR2(100)))), '[.]0$', ''
        ) AS NORM_VALUE
    FROM E911.E911_PSAPSIM_CALL_DETAILS_LTE
    WHERE PSAP_ID IS NOT NULL
    FETCH FIRST &&SAMPLE_ROWS ROWS ONLY
),
value_counts AS (
    SELECT NORM_VALUE, MIN(RAW_VALUE) AS EXAMPLE_RAW_VALUE, COUNT(*) AS CALL_ROWS
    FROM calls
    GROUP BY NORM_VALUE
),
legacy_nena AS (
    SELECT DISTINCT REGEXP_REPLACE(
        UPPER(TRIM(CAST(NENA_ID AS VARCHAR2(100)))), '[.]0$', ''
    ) AS ID_VALUE
    FROM E911.E911_PSAP_BOUNDARIES
    WHERE NENA_ID IS NOT NULL
)
SELECT
    v.EXAMPLE_RAW_VALUE AS CALL_PSAP_ID_RAW,
    v.NORM_VALUE AS CALL_PSAP_ID_NORM,
    v.CALL_ROWS,
    CASE WHEN EXISTS (
        SELECT 1 FROM legacy_nena b WHERE b.ID_VALUE = v.NORM_VALUE
    ) THEN 'Y' ELSE 'N' END AS IN_E911_PSAP_BOUNDARIES_NENA_ID
FROM value_counts v
ORDER BY v.CALL_ROWS DESC, v.NORM_VALUE
FETCH FIRST 200 ROWS ONLY;

SPOOL OFF

SET MARKUP CSV OFF
SET TERMOUT ON
SET FEEDBACK ON

PROMPT.
PROMPT PSAP boundary identifier audit finished.
PROMPT Review:
PROMPT   &&ROOT_DIR.\audit\psap_id_mapping_summary.csv
PROMPT   &&ROOT_DIR.\audit\fcc_psap_id_value_profile.csv
PROMPT   &&ROOT_DIR.\audit\psap_id_to_nena_value_profile.csv
PROMPT.

EXIT SUCCESS
