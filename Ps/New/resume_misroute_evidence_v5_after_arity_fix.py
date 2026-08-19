"""Resume the v5 evidence cell after the V5_PROBABLE_CAUSE arity error.

Run this in a NEW notebook cell, in the SAME live kernel that produced the
error.  Keep this file and misroute_evidence_next_cell_v5.py beside the
notebook.  It reuses the already-prepared SQLite/temp tables and therefore
does not repeat geometry labeling or the expensive evidence-base build.
"""

from pathlib import Path


# Correct the SQLite UDF registration in the current live connection.
con.create_function("V5_PROBABLE_CAUSE", 13, v5_probable_cause)

# Make the recovery idempotent if a prior retry partly created the table.
con.execute("DROP TABLE IF EXISTS temp.misroute_investigation_v5")

# Execute only the inexpensive final scoring/export portion of the patched
# add-on, starting after the costly evidence preparation has completed.
source_path = Path("misroute_evidence_next_cell_v5.py")
if not source_path.exists():
    raise FileNotFoundError(
        "Put misroute_evidence_next_cell_v5.py beside this notebook, then retry."
    )

source = source_path.read_text(encoding="utf-8")
resume_marker = 'fallback_delta = f"('
resume_at = source.find(resume_marker)
if resume_at < 0:
    raise RuntimeError("Could not locate the v5 final-scoring resume marker.")

print("Resuming v5 at final scoring/export; expensive preparation is not rerun...")
exec(
    compile(source[resume_at:], f"{source_path}#resume", "exec"),
    globals(),
    globals(),
)
