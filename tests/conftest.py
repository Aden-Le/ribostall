"""Make `scripts/` importable so the test files can `import dylan_table_checker`
and `import cross_tp_summary_checker` without packaging the script tree.
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
