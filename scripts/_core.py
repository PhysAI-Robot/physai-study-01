"""Make the core repository importable for these scripts.

`research/` is mounted at `<core>/research`, so the core's `src/` (for `physai`), its
root (for `research`) and its `scripts/` (for the shared CLI helpers) go on `sys.path`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE_SCRIPTS = ROOT / "scripts"
for _path in (CORE_SCRIPTS, ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
