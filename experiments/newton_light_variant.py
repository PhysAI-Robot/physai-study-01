"""Eval with the fixed Newton renderer and the key light mirrored across the robot's symmetry plane (y -> -y). Nothing else changes."""

import runpy, sys
from pathlib import Path

rest = sys.argv[sys.argv.index("--") + 1 :]
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
from physai.sim.newton import render as R

R.KEY_LIGHT_DIRECTION = (
    R.KEY_LIGHT_DIRECTION[0],
    -R.KEY_LIGHT_DIRECTION[1],
    R.KEY_LIGHT_DIRECTION[2],
)
sys.argv = ["eval_policy.py", *rest]
runpy.run_path(str(ROOT / "scripts" / "eval_policy.py"), run_name="__main__")
