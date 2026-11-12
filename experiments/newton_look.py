"""Eval with the fixed Newton renderer drawing a named look (floor tile, floor colours, sky, key light).

    python research/experiments/newton_look.py LOOK -- <eval_policy.py arguments>

LOOK is one of the looks outside the training ranges (floor_fine, floor_bw, sky_dark, light_low, all_unseen) or the held-out look of the black-and-white-tile runs (heldout_look,
fixed before those runs were scored). The look is set on the constants of `physai.sim.newton.render`.
"""

import runpy, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
from physai.sim.newton import render as R

LOOKS = {
    "floor_fine": {"TILE": 1 / 24},
    "floor_bw": {"FLOOR_LIGHT": (1.0, 1.0, 1.0), "FLOOR_DARK": (0.1, 0.1, 0.1)},
    "sky_dark": {"SKY": (45, 55, 95)},
    "light_low": {"KEY_LIGHT_DIRECTION": (0.9, -0.2, -0.45)},
    "heldout_look": {  # wood-coloured two-brown tiles of 1/4 m, dark blue sky, key light mirrored
        "TILE": 1 / 4,
        "FLOOR_LIGHT": (0.62, 0.42, 0.25),
        "FLOOR_DARK": (0.38, 0.25, 0.14),
        "SKY": (20, 40, 110),
        "KEY_LIGHT_DIRECTION": (0.35, -0.45, -1.0),
    },
}
LOOKS["all_unseen"] = {
    **LOOKS["floor_fine"],
    **LOOKS["floor_bw"],
    **LOOKS["sky_dark"],
    **LOOKS["light_low"],
}

look = sys.argv[1]
for name, value in LOOKS[look].items():
    assert hasattr(R, name), name
    setattr(R, name, value)
sys.argv = ["eval_policy.py", *sys.argv[sys.argv.index("--") + 1 :]]
runpy.run_path(str(ROOT / "scripts" / "eval_policy.py"), run_name="__main__")
