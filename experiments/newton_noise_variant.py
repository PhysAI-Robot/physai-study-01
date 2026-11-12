"""Eval with the fixed Newton renderer plus a tiny deterministic noise (+-1 grey level) on both cameras: a control for 'any small change flips these seeds'."""

import runpy, sys
from pathlib import Path

rest = sys.argv[sys.argv.index("--") + 1 :]
seed = int(rest[rest.index("--seed") + 1])
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
import numpy as np
from physai.sim.newton import render as R

_render = R.NewtonRenderer.render
calls = {"n": 0}


def render(self, name):
    img = _render(self, name).astype(np.int16)
    rng = np.random.default_rng(seed * 1000 + calls["n"])
    calls["n"] += 1
    return np.clip(img + rng.integers(-1, 2, img.shape), 0, 255).astype(np.uint8)


R.NewtonRenderer.render = render
sys.argv = ["eval_policy.py", *rest]
runpy.run_path(str(ROOT / "scripts" / "eval_policy.py"), run_name="__main__")
