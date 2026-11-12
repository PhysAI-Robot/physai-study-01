"""Eval where only the named cameras are drawn by the renderer as it was before the fix (with the contact pads), the rest by the fixed one.

python newton_camera_variant.py --old-cams front|wrist|both|none -- <eval_policy.py arguments>
"""

import runpy, sys
from pathlib import Path

old_cams = {
    "front": ("front",),
    "wrist": ("wrist",),
    "both": ("front", "wrist"),
    "none": (),
}[sys.argv[sys.argv.index("--old-cams") + 1]]
rest = sys.argv[sys.argv.index("--") + 1 :]
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
from physai.sim.newton import core, render as R

Renderer = R.NewtonRenderer
_init, _render = Renderer.__init__, Renderer.render


def init(self, *args, **kwargs):
    self._ctor = (args, kwargs)
    self._twin = None
    _init(self, *args, **kwargs)


def render(self, name):
    if name not in old_cams:
        return _render(self, name)
    if self._twin is None:
        fixed = R.drawable_scene_mjcf
        R.drawable_scene_mjcf = lambda spec, model, robot_dir: core.scene_mjcf(
            spec, model, robot_dir
        )  # the scene as exported, with the pads
        try:
            self._twin = Renderer(*self._ctor[0], **self._ctor[1])
        finally:
            R.drawable_scene_mjcf = fixed
    return _render(self._twin, name)


Renderer.__init__, Renderer.render = init, render
sys.argv = ["eval_policy.py", *rest]
runpy.run_path(str(ROOT / "scripts" / "eval_policy.py"), run_name="__main__")
