"""Run another script with the SO-101's two contact pads drawn by MuJoCo (study 1).

    python research/scripts/with_pad_visible.py research/scripts/collect_demos.py --manifest ... --dataset ...
    python research/scripts/with_pad_visible.py scripts/eval_policy.py --manifest ... --checkpoint ...

The pads (`pad_static`, `pad_moving`) are collision-only boxes of MuJoCo group 3, which MuJoCo does not draw, so no image of
the study shows them. Here they are moved to group 2 and drawn opaque, as Newton drew them before the renderer fix
(`src/physai/sim/newton/render.py`). Only the picture changes; physics are the same.
"""

from __future__ import annotations

import runpy
import sys

import _core  # noqa: F401
import mujoco

import physai.runtime

PADS = ("pad_static", "pad_moving")
PAD_RGBA = (0.95, 0.6, 0.1, 1.0)


def show_pads(model: mujoco.MjModel) -> int:
    shown = 0
    for name in PADS:
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom >= 0:
            model.geom_group[geom] = 2
            model.geom_rgba[geom] = PAD_RGBA
            shown += 1
    return shown


_create_session = physai.runtime.create_session


def create_session(*args, **kwargs):
    session = _create_session(*args, **kwargs)
    show_pads(session.runtime.robot.model)
    return session


physai.runtime.create_session = create_session

if __name__ == "__main__":
    target, *rest = sys.argv[1:]
    sys.argv = [target, *rest]
    runpy.run_path(target, run_name="__main__")
