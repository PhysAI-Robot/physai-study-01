"""Run scripts/eval_policy.py with Newton drawing the scene the way MuJoCo does (the "hidden" variant).

    python research/experiments/newton_renderer_variant.py --variant hidden -- <eval_policy.py arguments>

hidden: the geoms of group 3 and above (MuJoCo does not draw them) are dropped from the scene Newton draws, and a
textured material is replaced by its texture's mean colour. control: the renderer as it is.
"""

import os, runpy, sys
from contextlib import contextmanager
from pathlib import Path

variant = sys.argv[sys.argv.index("--variant") + 1]
rest = sys.argv[sys.argv.index("--") + 1 :]
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]

if variant == "old":
    # the renderer before the fix of commit e95354a: the scene exactly as exported
    from physai.sim.newton import render, core

    render.drawable_scene_mjcf = lambda spec, model, robot_dir: core.scene_mjcf(
        spec, model, robot_dir
    )
elif variant in ("hidden", "pads", "meshes"):
    import mujoco, numpy as np
    from physai.sim.newton import render, core

    @contextmanager
    def drawable_scene_mjcf(spec, model, robot_dir, geom_styles=None):
        with core.scene_mjcf(spec, model, robot_dir, geom_styles) as path:
            drawable = mujoco.MjSpec.from_file(path)
        rgb = mujoco.mjtTextureRole.mjTEXROLE_RGB
        for material in drawable.materials:
            name = material.textures[rgb]
            tex = (
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, name)
                if name
                else -1
            )
            if tex < 0 or model.tex_type[tex] != mujoco.mjtTexture.mjTEXTURE_2D:
                continue
            size = model.tex_width[tex] * model.tex_height[tex]
            ch = model.tex_nchannel[tex]
            pixels = model.tex_data[
                model.tex_adr[tex] : model.tex_adr[tex] + size * ch
            ].reshape(size, ch)
            material.rgba = np.append(
                np.asarray(material.rgba)[:3] * pixels[:, :3].mean(axis=0) / 255.0,
                material.rgba[3],
            )
            textures = list(material.textures)
            textures[rgb] = ""
            material.textures = textures
        named = lambda g: g.name.startswith("pad_")
        keep = {
            "hidden": lambda g: False,
            "pads": lambda g: not named(g),
            "meshes": named,
        }[variant]
        for geom in [g for g in drawable.geoms if g.group >= 3 and not keep(g)]:
            drawable.delete(geom)
        drawable.compile()
        out = Path(robot_dir) / f"_newton_hidden_{os.getpid()}.xml"
        out.write_text(drawable.to_xml(), encoding="utf-8")
        try:
            yield str(out)
        finally:
            out.unlink(missing_ok=True)

    render.drawable_scene_mjcf = (
        drawable_scene_mjcf  # the function NewtonRenderer calls since e95354a
    )

sys.argv = ["eval_policy.py", *rest]
runpy.run_path(str(ROOT / "scripts" / "eval_policy.py"), run_name="__main__")
