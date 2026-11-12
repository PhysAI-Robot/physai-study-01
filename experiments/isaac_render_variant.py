"""Run a script on Isaac Sim with one render or light setting moved toward MuJoCo's (the rendering settings of Section "attribution"). Core is not changed.

    python research/experiments/isaac_render_variant.py VARIANT -- scripts/eval_policy.py <arguments>

VARIANT, one setting each unless it says "all": gi_off, ao_off, reflections_off (RTX effects MuJoCo does not have), hard_shadows (the key light's
angle 0, MuJoCo's shadows have no penumbra), spot_light (MuJoCo's second, off-axis light: the key's intensity split 70/30 and the 30% sent along
(-0.4, -0.4, -1) from the front right), sky_dome (the dome light takes the studio's sky colour), floor_raw (the floor texture is read as raw values, not sRGB, so with the linear output its tiles keep MuJoCo's numbers), all, none.
"""

import runpy, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "scripts")]
import os

if (
    "ISAAC_VARIANT" in os.environ
):  # as the eval script of eval_policy_sharded.py: the arguments are eval_policy.py's
    variant, script, args = (
        os.environ["ISAAC_VARIANT"],
        "scripts/eval_policy.py",
        sys.argv[1:],
    )
else:
    variant = sys.argv[1]
    script, *args = sys.argv[sys.argv.index("--") + 1 :]

RTX = {
    "gi_off": {"/rtx/indirectDiffuse/enabled": False},
    "ao_off": {"/rtx/ambientOcclusion/enabled": False},
    "reflections_off": {"/rtx/reflections/enabled": False},
}
LIGHTS = {"hard_shadows", "spot_light", "sky_dome", "floor_raw"}
CUMULATIVE = {
    "floor_raw",
    "hard_shadows",
    "gi_off",
    "ao_off",
    "reflections_off",
}  # the settings that moved the picture toward MuJoCo (frames test)
steps = (
    CUMULATIVE
    if variant == "cumulative"
    else {
        "gi_off",
        "ao_off",
        "reflections_off",
        "hard_shadows",
        "spot_light",
        "sky_dome",
        "floor_raw",
    }
    if variant == "all"
    else {variant} - {"none"}
)
assert steps <= set(RTX) | LIGHTS, variant
print("isaac_variant:", variant, sorted(steps), file=sys.stderr)

from physai.sim.isaac import core, scene
import physai.robots.so101.isaac_env as so101_env
import physai.robots.viperx300s.isaac_env as viperx_env
from physai.sim.studio import STUDIO_SKY_RGB, ISAAC_KEY_INTENSITY

original_output = core.configure_render_output


def configure_render_output():
    original_output()
    import carb.settings

    settings = carb.settings.get_settings()
    for step in steps & set(RTX):
        for key, value in RTX[step].items():
            settings.set(key, value)


core.configure_render_output = configure_render_output

original_lighting = scene.add_studio_lighting


def add_studio_lighting(stage, **kwargs):
    original_lighting(stage, **kwargs)
    from pxr import Gf, UsdGeom, UsdLux

    scale = kwargs.get("scale", 1.0)
    key = UsdLux.DistantLight(stage.GetPrimAtPath("/World_lights/key"))
    if "hard_shadows" in steps:
        key.GetAngleAttr().Set(0.0)
    if "spot_light" in steps:
        key.GetIntensityAttr().Set(0.7 * ISAAC_KEY_INTENSITY * scale)
        spot = UsdLux.DistantLight.Define(stage, "/World_lights/spot")
        spot.CreateIntensityAttr(0.3 * ISAAC_KEY_INTENSITY * scale)
        spot.CreateAngleAttr(0.0 if "hard_shadows" in steps else 1.0)
        direction = Gf.Vec3d(-0.4, -0.4, -1.0).GetNormalized()
        rotation = Gf.Rotation(
            Gf.Vec3d(0, 0, -1), direction
        )  # a distant light shines along its -Z
        UsdGeom.Xformable(spot.GetPrim()).AddOrientOp().Set(
            Gf.Quatf(rotation.GetQuat())
        )
    if "sky_dome" in steps:
        UsdLux.DomeLight(stage.GetPrimAtPath("/World_lights/dome")).CreateColorAttr(
            Gf.Vec3f(*STUDIO_SKY_RGB)
        )


scene.add_studio_lighting = add_studio_lighting
so101_env.add_studio_lighting = add_studio_lighting
viperx_env.add_studio_lighting = add_studio_lighting

original_ground = scene.add_ground_plane


def add_ground_plane(stage, **kwargs):
    original_ground(stage, **kwargs)
    if "floor_raw" in steps:
        from pxr import Sdf, UsdShade

        texture = UsdShade.Shader.Get(stage, "/World_ground_material/checker")
        texture.GetInput("sourceColorSpace").Set("raw")


scene.add_ground_plane = add_ground_plane
so101_env.add_ground_plane = add_ground_plane
viperx_env.add_ground_plane = add_ground_plane

sys.argv = [script, *args]
runpy.run_path(str(ROOT / script), run_name="__main__")
