import pytest
from tests.conftest import requires_viperx300s_assets

pytestmark = pytest.mark.assets


def run(env, seed, stop=None):
    """Drive the ViperX expert until it is done (or `stop(policy)` says so); yield each step."""
    from research.scripted_experts.viperx300s_pick_place_expert import (
        ViperX300SPickPlaceExpert,
    )

    obs = env.reset(seed=seed)
    policy = ViperX300SPickPlaceExpert(env.kin, env)
    policy.reset(obs)
    for _ in range(env.cfg.max_steps):
        obs, _, _, _, info = env.step(policy.act(obs))
        yield policy, info
        if policy.done or (stop and stop(policy)):
            break


@requires_viperx300s_assets
def test_expert_state_machine_reaches_the_end(viperx_env):
    from research.scripted_experts.so101_pick_place_expert import Phase

    seen = {policy.phase for policy, _ in run(viperx_env, seed=0)}
    assert Phase.CLOSE in seen and Phase.SQUEEZE in seen and Phase.LIFT in seen


@requires_viperx300s_assets
def test_expert_actually_closes_on_the_cube(viperx_env):
    """The grasp must be a real contact, not the fingers shutting on empty air."""
    import mujoco

    from research.scripted_experts.so101_pick_place_expert import Phase

    env = viperx_env
    pads = {
        geom
        for geom in range(env.model.ngeom)
        if (
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        ).startswith(("left_finger_pad_", "right_finger_pad_"))
    }
    pad_contact = False
    for policy, _ in run(env, seed=0):
        if policy.phase in (Phase.SQUEEZE, Phase.LIFT):
            for c in range(env.data.ncon):
                contact = env.data.contact[c]
                if env.cube_geom_id in (contact.geom1, contact.geom2) and pads & {
                    contact.geom1,
                    contact.geom2,
                }:
                    pad_contact = True
    assert pad_contact, "expert never made finger-pad-to-cube contact"


@requires_viperx300s_assets
@pytest.mark.parametrize("seed", [900, 901, 902])
def test_expert_places_the_cube_on_the_target(viperx_env, seed):
    """The gate of the ViperX task is the expert at 100%: a few validation seeds, end to end."""
    info = {}
    for _, info in run(viperx_env, seed):
        if info.get("success"):
            break
    assert info.get("success"), info.get("dist_cube_target")
