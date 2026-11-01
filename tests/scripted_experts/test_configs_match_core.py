"""The research configs and experts keep their own copy of numbers the core also states.

The core ships a manifest for each randomized task and a `GraspProfile` for each arm; the
research config adds `plugins:` and the expert has its own heights. Nothing makes the copies
agree, so a test does.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]

SAMPLER = (
    "randomize_cube",
    "cube_x_range",
    "cube_y_range",
    "randomize_target",
    "target_x_range",
    "target_y_range",
    "spawn_radius_range",
    "min_cube_target_distance",
)


def _load(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("research", "core"),
    [
        (
            "research/configs/so101_randomized_pick_place.yaml",
            "configs/manifests/so101_single_cube_randomized_place.yaml",
        ),
        (
            "research/configs/viperx300s_single_cube_randomized_place.yaml",
            "configs/manifests/viperx300s_single_cube_randomized_place.yaml",
        ),
    ],
)
def test_the_research_sampler_is_the_core_manifests(research, core):
    ours, theirs = _load(research), _load(core)
    assert ours["scene"] == theirs["scene"]
    ours_config, theirs_config = (
        ours["robots"][0]["config"],
        theirs["robots"][0]["config"],
    )
    for key in SAMPLER:
        assert ours_config.get(key) == theirs_config.get(key), key


def test_the_so101_expert_uses_the_cores_heights():
    from physai.robots.so101.mujoco_env import SO101_GRASP
    from research.scripted_experts.so101_pick_place_expert import ExpertConfig

    cfg = ExpertConfig()
    assert cfg.hover_height == SO101_GRASP.hover
    assert cfg.grasp_height == SO101_GRASP.grasp_dz
    assert cfg.gripper_open == SO101_GRASP.grip_open


def test_the_viperx_expert_uses_the_cores_heights():
    from physai.robots.viperx300s.mujoco_env import VIPERX300S_GRASP
    from research.scripted_experts.viperx300s_pick_place_expert import (
        ViperX300SExpertConfig,
    )

    cfg = ViperX300SExpertConfig()
    assert cfg.hover_height == VIPERX300S_GRASP.hover
    assert cfg.grasp_height == VIPERX300S_GRASP.grasp_dz
    assert cfg.lift_height == VIPERX300S_GRASP.carry
    assert cfg.gripper_open == VIPERX300S_GRASP.grip_open
    assert cfg.gripper_grip == VIPERX300S_GRASP.grip_hold
