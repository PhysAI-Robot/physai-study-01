"""Share the framework's fixtures (tests/conftest.py of the outer repo) with these tests:
they live in the `research/` checkout, outside that conftest's scope.

`env` is the SO-101's; `viperx_env` is the ViperX 300 S's, built from the research manifest
(its randomized sampler) without cameras.
"""

from pathlib import Path

import pytest
from tests.conftest import env  # noqa: F401
from tests.conftest import requires_viperx300s_assets  # noqa: F401

VIPERX_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "configs"
    / "viperx300s_single_cube_randomized_place.yaml"
)


@pytest.fixture
def viperx_env():
    from physai.config import load_manifest
    from physai.config.compat import with_overrides
    from physai.runtime import create_session

    manifest = with_overrides(
        load_manifest(VIPERX_MANIFEST),
        simulator="mujoco",
        policy="idle",
        seed=0,
        max_steps=400,
    )
    session = create_session(manifest, render=False)
    try:
        yield session.runtime.robot
    finally:
        session.close()
