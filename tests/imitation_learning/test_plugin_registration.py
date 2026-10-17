"""Importing the research plugin registers every research policy."""

from tests.conftest import requires_assets


def test_the_research_plugin_registers_its_policies():
    import research.plugin  # noqa: F401
    from physai.policy import available_policies

    assert {"scripted", "visual_servo", "lerobot"} <= set(available_policies("so101"))


@requires_assets
def test_research_manifests_load_the_plugin_and_build_for_either_simulator():
    from dataclasses import fields
    from pathlib import Path

    from physai.config import load_manifest
    from physai.config.compat import with_overrides
    from physai.robots import create_env_config

    root = Path(__file__).resolve().parents[3]
    accepted = {f.name for f in fields(create_env_config("so101", simulator="isaac"))}
    for path in sorted((root / "research" / "configs").glob("*.yaml")):
        manifest = load_manifest(path)
        assert manifest.plugins == ("research.plugin",), path.name
        isaac = with_overrides(manifest, simulator="isaac")
        assert set(isaac.robots[0].config) <= accepted, path.name
