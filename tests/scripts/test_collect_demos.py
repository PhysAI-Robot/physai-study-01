import json
import sys
from pathlib import Path

import pytest
from tests.conftest import requires_viperx300s_assets

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))

MANIFEST = (
    Path(__file__).parents[2]
    / "configs"
    / "viperx300s_single_cube_randomized_place.yaml"
)


@requires_viperx300s_assets
def test_collect_demos_drives_the_robots_own_expert(tmp_path, monkeypatch):
    """The ViperX expert, not the SO-101's, records the episode (the SO-101's fails on this arm)."""
    import collect_demos

    dataset = tmp_path / "demos"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_demos.py",
            "--manifest",
            str(MANIFEST),
            "--episodes",
            "1",
            "--seed",
            "900",
            "--no-images",
            "--dataset",
            str(dataset),
        ],
    )
    assert collect_demos.main() == 0
    meta = json.loads((dataset / "meta.json").read_text(encoding="utf-8"))
    assert len(meta["episodes"]) == 1
