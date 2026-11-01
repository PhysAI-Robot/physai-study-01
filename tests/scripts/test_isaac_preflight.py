import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))


def test_preflight_retries_after_a_glitch_and_logs_every_try(tmp_path, monkeypatch):
    import isaac_preflight

    results = iter(
        [
            {"glitch": True, "ok": False},
            {"glitch": True, "ok": False},
            {"glitch": False, "ok": True},
        ]
    )
    monkeypatch.setattr(
        isaac_preflight, "one_start", lambda manifest, timeout, policy: next(results)
    )
    monkeypatch.setattr(isaac_preflight.time, "sleep", lambda seconds: None)
    log = tmp_path / "starts.jsonl"
    monkeypatch.setattr(
        sys, "argv", ["isaac_preflight.py", "--log", str(log), "--tries", "3"]
    )

    assert isaac_preflight.main() == 0
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["ok"] for r in rows] == [False, False, True]
    assert [r["attempt"] for r in rows] == [1, 2, 3]


def test_preflight_fails_when_no_try_draws_the_robot(tmp_path, monkeypatch):
    import isaac_preflight

    monkeypatch.setattr(
        isaac_preflight,
        "one_start",
        lambda manifest, timeout, policy: {"glitch": True, "ok": False},
    )
    monkeypatch.setattr(isaac_preflight.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["isaac_preflight.py", "--log", str(tmp_path / "s.jsonl"), "--tries", "2"],
    )
    assert isaac_preflight.main() == 1
