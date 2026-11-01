"""Check that Isaac Sim draws the robot before a long evaluation, and log every start.

    python research/scripts/isaac_preflight.py --log outputs/isaac_starts.jsonl
    python research/scripts/isaac_preflight.py --log outputs/isaac_starts.jsonl --tries 3 --wait 60

One short Isaac episode with the `constant` policy (no model is loaded, nothing is moved): if the
robot is not drawn the start stops with "Isaac Sim is not drawing the robot" and the check tries
again after `--wait` seconds, up to `--tries` times. Exit code 0 means a start drew the robot, 1
that none of the tries did. Each try is appended to `--log` with the time, the minutes since the
machine booted and whether the start was a glitch, so the rate of the startup glitch and its
dependence on a warm-up (after a reboot or an update, before and after a first light run) can be
read back (hypothesis to test: a light start first removes the glitch, see
`research/classical_control/FINDINGS.md`, Isaac render glitch).

Research module: nothing in `src/physai` imports it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "research/configs/so101_randomized_pick_place.yaml"
GLITCH = "not drawing the robot"


def minutes_since_boot() -> float | None:
    """Windows only; None elsewhere."""
    try:
        import ctypes

        return round(ctypes.windll.kernel32.GetTickCount64() / 60000, 1)
    except (AttributeError, OSError):
        return None


def one_start(manifest: Path, timeout: int, policy: list[str]) -> dict:
    command = [
        sys.executable,
        str(ROOT / "scripts/eval_policy.py"),
        "--manifest",
        str(manifest),
        "--sim",
        "isaac",
        *policy,
        "--seed",
        "1000",
        "--episodes",
        "1",
        "--max-steps",
        "5",
    ]
    started = time.time()
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    try:
        done = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=timeout,
            env=env,
        )
        output, code = done.stdout, done.returncode
    except subprocess.TimeoutExpired:
        output, code = "", -1
    glitch = GLITCH in output
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "minutes_since_boot": minutes_since_boot(),
        "seconds": round(time.time() - started, 1),
        "exit_code": code,
        "glitch": glitch,
        "ok": code == 0 and not glitch and "policy=" in output,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", type=Path, required=True, help="JSON lines, appended")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--wait", type=int, default=60, help="seconds between tries")
    ap.add_argument("--timeout", type=int, default=900, help="seconds for one start")
    ap.add_argument(
        "--policy-args",
        default="--policy constant",
        help="eval_policy.py policy flags of the start (default: the constant policy)",
    )
    ap.add_argument("--kind", default="preflight", help="label written to the log")
    ap.add_argument(
        "--no-stop",
        action="store_true",
        help="make all --tries starts even after one draws the robot (for the cold-start test)",
    )
    args = ap.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, args.tries + 1):
        row = {
            "kind": args.kind,
            "attempt": attempt,
            **one_start(args.manifest, args.timeout, args.policy_args.split()),
        }
        with args.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if row["ok"] and not args.no_stop:
            return 0
        if attempt < args.tries:
            time.sleep(args.wait)
    return 0 if args.no_stop else 1


if __name__ == "__main__":
    sys.exit(main())
