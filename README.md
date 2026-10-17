# physai-study-01

Code to reproduce study 1 of [physai-robot-starter](https://github.com/PhysAI-Robot/physai-robot-starter):
randomized SO-101 pick-and-place, and how much a learned policy loses when only the renderer changes. It holds
the scripted expert, the camera-only visual servo, the ACT and Diffusion Policy training and evaluation code,
the probes, and the manifests. Nothing else: checkpoints and data are on Hugging Face.

| Part | Where |
| --- | --- |
| scripted expert (privileged state, collects demonstrations) | `scripted_experts/` |
| visual servo (camera only) | `classical_control/` |
| ACT and Diffusion Policy: dataset, training, loading, probes | `imitation_learning/` |
| task manifests (randomized pick-and-place, floor variants, randomized look) | `configs/` |
| camera comparison between simulators, workspace success maps | `scripts/` |
| tests | `tests/` |

## Setup

This repository is a plugin of the framework: it must be checked out as the folder `research/` inside a clone of
the framework at the version it was run with (a manifest loads it with `plugins: [research.plugin]`). Both
repositories carry the tag `study-01-v1`:

```bash
git clone --branch study-01-v1 https://github.com/PhysAI-Robot/physai-robot-starter
git clone --branch study-01-v1 https://github.com/PhysAI-Robot/physai-study-01 physai-robot-starter/research
cd physai-robot-starter
uv sync --extra training                          # ACT and Diffusion Policy; add --extra isaac for Isaac Sim
uv run python scripts/fetch_assets.py --robot so101
```

Newton (a third renderer; MuJoCo keeps the physics) needs its own Python:
`uv venv nv && uv pip install --python nv/Scripts/python.exe "newton==1.6.1" scipy trimesh` (`nv/bin/python` on Linux).

Seeds: demonstrations use episode seeds 0-199 (200 demos) or 0-399 (400 demos); settings are chosen on 900-999;
every reported score uses 1000-1299 (1000-1099 on Isaac Sim) and never the demonstration seeds.

## Reproduce

Scripted expert and visual servo (MuJoCo; add `--sim isaac` for Isaac Sim):

```bash
M=research/configs/so101_randomized_pick_place.yaml
uv run python scripts/eval_policy.py --manifest $M --policy scripted --episodes 300 --seed 1000 --json outputs/scripted.json
uv run python scripts/eval_policy.py --manifest $M --policy visual_servo --episodes 300 --seed 1000 --json outputs/visual_servo.json
```

Demonstrations are the scripted expert's episodes, recorded with `--save-dataset` (the same seeds give the same
states and actions; pixels can differ by one grey level). Every episode is kept, with its outcome in `meta.json`
(`success` per episode, `num_successful_episodes`); the expert delivers all 200 and all 400:

```bash
uv run python scripts/eval_policy.py --manifest $M --policy scripted --seed 0 --episodes 200 --save-dataset data/randomized_v1
uv run python scripts/eval_policy.py --manifest research/configs/so101_randomized_pick_place_visual_dr.yaml --policy scripted \
  --seed 0 --episodes 400 --save-dataset data/visual_dr     # the look is randomized in every episode
```

ACT (chunk 100 played in full; add `--seed N` for a repeatable run):

```bash
uv run python research/imitation_learning/train_act.py --dataset data/randomized_v1 --chunk-size 100 --steps 60000 --num-workers 0 --out outputs/act_nominal
uv run python research/imitation_learning/train_act.py --dataset data/visual_dr --chunk-size 100 --steps 120000 --num-workers 0 --out outputs/act_randomized_look
uv run python scripts/eval_policy.py --manifest $M --policy lerobot --checkpoint outputs/act_nominal --episodes 300 --seed 1000 --json outputs/act_nominal_mujoco.json
```

Diffusion Policy (LeRobot; plays 16 of its 64 predicted actions; the evaluation samples noise and does not fix the
torch seed, so reruns differ in the details of an episode):

```bash
uv run python research/imitation_learning/train_diffusion.py --dataset data/randomized_v1 --steps 100000 --lr 1e-4 --num-workers 0 --seed 0 --out outputs/dp_nominal
uv run python scripts/eval_policy.py --manifest $M --policy lerobot --checkpoint outputs/dp_nominal --policy-arg n_action_steps=16 --episodes 300 --seed 1000
```

Isaac Sim (shards of 10 episodes, each in its own process; a shard that hits the startup render glitch is repeated):

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/eval_policy_sharded.py --out outputs/act_nominal_isaac --seed 1000 --episodes 100 --shard-size 10 \
  -- --manifest $M --sim isaac --policy lerobot --checkpoint outputs/act_nominal --video all --out outputs/act_nominal_isaac_videos
```

Models trained on Isaac Sim frames: replay the MuJoCo demonstrations' actions on Isaac while recording Isaac's frames, then train on them
(alone, or together with the MuJoCo demonstrations):

```bash
OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/eval_policy.py --manifest $M --sim isaac --policy replay --dataset data/randomized_v1 \
  --seed 0 --episodes 100 --save-dataset data/randomized_isaac_v1
# the second hundred demonstrations: `--seed` is the index of the first episode of the dataset to replay
OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/eval_policy.py --manifest $M --sim isaac --policy replay --dataset data/randomized_v1 \
  --seed 100 --episodes 100 --save-dataset data/randomized_isaac_v1b
uv run python research/imitation_learning/train_act.py --dataset data/randomized_v1 data/randomized_isaac_v1 data/randomized_isaac_v1b \
  --chunk-size 100 --steps 120000 --num-workers 0 --out outputs/act_both_renderers
```

Newton as the renderer, a one-colour or shifted floor, and clutter:

```bash
uv run python research/imitation_learning/newton_eval.py --newton-python nv/Scripts/python.exe -- --manifest $M --policy lerobot --checkpoint outputs/act_nominal --seed 1000 --episodes 300
uv run python scripts/eval_policy.py --manifest research/configs/so101_randomized_pick_place_plainfloor.yaml --policy lerobot --checkpoint outputs/act_nominal --seed 1000 --episodes 100
uv run python scripts/sweep_difficulty.py --policy lerobot --checkpoint outputs/act_nominal --manifest $M --seed 1000 --episodes 100
```

Probes of what the policy reads: `imitation_learning/image_edits.py` (`--policy-arg image_edit=...`),
`state_edits.py`, `read_probe.py`, `feature_shift.py`; each module's docstring has its command.

## Checkpoints and data

- Checkpoints, 17 models with their scores and training seeds:
  [rizgiak/physai-study-01-checkpoints](https://huggingface.co/rizgiak/physai-study-01-checkpoints). Use one
  directly: `--checkpoint hf:rizgiak/physai-study-01-checkpoints/act_randomized_look_seed0@e9cdc9feb6f1093cc65890d04a33e21031e52775`
  (downloaded into `checkpoints/`; `scripts/fetch_checkpoints.py` downloads ahead of time).
- The Isaac Sim replays (400 episodes): [rizgiak/physai-study-01-isaac-replays](https://huggingface.co/datasets/rizgiak/physai-study-01-isaac-replays).
  The MuJoCo demonstrations are regenerated with the `--save-dataset` commands above.

## Notes

- A 6 GB laptop GPU is enough. Approximate times on one: ACT 200 demos 60k steps about 2 h, Diffusion Policy 100k steps
  about 3.6 h, a 300-episode MuJoCo evaluation 20 min (ACT) to 2 h (Diffusion Policy).
- MuJoCo evaluations of ACT repeat step for step on the same versions; Diffusion Policy evaluations and every Isaac
  Sim run reproduce only within sampling noise.

## License

Apache License 2.0, see `LICENSE`.
