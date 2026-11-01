# physai-study-01

Code to reproduce study 1 of [physai-robot-starter](https://github.com/PhysAI-Robot/physai-robot-starter):
randomized SO-101 and ViperX 300 S pick-and-place, and how much of a policy's loss between two simulators is the
renderer's. It holds the scripted experts, the camera-only visual servo, the ACT and Diffusion Policy training and
evaluation code, the probes and the checks, and the manifests. Nothing else: checkpoints and data are on Hugging Face.

Two tags: `study-01-v1` is the first version of the study (SO-101, three engines); `study-01-v2` adds the ViperX 300 S,
the physics-by-renderer cells, and the checks of the paper's diagnostic ladder. `study-01-v2` includes everything of v1.

| Part | Where |
| --- | --- |
| scripted experts (privileged state, collect demonstrations): SO-101 and ViperX 300 S | `scripted_experts/` |
| visual servo (camera only) | `classical_control/` |
| ACT and Diffusion Policy: dataset, training, loading, probes | `imitation_learning/` |
| task manifests (randomized pick-and-place, floor variants, randomized look; SO-101 and ViperX 300 S) | `configs/` |
| camera comparison between simulators, render-parity check, ViperX checks, pad-visible wrapper, workspace success maps | `scripts/` |
| the queues, gates and analyses of the numbered items N9 to N14 of the second part, as they were run | `experiments/` |
| the scripts that wrote the Hugging Face release (model card, per-seed results, checkpoint folders) | `scripts/build_model_card.py`, `scripts/stage_*.py` |
| tests | `tests/` |

## Setup

This repository is a plugin of the framework: it must be checked out as the folder `research/` inside a clone of
the framework at the version it was run with (a manifest loads it with `plugins: [research.plugin]`). Both
repositories carry the tags `study-01-v1` and `study-01-v2`; use `study-01-v2` for everything below (the tests and
the ViperX parts need it):

```bash
git clone --branch study-01-v2 https://github.com/PhysAI-Robot/physai-robot-starter
git clone --branch study-01-v2 https://github.com/PhysAI-Robot/physai-study-01 physai-robot-starter/research
cd physai-robot-starter
uv sync                                           # MuJoCo, Newton, ACT and Diffusion Policy; add --extra isaac for Isaac Sim
uv run python scripts/fetch_assets.py --robot so101
uv run python scripts/fetch_assets.py --robot viperx300s
```

At `study-01-v1` the training stack was an extra (`uv sync --extra training`) and Newton needed its own Python
(`imitation_learning/newton_eval.py`, kept for that version); at `study-01-v2` both come with `uv sync`, and any
simulator pairs with any renderer through `--sim` and `--render`.

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

Newton as the renderer (`study-01-v1` used `newton_eval.py` for this), a one-colour or shifted floor, and clutter:

```bash
uv run python scripts/eval_policy.py --manifest $M --render newton --policy lerobot --checkpoint outputs/act_nominal --seed 1000 --episodes 300
uv run python scripts/eval_policy.py --manifest research/configs/so101_randomized_pick_place_plainfloor.yaml --policy lerobot --checkpoint outputs/act_nominal --seed 1000 --episodes 100
uv run python scripts/sweep_difficulty.py --policy lerobot --checkpoint outputs/act_nominal --manifest $M --seed 1000 --episodes 100
```

**Where a loss comes from: the physics-by-renderer cells, the ViperX 300 S, and the checks (`study-01-v2`).**

Every evaluation takes `--sim` (the physics: `mujoco`, `isaac`, `newton`) and `--render` (the engine that draws the
cameras; default: the simulator's own). A cell is one pairing. The four cells of MuJoCo and Isaac Sim, for one model:

```bash
for CELL in "mujoco mujoco" "mujoco isaac" "isaac mujoco" "isaac isaac"; do set -- $CELL
  OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/eval_policy_sharded.py --out outputs/act_nominal_$1_$2 --seed 1000 --episodes 100 --shard-size 10 \
    -- --manifest $M --sim $1 --render $2 --policy lerobot --checkpoint outputs/act_nominal
done
```

(A cell that does not involve Isaac Sim can run `eval_policy.py` directly with the same flags.) The paper's ladder, in
the order it asks the questions:

1. Is the physics executing the actions? Replay the demonstrations' actions on the other engine without images:
   `eval_policy.py --manifest $M --sim isaac --policy replay --dataset data/randomized_v1 --seed 0 --episodes 100 --json outputs/replay_isaac.json`.
2. How much is visual? The cells above: the loss with the other engine's pictures on your physics, with the other
   engine's physics on your pictures, and with both.
3. Do two renderers draw the same scene? `scripts/render_parity.py` (poses, dump, compare; its docstring has the
   commands). It is a check to run before comparing scores, and the paper reports where it failed its own validation.
   `scripts/viperx300s_camera_parity.py` is the per-camera version for the ViperX (cube and pad found by colour).
4. Is the score a property of the recipe or of the run? Train at least three models per recipe
   (`train_act.py ... --seed N`) and report all of them.
5. Is a randomization better than a matched nominal control, and does it hold outside its ranges? The manifests
   `so101_randomized_pick_place_visual_dr.yaml` and `so101_randomized_pick_place_visual_dr_bw.yaml` (floor tiles
   black and white in 30% of the episodes) are the randomized looks; evaluate against the nominal manifest `$M`.

`scripts/with_pad_visible.py` runs another script with the SO-101's contact pads drawn (they are collision-only boxes
that MuJoCo does not draw):
`python research/scripts/with_pad_visible.py scripts/eval_policy.py --manifest $M --policy scripted --seed 0 --episodes 200 --save-dataset data/pad_visible`.

ViperX 300 S (front and wrist cameras, seven-dimensional state and action; `train_act.py` reads the dimensions from
the dataset):

```bash
V=research/configs/viperx300s_single_cube_randomized_place.yaml           # nominal look
VR=research/configs/viperx300s_randomized_place_visual_dr.yaml            # randomized look
uv run python research/scripts/viperx300s_check.py --seeds 20             # robot and expert end to end, writes outputs/viperx300s/
uv run python scripts/eval_policy.py --manifest $V --policy scripted --seed 0 --episodes 400 --save-dataset data/viperx_nominal
uv run python research/imitation_learning/train_act.py --dataset data/viperx_nominal --chunk-size 100 --steps 120000 --num-workers 0 --seed 1 --out outputs/viperx_act_nominal_seed1
uv run python scripts/eval_policy.py --manifest $V --policy lerobot --checkpoint outputs/viperx_act_nominal_seed1 --episodes 300 --seed 1000 --json outputs/viperx_act_nominal_seed1_mujoco.json
```

The same cells apply (`--sim isaac --render mujoco` and so on). `viperx300s_visibility.py` counts the cube's and the
target's pixels in the ViperX cameras; `viperx300s_diag.py` is a short diagnostic of the expert.

The numbered items of the second part that score the cells and the checks (`experiments/`; copy the folder to `outputs/study01/v2/` of the
framework, where the scripts find each other and write their results; the checkpoints are the Hugging Face ones): `n9_*` is the MuJoCo-physics, Isaac-pictures
cell (a gate on the first frames, a queue of 10-episode shards, the analysis against the bars of the study), `n13_*` the
Isaac-physics, MuJoCo-pictures cell, `n14_*` the ViperX repeat of both, `n11_queue.sh` the Newton controls and `n12_*` the
sensitivity pilot. The gate is `n9_gate.py` (`dump` the cell's frames, then `check` or `check_im`); Isaac Sim exits without writing
its file when it hits the start-up render glitch, so the queues test for the file and repeat.

Probes of what the policy reads: `imitation_learning/image_edits.py` (`--policy-arg image_edit=...`),
`state_edits.py`, `read_probe.py`, `feature_shift.py`; each module's docstring has its command.

## Checkpoints and data

- Checkpoints, 34 models (26 ACT and 4 Diffusion Policy on the SO-101, 4 ACT on the ViperX 300 S) with their scores,
  training seeds and, in `results/<checkpoint>/` (index: `results/INDEX.tsv`), the per-seed success lists the paired tests are
  computed from, for every score of both parts and for the cells and checks of the second part:
  [rizgiak/physai-study-01-checkpoints](https://huggingface.co/rizgiak/physai-study-01-checkpoints). Use one
  directly: `--checkpoint hf:rizgiak/physai-study-01-checkpoints/act_randomized_look_seed0@e9cdc9feb6f1093cc65890d04a33e21031e52775`
  (downloaded into `checkpoints/`; `scripts/fetch_checkpoints.py` downloads ahead of time).
- The Isaac Sim replays of v1 (400 episodes): [rizgiak/physai-study-01-isaac-replays](https://huggingface.co/datasets/rizgiak/physai-study-01-isaac-replays).
  The MuJoCo demonstrations are regenerated with the `--save-dataset` commands above.

## Notes

- A 6 GB laptop GPU is enough; the ViperX models were trained on the same one. Approximate times on one: ACT 200 demos 60k steps about 2 h, Diffusion Policy 100k steps
  about 3.6 h, a 300-episode MuJoCo evaluation 20 min (ACT) to 2 h (Diffusion Policy).
- MuJoCo evaluations of ACT repeat step for step on the same versions; Diffusion Policy evaluations and every Isaac
  Sim run reproduce only within sampling noise.

## License

Apache License 2.0, see `LICENSE`.
