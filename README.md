# physai-study-01

Code to reproduce study 1 of [physai-robot-starter](https://github.com/PhysAI-Robot/physai-robot-starter):
randomized SO-101 and ViperX 300 S pick-and-place, and how much of a policy's loss between two simulators is the
renderer's. It holds the scripted experts, the camera-only visual servo, the ACT, Diffusion Policy and SmolVLA training and
evaluation code, the probes and the checks, and the manifests. Nothing else: checkpoints and data are on Hugging Face.

The Isaac Sim numbers of the study were all scored on one stack, Isaac Sim 6.2.0.0.

| Part | Where |
| --- | --- |
| scripted experts (privileged state, collect demonstrations): SO-101 and ViperX 300 S | `scripted_experts/` |
| visual servo (camera only) | `classical_control/` |
| ACT, Diffusion Policy and SmolVLA: dataset, training, loading, probes | `imitation_learning/` |
| task manifests (randomized pick-and-place, floor variants, randomized look; SO-101 and ViperX 300 S) | `configs/` |
| camera comparison between simulators, render-parity check, ViperX checks, pad-visible wrapper, workspace success maps, cube-start table (`cube_start_table.py`) | `scripts/` |
| the queues and analyses behind the paper's tables, named by what they do (the table in "Reproduce" maps them to the paper), with the cube-start tables they read | `experiments/` |
| tests | `tests/` |

## Setup

This repository is a plugin of the framework: it must be checked out as the folder `research/` inside a clone of
the framework at the version it was run with (a manifest loads it with `plugins: [research.plugin]`). Both
repositories carry the tag `study-01`; check both out at it:

```bash
git clone --branch study-01 https://github.com/PhysAI-Robot/physai-robot-starter
git clone --branch study-01 https://github.com/PhysAI-Robot/physai-study-01 physai-robot-starter/research
cd physai-robot-starter
uv sync                                           # MuJoCo, Newton, ACT and Diffusion Policy; add --extra isaac for Isaac Sim, --extra smolvla for SmolVLA
uv run python scripts/fetch_assets.py --robot so101
uv run python scripts/fetch_assets.py --robot viperx300s
```

Any simulator pairs with any renderer through `--sim` and `--render`.

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
uv run python scripts/eval_policy.py --manifest $M --policy scripted --seed 0 --episodes 200 --save-dataset data/mujoco_nominal_200_seeds0-199
uv run python scripts/eval_policy.py --manifest research/configs/so101_randomized_pick_place_visual_dr.yaml --policy scripted \
  --seed 0 --episodes 400 --save-dataset data/visual_dr     # the look is randomized in every episode
```

ACT (chunk 100 played in full; add `--seed N` for a repeatable run):

```bash
uv run python research/imitation_learning/train_act.py --dataset data/mujoco_nominal_200_seeds0-199 --chunk-size 100 --steps 60000 --num-workers 0 --out outputs/act_nominal
uv run python research/imitation_learning/train_act.py --dataset data/visual_dr --chunk-size 100 --steps 120000 --num-workers 0 --out outputs/act_randomized_look
uv run python scripts/eval_policy.py --manifest $M --policy lerobot --checkpoint outputs/act_nominal --episodes 300 --seed 1000 --json outputs/act_nominal_mujoco.json
```

Diffusion Policy (LeRobot; plays 16 of its 64 predicted actions; the evaluation samples noise and does not fix the
torch seed, so reruns differ in the details of an episode):

```bash
uv run python research/imitation_learning/train_diffusion.py --dataset data/mujoco_nominal_200_seeds0-199 --steps 100000 --lr 1e-4 --num-workers 0 --seed 0 --out outputs/dp_nominal
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
OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/eval_policy.py --manifest $M --sim isaac --policy replay --dataset data/mujoco_nominal_200_seeds0-199 \
  --seed 0 --episodes 100 --save-dataset data/isaac_replay_100_seeds0-99
# the second hundred demonstrations: `--seed` is the index of the first episode of the dataset to replay
OMNI_KIT_ACCEPT_EULA=YES uv run python scripts/eval_policy.py --manifest $M --sim isaac --policy replay --dataset data/mujoco_nominal_200_seeds0-199 \
  --seed 100 --episodes 100 --save-dataset data/isaac_replay_100_seeds100-199
uv run python research/imitation_learning/train_act.py --dataset data/mujoco_nominal_200_seeds0-199 data/isaac_replay_100_seeds0-99 data/isaac_replay_100_seeds100-199 \
  --chunk-size 100 --steps 120000 --num-workers 0 --out outputs/act_both_renderers
```

Newton as the renderer, a one-colour or shifted floor, and clutter:

```bash
uv run python scripts/eval_policy.py --manifest $M --render newton --policy lerobot --checkpoint outputs/act_nominal --seed 1000 --episodes 300
uv run python scripts/eval_policy.py --manifest research/configs/so101_randomized_pick_place_plainfloor.yaml --policy lerobot --checkpoint outputs/act_nominal --seed 1000 --episodes 100
uv run python scripts/sweep_difficulty.py --policy lerobot --checkpoint outputs/act_nominal --manifest $M --seed 1000 --episodes 100
```

SmolVLA (`uv sync --extra smolvla`; fine-tunes `lerobot/smolvla_base` with the vision-language model frozen and plays
its 50-action chunk in full; the recipe of the paper is 20,000 steps, batch 64 as 2 x 32 in float32, picture size 256; `--steps 40000`
gives the longer runs; the 400 demonstrations are two datasets of 200, collected as above with `--seed 0` and `--seed 200`):

```bash
uv run python research/imitation_learning/train_smolvla.py --dataset data/nominal_a data/nominal_b --steps 20000 --batch-size 64 --accum 2   --model-image-size 256 --seed 1 --log-every 100 --save-every 2000 --out outputs/smolvla_nominal_seed1
uv run python scripts/eval_policy.py --manifest $M --policy lerobot --checkpoint outputs/smolvla_nominal_seed1 --episodes 300 --seed 1000 --json outputs/smolvla_mujoco.json
# fewer actions per chunk: --policy-arg n_action_steps=25
```

**Where a loss comes from: the physics-by-renderer cells, the ViperX 300 S, and the checks.**

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
   `eval_policy.py --manifest $M --sim isaac --policy replay --dataset data/mujoco_nominal_200_seeds0-199 --seed 0 --episodes 100 --json outputs/replay_isaac.json`.
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
target's pixels in the ViperX cameras.

`experiments/` holds the queues and analyses behind the paper's tables, named by what they do. Run them from the framework root
(`bash research/experiments/smolvla_ladder.sh`, `python research/experiments/smolvla_ladder_analyse.py`): they find each other in
`research/experiments/` and read and write their results in `outputs/study01/` (one folder per experiment, logs in `outputs/study01/logs/`) and
`data/study01/`. The datasets they train on are expected there under the names the scripts use (`data/study01/mujoco_nominal_200_seeds0-199`,
`mujoco_nominal_200_seeds200-399`, `mujoco_randomized-look_400_seeds0-399`, and the probe sets of `probe_first_frames.sh` and
`isaac_scores_controls_b.sh`), collected with the commands above; the checkpoints are the Hugging Face ones, so change a local `--checkpoint` path to
`hf:rizgiak/physai-study-01-checkpoints/<folder>`. A queue (`.sh`) runs the evaluations and writes per-episode files; the `_analyse.py` next to it
reads them and prints the numbers of the table.

| Paper | Scripts in `experiments/` | What they do |
| --- | --- | --- |
| Section "sim-to-sim gap", cells table: the MI cell (MuJoCo physics, Isaac Sim pictures) | `cell_first_frame_gate.py`, `cell_mujoco_physics_isaac_pictures.sh`, `cell_mujoco_physics_isaac_pictures_analyse.py` | the gate on the first frames, 100 seeds of each of three ACT models, the loss against the bars written beforehand |
| the IM cell (Isaac Sim physics, MuJoCo pictures) | `cell_isaac_physics_mujoco_pictures.sh`, `cell_isaac_physics_mujoco_pictures_analyse.py` | the same for the other cell |
| the shares of the loss, with bootstrap intervals | `cell_shares_ci.py` | needs the four cells of the SO-101, ViperX and SmolVLA runs |
| ViperX table | `viperx_cells.sh mi` / `im`, `viperx_cells_analyse.py` | the two nominal ViperX models in the MI and IM cells, 300 seeds |
| Appendix, further results: Isaac Sim's cube start | `isaac_cube_start_measure.sh`, `isaac_cube_start_tables.py`, `isaac_cube_start_gate.sh`, `isaac_cube_start_mujoco_cell.sh`, `isaac_cube_start_isaac_pictures_cell.sh`, `isaac_cube_start_analyse.py`, `isaac_cube_start_isaac_pictures_analyse.py` (inputs: `cube_*.json`; `scripts/cube_start_table.py` measures) | where Isaac Sim puts the cube, and the cells scored with that start |
| Newton controls table | `newton_controls.sh` | the tile-free runs on the held-out look, the tile runs on the look outside every range |
| proxies table | `proxy_sensitivity_run.sh`, `proxy_sensitivity_models.json`, `proxy_sensitivity_analyse.py` | the sensitivity of each model to edits on MuJoCo against its loss |
| budget table, SmolVLA table | `smolvla_memory_fit.sh`, `smolvla_memory_fit_picture_size.sh`, `smolvla_ladder.sh`, `smolvla_ladder_analyse.py` | the memory test of the recipe; the six fine-tunes and their ladder of cells |
| tuning table, "SmolVLA with more steps" | `tuning_search.sh`, `smolvla_longer_runs_scores.sh`, `smolvla_longer_runs_analyse.py`, `act_learning_rate_run.sh` | the later tuning search, the four longer SmolVLA checkpoints on the test seeds, the last ACT learning-rate run |
| paired table, failure counts, y bands, replay controls, actions per chunk | `isaac_scores_controls_a.sh`, `isaac_scores_controls_b.sh`, `paired_renderer_gap.py`, `failure_and_replay_numbers.py` | the Isaac Sim scores of the control runs, then the paired McNemar table and the failure numbers |
| probes and reads tables, probes figure, measured picture differences, gripper reading | `probe_first_frames.sh`, `camera_dump.sh`, `image_statistics.py`, `gripper_trace.py`, `gripper_trace_retry.sh` | the first-frame swap, the encoder shift, the read probe, the image statistics and the gripper trace |
| rendering-settings table | `isaac_render_settings.sh`, `isaac_render_variant.py`, `isaac_render_settings_analyse.py` | the eight Isaac Sim settings moved toward MuJoCo's, one at a time and together |
| looks tables: outside the ranges, black-and-white tiles, ViperX held-out look | `bw_look_runs.sh`, `newton_looks.sh`, `newton_look.py`, `viperx_heldout_look.sh`, `viperx_heldout_look_analyse.py` | the black-and-white-tile runs, the looks no run was trained on, the same on the second robot |
| pads table, Newton before and after the fix, the one-grey-level control | `newton_after_fix_scores.sh`, `newton_before_fix_scores.sh`, `newton_pad_test.sh` (seeds: `newton_pad_test_seeds.json`), `newton_renderer_variant.py`, `newton_camera_variant.py`, `newton_noise_variant.py`, `pad_visible_run.sh` | the models on both Newton renderers, the failed seeds with the contact pads hidden, one camera at a time, the model trained with the pads drawn |
| left/right failures and the light direction | `left_right_isaac_scores.sh`, `left_right_failures.py`, `newton_light_mirrored.sh`, `newton_light_variant.py` | Isaac Sim seeds 1100-1299 of the nine nominal runs, the failure rate by side, Newton with the key light mirrored |
| any paired row from the Hugging Face files | `hub_counts.py` | counts and exact McNemar p of two `results/` files, without a queue |

The `_analyse.py` scripts read the per-episode files that the queues write under `outputs/study01/` (they hold fields such as the failure reason and the
distance to the target). `results/` on Hugging Face holds the success of every seed of the same runs, from which the counts and the paired tests of the tables
are recomputed (`results/INDEX.tsv` lists the files; `hub_counts.py` does the pairing). The training of the runs that only repeat a recipe (more seeds of ACT and
Diffusion Policy, the ViperX models) is the commands of "Reproduce" above.

Isaac Sim exits without writing its file when it hits the start-up render glitch, so the queues test for the file and repeat;
`eval_policy_sharded.py` repeats such a shard.

Probes of what the policy reads: `imitation_learning/image_edits.py` (`--policy-arg image_edit=...`),
`state_edits.py`, `read_probe.py`, `feature_shift.py`; each module's docstring has its command.

## Checkpoints and data

- Checkpoints, 45 models (27 ACT and 4 Diffusion Policy on the SO-101, 4 ACT on the ViperX 300 S, 10 SmolVLA) with their scores, training seeds and, in
  `results/<checkpoint>/` (index: `results/INDEX.tsv`), the per-seed success lists the paired tests are computed from, for every score of the study.
  It also publishes the per-seed validation files (MuJoCo seeds 900-999) of runs whose weights are not published, and the colour-jitter ACT model
  of the image-edit probes: [rizgiak/physai-study-01-checkpoints](https://huggingface.co/rizgiak/physai-study-01-checkpoints). Use one
  directly: `--checkpoint hf:rizgiak/physai-study-01-checkpoints/act_randomized_look_seed0@9d9ff23af0c15340a7ed8cf50132128122c0ec3a`
  (downloaded into `checkpoints/`; `scripts/fetch_checkpoints.py` downloads ahead of time).
- The Isaac Sim replays (400 episodes): [rizgiak/physai-study-01-isaac-replays](https://huggingface.co/datasets/rizgiak/physai-study-01-isaac-replays).
  The MuJoCo demonstrations are regenerated with the `--save-dataset` commands above.

## Notes

- A 6 GB laptop GPU is enough; the ViperX models were trained on the same one. Approximate times on one: ACT 200 demos 60k steps about 2 h, Diffusion Policy 100k steps
  about 3.6 h, SmolVLA 20k steps about 6.2 h (peak 4.6 GB; 40k steps about 12.5 h), a 300-episode MuJoCo evaluation 20 min (ACT) to 2 h (Diffusion Policy).
- MuJoCo evaluations of ACT repeat step for step on the same versions; Diffusion Policy evaluations and every Isaac
  Sim run reproduce only within sampling noise.

## License

Apache License 2.0, see `LICENSE`.
