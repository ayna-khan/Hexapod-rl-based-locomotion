# Hexapod RL — 3-Stage Training Pipeline

Reinforcement learning pipeline for a 6-legged (hexapod) robot trained entirely in simulation using **PPO** (Proximal Policy Optimization) and **PyBullet**. The robot learns locomotion from scratch across three progressive stages: standing up from a resting pose, walking forward, and navigating to a goal.

---

## Architecture Overview
Stage 1 → Stand Up (crouched resting pose → stable upright stance)

Stage 2 → Walk Forwar (delta joint control → tripod gait locomotion)

Stage 3 → Reach Goal (goal-conditioned navigation with A* path planning)

Each stage warm-starts from the previous stage's policy weights, enabling efficient transfer learning.

---

## Robot

- **6 legs**, 3 DOF each (coxa, femur, tibia) → **18 actuated joints**
- Simulated in PyBullet with the included custom URDF
- Servo-style position control at 50 Hz control frequency (240 Hz physics)

---

## Requirements

Python 3.8+ and the following packages:

```bash
pip install -r requirements.txt
```

> **Linux users:** PyBullet needs system dependencies — install with:
> `sudo apt install cmake libgl1 libxrender1 libglib2.0-0`

---

## Project Structure
```

hexapod-rl-bases-locomtion/
│
├── urdf/
│   └── hexapod_generated.urdf      # Robot description
│
├── hexapod_env_stage1.py           # Stage 1 Gymnasium environment
├── hexapod_env_stage2.py           # Stage 2 Gymnasium environment
├── hexapod_env_stage3.py           # Stage 3 Gymnasium environment 
│
├── train_stage1.py                 # Stage 1 PPO training script
├── train_stage2.py                 # Stage 2 PPO training script
├── train_stage3.py                 # Stage 3 PPO training script
│
├── visualize_stage1.py             # Watch Stage 1 trained agent
├── visualize_stage2.py             # Watch Stage 2 trained agent
├── visualize_stage3.py             # Watch Stage 3 trained agent
│
├── requirements.txt
└── README.md
```
    
---

## Training

Stages must be run **in order**. Each stage saves checkpoints and normalizer stats automatically.

### Stage 1 — Stand Up

Trains the hexapod to rise from a crouched resting position to a stable standing height.

```bash
python train_stage1.py
```

Stop when `mean_reward > 80`. Watch the result:

```bash
python visualize_stage1.py
```

---

### Stage 2 — Walk Forward

Trains the hexapod to walk in a straight line using delta joint actions. Automatically warm-starts from the best Stage 1 policy.

```bash
python train_stage2.py
```

Stop when `mean_reward > 200` AND `mean_ep_len > 500`. Watch:

```bash
python visualize_stage2.py
```

---

### Stage 3 — Reach a Goal

Trains the hexapod to navigate to a randomly placed goal marker. Uses a distance curriculum (0.8 m → 3.0 m) and warm-starts from Stage 2.

```bash
python train_stage3.py
```

Watch with random goals:

```bash
python visualize_stage3.py
```

Watch with a fixed goal position:

```bash
python visualize_stage3.py --goal 1.5 0.0
```

---

## Resuming Training

All stages support `--resume`:

```bash
python train_stage1.py --resume
python train_stage2.py --resume
python train_stage3.py --resume
```

---

## Monitoring

```bash
tensorboard --logdir logs_stage1   # or logs_stage2 / logs_stage3
```

---

## CLI Flags

| Flag | Scripts | Effect |
|---|---|---|
| `--resume` | all train | Resume from last checkpoint |
| `--timesteps N` | all train | Override total training steps |
| `--no-warmstart` | train_stage3 | Skip Stage 2 weight transfer |
| `--slow` | all visualize | Run at 0.5× real-time |
| `--fast` | all visualize | No sleep (max speed) |
| `--model path/to.zip` | all visualize | Load a specific checkpoint |
| `--goal X Y` | visualize_stage3 | Set a fixed goal (metres) |
| `--episodes N` | visualize_stage3 | Stop after N episodes |

---

## Reward Design Summary

**Stage 1** — Height error (Gaussian), tilt penalty, alive bonus, foot contacts, posture match

**Stage 2** — Forward velocity, height stability, tilt penalty, tripod gait bonus, energy cost

**Stage 3** — Potential-based progress to goal, velocity toward goal, sparse success bonus (+200), height/tilt preservation, stall penalty

---

## Checkpoints

Saved automatically under `checkpoints_stage*/`. The best model (by evaluation reward) is saved as `best_model_stage*.zip`. Normalizer statistics are saved as `vec_normalize_stage*.pkl` and must be loaded alongside the model for correct inference.

---

## Notes

- The robot spawns **upside-down (legs pointing up)** in the resting pose for Stage 1 and must learn to self-right and stand.
- Stage 2 uses **delta joint actions** rather than absolute targets, which produces smoother and more stable gaits.
- Stage 3 uses an **A\* planner** on a 2D occupancy grid to compute waypoints; on the flat arena this degenerates to a straight line but the planner is extensible to obstacle avoidance.

---
