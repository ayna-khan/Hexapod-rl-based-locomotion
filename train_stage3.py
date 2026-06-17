"""
HEXAPOD — Stage 3 Training: Goal Reaching
==========================================
Usage:
  python3 train_stage3.py
  python3 train_stage3.py --resume
  python3 train_stage3.py --no-warmstart
  python3 train_stage3.py --timesteps 15000000
"""

import os
import shutil
import argparse
import glob

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

from hexapod_env_stage3 import (
    HexapodGoalEnv,
    JOINT_LOWER, JOINT_UPPER, STAND_POSE,
    NUM_JOINTS, SIM_HZ, CTRL_HZ, SIM_STEPS_PER_CTRL,
    TARGET_HEIGHT, SPAWN_Z, MIN_HEIGHT, MAX_TILT,
    ACTION_SCALE, SERVO_FORCE,
    GOAL_RADIUS, GOAL_MIN_DIST,
    ACT_DIM,
)

import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STAGE2_DIR = os.path.join(BASE_DIR, "checkpoints_stage2")
STAGE3_DIR = os.path.join(BASE_DIR, "checkpoints_stage3")
LOG_DIR    = os.path.join(BASE_DIR, "logs_stage3")
BEST_MODEL = os.path.join(STAGE3_DIR, "best_model_stage3")
LAST_MODEL = os.path.join(STAGE3_DIR, "last_model_stage3")
NORM_STATS = os.path.join(STAGE3_DIR, "vec_normalize_stage3.pkl")
os.makedirs(STAGE3_DIR, exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)

TOTAL_TIMESTEPS = 12_000_000
N_ENVS          = 8
N_STEPS         = 8192
BATCH_SIZE      = 512
N_EPOCHS        = 10
LEARNING_RATE   = 3e-4
GAMMA           = 0.995
GAE_LAMBDA      = 0.95
CLIP_RANGE      = 0.2
ENT_COEF        = 0.03

CURRICULUM = [
    (0.00, 0.8),
    (0.10, 1.5),
    (0.30, 2.5),
    (0.60, 3.0),
]

W_VEL_TOWARD  =  5.0
W_PROGRESS    = 15.0
W_GOAL        = 200.0
W_HEIGHT      =  2.0
W_TILT        = -2.0
W_ALIVE       =  0.1
W_ENERGY      = -0.001
W_ACTION_RATE = -0.002
W_STALL       = -2.0

STALL_STEPS   = 20
STALL_THRESH  = 0.005
GRACE_STEPS   = 200

OBS_DIM           = 72
GOAL_MAX_DIST_DEFAULT = 3.0


class HexapodGoalEnvFixed(HexapodGoalEnv):
    """
    Drop-in replacement for HexapodGoalEnv with two changes:
      1. Obs uses a direct vector to goal instead of A* waypoints.
      2. Debug line is a single straight line from spawn to goal.
    Everything else (reward, physics, termination) is identical to the base env.
    """

    def __init__(self, max_goal_dist=GOAL_MAX_DIST_DEFAULT, **kwargs):
        super().__init__(**kwargs)
        self._max_goal_dist = float(max_goal_dist)
        self._min_goal_dist = GOAL_MIN_DIST

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)

        self._stall_buf = []

    def set_max_goal_dist(self, d: float):
        self._max_goal_dist = float(d)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Resample goal with curriculum distance range
        if self.fixed_goal is None:
            rng = self.np_random
            for _ in range(500):
                angle = rng.uniform(0, 2 * np.pi)
                dist  = rng.uniform(self._min_goal_dist, self._max_goal_dist)
                gx    = float(dist * np.cos(angle))
                gy    = float(dist * np.sin(angle))
                self._goal_xy = np.array([gx, gy], dtype=np.float32)
                break
            try:
                self._spawn_goal_marker(*self._goal_xy)
            except Exception:
                pass

        # line from the spawn point directly to the goal.
        if self.render_mode == "human":
            try:
                p.removeAllUserDebugItems(physicsClientId=self._client)
                p.addUserDebugLine(
                    [0.0, 0.0, 0.02],
                    [float(self._goal_xy[0]), float(self._goal_xy[1]), 0.02],
                    lineColorRGB=[0, 1, 0],
                    lineWidth=3.0,
                    physicsClientId=self._client)
            except Exception:
                pass

        # Use goal itself as the single waypoint 
        self._waypoints = [tuple(self._goal_xy)]
        self._wp_idx    = 0

        try:
            pos, _ = p.getBasePositionAndOrientation(
                self._robot, physicsClientId=self._client)
            self._prev_dist = float(np.linalg.norm(
                np.array([pos[0], pos[1]]) - self._goal_xy))
        except Exception:
            self._prev_dist = float(self._max_goal_dist)

        self._stall_buf = [self._prev_dist] * STALL_STEPS

        return self._get_obs_fixed(), info

    def _get_obs_fixed(self):
        pos, orn         = p.getBasePositionAndOrientation(
            self._robot, physicsClientId=self._client)
        lin_vel, ang_vel = p.getBaseVelocity(
            self._robot, physicsClientId=self._client)
        euler            = p.getEulerFromQuaternion(orn)

        jpos, jvel = [], []
        for jid in self._joint_ids:
            js = p.getJointState(self._robot, jid,
                                 physicsClientId=self._client)
            jpos.append(js[0])
            jvel.append(js[1])

        xy = np.array([pos[0], pos[1]], dtype=np.float32)

        rel_goal  = self._goal_xy - xy
        dist_goal = float(np.linalg.norm(rel_goal))

        if dist_goal > 1e-6:
            goal_dir_norm = rel_goal / dist_goal
        else:
            goal_dir_norm = np.zeros(2, dtype=np.float32)

        dist_goal_norm = float(np.clip(dist_goal / self._max_goal_dist, 0.0, 1.0))

        yaw         = float(euler[2])
        goal_angle  = float(np.arctan2(
            self._goal_xy[1] - pos[1],
            self._goal_xy[0] - pos[0]))
        heading_err = float(np.arctan2(
            np.sin(goal_angle - yaw),
            np.cos(goal_angle - yaw)))

        return np.concatenate([
            lin_vel,
            ang_vel,
            [euler[0], euler[1]],
            jpos,
            jvel,
            self._prev_action,
            goal_dir_norm,
            [dist_goal_norm],
            [heading_err / np.pi],
        ]).astype(np.float32)

    def _get_obs(self):
        return self._get_obs_fixed()

    def _compute_reward(self, action):
        pos, orn   = p.getBasePositionAndOrientation(
            self._robot, physicsClientId=self._client)
        lin_vel, _ = p.getBaseVelocity(
            self._robot, physicsClientId=self._client)
        euler      = p.getEulerFromQuaternion(orn)

        xy        = np.array([pos[0], pos[1]], dtype=np.float32)
        height    = float(pos[2])
        tilt      = float(abs(euler[0]) + abs(euler[1]))
        dist_goal = float(np.linalg.norm(xy - self._goal_xy))

        progress        = self._prev_dist - dist_goal
        self._prev_dist = dist_goal

        self._stall_buf.pop(0)
        self._stall_buf.append(dist_goal)
        stall_progress = self._stall_buf[0] - self._stall_buf[-1]
        stall_penalty  = W_STALL if stall_progress < STALL_THRESH else 0.0

        rel_goal     = self._goal_xy - xy
        dist_to_goal = float(np.linalg.norm(rel_goal))
        if dist_to_goal > 1e-6:
            goal_unit = rel_goal / dist_to_goal
        else:
            goal_unit = np.zeros(2, dtype=np.float32)
        vel_toward = float(np.dot(
            np.array([lin_vel[0], lin_vel[1]], dtype=np.float32),
            goal_unit))

        height_rew = float(np.exp(-20.0 * abs(height - TARGET_HEIGHT)))

        torques = [p.getJointState(self._robot, jid,
                    physicsClientId=self._client)[3]
                   for jid in self._joint_ids]
        energy  = float(np.sum(np.abs(torques)))

        action_rate = float(np.sum(np.square(action - self._prev_action)))

        reward = (
            W_VEL_TOWARD  * vel_toward   +
            W_PROGRESS    * progress     +
            stall_penalty               +
            W_HEIGHT      * height_rew   +
            W_TILT        * tilt         +
            W_ALIVE                      +
            W_ENERGY      * energy       +
            W_ACTION_RATE * action_rate
        )

        terminated = False
        info       = {"dist_to_goal": dist_goal, "success": False}

        if dist_goal < GOAL_RADIUS:
            reward    += W_GOAL
            terminated = True
            info["success"] = True
        elif self._step_count > GRACE_STEPS and (
                height < MIN_HEIGHT or tilt > MAX_TILT):
            reward    -= 10.0
            terminated = True

        return float(reward), terminated, info


class CurriculumGoalEnv(HexapodGoalEnvFixed):
    pass


class CurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps, vec_env, verbose=1):
        super().__init__(verbose)
        self.total   = total_timesteps
        self.vec_env = vec_env
        self._phase  = -1

    def _on_step(self):
        frac  = self.num_timesteps / self.total
        phase = 0
        for i, (thresh, _) in enumerate(CURRICULUM):
            if frac >= thresh:
                phase = i

        if phase != self._phase:
            self._phase = phase
            _, max_d    = CURRICULUM[phase]
            _apply_max_dist(self.vec_env, max_d)
            if self.verbose:
                print(f"\n  [Curriculum] Phase {phase+1} → "
                      f"max_goal_dist = {max_d:.1f} m  "
                      f"(step {self.num_timesteps:,})\n")
        return True


def _apply_max_dist(vec_env, d: float):
    env = vec_env
    while hasattr(env, "venv"):
        env = env.venv
    if hasattr(env, "envs"):
        for e in env.envs:
            inner = e
            while hasattr(inner, "env"):
                inner = inner.env
            if hasattr(inner, "set_max_goal_dist"):
                inner.set_max_goal_dist(d)


class ProgressCallback(BaseCallback):
    def __init__(self, log_every: int = 20_000):
        super().__init__()
        self._log_every = log_every
        self._last_log  = 0
        self._successes = 0
        self._episodes  = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if info.get("success", False):
                self._successes += 1
            if "episode" in info:
                self._episodes  += 1

        if self.num_timesteps - self._last_log < self._log_every:
            return True
        self._last_log = self.num_timesteps

        buf = self.model.ep_info_buffer
        if not buf:
            return True

        rewards  = [ep["r"] for ep in buf]
        lengths  = [ep["l"] for ep in buf]
        mean_r   = sum(rewards) / len(rewards)
        mean_l   = sum(lengths) / len(lengths)
        sr       = (self._successes / max(self._episodes, 1)) * 100

        if sr > 60 and mean_r > 150:
            status = "✓  CONVERGED"
        elif sr > 30:
            status = "→  navigating"
        elif mean_l < 200 and self.num_timesteps > 1_500_000:
            status = "⚠  FALLING"
        elif mean_r < 0 and self.num_timesteps > 800_000:
            status = "⚠  STUCK"
        else:
            status = "→  exploring"

        print(
            f"  steps {self.num_timesteps:>10,}  |  "
            f"mean_r {mean_r:>8.2f}  |  "
            f"mean_len {mean_l:>6.0f}  |  "
            f"success {sr:>5.1f}%  |  {status}",
            flush=True)

        self._successes = 0
        self._episodes  = 0
        return True


def make_env():
    def _init():
        env = CurriculumGoalEnv(render_mode=None)
        env = Monitor(env)
        return env
    return _init


def find_stage2_model():
    for name in ["best_model_stage2.zip", "best_model.zip", "last_model_stage2.zip"]:
        path = os.path.join(STAGE2_DIR, name)
        if os.path.exists(path):
            return path.replace(".zip", "")
    ckpts = glob.glob(os.path.join(STAGE2_DIR, "stage2_ppo_*.zip"))
    if ckpts:
        return max(ckpts, key=os.path.getctime).replace(".zip", "")
    return None


def _load_stage2_weights(model: PPO, stage2_path: str):
    """
    Copy weights from Stage-2 into Stage-3 where tensor shapes match.
    The first MLP layer is skipped because Stage-2 has 62-D obs vs 72-D here;
    all hidden layers and output heads transfer if net_arch is the same.
    """
    import torch
    try:
        s2 = PPO.load(stage2_path)
    except Exception as e:
        print(f"  Warning: could not load Stage-2 ({e}). Training from scratch.")
        return

    s2_sd  = s2.policy.state_dict()
    s3_sd  = model.policy.state_dict()
    copied = 0
    skipped = 0

    for k, v in s2_sd.items():
        if k in s3_sd and s3_sd[k].shape == v.shape:
            s3_sd[k] = v.clone()
            copied  += 1
        else:
            skipped += 1

    model.policy.load_state_dict(s3_sd)
    print(f"  Warm-start: {copied} tensors copied, {skipped} skipped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--no-warmstart", action="store_true")
    parser.add_argument("--stage2",       type=str, default=None)
    parser.add_argument("--timesteps",    type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    print(f"\n{'='*68}")
    print(f"  HEXAPOD — Stage 3: Goal Reaching")
    print(f"  obs={OBS_DIM}  act={ACT_DIM}  envs={N_ENVS}  total={args.timesteps:,}")
    print(f"{'='*68}\n")

    env_fns = [make_env() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    if args.resume and os.path.exists(NORM_STATS):
        try:
            vec_env = VecNormalize.load(NORM_STATS, vec_env)
            print("  Normalizer loaded.")
        except Exception as e:
            print(f"  Normalizer mismatch ({e}) — fresh stats.")
            vec_env = VecNormalize(
                vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    else:
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training    = False
    eval_env.norm_reward = False

    checkpoint_cb = CheckpointCallback(
        save_freq=max(250_000 // N_ENVS, 1),
        save_path=STAGE3_DIR,
        name_prefix="stage3_ppo",
        verbose=1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=STAGE3_DIR,
        log_path=LOG_DIR,
        eval_freq=max(50_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=0)

    progress_cb   = ProgressCallback(log_every=20_000)
    curriculum_cb = CurriculumCallback(
        total_timesteps=args.timesteps, vec_env=vec_env, verbose=1)

    if args.resume and os.path.exists(LAST_MODEL + ".zip"):
        print(f"  Resuming from {LAST_MODEL}.zip …")
        model = PPO.load(LAST_MODEL, env=vec_env, tensorboard_log=LOG_DIR)
        model.learning_rate = LEARNING_RATE
    else:
        model = PPO(
            "MlpPolicy",
            env=vec_env,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            verbose=0,
            tensorboard_log=LOG_DIR,
            policy_kwargs=dict(net_arch=[512, 256, 128]))

        if not args.no_warmstart:
            s2_path = args.stage2 or find_stage2_model()
            if s2_path and os.path.exists(s2_path + ".zip"):
                print(f"  Warm-starting from: {os.path.basename(s2_path)}.zip")
                _load_stage2_weights(model, s2_path)
            else:
                print("  No Stage-2 model found — training from scratch.")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_cb, eval_cb, progress_cb, curriculum_cb],
            reset_num_timesteps=not args.resume,
            tb_log_name="stage3_ppo")
    except KeyboardInterrupt:
        print("\n  Interrupted — saving …")
    finally:
        model.save(LAST_MODEL)
        vec_env.save(NORM_STATS)
        best_src = os.path.join(STAGE3_DIR, "best_model.zip")
        if os.path.exists(best_src):
            shutil.copy(best_src, BEST_MODEL + ".zip")
            print(f"  Best model → {BEST_MODEL}.zip")
        print(f"  Last model → {LAST_MODEL}.zip")
        vec_env.close()
        eval_env.close()

    print("\nDone!")
    print("  Watch: python3 enjoy_stage3.py")
    print("  Monitor: tensorboard --logdir logs_stage3")


if __name__ == "__main__":
    main()