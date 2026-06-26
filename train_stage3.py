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
    HexapodGoalEnv, ACT_DIM, OBS_DIM,
    GOAL_MIN_DIST, GOAL_MAX_DIST,
)

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
        env = HexapodGoalEnv(render_mode=None)
        return Monitor(env)
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
    The first MLP layer is skipped because Stage-2 has a 53-D obs vs 57-D here;
    all hidden layers and output heads transfer since net_arch and ACT_DIM (9) match.
    """
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