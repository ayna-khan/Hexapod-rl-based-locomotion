"""
CRAWL — Stage 1 Training: Standing posture
==========================================
Trains the hexapod to stand up and hold a stable position.

Usage:
  python3 train_stage1.py            # fresh training
  python3 train_stage1.py --resume   # continue from last checkpoint

When mean_reward consistently exceeds 80, stop training.
The best_model_stage1.zip will be used as the starting point for Stage 2.
"""

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from hexapod_env_stage1 import HexapodStandEnv

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
STAGE1_DIR      = os.path.join(BASE_DIR, "checkpoints_stage1")
LOG_DIR         = os.path.join(BASE_DIR, "logs_stage1")
BEST_MODEL      = os.path.join(STAGE1_DIR, "best_model_stage1")
LAST_MODEL      = os.path.join(STAGE1_DIR, "last_model_stage1")
NORM_STATS      = os.path.join(STAGE1_DIR, "vec_normalize_stage1.pkl")
os.makedirs(STAGE1_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Hyperparameters
TOTAL_TIMESTEPS = 1_500_000
N_ENVS          = 4
N_STEPS         = 2048
BATCH_SIZE      = 256
N_EPOCHS        = 10
LEARNING_RATE   = 3e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_RANGE      = 0.2
ENT_COEF        = 0.0
CHECKPOINT_FREQ = 50_000


class ProgressCallback(BaseCallback):
    def __init__(self, log_every=2000):
        super().__init__()
        self.log_every  = log_every
        self._last_log  = 0

    def _on_step(self):
        if self.num_timesteps - self._last_log >= self.log_every:
            self._last_log = self.num_timesteps
            if len(self.model.ep_info_buffer) > 0:
                rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                lengths = [ep["l"] for ep in self.model.ep_info_buffer]
                mean_r  = sum(rewards) / len(rewards)
                mean_l  = sum(lengths) / len(lengths)
                print(
                    f"  steps: {self.num_timesteps:>8,}  |  "
                    f"mean_reward: {mean_r:>8.2f}  |  "
                    f"mean_ep_len: {mean_l:>6.0f}",
                    flush=True
                )
                if mean_r > 80:
                    print("\n  *** Stage 1 solved (mean_reward > 80)! ***")
                    print(f"  Best model saved to: {BEST_MODEL}.zip")
                    print("  You can stop training and move to Stage 2.\n")
        return True


def make_env():
    def _init():
        env = HexapodStandEnv(render_mode=None)
        env = Monitor(env)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  CRAWL — Stage 1: Standing posture")
    print(f"  envs: {N_ENVS}  |  total steps: {TOTAL_TIMESTEPS:,}")
    print(f"  target: mean_reward > 80")
    print(f"  checkpoints -> {STAGE1_DIR}")
    print(f"{'='*55}\n")

    env_fns = [make_env() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    if args.resume and os.path.exists(NORM_STATS):
        vec_env = VecNormalize.load(NORM_STATS, vec_env)
        print("  Loaded normalizer stats.")
    else:
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0
        )

    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )
    eval_env.training   = False
    eval_env.norm_reward = False

    checkpoint_cb = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path=STAGE1_DIR,
        name_prefix="stage1_ppo",
        verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=STAGE1_DIR,
        log_path=LOG_DIR,
        eval_freq=max(5_000 // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=0
    )
    progress_cb = ProgressCallback(log_every=2_000)

    if args.resume and os.path.exists(LAST_MODEL + ".zip"):
        print(f"  Resuming from {LAST_MODEL}.zip ...")
        model = PPO.load(LAST_MODEL, env=vec_env, tensorboard_log=LOG_DIR)
    else:
        model = PPO(
            policy="MlpPolicy",
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
            policy_kwargs=dict(net_arch=[128, 128], log_std_init=-2.0)
        )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_cb, eval_cb, progress_cb],
            reset_num_timesteps=not args.resume,
            tb_log_name="stage1_ppo"
        )
    except KeyboardInterrupt:
        print("\n  Training interrupted — saving...")
    finally:
        model.save(LAST_MODEL)
        vec_env.save(NORM_STATS)
        import shutil
        best_src = os.path.join(STAGE1_DIR, "best_model.zip")
        if os.path.exists(best_src):
            shutil.copy(best_src, BEST_MODEL + ".zip")
            print(f"  Stage 1 best model -> {BEST_MODEL}.zip")
        print(f"  Last model -> {LAST_MODEL}.zip")
        print(f"  Normalizer -> {NORM_STATS}")
        vec_env.close()
        eval_env.close()

    print("\nStage 1 done. Next: python3 train_stage2.py")


if __name__ == "__main__":
    main()