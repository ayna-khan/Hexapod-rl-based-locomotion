"""
CRAWL — Stage 2 Training: Straight-line locomotion 
===============================================================

  - femur=-1.40 (legs DOWN, not up)
  - N_ENVS 4→8: more parallel experience speeds gait discovery
  - N_STEPS 4096→8192: captures full gait cycles in each rollout
  - ENT_COEF 0.01→0.02: more exploration to escape standing-still basin
  - net_arch [256,256]→[512,256,128]: capacity for gait coordination
  - Stop criterion: mean_reward > 200 AND mean_ep_len > 500

Usage:
  python3 train_stage2.py               # fresh training
  python3 train_stage2.py --resume      # continue from last checkpoint
  python3 train_stage2.py --timesteps 12000000  # longer run
"""

import os, shutil, argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from hexapod_env_stage2 import HexapodWalkEnv

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STAGE2_DIR = os.path.join(BASE_DIR, "checkpoints_stage2")
LOG_DIR    = os.path.join(BASE_DIR, "logs_stage2")
BEST_MODEL = os.path.join(STAGE2_DIR, "best_model_stage2")
LAST_MODEL = os.path.join(STAGE2_DIR, "last_model_stage2")
NORM_STATS = os.path.join(STAGE2_DIR, "vec_normalize_stage2.pkl")
os.makedirs(STAGE2_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 8_000_000
N_ENVS          = 8
N_STEPS         = 8192
BATCH_SIZE      = 512
N_EPOCHS        = 10
LEARNING_RATE   = 2e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_RANGE      = 0.2
ENT_COEF        = 0.02   # higher entropy → more exploration early on


class ProgressCallback(BaseCallback):
    """Logs progress and detects if bot is still stuck standing."""

    def __init__(self, log_every=16_000):
        super().__init__()
        self.log_every  = log_every
        self._last_log  = 0
        self._announced = False

    def _on_step(self):
        if self.num_timesteps - self._last_log < self.log_every:
            return True
        self._last_log = self.num_timesteps
        buf = self.model.ep_info_buffer
        if not buf:
            return True

        rewards = [ep["r"] for ep in buf]
        lengths = [ep["l"] for ep in buf]
        mean_r  = sum(rewards) / len(rewards)
        mean_l  = sum(lengths) / len(lengths)

        # Diagnose
        if mean_r < 30 and self.num_timesteps > 500_000:
            status = " LOW — may still be standing still"
        elif mean_r > 200 and mean_l > 500:
            status = "  WALKING"
            if not self._announced:
                self._announced = True
                print("\n  *** Stage 2 SUCCESS: bot is walking forward! ***\n")
        else:
            status = "→  learning..."

        print(
            f"  steps {self.num_timesteps:>9,}  |  "
            f"mean_r {mean_r:>8.2f}  |  "
            f"mean_len {mean_l:>6.0f}  |  {status}",
            flush=True)
        return True


def make_env():
    def _init():
        return Monitor(HexapodWalkEnv(render_mode=None))
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",    action="store_true")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  CRAWL — Stage 2: Locomotion (CORRECTED)")
    print(f"  Pose: femur=-1.40, tibia=+0.90  (feet DOWN, body UP)")
    print(f"  envs={N_ENVS}  steps/rollout={N_STEPS}  total={args.timesteps:,}")
    print(f"")
    print(f"  Reward sanity check:")
    print(f"    Standing still  ≈  0.1/step")
    print(f"    Walking 0.4m/s  ≈  4.5/step  (walking >> standing)")
    print(f"    Tripod gait     +0.3/step bonus")
    print(f"    All 6 feet down -0.25/step penalty (no more free lunch)")
    print(f"")
    print(f"  Stop when: mean_reward > 200 AND mean_ep_len > 500")
    print(f"{'='*60}\n")

    env_fns = [make_env() for _ in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)

    if args.resume and os.path.exists(NORM_STATS):
        vec_env = VecNormalize.load(NORM_STATS, vec_env)
        print("  Loaded normalizer stats.")
    else:
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env()])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training    = False
    eval_env.norm_reward = False

    checkpoint_cb = CheckpointCallback(
        save_freq=max(200_000 // N_ENVS, 1),
        save_path=STAGE2_DIR,
        name_prefix="stage2_ppo",
        verbose=1)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=STAGE2_DIR,
        log_path=LOG_DIR,
        eval_freq=max(40_000 // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=0)

    progress_cb = ProgressCallback(log_every=16_000)

    if args.resume and os.path.exists(LAST_MODEL + ".zip"):
        print(f"  Resuming from {LAST_MODEL}.zip ...")
        model = PPO.load(LAST_MODEL, env=vec_env, tensorboard_log=LOG_DIR)
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
            policy_kwargs=dict(net_arch=[512, 256, 128])
        )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_cb, eval_cb, progress_cb],
            reset_num_timesteps=not args.resume,
            tb_log_name="stage2_ppo")
    except KeyboardInterrupt:
        print("\n  Interrupted — saving...")
    finally:
        model.save(LAST_MODEL)
        vec_env.save(NORM_STATS)
        best_src = os.path.join(STAGE2_DIR, "best_model.zip")
        if os.path.exists(best_src):
            shutil.copy(best_src, BEST_MODEL + ".zip")
            print(f"  Best model  → {BEST_MODEL}.zip")
        print(f"  Last model  → {LAST_MODEL}.zip")
        vec_env.close()
        eval_env.close()

    print("\nDone. Run:  python3 enjoy_stage2.py")
    print("Monitor:    tensorboard --logdir logs_stage2")


if __name__ == "__main__":
    main()