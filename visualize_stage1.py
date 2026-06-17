"""
CRAWL — Watch the trained hexapod (Stage 1)

Usage:
  python3 enjoy.py
  python3 enjoy.py --steps 2000
  python3 enjoy.py --no-norm    # skip normalizer (debug)
"""

import os
import glob
import time
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_stage1")
NORM_STATS     = os.path.join(CHECKPOINT_DIR, "vec_normalize_stage1.pkl")


def find_latest_model():
    # 1. Final named model
    final = os.path.join(CHECKPOINT_DIR, "best_model_stage1.zip")
    if os.path.exists(final):
        return final, "best_model_stage1 (final)"
    # 2. Live best during training
    live = os.path.join(CHECKPOINT_DIR, "best_model.zip")
    if os.path.exists(live):
        return live, "best_model (live during training)"
    # 3. Latest periodic checkpoint
    ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "stage1_ppo_*.zip"))
    if ckpts:
        latest = max(ckpts, key=os.path.getctime)
        return latest, os.path.basename(latest)
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",   type=int,  default=1000)
    parser.add_argument("--no-norm", action="store_true")
    args = parser.parse_args()

    model_path, model_name = find_latest_model()
    if model_path is None:
        print(f"\n No model found in: {CHECKPOINT_DIR}")
        print(f"    Run first: python3 train_stage1.py")
        return

    print(f"\n  Loading: {model_name}")

    from hexapod_env_stage1 import HexapodStandEnv

    # Build the display env
    vec_env = DummyVecEnv([lambda: HexapodStandEnv(render_mode="human")])

    # ──  load normalizer ONTO the env, then load model ──────────────
    use_norm = not args.no_norm and os.path.exists(NORM_STATS)
    if use_norm:
        vec_env = VecNormalize.load(NORM_STATS, vec_env)
        vec_env.training    = False   # freeze running mean/var
        vec_env.norm_reward = False
        print("   Normalizer loaded and frozen.")
    else:
        if not args.no_norm:
            print("     No normalizer found — behaviour may be erratic.")

    model = PPO.load(model_path.replace(".zip", ""), env=vec_env)

    obs      = vec_env.reset()
    total_r  = 0.0
    episode  = 1

    print(f"\n  Watching '{model_name}'  |  Ctrl+C to stop")
    print("   Robot should start crouched (base on ground) then stand up.\n")

    try:
        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_r += float(reward[0])
            time.sleep(1.0 / 50)

            if done[0]:
                print(f"  Episode {episode:>3}  |  reward: {total_r:>9.2f}")
                episode += 1
                total_r = 0.0
                obs = vec_env.reset()

    except KeyboardInterrupt:
        print("\n  Stopped by user.")

    vec_env.close()
    print("\n  Done.")


if __name__ == "__main__":
    main()