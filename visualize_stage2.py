"""
CRAWL — Stage 2 Visualize: Watch the hexapod walk
==============================================
Shows live telemetry per step: x-displacement, body height, feet on ground, vx.

Usage:
  python3 visualize_stage2.py                      # watch best model
  python3 visualize_stage2.py --slow               # 0.5x speed
  python3 visualize_stage2.py --model path/to.zip  # specific checkpoint
"""

import os, argparse, time, glob
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from hexapod_env_stage2 import HexapodWalkEnv, ACT_DIM

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STAGE2_DIR = os.path.join(BASE_DIR, "checkpoints_stage2")
NORM_STATS = os.path.join(STAGE2_DIR, "vec_normalize_stage2.pkl")


def find_model():
    for name in ["best_model_stage2.zip", "best_model.zip"]:
        path = os.path.join(STAGE2_DIR, name)
        if os.path.exists(path):
            return path
    checkpoints = glob.glob(os.path.join(STAGE2_DIR, "stage2_ppo_*.zip"))
    return max(checkpoints, key=os.path.getctime) if checkpoints else None


def get_raw_env(vec_env):
    """Unwrap VecNormalize and DummyVecEnv to get the actual HexapodWalkEnv."""
    env = vec_env
    while hasattr(env, "venv"):
        env = env.venv
    if hasattr(env, "envs"):
        env = env.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps",  type=int,   default=5000)
    parser.add_argument("--slow",   action="store_true", help="0.5x speed")
    parser.add_argument("--fast",   action="store_true", help="no sleep")
    parser.add_argument("--model",  type=str,   default=None)
    args = parser.parse_args()

    path = args.model or find_model()
    if not path:
        print("No model found in checkpoints_stage2/. Run train_stage2.py first.")
        return
    path = path.replace(".zip", "")
    print(f"\nLoading: {os.path.basename(path)}.zip")

    vec_env = DummyVecEnv([lambda: HexapodWalkEnv(render_mode="human")])
    if os.path.exists(NORM_STATS):
        vec_env = VecNormalize.load(NORM_STATS, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
        print("Normalizer loaded.")
    else:
        print("Warning: no normalizer found — obs not normalized.")

    model = PPO.load(path, env=vec_env)

    sleep_time = 0.0 if args.fast else (2.0/50 if args.slow else 1.0/50)

    obs       = vec_env.reset()
    ep        = 1
    ep_reward = 0.0
    ep_steps  = 0
    start_x   = None
    raw_env   = get_raw_env(vec_env)

    print(f"\nWatching {args.steps} steps | Ctrl+C to stop")
    print(f"{'─'*72}")
    print(f"{'Step':>6}  {'X-dist':>8}  {'Height':>8}  {'Feet':>5}  "
          f"{'Vx':>7}  {'EpRew':>9}  {'Ep':>3}")
    print(f"{'─'*72}")

    try:
        camera_distance = 1.2
        camera_yaw = 45
        camera_pitch = -25
        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += float(reward[0])
            ep_steps  += 1
            
            keys = p.getKeyboardEvents()

            if ord("z") in keys and keys[ord("z")] & p.KEY_WAS_TRIGGERED:
                camera_distance = max(0.25, camera_distance - 0.15)

            if ord("x") in keys and keys[ord("x")] & p.KEY_WAS_TRIGGERED:
                camera_distance = min(4.0, camera_distance + 0.15)

            raw_env = get_raw_env(vec_env)
            pos, _ = p.getBasePositionAndOrientation(
                raw_env._robot,
                physicsClientId=raw_env._client,
            )

            p.resetDebugVisualizerCamera(
                cameraDistance=camera_distance,
                cameraYaw=camera_yaw,
                cameraPitch=camera_pitch,
                cameraTargetPosition=[pos[0], pos[1], 0.10],
                physicsClientId=raw_env._client,
            )
            # Telemetry from raw env
            try:
                pos, _ = p.getBasePositionAndOrientation(
                    raw_env._robot, physicsClientId=raw_env._client)
                lin_vel, _ = p.getBaseVelocity(
                    raw_env._robot, physicsClientId=raw_env._client)
                if start_x is None:
                    start_x = float(pos[0])
                x_dist = float(pos[0]) - start_x
                height = float(pos[2])
                vx     = float(lin_vel[0])
                n_feet = raw_env._get_foot_contacts()
            except Exception:
                x_dist = height = vx = 0.0
                n_feet = 0

            if step % 25 == 0:
                print(f"{step:>6}  {x_dist:>+8.3f}m  {height:>8.4f}m  "
                      f"{n_feet:>5}  {vx:>+7.3f}  {ep_reward:>9.1f}  {ep:>3}")

            if sleep_time:
                time.sleep(sleep_time)

            if done[0]:
                print(f"\n  ── Ep {ep} ended: steps={ep_steps}  "
                      f"reward={ep_reward:.1f}  x-travel={x_dist:+.3f}m ──\n")
                ep        += 1
                ep_reward  = 0.0
                ep_steps   = 0
                start_x    = None
                obs        = vec_env.reset()
                raw_env    = get_raw_env(vec_env)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        vec_env.close()
    print(f"\nEpisodes watched: {ep - 1}")


if __name__ == "__main__":
    main()