"""
HEXAPOD — Stage 3 Enjoy: Watch the hexapod reach a goal 
====================================================================

Usage:
  python3 enjoy_stage3.py                      # random goals, best model
  python3 enjoy_stage3.py --goal 1.5 0.0       # fixed goal at (1.5, 0.0)
  python3 enjoy_stage3.py --slow               # 0.5× real-time
  python3 enjoy_stage3.py --fast               # no sleep
  python3 enjoy_stage3.py --model path/to.zip  # specific checkpoint
  python3 enjoy_stage3.py --episodes 5         # run N episodes then exit
"""

import os
import argparse
import time
import glob

import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from train_stage3 import HexapodGoalEnvFixed, ACT_DIM

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STAGE3_DIR = os.path.join(BASE_DIR, "checkpoints_stage3")
NORM_STATS = os.path.join(STAGE3_DIR, "vec_normalize_stage3.pkl")


def find_model():
    for name in ["best_model_stage3.zip", "best_model.zip",
                 "last_model_stage3.zip"]:
        path = os.path.join(STAGE3_DIR, name)
        if os.path.exists(path):
            return path
    ckpts = glob.glob(os.path.join(STAGE3_DIR, "stage3_ppo_*.zip"))
    return max(ckpts, key=os.path.getctime) if ckpts else None


def get_raw_env(vec_env):
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
    parser.add_argument("--steps",    type=int,   default=10_000)
    parser.add_argument("--episodes", type=int,   default=None)
    parser.add_argument("--slow",     action="store_true")
    parser.add_argument("--fast",     action="store_true")
    parser.add_argument("--model",    type=str,   default=None)
    parser.add_argument("--goal",     type=float, nargs=2, default=None,
                        metavar=("X", "Y"))
    args = parser.parse_args()

    path = args.model or find_model()
    if not path:
        print("No model found in checkpoints_stage3/. Run train_stage3.py first.")
        return
    path = path.replace(".zip", "")
    print(f"\nLoading: {os.path.basename(path)}.zip")
    print(f"Navigation: STRAIGHT-LINE (direct goal vector, no A*)\n")

    goal_xy = args.goal

    def _make():
        env = HexapodGoalEnvFixed(
            render_mode="human",
            goal_xy=goal_xy)
        return Monitor(env)

    vec_env = DummyVecEnv([_make])

    if os.path.exists(NORM_STATS):
        try:
            vec_env = VecNormalize.load(NORM_STATS, vec_env)
            vec_env.training    = False
            vec_env.norm_reward = False
            print("Normalizer loaded.")
        except Exception as e:
            print(f"Normalizer mismatch ({e}) — fresh VecNormalize.")
            vec_env = VecNormalize(
                vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
            vec_env.training    = False
            vec_env.norm_reward = False
    else:
        print("No normalizer — fresh VecNormalize.")
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        vec_env.training    = False
        vec_env.norm_reward = False

    model   = PPO.load(path, env=vec_env)
    raw_env = get_raw_env(vec_env)

    if args.fast:
        sleep_time = 0.0
    elif args.slow:
        sleep_time = 2.0 / 50
    else:
        sleep_time = 1.0 / 50

    goal_desc = f"({goal_xy[0]:.2f}, {goal_xy[1]:.2f})" if goal_xy else "random"
    print(f"Goal mode   : {goal_desc}")
    print(f"Max steps   : {args.steps}")
    if args.episodes:
        print(f"Max episodes: {args.episodes}")
    print()
    print(f"{'─'*92}")
    print(f"{'Step':>6}  {'X':>7}  {'Y':>7}  {'GoalX':>7}  {'GoalY':>7}  "
          f"{'Dist':>7}  {'Height':>8}  {'Feet':>5}  {'EpRew':>9}  {'Status':>12}")
    print(f"{'─'*92}")

    obs          = vec_env.reset()
    ep           = 1
    ep_reward    = 0.0
    ep_steps     = 0
    ep_successes = 0

    try:
        for step in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            ep_reward += float(reward[0])
            ep_steps  += 1

            try:
                pos, _ = p.getBasePositionAndOrientation(
                    raw_env._robot, physicsClientId=raw_env._client)
                height    = float(pos[2])
                robot_xy  = np.array([pos[0], pos[1]], dtype=np.float32)
                goal_xy_t = raw_env._goal_xy
                dist      = float(np.linalg.norm(robot_xy - goal_xy_t))
                n_feet    = raw_env._get_foot_contacts()
            except Exception:
                pos = (0.0, 0.0, 0.0)
                dist = height = 0.0
                n_feet = 0
                goal_xy_t = np.zeros(2)

            raw_info = info[0] if isinstance(info, list) else info
            if isinstance(raw_info, dict) and raw_info.get("success", False):
                status = " REACHED"
            elif dist < 0.20:
                status = " almost"
            elif dist > 1.5:
                status = " en route"
            else:
                status = " closing"

            if step % 25 == 0:
                print(
                    f"{step:>6}  "
                    f"{pos[0]:>+7.3f}  {pos[1]:>+7.3f}  "
                    f"{goal_xy_t[0]:>+7.3f}  {goal_xy_t[1]:>+7.3f}  "
                    f"{dist:>7.3f}m  "
                    f"{height:>8.4f}m  "
                    f"{n_feet:>5}  "
                    f"{ep_reward:>9.1f}  "
                    f"{status:>12}")

            if sleep_time:
                time.sleep(sleep_time)

            if done[0]:
                succeeded    = isinstance(raw_info, dict) and \
                               raw_info.get("success", False)
                ep_successes += int(succeeded)
                outcome      = "SUCCESS ✓" if succeeded else "FAILED ✗"
                print(f"\n  ── Ep {ep:>3}  steps={ep_steps:>5}  "
                      f"reward={ep_reward:>8.1f}  dist={dist:.3f}m  "
                      f"{outcome} ──\n")

                if args.episodes and ep >= args.episodes:
                    print(f"  Reached episode limit ({args.episodes}).")
                    break

                ep        += 1
                ep_reward  = 0.0
                ep_steps   = 0
                obs        = vec_env.reset()
                raw_env    = get_raw_env(vec_env)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        vec_env.close()

    total_eps = ep - 1 + (1 if ep_steps > 0 else 0)
    sr = ep_successes / max(total_eps, 1) * 100
    print(f"\n{'='*50}")
    print(f"  Episodes     : {total_eps}")
    print(f"  Successes    : {ep_successes}")
    print(f"  Success rate : {sr:.1f}%")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()