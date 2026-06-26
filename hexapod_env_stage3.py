"""
HEXAPOD — Stage 3: Goal Reaching
=================================
The hexapod must navigate from its spawn position to a goal marker
placed randomly in the arena.

Key design decisions:
  ─ A* path planner computes a 2-D waypoint sequence from spawn → goal
  ─ The agent receives: all Stage-2 obs + relative_goal_vec (2D) +
    distance_to_goal (1D) + heading_error (1D) = OBS_DIM 66
  ─ Action space identical to Stage 2 (18 joint deltas)
  ─ Reward shaped around:
      • Progress toward goal  (dense, potential-based)
      • Reaching the goal     (large sparse bonus)
      • Height / tilt / alive (copied from Stage 2 so gait is preserved)
      • Energy penalty
  ─ Episode terminates when:
      • Goal reached (distance < GOAL_RADIUS)
      • Body falls (height < MIN_HEIGHT or tilt > MAX_TILT) after 100 grace steps
      • MAX_EPISODE_STEPS exceeded

A* is run on a 2-D occupancy grid; for the flat arena it is trivially
free space but the planner is already present for future obstacle
extension.  The waypoint list is used to compute a "local subgoal"
vector that helps the agent avoid getting confused by large goal
distances.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import hexapod_env_stage2 as stage2

from hexapod_env_stage2 import (
    URDF, SPAWN_Z, TARGET_HEIGHT,
    SIM_HZ, CTRL_HZ, SIM_STEPS_PER_CTRL, NUM_JOINTS, SERVO_FORCE, MIN_HEIGHT, MAX_TILT,
)

ACT_DIM = 9
OBS_DIM = 57          # Stage2's 53 + goal_dir(2) + dist_norm(1) + heading_err(1)
MAX_EPISODE_STEPS = 2000

GOAL_RADIUS   = 0.15
GOAL_MIN_DIST = 0.8
GOAL_MAX_DIST = 3.0
GRACE_STEPS   = 150

W_VEL_TOWARD  =  5.0
W_PROGRESS    = 15.0
W_GOAL        = 200.0
W_HEIGHT      =  2.0
W_TILT        = -2.0
W_ALIVE       =  0.1
W_ENERGY      = -0.001
W_ACTION_RATE = -0.01
W_STALL       = -2.0
STALL_STEPS   = 20
STALL_THRESH  = 0.005
W_HEADING     = 1.5

def _slice(leg_id):
    i = leg_id - 1
    return slice(3 * i, 3 * i + 3)

def expand_goal_steering_targets(action9, heading_err, step_count):
    """
    Stage 3 walking target with goal steering.
    heading_err:
      positive = goal is to the left
      negative = goal is to the right
    """
    action9 = np.clip(action9, -1.0, 1.0).astype(np.float32)

    full = np.zeros(18, dtype=np.float32)

    phase = 2.0 * np.pi * stage2.GAIT_FREQ * (step_count / CTRL_HZ)

    stand_coxa = 0.0
    stand_femur = 0.35
    stand_tibia = 1.05

    base_coxa_amp = -0.38
    femur_lift_amp = 0.34
    tibia_lift_amp = 0.22

    # Clamp steering so it does not flip the robot.
    steer = float(np.clip(heading_err / 1.2, -1.0, 1.0))

    # If goal is left, right side takes bigger steps.
    left_scale = np.clip(1.0 - 0.45 * steer, 0.45, 1.55)
    right_scale = np.clip(1.0 + 0.45 * steer, 0.45, 1.55)

    # left leg IDs and corresponding right IDs
    pairs = [
        (2, 1, phase),             # front
        (3, 6, phase + np.pi),     # middle
        (4, 5, phase),             # rear
    ]

    groups = [action9[0:3], action9[3:6], action9[6:9]]

    for (left_id, right_id, ph), g in zip(pairs, groups):
        swing = np.sin(ph)
        lift = max(0.0, np.cos(ph))

        right_swing = -swing
        right_lift = max(0.0, -np.cos(ph))

        coxa_res = 0.06 * float(g[0])
        femur_res = 0.06 * float(g[1])
        tibia_res = 0.06 * float(g[2])

        left_coxa = stand_coxa + left_scale * base_coxa_amp * swing + coxa_res
        left_femur = stand_femur - femur_lift_amp * lift + femur_res
        left_tibia = stand_tibia - tibia_lift_amp * lift + tibia_res

        right_coxa = -(stand_coxa + right_scale * base_coxa_amp * right_swing + coxa_res)
        right_femur = stand_femur - femur_lift_amp * right_lift + femur_res
        right_tibia = stand_tibia - tibia_lift_amp * right_lift + tibia_res

        full[_slice(left_id)] = [left_coxa, left_femur, left_tibia]
        full[_slice(right_id)] = [right_coxa, right_femur, right_tibia]

    return np.clip(full, stage2.JOINT_LOWER, stage2.JOINT_UPPER).astype(np.float32)


class HexapodGoalEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": CTRL_HZ}

    def __init__(self, render_mode=None, goal_xy=None, max_goal_dist=GOAL_MAX_DIST):
        super().__init__()
        self.render_mode = render_mode
        self.fixed_goal = goal_xy
        self._max_goal_dist = float(max_goal_dist)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
        self._client = None
        self._robot = None
        self._goal_body = None
        self._joint_ids = []
        self._step_count = 0
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        self._goal_xy = np.zeros(2, dtype=np.float32)
        self._prev_dist = 0.0
        self._stall_buf = []

    def set_max_goal_dist(self, d):
        self._max_goal_dist = float(d)

    def _build_joint_index(self):
        self._joint_ids = []
        for i in range(p.getNumJoints(self._robot, physicsClientId=self._client)):
            info = p.getJointInfo(self._robot, i, physicsClientId=self._client)
            if info[2] == p.JOINT_REVOLUTE:
                self._joint_ids.append(i)
        assert len(self._joint_ids) == NUM_JOINTS

    def _spawn_goal_marker(self, gx, gy):
        if self._goal_body is not None:
            try:
                p.removeBody(self._goal_body, physicsClientId=self._client)
            except Exception:
                pass
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 0.8],
                                   physicsClientId=self._client)
        self._goal_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                              basePosition=[gx, gy, 0.05], physicsClientId=self._client)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._client is None:
            if self.render_mode == "human":
                self._client = p.connect(p.GUI)
                p.resetDebugVisualizerCamera(2.0, 45, -30, [0, 0, 0.10], physicsClientId=self._client)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._client)
            else:
                self._client = p.connect(p.DIRECT)

        p.resetSimulation(physicsClientId=self._client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / SIM_HZ, physicsClientId=self._client)
        p.loadURDF("plane.urdf", physicsClientId=self._client)

        self._robot = p.loadURDF(URDF, [0, 0, SPAWN_Z], p.getQuaternionFromEuler([0, 0, 0]),
                                  useFixedBase=False, physicsClientId=self._client)
        self._build_joint_index()
        self._goal_body = None

        for link_idx in range(-1, p.getNumJoints(self._robot, physicsClientId=self._client)):
            p.changeDynamics(self._robot, link_idx, lateralFriction=2.5, spinningFriction=0.3,
                              rollingFriction=0.01, frictionAnchor=1, physicsClientId=self._client)

        stage2.CURRENT_STEP_FRAC = 0.0
        neutral18 = stage2.expand_mirrored_targets(np.zeros(9, dtype=np.float32))
        for idx, jid in enumerate(self._joint_ids):
            p.resetJointState(self._robot, jid, float(neutral18[idx]), physicsClientId=self._client)
            p.setJointMotorControl2(self._robot, jid, p.POSITION_CONTROL,
                targetPosition=float(neutral18[idx]), force=SERVO_FORCE, physicsClientId=self._client)
        for _ in range(20):
            p.stepSimulation(physicsClientId=self._client)
        p.resetBaseVelocity(self._robot, [0, 0, 0], [0, 0, 0], physicsClientId=self._client)

        if self.fixed_goal is not None:
            self._goal_xy = np.array(self.fixed_goal, dtype=np.float32)
        else:
            rng = self.np_random
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(GOAL_MIN_DIST, self._max_goal_dist)
            self._goal_xy = np.array([dist * np.cos(angle), dist * np.sin(angle)], dtype=np.float32)

        self._spawn_goal_marker(*self._goal_xy)
        if self.render_mode == "human":
            try:
                p.removeAllUserDebugItems(physicsClientId=self._client)
                p.addUserDebugLine([0, 0, 0.02], [float(self._goal_xy[0]), float(self._goal_xy[1]), 0.02],
                                    lineColorRGB=[0, 1, 0], lineWidth=3.0, physicsClientId=self._client)
            except Exception:
                pass

        self._step_count = 0
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        pos, _ = p.getBasePositionAndOrientation(self._robot, physicsClientId=self._client)
        self._prev_dist = float(np.linalg.norm(np.array([pos[0], pos[1]]) - self._goal_xy))
        self._stall_buf = [self._prev_dist] * STALL_STEPS

        return self._get_obs(), {"goal_xy": self._goal_xy.tolist()}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        pos, orn = p.getBasePositionAndOrientation(
            self._robot,
            physicsClientId=self._client,
        )
        yaw = p.getEulerFromQuaternion(orn)[2]

        goal_angle = float(np.arctan2(
            self._goal_xy[1] - pos[1],
            self._goal_xy[0] - pos[0],
        ))

        heading_err = float(np.arctan2(
            np.sin(goal_angle - yaw),
            np.cos(goal_angle - yaw),
        ))

        target18 = expand_goal_steering_targets(
            action,
            heading_err,
            self._step_count,
        )
        for idx, jid in enumerate(self._joint_ids):
            p.setJointMotorControl2(self._robot, jid, p.POSITION_CONTROL,
                targetPosition=float(target18[idx]), force=SERVO_FORCE, physicsClientId=self._client)
        for _ in range(SIM_STEPS_PER_CTRL):
            p.stepSimulation(physicsClientId=self._client)

        obs = self._get_obs()
        reward, terminated, info = self._compute_reward(action)
        self._step_count += 1
        truncated = self._step_count >= MAX_EPISODE_STEPS
        self._prev_action = action.copy()
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self._robot, physicsClientId=self._client)
        lin_vel, ang_vel = p.getBaseVelocity(self._robot, physicsClientId=self._client)
        euler = p.getEulerFromQuaternion(orn)
        jpos, jvel = [], []
        for jid in self._joint_ids:
            js = p.getJointState(self._robot, jid, physicsClientId=self._client)
            jpos.append(js[0]); jvel.append(js[1])

        xy = np.array([pos[0], pos[1]], dtype=np.float32)
        rel_goal = self._goal_xy - xy
        dist_goal = float(np.linalg.norm(rel_goal))
        goal_dir = rel_goal / dist_goal if dist_goal > 1e-6 else np.zeros(2, dtype=np.float32)
        dist_norm = float(np.clip(dist_goal / self._max_goal_dist, 0.0, 1.0))

        yaw = float(euler[2])
        goal_angle = float(np.arctan2(self._goal_xy[1] - pos[1], self._goal_xy[0] - pos[0]))
        heading_err = float(np.arctan2(np.sin(goal_angle - yaw), np.cos(goal_angle - yaw)))

        return np.concatenate([
            lin_vel, ang_vel, [euler[0], euler[1]], jpos, jvel, self._prev_action,
            goal_dir, [dist_norm], [heading_err / np.pi],
        ]).astype(np.float32)

    def _compute_reward(self, action):
        pos, orn = p.getBasePositionAndOrientation(self._robot, physicsClientId=self._client)
        lin_vel, _ = p.getBaseVelocity(self._robot, physicsClientId=self._client)
        euler = p.getEulerFromQuaternion(orn)

        xy = np.array([pos[0], pos[1]], dtype=np.float32)
        height = float(pos[2])
        tilt = float(abs(euler[0]) + abs(euler[1]))
        dist_goal = float(np.linalg.norm(xy - self._goal_xy))

        progress = self._prev_dist - dist_goal
        self._prev_dist = dist_goal

        self._stall_buf.pop(0)
        self._stall_buf.append(dist_goal)
        stall_penalty = W_STALL if (self._stall_buf[0] - self._stall_buf[-1]) < STALL_THRESH else 0.0

        rel_goal = self._goal_xy - xy
        goal_unit = rel_goal / dist_goal if dist_goal > 1e-6 else np.zeros(2, dtype=np.float32)
        vel_toward = float(np.dot([lin_vel[0], lin_vel[1]], goal_unit))

        yaw = float(euler[2])
        goal_angle = float(np.arctan2(
            self._goal_xy[1] - pos[1],
            self._goal_xy[0] - pos[0],
        ))
        heading_err = float(np.arctan2(
            np.sin(goal_angle - yaw),
            np.cos(goal_angle - yaw),
        ))
        heading_reward = float(np.cos(heading_err))

        height_rew = float(np.exp(-20.0 * abs(height - TARGET_HEIGHT)))
        torques = [p.getJointState(self._robot, jid, physicsClientId=self._client)[3] for jid in self._joint_ids]
        energy = float(np.sum(np.abs(torques)))
        action_rate = float(np.sum(np.square(action - self._prev_action)))

        reward = (
            W_VEL_TOWARD * vel_toward + W_PROGRESS * progress + stall_penalty
            + W_HEIGHT * height_rew + W_TILT * tilt + W_ALIVE
            + W_ENERGY * energy + W_ACTION_RATE * action_rate
            + W_HEADING * heading_reward
        )

        terminated = False
        info = {"dist_to_goal": dist_goal, "success": False}
        if dist_goal < GOAL_RADIUS:
            reward += W_GOAL
            terminated = True
            info["success"] = True
        elif self._step_count > GRACE_STEPS and (height < MIN_HEIGHT or tilt > MAX_TILT):
            reward -= 10.0
            terminated = True

        return float(reward), terminated, info

    def _get_foot_contacts(self):
        contacts = p.getContactPoints(self._robot, physicsClientId=self._client)
        feet = set()
        if contacts:
            for c in contacts:
                if c[2] != self._robot:
                    feet.add(c[3])
        return len(feet)

    def render(self):
        if self.render_mode == "rgb_array":
            w, h = 640, 480
            pos, _ = p.getBasePositionAndOrientation(self._robot, physicsClientId=self._client)
            view = p.computeViewMatrixFromYawPitchRoll([pos[0], pos[1], 0.10], 2.0, 45, -30, 0, 2, physicsClientId=self._client)
            proj = p.computeProjectionMatrixFOV(60, w / h, 0.01, 100, physicsClientId=self._client)
            _, _, rgb, _, _ = p.getCameraImage(w, h, view, proj, physicsClientId=self._client)
            return np.array(rgb, dtype=np.uint8)[:, :, :3]

    def close(self):
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
            self._client = None


if __name__ == "__main__":
    import time
    env = HexapodGoalEnv(render_mode="human", goal_xy=[1.5, 0.0])
    obs, info = env.reset()
    print(f"Obs shape: {obs.shape} (expect {OBS_DIM})  Goal: {info['goal_xy']}")
    for i in range(1500):
        obs, r, term, trunc, info = env.step(np.zeros(9, dtype=np.float32))
        if i % 50 == 0:
            print(f"step={i}, dist={info['dist_to_goal']:.3f}, reward={r:.2f}")
        time.sleep(1 / CTRL_HZ)
        if term or trunc:
            print(f"Ended step {i}, dist={info['dist_to_goal']:.3f}")
            break
    env.close()