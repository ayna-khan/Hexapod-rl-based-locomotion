"""
CRAWL — Stage 1: Crouched spawn → Stand up
============================================

  - SIT_POSE  (coxa=0, femur=-0.6458, tibia=+1.4522) -> foot_z = -0.012 rel. to base
  - STAND_POSE(coxa=0, femur=+0.45,   tibia=+1.05)   -> foot_z = -0.143 rel. to base

Mirror law 
  coxa_right  = -coxa_left
  femur_right =  femur_left
  tibia_right =  tibia_left

Leg index map (URDF order, 3 joints/leg: coxa, femur, tibia):
  leg1 = front-right   leg2 = front-left   leg3 = mid-left
  leg4 = rear-left      leg5 = rear-right   leg6 = mid-right

Mirrored pairs:  (left -> right)
  front: leg2 -> leg1
  mid:   leg3 -> leg6
  rear:  leg4 -> leg5
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "urdf", "hexapod_generated.urdf")

# ── joint limits (per-leg: coxa, femur, tibia) ──────────────────────────────
JOINT_LOWER_1LEG = np.array([-0.785, -1.571, -1.571], dtype=np.float32)
JOINT_UPPER_1LEG = np.array([ 0.785,  0.524,  1.571], dtype=np.float32)
JOINT_MID_1LEG   = (JOINT_LOWER_1LEG + JOINT_UPPER_1LEG) / 2.0
JOINT_RANGE_1LEG = (JOINT_UPPER_1LEG - JOINT_LOWER_1LEG) / 2.0

JOINT_LOWER = np.tile(JOINT_LOWER_1LEG, 6).astype(np.float32)
JOINT_UPPER = np.tile(JOINT_UPPER_1LEG, 6).astype(np.float32)

NUM_JOINTS         = 18
ACT_DIM            = 3         
OBS_DIM            = 48
SIM_HZ             = 240
CTRL_HZ            = 50
SIM_STEPS_PER_CTRL = SIM_HZ // CTRL_HZ
MAX_EPISODE_STEPS  = 500

# leg_id -> joint slice index in the 18-vector (leg_id is 1-indexed, in URDF build order)
def _slice(leg_id):
    i = leg_id - 1
    return slice(3 * i, 3 * i + 3)

# (master/left leg_id, slave/right leg_id) for front, mid, rear
MIRROR_PAIRS = [(2, 1), (3, 6), (4, 5)]

SIT_POSE_1LEG   = np.array([0.0, -0.6458, 1.4522], dtype=np.float32)
STAND_POSE_1LEG = np.array([0.0, 0.25, 1.15], dtype=np.float32)

def expand_mirrored(action_9):
    """9-dim [front(coxa,femur,tibia), mid(...), rear(...)] (LEFT side)
       -> full 18-dim joint target vector, mirrored onto the right side."""
    full = np.zeros(18, dtype=np.float32)
    groups = [action_9[0:3], action_9[3:6], action_9[6:9]]
    for (left_id, right_id), g in zip(MIRROR_PAIRS, groups):
        coxa, femur, tibia = g
        full[_slice(left_id)]  = [coxa,  femur, tibia]
        full[_slice(right_id)] = [-coxa, femur, tibia]
    return full

def build_full_pose(pose_1leg):
    """Same pose on every leg"""
    a9 = np.tile(pose_1leg, 3)
    return expand_mirrored(a9)

SIT_POSE   = build_full_pose(SIT_POSE_1LEG)     # 18-dim, all legs
STAND_POSE = build_full_pose(STAND_POSE_1LEG)   # 18-dim, all legs

INITIAL_BASE_HEIGHT = 0.012  
TARGET_HEIGHT       = 0.105    
MIN_HEIGHT          = 0.050
MAX_TILT            = 0.70

W_HEIGHT      = 12.0
W_ALIVE       =  2.0
W_TILT        = -8.0
W_VEL         = -2.0
W_ACTION_RATE = -0.05
W_FOOT        =  1.5
W_POSTURE     =  4.0
SERVO_FORCE   = 12.0


class HexapodStandEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": CTRL_HZ}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
        self._client      = None
        self._robot       = None
        self._joint_ids   = None
        self._step_count  = 0
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)

    def _build_joint_index(self):
        self._joint_ids = []
        for i in range(p.getNumJoints(self._robot, physicsClientId=self._client)):
            info = p.getJointInfo(self._robot, i, physicsClientId=self._client)
            if info[2] == p.JOINT_REVOLUTE:
                self._joint_ids.append(i)
        assert len(self._joint_ids) == NUM_JOINTS

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self._client is None:
            if self.render_mode == "human":
                self._client = p.connect(p.GUI)
                p.resetDebugVisualizerCamera(
                    0.7, 45, -25, [0, 0, 0.10], physicsClientId=self._client)
            else:
                self._client = p.connect(p.DIRECT)

        p.resetSimulation(physicsClientId=self._client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / SIM_HZ, physicsClientId=self._client)
        p.loadURDF("plane.urdf", physicsClientId=self._client)

        self._robot = p.loadURDF(
            URDF, [0, 0, INITIAL_BASE_HEIGHT],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False, physicsClientId=self._client)
        self._build_joint_index()

        for jid in self._joint_ids:
            p.changeDynamics(self._robot, jid,
                lateralFriction=2.5, spinningFriction=0.3,
                rollingFriction=0.01, frictionAnchor=1,
                physicsClientId=self._client)

        #(mirrored, symmetric)
        for idx, jid in enumerate(self._joint_ids):
            p.resetJointState(self._robot, jid, float(SIT_POSE[idx]), physicsClientId=self._client)
            p.setJointMotorControl2(
                self._robot, jid, p.POSITION_CONTROL,
                targetPosition=float(SIT_POSE[idx]),
                force=SERVO_FORCE, physicsClientId=self._client)

        for step_i in range(SIM_HZ):
            if step_i < 30:
                force = SERVO_FORCE * (step_i / 30.0)
                for idx, jid in enumerate(self._joint_ids):
                    p.setJointMotorControl2(
                        self._robot, jid, p.POSITION_CONTROL,
                        targetPosition=float(SIT_POSE[idx]),
                        force=force, physicsClientId=self._client)
            p.stepSimulation(physicsClientId=self._client)

        p.resetBaseVelocity(self._robot, [0, 0, 0], [0, 0, 0], physicsClientId=self._client)

        self._step_count  = 0
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        target_1leg = STAND_POSE_1LEG + action * np.array([0.0, 0.10, 0.10], dtype=np.float32)
        target_1leg[0] = 0.0 

        target_1leg = np.clip(target_1leg, JOINT_LOWER_1LEG, JOINT_UPPER_1LEG)
        target18 = build_full_pose(target_1leg)

        for idx, jid in enumerate(self._joint_ids):
            p.setJointMotorControl2(
                self._robot,
                jid,
                p.POSITION_CONTROL,
                targetPosition=float(target18[idx]),
                force=SERVO_FORCE,
                positionGain=0.35,
                velocityGain=0.35,
                physicsClientId=self._client,
            )

        for _ in range(SIM_STEPS_PER_CTRL):
            p.stepSimulation(physicsClientId=self._client)

        obs = self._get_obs()
        reward, terminated = self._compute_reward(action)
        self._step_count += 1
        truncated = self._step_count >= MAX_EPISODE_STEPS
        self._prev_action = action.copy()
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self._robot, physicsClientId=self._client)
        lin_vel, ang_vel = p.getBaseVelocity(self._robot, physicsClientId=self._client)
        euler = p.getEulerFromQuaternion(orn)
        jpos, jvel = [], []
        for jid in self._joint_ids:
            js = p.getJointState(self._robot, jid, physicsClientId=self._client)
            jpos.append(js[0]); jvel.append(js[1])
        return np.concatenate([
            lin_vel,
            ang_vel,
            [pos[2], euler[0], euler[1]],
            jpos,
            jvel,
            self._prev_action,
        ]).astype(np.float32)

    def _compute_reward(self, action):
        pos, orn = p.getBasePositionAndOrientation(self._robot, physicsClientId=self._client)
        lin_vel, ang_vel = p.getBaseVelocity(self._robot, physicsClientId=self._client)
        euler = p.getEulerFromQuaternion(orn)

        height = float(pos[2])
        roll = float(euler[0])
        pitch = float(euler[1])
        tilt = abs(roll) + abs(pitch)

        xy_speed = float(np.sqrt(lin_vel[0] ** 2 + lin_vel[1] ** 2))
        ang_speed = float(np.sqrt(ang_vel[0] ** 2 + ang_vel[1] ** 2 + ang_vel[2] ** 2))

        jpos = np.array([
            p.getJointState(self._robot, jid, physicsClientId=self._client)[0]
            for jid in self._joint_ids
        ], dtype=np.float32)

        posture_error = float(np.mean(np.square(jpos - STAND_POSE)))

        height_reward = np.exp(-35.0 * abs(height - TARGET_HEIGHT))
        posture_reward = np.exp(-8.0 * posture_error)

        action_rate = float(np.sum(np.square(action - self._prev_action)))

        reward = (
            10.0 * height_reward
            + 8.0 * posture_reward
            + 1.0
            - 10.0 * tilt
            - 8.0 * xy_speed
            - 2.0 * ang_speed
            - 0.10 * action_rate
        )

        terminated = bool(self._step_count > 30 and (height < MIN_HEIGHT or tilt > MAX_TILT))
        if terminated:
            reward -= 25.0

        return float(reward), terminated

    def close(self):
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
            self._client = None


if __name__ == "__main__":
    import time
    print("Sanity check — should settle in SIT pose (belly+feet on ground), no drifting.")
    env = HexapodStandEnv(render_mode="human")
    obs, _ = env.reset()
    base_pos, _ = p.getBasePositionAndOrientation(env._robot, physicsClientId=env._client)
    print(f"  Body COM z after settle = {base_pos[2]:.4f} m  (target ~0.012)")

    stand_action = np.zeros(3, dtype=np.float32)

    total = 0
    for i in range(1500):
        obs, reward, terminated, truncated, _ = env.step(stand_action)
        total += reward
        time.sleep(1 / CTRL_HZ)

        if i % 50 == 0:
            base_pos, orn = p.getBasePositionAndOrientation(env._robot, physicsClientId=env._client)
            lin_vel, ang_vel = p.getBaseVelocity(env._robot, physicsClientId=env._client)
            print(
                f"step={i}, z={base_pos[2]:.3f}, "
                f"xy_speed={(lin_vel[0]**2 + lin_vel[1]**2)**0.5:.4f}, "
                f"ang_speed={(ang_vel[0]**2 + ang_vel[1]**2 + ang_vel[2]**2)**0.5:.4f}, "
                f"reward={reward:.2f}"
            )

        if terminated or truncated:
            print(f"Ended at step {i}, reward={total:.1f}")
            break