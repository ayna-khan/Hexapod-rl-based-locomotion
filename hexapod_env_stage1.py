"""
CRAWL — Stage 1: Crouched spawn → Stand up
============================================
CROUCH pose calculated from URDF geometry:
  base_link box height = 0.024 m → COM at z=0.012 when bottom face on ground
  femur = -0.6458 rad, tibia = +1.4522 rad → foot_z = 0.000 m  
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "urdf", "hexapod_generated.urdf")

JOINT_LOWER = np.array([-0.785, -1.571, -1.571] * 6, dtype=np.float32)
JOINT_UPPER = np.array([ 0.785,  0.524,  1.571] * 6, dtype=np.float32)
JOINT_MID   = (JOINT_LOWER + JOINT_UPPER) / 2.0
JOINT_RANGE = (JOINT_UPPER - JOINT_LOWER) / 2.0

NUM_JOINTS         = 18
OBS_DIM            = 62
ACT_DIM            = 18
SIM_HZ             = 240
CTRL_HZ            = 50
SIM_STEPS_PER_CTRL = SIM_HZ // CTRL_HZ
MAX_EPISODE_STEPS  = 500

# ── POSES ──────────────────────────────────────────────────────────────────────
# CROUCH: base bottom face on ground, feet also on ground.
# Geometry (URDF): base half-h=0.012, femur=0.10m, tibia=0.10m
#   body_z = FEMUR*sin(-0.6458) + TIBIA*sin(-0.6458+1.4522) = 0.012 
#   foot_z = 0.000 
CROUCH_POSE = [0.0, -0.6458, 1.4522] * 6

# STAND: legs push body up to a stable height
STAND_POSE  = [0.0, -0.40, 0.90] * 6

# Spawn COM exactly at base half-height so bottom face touches ground
INITIAL_BASE_HEIGHT = 0.012   # metres

# ── HEIGHT TARGETS ─────────────────────────────────────────────────────────────
TARGET_HEIGHT = 0.1508    # what we reward agent for standing at
MIN_HEIGHT    = 0.030   # below this (after grace period) → fallen
MAX_TILT      = 0.40    # radians

# ── REWARD WEIGHTS ─────────────────────────────────────────────────────────────
W_HEIGHT      = 12.0
W_ALIVE       =  2.0
W_TILT        = -8.0
W_VEL         = -2.0
W_ACTION_RATE = -0.05
W_FOOT        =  1.5
W_POSTURE     =  4.0
SERVO_FORCE   = 25.0


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
                    0.7, 45, -25, [0, 0, 0.15],
                    physicsClientId=self._client)
            else:
                self._client = p.connect(p.DIRECT)

        p.resetSimulation(physicsClientId=self._client)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / SIM_HZ, physicsClientId=self._client)
        p.loadURDF("plane.urdf", physicsClientId=self._client)

        # Spawn upright with base bottom touching ground
        self._robot = p.loadURDF(
            URDF,
            [0, 0, INITIAL_BASE_HEIGHT],
            p.getQuaternionFromEuler([0, 0, 0]),  # always upright
            useFixedBase=False,
            physicsClientId=self._client)
        self._build_joint_index()

        for jid in self._joint_ids:
            p.changeDynamics(self._robot, jid,
                lateralFriction=1.2, spinningFriction=0.1,
                rollingFriction=0.01, physicsClientId=self._client)

        # Apply crouch — body on ground, feet on ground
        for idx, jid in enumerate(self._joint_ids):
            p.resetJointState(self._robot, jid, CROUCH_POSE[idx],
                              physicsClientId=self._client)
            p.setJointMotorControl2(
                self._robot, jid, p.POSITION_CONTROL,
                targetPosition=CROUCH_POSE[idx],
                force=SERVO_FORCE, physicsClientId=self._client)

        # Let everything settle physically for 1 second
        for _ in range(SIM_HZ):
            p.stepSimulation(physicsClientId=self._client)

        # Kill residual velocity
        p.resetBaseVelocity(self._robot, [0, 0, 0], [0, 0, 0],
                            physicsClientId=self._client)

        self._step_count  = 0
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action):
        target = np.clip(
            JOINT_MID + action * JOINT_RANGE,
            JOINT_LOWER, JOINT_UPPER)
        for idx, jid in enumerate(self._joint_ids):
            p.setJointMotorControl2(
                self._robot, jid, p.POSITION_CONTROL,
                targetPosition=float(target[idx]),
                force=SERVO_FORCE, physicsClientId=self._client)
        for _ in range(SIM_STEPS_PER_CTRL):
            p.stepSimulation(physicsClientId=self._client)
        obs = self._get_obs()
        reward, terminated = self._compute_reward(action)
        self._step_count += 1
        truncated = self._step_count >= MAX_EPISODE_STEPS
        self._prev_action = action.copy()
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(
            self._robot, physicsClientId=self._client)
        lin_vel, ang_vel = p.getBaseVelocity(
            self._robot, physicsClientId=self._client)
        euler = p.getEulerFromQuaternion(orn)
        jpos, jvel = [], []
        for jid in self._joint_ids:
            js = p.getJointState(self._robot, jid,
                                 physicsClientId=self._client)
            jpos.append(js[0])
            jvel.append(js[1])
        return np.concatenate([
            lin_vel, ang_vel,
            [euler[0], euler[1]],
            jpos, jvel,
            self._prev_action
        ]).astype(np.float32)

    def _compute_reward(self, action):
        pos, orn = p.getBasePositionAndOrientation(
            self._robot, physicsClientId=self._client)
        lin_vel, _ = p.getBaseVelocity(
            self._robot, physicsClientId=self._client)
        euler = p.getEulerFromQuaternion(orn)
        height = pos[2]
        tilt   = abs(euler[0]) + abs(euler[1])
        speed  = float(np.sqrt(lin_vel[0]**2 + lin_vel[1]**2))

        jpos = np.array([
            p.getJointState(self._robot, jid,
                            physicsClientId=self._client)[0]
            for jid in self._joint_ids
        ])
        posture_error = np.mean(np.abs(jpos - np.array(STAND_POSE)))

        height_rew  = np.exp(-15.0 * abs(height - TARGET_HEIGHT))
        contacts    = p.getContactPoints(self._robot,
                                         physicsClientId=self._client)
        n_feet      = len(set(c[3] for c in contacts)) if contacts else 0
        action_rate = float(np.sum(np.square(action - self._prev_action)))

        reward = (
            W_HEIGHT      * height_rew
            + W_ALIVE
            + W_TILT      * tilt
            + W_VEL       * speed
            + W_ACTION_RATE * action_rate
            + W_FOOT      * n_feet
            + W_POSTURE   * (1.0 - np.clip(posture_error * 2.0, 0, 1.0))
        )

        terminated = bool(
            self._step_count > 20
            and (height < MIN_HEIGHT or tilt > MAX_TILT)
        )
        if terminated:
            reward -= 20.0
        return float(reward), terminated

    def render(self):
        if self.render_mode == "rgb_array":
            w, h = 640, 480
            view = p.computeViewMatrixFromYawPitchRoll(
                [0, 0, 0.15], 0.7, 45, -25, 0, 2,
                physicsClientId=self._client)
            proj = p.computeProjectionMatrixFOV(
                60, w/h, 0.01, 100, physicsClientId=self._client)
            _, _, rgb, _, _ = p.getCameraImage(
                w, h, view, proj, physicsClientId=self._client)
            return np.array(rgb, dtype=np.uint8)[:, :, :3]

    def close(self):
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
            self._client = None


if __name__ == "__main__":
    import time
    print("Testing — body should rest on ground in crouch, then lift to stand...")
    env = HexapodStandEnv(render_mode="human")
    obs, _ = env.reset()

    base_pos, _ = p.getBasePositionAndOrientation(
        env._robot, physicsClientId=env._client)
    print(f"  Body COM z after settle = {base_pos[2]:.4f} m  (target ~0.012)")

    stand_action = np.clip(
        (np.array(STAND_POSE, dtype=np.float32) - JOINT_MID) / JOINT_RANGE,
        -1, 1)
    total = 0
    for i in range(500):
        obs, reward, terminated, truncated, _ = env.step(stand_action)
        total += reward
        time.sleep(1 / CTRL_HZ)
        if terminated or truncated:
            print(f"  Ended at step {i}  reward={total:.1f}")
            break
    print(f"  Total reward = {total:.1f}")
    env.close()