"""
CRAWL — Stage 2: Locomotion with delta joint actions
=====================================================
Key design decisions:
  - Spawn at EXACT standing height (0.1508m) — confirmed stable, no bounce
  - RL outputs joint angle DELTAS 
  - Small action scale per joint type prevents violent launches
  - Terminate only after 100 grace steps
  - Confirmed pose: femur=0.52, tibia=1.00 → 0.1508m, 6 feet
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

NUM_JOINTS         = 18
OBS_DIM            = 62
ACT_DIM            = 18
SIM_HZ             = 240
CTRL_HZ            = 50
SIM_STEPS_PER_CTRL = SIM_HZ // CTRL_HZ
MAX_EPISODE_STEPS  = 1000
SERVO_FORCE        = 25.0

# CONFIRMED: femur=0.52, tibia=1.00 → height=0.1508m, 6 feet 
STAND_POSE    = [0.0, 0.52, 1.00] * 6
TARGET_HEIGHT = 0.1508
SPAWN_Z       = 0.1508

MIN_HEIGHT    = 0.08
MAX_TILT      = 0.60

# Per-joint action scale (radians per step)
# coxa can swing more, femur/tibia smaller to keep stable
ACTION_SCALE  = np.array([0.25, 0.15, 0.20] * 6, dtype=np.float32)

W_FORWARD     =  5.0
W_LATERAL     = -1.0
W_HEIGHT      =  2.0
W_TILT        = -2.0
W_ALIVE       =  0.2
W_ENERGY      = -0.002
W_ACTION_RATE = -0.005
W_FOOT        =  0.2


class HexapodWalkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": CTRL_HZ}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode      = render_mode
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space      = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
        self._client           = None
        self._robot            = None
        self._joint_ids        = None
        self._step_count       = 0
        self._prev_action      = np.zeros(ACT_DIM, dtype=np.float32)
        self._current_joints   = np.array(STAND_POSE, dtype=np.float32)

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
                    1.2, 45, -25, [0, 0, 0.15],
                    physicsClientId=self._client)
                p.configureDebugVisualizer(
                    p.COV_ENABLE_GUI, 0, physicsClientId=self._client)
            else:
                self._client = p.connect(p.DIRECT)

        p.resetSimulation(physicsClientId=self._client)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / SIM_HZ, physicsClientId=self._client)
        p.loadURDF("plane.urdf", physicsClientId=self._client)

        # Spawn at EXACT confirmed standing height
        self._robot = p.loadURDF(
            URDF, [0, 0, SPAWN_Z],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False,
            physicsClientId=self._client)
        self._build_joint_index()

        for link_idx in range(-1, p.getNumJoints(
                self._robot, physicsClientId=self._client)):
            p.changeDynamics(self._robot, link_idx,
                lateralFriction=1.5, spinningFriction=0.1,
                rollingFriction=0.01, physicsClientId=self._client)

        # Apply confirmed standing pose
        for idx, jid in enumerate(self._joint_ids):
            p.resetJointState(
                self._robot, jid, STAND_POSE[idx],
                physicsClientId=self._client)
            p.setJointMotorControl2(
                self._robot, jid, p.POSITION_CONTROL,
                targetPosition=STAND_POSE[idx],
                force=SERVO_FORCE, physicsClientId=self._client)

        # 10 steps only — just enough for contacts to register
        for _ in range(10):
            p.stepSimulation(physicsClientId=self._client)

        # Zero velocity — clean start
        p.resetBaseVelocity(
            self._robot, [0, 0, 0], [0, 0, 0],
            physicsClientId=self._client)

        self._step_count     = 0
        self._prev_action    = np.zeros(ACT_DIM, dtype=np.float32)
        self._current_joints = np.array(STAND_POSE, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action):
        # DELTA actions: add small change to current joint positions
        delta = action * ACTION_SCALE
        self._current_joints = np.clip(
            self._current_joints + delta,
            JOINT_LOWER, JOINT_UPPER)

        for idx, jid in enumerate(self._joint_ids):
            p.setJointMotorControl2(
                self._robot, jid, p.POSITION_CONTROL,
                targetPosition=float(self._current_joints[idx]),
                force=SERVO_FORCE, physicsClientId=self._client)

        for _ in range(SIM_STEPS_PER_CTRL):
            p.stepSimulation(physicsClientId=self._client)

        obs               = self._get_obs()
        reward, terminated = self._compute_reward(action)
        self._step_count  += 1
        truncated          = self._step_count >= MAX_EPISODE_STEPS
        self._prev_action  = action.copy()
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        pos, orn         = p.getBasePositionAndOrientation(
            self._robot, physicsClientId=self._client)
        lin_vel, ang_vel = p.getBaseVelocity(
            self._robot, physicsClientId=self._client)
        euler            = p.getEulerFromQuaternion(orn)
        jpos, jvel       = [], []
        for jid in self._joint_ids:
            js = p.getJointState(self._robot, jid, physicsClientId=self._client)
            jpos.append(js[0]); jvel.append(js[1])
        return np.concatenate([
            lin_vel, ang_vel, [euler[0], euler[1]],
            jpos, jvel, self._prev_action
        ]).astype(np.float32)

    def _compute_reward(self, action):
        pos, orn   = p.getBasePositionAndOrientation(
            self._robot, physicsClientId=self._client)
        lin_vel, _ = p.getBaseVelocity(
            self._robot, physicsClientId=self._client)
        euler      = p.getEulerFromQuaternion(orn)

        height = pos[2]
        tilt   = abs(euler[0]) + abs(euler[1])
        vx     = lin_vel[0]
        vy     = lin_vel[1]

        height_rew  = np.exp(-20.0 * abs(height - TARGET_HEIGHT))
        forward_rew = max(0.0, vx)   # only reward forward motion

        torques = [p.getJointState(self._robot, jid,
                    physicsClientId=self._client)[3]
                   for jid in self._joint_ids]
        energy  = float(np.sum(np.abs(torques)))

        contacts = p.getContactPoints(self._robot, physicsClientId=self._client)
        ground_feet = set()
        if contacts:
            for c in contacts:
                if c[2] != self._robot:
                    ground_feet.add(c[3])
        n_feet = len(ground_feet)

        action_rate = float(np.sum(np.square(action - self._prev_action)))

        reward = (
            W_FORWARD     * forward_rew +
            W_LATERAL     * abs(vy) +
            W_HEIGHT      * height_rew +
            W_TILT        * tilt +
            W_ALIVE +
            W_ENERGY      * energy +
            W_ACTION_RATE * action_rate +
            W_FOOT        * n_feet
        )

        terminated = bool(
            self._step_count > 100 and
            (height < MIN_HEIGHT or tilt > MAX_TILT))
        if terminated:
            reward -= 10.0
        return float(reward), terminated

    def render(self):
        if self.render_mode == "rgb_array":
            w, h = 640, 480
            pos, _ = p.getBasePositionAndOrientation(
                self._robot, physicsClientId=self._client)
            view = p.computeViewMatrixFromYawPitchRoll(
                [pos[0], pos[1], 0.15], 1.2, 45, -25, 0, 2,
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
    print("Sanity check — zero actions should hold pose for 500 steps...")
    env = HexapodWalkEnv(render_mode="human")
    obs, _ = env.reset()
    pos, _ = p.getBasePositionAndOrientation(
        env._robot, physicsClientId=env._client)
    print(f"  Height: {pos[2]:.4f}m  (must be ~0.1508)")

    total = 0
    for i in range(500):
        obs, reward, terminated, truncated, _ = env.step(np.zeros(ACT_DIM))
        total += reward
        time.sleep(1 / CTRL_HZ)
        if terminated or truncated:
            print(f"  Ended step {i}  reward={total:.1f}")
            break
    else:
        print(f"  PASS — 500 steps stable, reward={total:.1f}")
    env.close()