"""
CRAWL — Stage 2: Locomotion via IK-driven feet + master-slave mirroring.
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

FEMUR_LEN, TIBIA_LEN = 0.10, 0.10
JOINT_LOWER_1LEG = np.array([-0.785, -1.571, -1.571], dtype=np.float32)
JOINT_UPPER_1LEG = np.array([ 0.785,  0.524,  1.571], dtype=np.float32)
JOINT_LOWER = np.tile(JOINT_LOWER_1LEG, 6).astype(np.float32)
JOINT_UPPER = np.tile(JOINT_UPPER_1LEG, 6).astype(np.float32)

MIRROR_PAIRS = [(2, 1), (3, 6), (4, 5)]   # (left, right) for front/mid/rear

def _slice(leg_id):
    i = leg_id - 1
    return slice(3 * i, 3 * i + 3)

def ik_2link(X, Z):
    """Analytic 2-link IK, verified against forward kinematics."""
    l1, l2 = FEMUR_LEN, TIBIA_LEN
    W = -Z
    r2 = X * X + W * W
    r2 = np.clip(r2, (l1 - l2) ** 2 + 1e-9, (l1 + l2) ** 2 - 1e-9)
    cos_t2 = np.clip((r2 - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1.0, 1.0)
    t2 = np.arccos(cos_t2)                       # tibia
    k1 = l1 + l2 * np.cos(t2)
    k2 = l2 * np.sin(t2)
    t1 = np.arctan2(W, X) - np.arctan2(k2, k1)    # femur
    return float(t1), float(t2)

NEUTRAL_X, NEUTRAL_Z = 0.16, -0.10
NEUTRAL_FEMUR, NEUTRAL_TIBIA = ik_2link(NEUTRAL_X, NEUTRAL_Z)   # ~0.221, ~0.676
SPAWN_Z       = 0.10
TARGET_HEIGHT = 0.10

COXA_SCALE    = 0.30    # rad (~17 deg) swing range for stepping
FOOT_DX_SCALE = 0.025   # m, forward/back reach modulation
FOOT_DZ_SCALE = 0.025   # m, lift/push modulation

NUM_JOINTS        = 18
ACT_DIM           = 9     # [front(coxa,dx,dz), mid(...), rear(...)] LEFT side only
OBS_DIM           = 53
SIM_HZ            = 240
CTRL_HZ           = 50
SIM_STEPS_PER_CTRL = SIM_HZ // CTRL_HZ
MAX_EPISODE_STEPS = 1000
SERVO_FORCE       = 25.0

MIN_HEIGHT = 0.06
MAX_TILT   = 0.50

W_FORWARD     =  30.0
W_LATERAL     = -1.0
W_HEIGHT      =  1.0
W_TILT        = -2.0
W_ALIVE       =  0.0
W_ENERGY      = -0.0015
W_ACTION_RATE = -0.006
W_FOOT        =  0.15
W_STILL       = -3.0
STILL_THRESH  = 0.04   # m/s
GAIT_FREQ = 1.4
CURRENT_STEP_FRAC = 0.0


def expand_mirrored_targets(action9):
    """
    Joint-space tripod gait + small PPO residuals.
    action9:
      front-left residual: coxa, femur, tibia
      mid-left residual:   coxa, femur, tibia
      rear-left residual:  coxa, femur, tibia
    """
    action9 = np.clip(action9, -1.0, 1.0).astype(np.float32)

    full = np.zeros(18, dtype=np.float32)

    stand_coxa = 0.0
    stand_femur = 0.35
    stand_tibia = 1.05

    phase = 2.0 * np.pi * GAIT_FREQ * CURRENT_STEP_FRAC

    group_phases = [
        phase,             # front-left
        phase + np.pi,     # mid-left
        phase,             # rear-left
    ]

    coxa_amp = -0.38
    femur_lift_amp = 0.34
    tibia_lift_amp = 0.22

    groups = [action9[0:3], action9[3:6], action9[6:9]]

    for (left_id, right_id), g, ph in zip(MIRROR_PAIRS, groups, group_phases):
        swing = np.sin(ph)
        lift = max(0.0, np.cos(ph))

        coxa_res = 0.08 * float(g[0])
        femur_res = 0.08 * float(g[1])
        tibia_res = 0.08 * float(g[2])

        left_coxa = stand_coxa + coxa_amp * swing + coxa_res
        left_femur = stand_femur - femur_lift_amp * lift + femur_res
        left_tibia = stand_tibia - tibia_lift_amp * lift + tibia_res

        right_swing = -swing
        right_lift = max(0.0, -np.cos(ph))

        right_coxa = -(stand_coxa + coxa_amp * right_swing + coxa_res)
        right_femur = stand_femur - femur_lift_amp * right_lift + femur_res
        right_tibia = stand_tibia - tibia_lift_amp * right_lift + tibia_res

        full[_slice(left_id)] = [left_coxa, left_femur, left_tibia]
        full[_slice(right_id)] = [right_coxa, right_femur, right_tibia]

    return np.clip(full, JOINT_LOWER, JOINT_UPPER).astype(np.float32)


class HexapodWalkEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": CTRL_HZ}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)
        self._client = None
        self._robot = None
        self._joint_ids = None
        self._step_count = 0
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)

    def _build_joint_index(self):
        self._joint_ids = []
        for i in range(p.getNumJoints(self._robot, physicsClientId=self._client)):
            info = p.getJointInfo(self._robot, i, physicsClientId=self._client)
            if info[2] == p.JOINT_REVOLUTE:
                self._joint_ids.append(i)
        assert len(self._joint_ids) == NUM_JOINTS
        self._tibia_link_ids = set(self._joint_ids[i] for i in range(2, 18, 3))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._client is None:
            if self.render_mode == "human":
                self._client = p.connect(p.GUI)
                p.resetDebugVisualizerCamera(1.0, 45, -25, [0, 0, 0.10], physicsClientId=self._client)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._client)
            else:
                self._client = p.connect(p.DIRECT)

        p.resetSimulation(physicsClientId=self._client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / SIM_HZ, physicsClientId=self._client)
        p.loadURDF("plane.urdf", physicsClientId=self._client)

        self._robot = p.loadURDF(
            URDF, [0, 0, SPAWN_Z], p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False, physicsClientId=self._client)
        self._build_joint_index()

        for link_idx in range(-1, p.getNumJoints(self._robot, physicsClientId=self._client)):
            p.changeDynamics(self._robot, link_idx,
                lateralFriction=2.5, spinningFriction=0.3, rollingFriction=0.01,
                frictionAnchor=1, physicsClientId=self._client)

        neutral18 = expand_mirrored_targets(np.zeros(9, dtype=np.float32))
        for idx, jid in enumerate(self._joint_ids):
            p.resetJointState(self._robot, jid, float(neutral18[idx]), physicsClientId=self._client)
            p.setJointMotorControl2(self._robot, jid, p.POSITION_CONTROL,
                targetPosition=float(neutral18[idx]), force=SERVO_FORCE, physicsClientId=self._client)

        for _ in range(20):
            p.stepSimulation(physicsClientId=self._client)
        p.resetBaseVelocity(self._robot, [0, 0, 0], [0, 0, 0], physicsClientId=self._client)

        self._step_count = 0
        self._prev_action = np.zeros(ACT_DIM, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        global CURRENT_STEP_FRAC
        CURRENT_STEP_FRAC = self._step_count / CTRL_HZ
        target18 = expand_mirrored_targets(action)
        for idx, jid in enumerate(self._joint_ids):
            p.setJointMotorControl2(self._robot, jid, p.POSITION_CONTROL,
                targetPosition=float(target18[idx]), force=SERVO_FORCE, physicsClientId=self._client)
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
            lin_vel, ang_vel, [euler[0], euler[1]], jpos, jvel, self._prev_action
        ]).astype(np.float32)

    def _get_foot_contacts(self):
        contacts = p.getContactPoints(self._robot, physicsClientId=self._client)
        feet = set()
        if contacts:
            for c in contacts:
                if c[2] != self._robot and c[3] in self._tibia_link_ids:
                    feet.add(c[3])
        return len(feet)

    def _compute_reward(self, action):
        pos, orn = p.getBasePositionAndOrientation(self._robot, physicsClientId=self._client)
        lin_vel, _ = p.getBaseVelocity(self._robot, physicsClientId=self._client)
        euler = p.getEulerFromQuaternion(orn)
        height = pos[2]
        tilt = abs(euler[0]) + abs(euler[1])
        vx, vy = lin_vel[0], lin_vel[1]
        speed = float(np.sqrt(vx*vx + vy*vy))

        height_rew = np.exp(-20.0 * abs(height - TARGET_HEIGHT))
        forward_rew = max(0.0, vx)
        still_penalty = W_STILL if (self._step_count > 50 and speed < STILL_THRESH) else 0.0

        torques = [p.getJointState(self._robot, jid, physicsClientId=self._client)[3] for jid in self._joint_ids]
        energy = float(np.sum(np.abs(torques)))
        n_feet = self._get_foot_contacts()
        action_rate = float(np.sum(np.square(action - self._prev_action)))

        reward = (
            W_FORWARD * forward_rew + W_LATERAL * abs(vy) + W_HEIGHT * height_rew
            + W_TILT * tilt + still_penalty + W_ENERGY * energy
            + W_ACTION_RATE * action_rate + W_FOOT * n_feet
        )
        terminated = bool(self._step_count > 150 and (height < MIN_HEIGHT or tilt > MAX_TILT))
        if terminated:
            reward -= 10.0
        return float(reward), terminated

    def render(self):
        if self.render_mode == "rgb_array":
            w, h = 640, 480
            pos, _ = p.getBasePositionAndOrientation(self._robot, physicsClientId=self._client)
            view = p.computeViewMatrixFromYawPitchRoll([pos[0], pos[1], 0.10], 1.0, 45, -25, 0, 2, physicsClientId=self._client)
            proj = p.computeProjectionMatrixFOV(60, w / h, 0.01, 100, physicsClientId=self._client)
            _, _, rgb, _, _ = p.getCameraImage(w, h, view, proj, physicsClientId=self._client)
            return np.array(rgb, dtype=np.uint8)[:, :, :3]

    def close(self):
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
            self._client = None


if __name__ == "__main__":
    import time
    print("Tripod-style test — front+rear lift/swing together, mid opposite phase.")
    env = HexapodWalkEnv(render_mode="human")
    obs, _ = env.reset()
    pos, _ = p.getBasePositionAndOrientation(env._robot, physicsClientId=env._client)
    print(f"  Height: {pos[2]:.4f}m (expect ~0.10)")

    total = 0
    for i in range(1000):
        t = i * 0.15
        a = np.zeros(9, dtype=np.float32)
        a[0] = 0.6 * np.sin(t)        # front coxa
        a[2] = 0.6 * np.cos(t)        # front dz: lift during swing, push during stance
        a[6] = 0.6 * np.sin(t)
        a[8] = 0.6 * np.cos(t)
        a[3] = -0.6 * np.sin(t)
        a[5] = -0.6 * np.cos(t)

        obs, reward, terminated, truncated, _ = env.step(a)
        total += reward
        time.sleep(1 / CTRL_HZ)
        if terminated or truncated:
            print(f"  Ended step {i} reward={total:.1f}")
            break
    else:
        print(f"  1000 steps stable, reward={total:.1f}")
    env.close()