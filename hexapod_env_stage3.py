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
import heapq
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data

URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "urdf", "hexapod_generated.urdf")

# ── Joint limits (18 DOF: coxa, femur, tibia × 6) ───────────────────────────
JOINT_LOWER = np.array([-0.785, -1.571, -1.571] * 6, dtype=np.float32)
JOINT_UPPER = np.array([ 0.785,  0.524,  1.571] * 6, dtype=np.float32)

# ── Simulation constants ──────────────────────────────────────────────────────
NUM_JOINTS         = 18
SIM_HZ             = 240
CTRL_HZ            = 50
SIM_STEPS_PER_CTRL = SIM_HZ // CTRL_HZ
MAX_EPISODE_STEPS  = 2000         
SERVO_FORCE        = 25.0

# ── Standing pose ─────────────────
STAND_POSE    = [0.0, 0.52, 1.00] * 6
TARGET_HEIGHT = 0.1508
SPAWN_Z       = 0.1508

# ── Termination thresholds ───────────────────────────────────────────────────
MIN_HEIGHT    = 0.08
MAX_TILT      = 0.60

# ── Action scale per joint type (radians / step) ─────────────────────────────
ACTION_SCALE  = np.array([0.25, 0.15, 0.20] * 6, dtype=np.float32)

# ── Goal parameters ───────────────────────────────────────────────────────────
GOAL_RADIUS       = 0.15    # m — success threshold
GOAL_MIN_DIST     = 0.8     # m — minimum spawn-to-goal distance
GOAL_MAX_DIST     = 3.0     # m — maximum spawn-to-goal distance
SUBGOAL_LOOKAHEAD = 3       # waypoints ahead to use as local subgoal

# ── Observation dimension ────────────────────────────────────────────────────
#   Stage-2 obs : 68  (lin_vel×3, ang_vel×3, euler×2, jpos×18, jvel×18, prev_act×18)
#   goal_vec_2D :  2  (relative x,y to current subgoal, normalised)
#   dist_to_goal:  1  (scalar, normalised by GOAL_MAX_DIST)
#   heading_err :  1  (angle between body heading and goal direction)
OBS_DIM_STAGE2 = 68
EXTRA_OBS      = 4
OBS_DIM        = OBS_DIM_STAGE2 + EXTRA_OBS   # 72
ACT_DIM        = 18

# ── Reward weights ────────────────────────────────────────────────────────────
W_PROGRESS    =  8.0    # potential-based progress per step
W_GOAL        = 200.0   # sparse bonus on success
W_HEADING     = -0.5    # penalise yawing away from goal
W_HEIGHT      =  2.0
W_TILT        = -2.0
W_ALIVE       =  0.1
W_ENERGY      = -0.001
W_ACTION_RATE = -0.003


# ═══════════════════════════════════════════════════════════════════════════════
#  A*  PATH  PLANNER  (2-D grid)
# ═══════════════════════════════════════════════════════════════════════════════

class AStarPlanner:
    """
    Lightweight 2-D A* on a uniform occupancy grid.

    Parameters
    ----------
    resolution : float   — metres per cell
    half_extent : float  — arena half-size in metres (square grid)
    """

    def __init__(self, resolution: float = 0.05, half_extent: float = 4.0):
        self.res         = resolution
        self.half        = half_extent
        self.n_cells     = int(2 * half_extent / resolution)
        # occupancy grid — 0 free, 1 occupied  (flat arena → all 0)
        self.grid        = np.zeros((self.n_cells, self.n_cells), dtype=np.uint8)

    # ── coordinate helpers ────────────────────────────────────────────────────
    def _w2g(self, wx: float, wy: float):
        """World coords → grid indices (row, col)."""
        col = int((wx + self.half) / self.res)
        row = int((wy + self.half) / self.res)
        col = max(0, min(self.n_cells - 1, col))
        row = max(0, min(self.n_cells - 1, row))
        return row, col

    def _g2w(self, row: int, col: int):
        """Grid indices → world coords (centre of cell)."""
        wx = col * self.res - self.half + self.res / 2
        wy = row * self.res - self.half + self.res / 2
        return wx, wy

    # ── A* ────────────────────────────────────────────────────────────────────
    def plan(self, start_xy, goal_xy):
        """
        Returns a list of (x, y) world-coord waypoints from start to goal.
        Falls back to a straight line if start == goal or grid is tiny.
        """
        sr, sc = self._w2g(*start_xy)
        gr, gc = self._w2g(*goal_xy)

        if (sr, sc) == (gr, gc):
            return [goal_xy]

        open_heap = []
        heapq.heappush(open_heap, (0.0, sr, sc))
        came_from = {}
        g_cost    = {(sr, sc): 0.0}

        def h(r, c):
            return abs(r - gr) + abs(c - gc)          # Manhattan heuristic

        neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                     (-1,-1),(-1,1),(1,-1),(1,1)]     # 8-connected

        while open_heap:
            f, r, c = heapq.heappop(open_heap)
            if (r, c) == (gr, gc):
                break
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.n_cells and 0 <= nc < self.n_cells):
                    continue
                if self.grid[nr, nc]:
                    continue
                step = 1.414 if (dr != 0 and dc != 0) else 1.0
                ng   = g_cost[(r, c)] + step
                if ng < g_cost.get((nr, nc), float("inf")):
                    g_cost[(nr, nc)] = ng
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(open_heap, (ng + h(nr, nc), nr, nc))
        else:
            # No path found — return straight line
            return [goal_xy]

        # Reconstruct path
        path = []
        node = (gr, gc)
        while node in came_from:
            path.append(self._g2w(*node))
            node = came_from[node]
        path.append(self._g2w(sr, sc))
        path.reverse()

        # Thin the waypoints (keep every 4th + last)
        thinned = path[::4]
        if thinned[-1] != path[-1]:
            thinned.append(path[-1])
        return thinned


# ═══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class HexapodGoalEnv(gym.Env):
    """
    Stage-3 goal-reaching environment.

    Observation (66-D):
        [lin_vel(3), ang_vel(3), euler(2), jpos(18), jvel(18), prev_act(18),
         rel_subgoal_x, rel_subgoal_y, dist_to_goal_norm, heading_error]

    Action (18-D):
        Joint-angle deltas ∈ [-1, 1], scaled by ACTION_SCALE.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": CTRL_HZ}

    def __init__(self, render_mode=None,
                 goal_xy=None,
                 arena_half=3.0):
        super().__init__()
        self.render_mode  = render_mode
        self.fixed_goal   = goal_xy      # None → random every episode
        self.arena_half   = arena_half
        self.planner      = AStarPlanner(resolution=0.05,
                                         half_extent=arena_half + 1.0)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)

        self._client        = None
        self._robot         = None
        self._goal_body     = None
        self._joint_ids     = []
        self._step_count    = 0
        self._prev_action   = np.zeros(ACT_DIM, dtype=np.float32)
        self._current_joints = np.array(STAND_POSE, dtype=np.float32)

        # Navigation state
        self._goal_xy       = np.zeros(2, dtype=np.float32)
        self._waypoints     = []
        self._wp_idx        = 0
        self._prev_dist     = 0.0

    # ── Physics setup ─────────────────────────────────────────────────────────

    def _build_joint_index(self):
        self._joint_ids = []
        for i in range(p.getNumJoints(self._robot,
                                       physicsClientId=self._client)):
            info = p.getJointInfo(self._robot, i,
                                  physicsClientId=self._client)
            if info[2] == p.JOINT_REVOLUTE:
                self._joint_ids.append(i)
        assert len(self._joint_ids) == NUM_JOINTS, \
            f"Expected {NUM_JOINTS} revolute joints, got {len(self._joint_ids)}"

    def _spawn_goal_marker(self, gx, gy):
        """Draw a small red sphere at the goal location (visual only)."""
        if self._goal_body is not None:
            try:
                p.removeBody(self._goal_body,
                             physicsClientId=self._client)
            except Exception:
                pass
        vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.05,
            rgbaColor=[1, 0, 0, 0.8],
            physicsClientId=self._client)
        self._goal_body = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis,
            basePosition=[gx, gy, 0.05],
            physicsClientId=self._client)

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # ── Connect / reset simulator ─────────────────────────────────────────
        if self._client is None:
            if self.render_mode == "human":
                self._client = p.connect(p.GUI)
                p.resetDebugVisualizerCamera(
                    2.5, 45, -30, [0, 0, 0.15],
                    physicsClientId=self._client)
                p.configureDebugVisualizer(
                    p.COV_ENABLE_GUI, 0,
                    physicsClientId=self._client)
            else:
                self._client = p.connect(p.DIRECT)

        p.resetSimulation(physicsClientId=self._client)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / SIM_HZ, physicsClientId=self._client)
        p.loadURDF("plane.urdf", physicsClientId=self._client)

        # ── Spawn hexapod at confirmed standing height ────────────────────────
        self._robot = p.loadURDF(
            URDF, [0, 0, SPAWN_Z],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False,
            physicsClientId=self._client)
        self._build_joint_index()
        self._goal_body = None

        for link_idx in range(-1, p.getNumJoints(
                self._robot, physicsClientId=self._client)):
            p.changeDynamics(
                self._robot, link_idx,
                lateralFriction=1.5,
                spinningFriction=0.1,
                rollingFriction=0.01,
                physicsClientId=self._client)

        # Apply confirmed standing pose
        for idx, jid in enumerate(self._joint_ids):
            p.resetJointState(
                self._robot, jid, STAND_POSE[idx],
                physicsClientId=self._client)
            p.setJointMotorControl2(
                self._robot, jid, p.POSITION_CONTROL,
                targetPosition=STAND_POSE[idx],
                force=SERVO_FORCE,
                physicsClientId=self._client)

        # Settle physics (10 steps only — same as Stage 2)
        for _ in range(10):
            p.stepSimulation(physicsClientId=self._client)
        p.resetBaseVelocity(
            self._robot, [0, 0, 0], [0, 0, 0],
            physicsClientId=self._client)

        # ── Sample goal ───────────────────────────────────────────────────────
        if self.fixed_goal is not None:
            self._goal_xy = np.array(self.fixed_goal, dtype=np.float32)
        else:
            rng = self.np_random
            for _ in range(200):
                angle = rng.uniform(0, 2 * np.pi)
                dist  = rng.uniform(GOAL_MIN_DIST, GOAL_MAX_DIST)
                gx    = float(dist * np.cos(angle))
                gy    = float(dist * np.sin(angle))
                if abs(gx) < self.arena_half and abs(gy) < self.arena_half:
                    self._goal_xy = np.array([gx, gy], dtype=np.float32)
                    break

        # ── Plan path ─────────────────────────────────────────────────────────
        start = (0.0, 0.0)
        goal  = (float(self._goal_xy[0]), float(self._goal_xy[1]))
        self._waypoints = self.planner.plan(start, goal)
        self._wp_idx    = 0

        # ── Goal marker (visual) ──────────────────────────────────────────────
        self._spawn_goal_marker(*self._goal_xy)
        if self.render_mode == "human":
            self._draw_path_debug()

        # ── State reset ───────────────────────────────────────────────────────
        self._step_count    = 0
        self._prev_action   = np.zeros(ACT_DIM, dtype=np.float32)
        self._current_joints = np.array(STAND_POSE, dtype=np.float32)
        pos, _              = p.getBasePositionAndOrientation(
            self._robot, physicsClientId=self._client)
        self._prev_dist     = float(np.linalg.norm(
            np.array([pos[0], pos[1]]) - self._goal_xy))

        return self._get_obs(), {"goal_xy": self._goal_xy.tolist()}

    def _draw_path_debug(self):
        """Draw the A* path as green debug lines (GUI mode only)."""
        if not self._waypoints:
            return
        pts = [(wp[0], wp[1], 0.02) for wp in self._waypoints]
        for i in range(len(pts) - 1):
            p.addUserDebugLine(pts[i], pts[i+1],
                               lineColorRGB=[0, 1, 0],
                               lineWidth=2.0,
                               physicsClientId=self._client)

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        # Delta joint control (identical to Stage 2)
        delta = action * ACTION_SCALE
        self._current_joints = np.clip(
            self._current_joints + delta,
            JOINT_LOWER, JOINT_UPPER)

        for idx, jid in enumerate(self._joint_ids):
            p.setJointMotorControl2(
                self._robot, jid, p.POSITION_CONTROL,
                targetPosition=float(self._current_joints[idx]),
                force=SERVO_FORCE,
                physicsClientId=self._client)

        for _ in range(SIM_STEPS_PER_CTRL):
            p.stepSimulation(physicsClientId=self._client)

        # Advance waypoint index if close to current subgoal
        self._update_waypoint()

        obs               = self._get_obs()
        reward, terminated, info = self._compute_reward(action)
        self._step_count += 1
        truncated          = self._step_count >= MAX_EPISODE_STEPS
        self._prev_action  = action.copy()

        if terminated:
            info["success"] = True
        return obs, reward, terminated, truncated, info

    def _update_waypoint(self):
        """Pop waypoints as the robot passes close enough to them."""
        if self._wp_idx >= len(self._waypoints):
            return
        pos, _ = p.getBasePositionAndOrientation(
            self._robot, physicsClientId=self._client)
        xy = np.array([pos[0], pos[1]], dtype=np.float32)
        while self._wp_idx < len(self._waypoints):
            wp  = np.array(self._waypoints[self._wp_idx], dtype=np.float32)
            d   = float(np.linalg.norm(xy - wp))
            if d < GOAL_RADIUS * 1.5:
                self._wp_idx += 1
            else:
                break

    # ── Observations ──────────────────────────────────────────────────────────

    def _get_obs(self):
        pos, orn         = p.getBasePositionAndOrientation(
            self._robot, physicsClientId=self._client)
        lin_vel, ang_vel = p.getBaseVelocity(
            self._robot, physicsClientId=self._client)
        euler            = p.getEulerFromQuaternion(orn)

        jpos, jvel = [], []
        for jid in self._joint_ids:
            js = p.getJointState(self._robot, jid,
                                 physicsClientId=self._client)
            jpos.append(js[0])
            jvel.append(js[1])

        # ── Goal / navigation extras ──────────────────────────────────────────
        xy = np.array([pos[0], pos[1]], dtype=np.float32)

        # Current subgoal (lookahead along waypoint list)
        subgoal_idx = min(self._wp_idx + SUBGOAL_LOOKAHEAD,
                          len(self._waypoints) - 1)
        if self._waypoints:
            subgoal = np.array(self._waypoints[subgoal_idx], dtype=np.float32)
        else:
            subgoal = self._goal_xy.copy()

        rel_sg   = subgoal - xy                         # 2-D relative vector
        dist_sg  = float(np.linalg.norm(rel_sg))
        if dist_sg > 1e-6:
            rel_sg_norm = rel_sg / dist_sg
        else:
            rel_sg_norm = np.zeros(2, dtype=np.float32)

        # Distance to final goal (normalised)
        dist_goal      = float(np.linalg.norm(xy - self._goal_xy))
        dist_goal_norm = np.clip(dist_goal / GOAL_MAX_DIST, 0.0, 1.0)

        # Heading error: yaw vs direction to goal
        yaw             = float(euler[2])
        goal_dir_angle  = float(np.arctan2(
            self._goal_xy[1] - pos[1],
            self._goal_xy[0] - pos[0]))
        heading_err     = float(np.arctan2(
            np.sin(goal_dir_angle - yaw),
            np.cos(goal_dir_angle - yaw)))

        return np.concatenate([
            lin_vel,                        # 3
            ang_vel,                        # 3
            [euler[0], euler[1]],           # 2
            jpos,                           # 18
            jvel,                           # 18
            self._prev_action,              # 18
            rel_sg_norm,                    # 2
            [dist_goal_norm],               # 1
            [heading_err / np.pi],          # 1  (normalised to [-1,1])
        ]).astype(np.float32)               # total: 66

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, action):
        pos, orn   = p.getBasePositionAndOrientation(
            self._robot, physicsClientId=self._client)
        lin_vel, _ = p.getBaseVelocity(
            self._robot, physicsClientId=self._client)
        euler      = p.getEulerFromQuaternion(orn)

        xy         = np.array([pos[0], pos[1]], dtype=np.float32)
        height     = float(pos[2])
        tilt       = float(abs(euler[0]) + abs(euler[1]))
        dist_goal  = float(np.linalg.norm(xy - self._goal_xy))

        # ── Potential-based progress ──────────────────────────────────────────
        progress     = self._prev_dist - dist_goal   # positive = closer
        self._prev_dist = dist_goal

        # ── Height reward (same Gaussian as Stage 2) ──────────────────────────
        height_rew  = float(np.exp(-20.0 * abs(height - TARGET_HEIGHT)))

        # ── Heading reward ────────────────────────────────────────────────────
        yaw           = float(euler[2])
        goal_dir      = float(np.arctan2(
            self._goal_xy[1] - pos[1],
            self._goal_xy[0] - pos[0]))
        heading_err   = float(abs(np.arctan2(
            np.sin(goal_dir - yaw),
            np.cos(goal_dir - yaw))))

        # ── Energy ────────────────────────────────────────────────────────────
        torques = [p.getJointState(self._robot, jid,
                    physicsClientId=self._client)[3]
                   for jid in self._joint_ids]
        energy  = float(np.sum(np.abs(torques)))

        # ── Action rate ───────────────────────────────────────────────────────
        action_rate = float(np.sum(np.square(action - self._prev_action)))

        reward = (
            W_PROGRESS    * progress +
            W_HEADING     * heading_err +
            W_HEIGHT      * height_rew +
            W_TILT        * tilt +
            W_ALIVE +
            W_ENERGY      * energy +
            W_ACTION_RATE * action_rate
        )

        # ── Goal reached ──────────────────────────────────────────────────────
        reached    = dist_goal < GOAL_RADIUS
        terminated = False
        info       = {"dist_to_goal": dist_goal, "success": False}

        if reached:
            reward    += W_GOAL
            terminated = True
            info["success"] = True

        # ── Fall detection (grace period of 100 steps) ────────────────────────
        elif self._step_count > 100 and (height < MIN_HEIGHT or tilt > MAX_TILT):
            reward    -= 10.0
            terminated = True

        return float(reward), terminated, info

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_foot_contacts(self):
        """Count unique tibia links in contact with the ground."""
        contacts = p.getContactPoints(
            self._robot, physicsClientId=self._client)
        ground_feet = set()
        if contacts:
            for c in contacts:
                if c[2] != self._robot:
                    ground_feet.add(c[3])
        return len(ground_feet)

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "rgb_array":
            w, h   = 640, 480
            pos, _ = p.getBasePositionAndOrientation(
                self._robot, physicsClientId=self._client)
            view   = p.computeViewMatrixFromYawPitchRoll(
                [pos[0], pos[1], 0.15], 2.5, 45, -30, 0, 2,
                physicsClientId=self._client)
            proj   = p.computeProjectionMatrixFOV(
                60, w / h, 0.01, 100,
                physicsClientId=self._client)
            _, _, rgb, _, _ = p.getCameraImage(
                w, h, view, proj,
                physicsClientId=self._client)
            return np.array(rgb, dtype=np.uint8)[:, :, :3]

    def close(self):
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
            self._client = None


if __name__ == "__main__":
    import time

    print("Stage-3 sanity check — zero actions, goal at (1.5, 0.0) …")
    env = HexapodGoalEnv(render_mode="human", goal_xy=[1.5, 0.0])
    obs, info = env.reset()
    print(f"  Obs shape : {obs.shape}  (expected 66)")
    print(f"  Goal      : {info['goal_xy']}")
    print(f"  Waypoints : {len(env._waypoints)}")

    pos, _ = p.getBasePositionAndOrientation(
        env._robot, physicsClientId=env._client)
    print(f"  Height    : {pos[2]:.4f} m  (expected ~0.1508)")

    total = 0.0
    for i in range(300):
        obs, r, terminated, truncated, info = env.step(np.zeros(ACT_DIM))
        total += r
        time.sleep(1 / CTRL_HZ)
        if terminated or truncated:
            print(f"  Episode ended at step {i} | reward={total:.1f} | "
                  f"dist={info['dist_to_goal']:.3f} m")
            break
    else:
        print(f"  300 steps stable | reward={total:.1f} | "
              f"dist={info['dist_to_goal']:.3f} m")
    env.close()