"""
hexapod_deploy.py
=====================================================
  1. Joint angle INVERSION for upside-down mounting.
     In simulation the robot is right-side-up; on hardware the body
     is flipped 180° around the X-axis so femur/tibia signs flip.
  2. SERVO_OFFSET table — trim each servo individually once you
     measure the real zero-position vs URDF zero.
  3. --invert-check flag: runs a slow sweep and prints angles
     so you can confirm direction before powering servos.
  4. Final update including normalization and stablization with
     IMU reading generation.

Usage:
  python3 hexapod_deploy.py
  python3 hexapod_deploy.py --dry-run          # safe test, no serial
  python3 hexapod_deploy.py --invert-check     # slow sweep, prints all joints
  python3 hexapod_deploy.py --port-a /dev/ttyUSB0 --port-b /dev/ttyUSB1
  python3 hexapod_deploy.py --port-a COM3 --port-b COM4   # Windows
"""

import argparse
import struct
import time
import numpy as np
import os

# ── Optional serial import ─────────────────────────────────────────────────────
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STAGE2_DIR = os.path.join(BASE_DIR, "checkpoints_stage2")
NORM_STATS = os.path.join(STAGE2_DIR, "vec_normalize_stage2.pkl")

# ── Joint layout (18 joints: coxa, femur, tibia × 6 legs) ────────────────────
JOINT_LOWER = np.array([-0.785, -1.571, -1.571] * 6, dtype=np.float32)
JOINT_UPPER = np.array([ 0.785,  0.524,  1.571] * 6, dtype=np.float32)

STAND_POSE   = np.array([0.0, 0.52, 1.00] * 6, dtype=np.float32)
# ACTION_SCALE = np.array([0.25, 0.15, 0.20] * 6, dtype=np.float32)
ACTION_SCALE = np.array([0.25, 0.05, 0.08] * 6, dtype=np.float32)
OBS_DIM      = 62
ACT_DIM      = 18
CTRL_HZ      = 50

# ── INVERSION TABLE ────────────────────────────────────────────────────────────
# The robot is mounted UPSIDE-DOWN (body on top, legs pointing downward).
# In simulation it is right-side-up, so the femur/tibia axes are flipped
# on the physical hardware.
#
# For each of the 18 joints (coxa0, femur0, tibia0, coxa1, femur1, tibia1, …):
#   +1  = same direction as simulation
#   -1  = physically reversed
#
# COXA  (rotates around Z-axis, vertical): mounting flip does NOT reverse Z → +1
# FEMUR (rotates around Y-axis, horizontal): body flip reverses Y → -1
# TIBIA (rotates around Y-axis, horizontal): body flip reverses Y → -1
#
# !! Verify with --invert-check before connecting all servos !!
JOINT_SIGN = np.array([
    +1, -1, -1,   # leg 1 (front-right):  coxa, femur, tibia
    +1, -1, -1,   # leg 2 (front-left)
    +1, -1, -1,   # leg 3 (mid-left)
    +1, -1, -1,   # leg 4 (rear-left)
    +1, -1, -1,   # leg 5 (rear-right)
    +1, -1, -1,   # leg 6 (mid-right)
], dtype=np.float32)

# ── PER-SERVO TRIM (radians) ──────────────────────────────────────────────────
# After verifying sign, measure how many degrees each servo is off at its
# "zero" position and fill in here.  Start with all zeros; tune one leg
# at a time.
SERVO_TRIM = np.zeros(ACT_DIM, dtype=np.float32)
# Example: leg-1 femur is 5° (0.087 rad) off:
# SERVO_TRIM[1] = 0.087

# ── Smoothing ─────────────────────────────────────────────────────────────────
SMOOTHING_ALPHA = 0.35   # lower = more smoothing


# ── Serial frame ──────────────────────────────────────────────────────────────
def pack_frame(angles_rad: np.ndarray) -> bytes:
    """Pack 18 floats → header 0xAA 0xBB + 72 bytes little-endian floats."""
    header  = bytes([0xAA, 0xBB])
    payload = struct.pack('<18f', *angles_rad.tolist())
    return header + payload


def sim_to_hardware(sim_angles):
    hw = JOINT_SIGN * sim_angles + SERVO_TRIM
    hw = np.clip(hw, JOINT_LOWER, JOINT_UPPER)
    
    # saturation warning
    saturated = np.where(
        (sim_angles >= JOINT_UPPER - 0.01) | (sim_angles <= JOINT_LOWER + 0.01)
    )[0]
    if len(saturated) > 3:
        print(f"WARNING: {len(saturated)} joints saturated: {saturated.tolist()}")
    
    return hw


# ── Fake hardware observation (open-loop, no encoders) ───────────────────────
class FakeHardwareObs:
    """
    Builds a synthetic 62-D observation for the PPO policy.

    On real hardware you would read an IMU (MPU-6050 etc.) for
    lin_vel / ang_vel / euler.  Without one, zeros are used — the
    policy degrades gracefully because forward velocity dominates.
    """

    def __init__(self):
        self.current_joints = STAND_POSE.copy()
        self.prev_action    = np.zeros(ACT_DIM, dtype=np.float32)
        self.lin_vel        = np.zeros(3, dtype=np.float32)
        self.ang_vel        = np.zeros(3, dtype=np.float32)
        self.euler_rp       = np.zeros(2, dtype=np.float32)  # roll, pitch

    def get_obs(self) -> np.ndarray:
        jpos = self.current_joints.copy()
        jvel = np.zeros(ACT_DIM, dtype=np.float32)
        obs  = np.concatenate([
            self.lin_vel, self.ang_vel, self.euler_rp,
            jpos, jvel, self.prev_action
        ]).astype(np.float32)
        assert obs.shape == (OBS_DIM,), f"Obs shape {obs.shape} ≠ {OBS_DIM}"
        return obs

    def apply_action(self, action: np.ndarray):
        delta = action * ACTION_SCALE
        self.current_joints = np.clip(
            self.current_joints + delta, JOINT_LOWER, JOINT_UPPER)
        self.prev_action = action.copy()

    def get_joint_angles(self) -> np.ndarray:
        return self.current_joints.copy()


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from hexapod_env_stage2 import HexapodWalkEnv

    dummy_env = DummyVecEnv([lambda: HexapodWalkEnv(render_mode=None)])

    if os.path.exists(NORM_STATS):
        vec_env = VecNormalize.load(NORM_STATS, dummy_env)
        vec_env.training    = False
        vec_env.norm_reward = False
        print("Normalizer loaded.")
    else:
        vec_env = dummy_env
        print("WARNING: No normalizer found. Policy may behave erratically.")

    model_path = None
    for name in ["best_model_stage2", "best_model", "last_model_stage2"]:
        p = os.path.join(STAGE2_DIR, name + ".zip")
        if os.path.exists(p):
            model_path = p.replace(".zip", "")
            break

    if model_path is None:
        raise FileNotFoundError("No Stage 2 model found in checkpoints_stage2/")

    model = PPO.load(model_path, env=vec_env)
    print(f"Model loaded: {os.path.basename(model_path)}.zip")
    return model, vec_env


def normalize_obs(obs: np.ndarray, vec_env) -> np.ndarray:
    from stable_baselines3.common.vec_env import VecNormalize
    if isinstance(vec_env, VecNormalize):
        return vec_env.normalize_obs(obs.reshape(1, -1)).reshape(-1).astype(np.float32)
    return obs


# ── Invert-check: slow sweep to verify sign table ─────────────────────────────
def run_invert_check():
    """
    Moves each joint slowly from 0 → +0.3 rad → 0 → -0.3 rad → 0.
    Watch each servo: if it moves AWAY from neutral on the + command,
    flip its sign in JOINT_SIGN above.
    No model needed — purely mechanical verification.
    """
    print("\n=== INVERT-CHECK (dry-run only, no serial) ===")
    print("Each joint sweeps ±0.3 rad.  Verify direction by watching servos.\n")
    joints = STAND_POSE.copy()
    header = "  " + "  ".join(f"J{i:02d}" for i in range(ACT_DIM))
    print(header)

    for j in range(ACT_DIM):
        for angle in [0.0, 0.3, 0.0, -0.3, 0.0]:
            sim_j         = joints.copy()
            sim_j[j]      = angle
            hw_j          = sim_to_hardware(sim_j)
            row = "  ".join(f"{v:+.3f}" for v in hw_j)
            print(f"  {row}    ← J{j:02d}={angle:+.2f}")
            time.sleep(0.3)

    print("\nInvert-check done.  Adjust JOINT_SIGN if any servo moved backwards.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-a",      type=str, default="/dev/ttyUSB0")
    parser.add_argument("--port-b",      type=str, default="/dev/ttyUSB1")
    parser.add_argument("--baud",        type=int, default=115200)
    parser.add_argument("--steps",       type=int, default=5000)
    parser.add_argument("--dry-run",     action="store_true",
                        help="No serial output — print angles only")
    parser.add_argument("--slow",        action="store_true",
                        help="0.2× speed for debugging")
    parser.add_argument("--invert-check", action="store_true",
                        help="Mechanical sign verification (no model needed)")
    args = parser.parse_args()

    if args.invert_check:
        run_invert_check()
        return

    print("Loading model...")
    model, vec_env = load_model()

    ser_a = ser_b = None
    if not args.dry_run:
        if not HAS_SERIAL:
            print("ERROR: pyserial not installed.  Run: pip install pyserial")
            return
        print(f"Opening serial: A={args.port_a}  B={args.port_b}")
        ser_a = serial.Serial(args.port_a, args.baud, timeout=0.1)
        ser_b = serial.Serial(args.port_b, args.baud, timeout=0.1)
        time.sleep(2.0)

        deadline = time.time() + 5.0
        got_a = got_b = False
        while time.time() < deadline and not (got_a and got_b):
            if ser_a.in_waiting:
                line = ser_a.readline().decode(errors="ignore").strip()
                if "READY" in line:
                    got_a = True
                    print("ESP32 A: READY")
            if ser_b.in_waiting:
                line = ser_b.readline().decode(errors="ignore").strip()
                if "READY" in line:
                    got_b = True
                    print("ESP32 B: READY")
        if not (got_a and got_b):
            print("WARNING: Did not receive READY from both ESP32s. Continuing.")

    hw            = FakeHardwareObs()
    smoothed_sim  = STAND_POSE.copy()   # smoothing in SIM space, convert last
    ctrl_period   = (1.0 / CTRL_HZ) * (5.0 if args.slow else 1.0)

    leg_labels = ["FR","FL","ML","RL","RR","MR"]
    print(f"\nRunning at {'0.2×' if args.slow else '1×'} speed | Ctrl+C to stop")
    print(f"\n{'Step':>6}  " + "  ".join(
        f"{lb+'-C':>7} {lb+'-F':>7} {lb+'-T':>7}"
        for lb in leg_labels))
    print("-" * (8 + 24 * 6))

    try:
        for step in range(args.steps):
            t_start = time.time()

            raw_obs  = hw.get_obs()
            norm_obs = normalize_obs(raw_obs, vec_env)

            action, _ = model.policy.predict(
                norm_obs.reshape(1, -1), deterministic=True)
            action = np.clip(action.flatten().astype(np.float32), -1.0, 1.0)

            hw.apply_action(action)
            raw_sim_joints = hw.get_joint_angles()

            # Smooth in simulation space
            alpha        = SMOOTHING_ALPHA
            smoothed_sim = alpha * raw_sim_joints + (1.0 - alpha) * smoothed_sim

            # Convert to hardware angles (flip signs + trim)
            hw_angles = sim_to_hardware(smoothed_sim)

            frame = pack_frame(hw_angles)
            if ser_a:
                ser_a.write(frame)
            if ser_b:
                ser_b.write(frame)

            if step % 50 == 0:
                vals = "  ".join(
                    f"{hw_angles[i*3]:>+7.3f} {hw_angles[i*3+1]:>+7.3f} {hw_angles[i*3+2]:>+7.3f}"
                    for i in range(6))
                print(f"{step:>6}  {vals}")

            if ser_a and ser_a.in_waiting:
                ser_a.readline()
            if ser_b and ser_b.in_waiting:
                ser_b.readline()

            elapsed = time.time() - t_start
            sleep_t = ctrl_period - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if ser_a:
            ser_a.close()
        if ser_b:
            ser_b.close()
        vec_env.close()
        print("Done.")


if __name__ == "__main__":
    main()