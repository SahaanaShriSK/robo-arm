"""
robot_simulator.py — KUKA iiwa draws text and sends 3DOF angles to ESP32
"""

import time
import threading
import queue
import requests

import pybullet as p
import pybullet_data

# ====== ESP CONFIG ======
ESP_IP = "172.27.131.184"   # ← CHANGE TO YOUR ESP32 IP

# ====== SIM SETTINGS ======
SETTLE_STEPS = 50
STEP_DELAY   = 0.004
SIM_FORCE    = 500
NUM_JOINTS   = 7
EEF_LINK     = 6

PZ = 0.012


# ====== SEND ANGLES TO ESP32 ======
def send_to_esp(base, shoulder, elbow):
    try:
        url = f"http://{ESP_IP}/?b={int(base)}&s={int(shoulder)}&e={int(elbow)}"
        requests.get(url, timeout=0.05)
    except:
        pass


class RobotSimulator:

    def __init__(self):
        self._wq       = queue.Queue()
        self._running  = False
        self._thread   = None
        self._robot    = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit_waypoints(self, wps):
        self._wq.put(wps)

    def shutdown(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        self._setup()
        self._home()

        while self._running:
            try:
                wps = self._wq.get(timeout=0.05)
            except queue.Empty:
                p.stepSimulation()
                time.sleep(STEP_DELAY)
                continue

            self._draw(wps)

        p.disconnect()

    def _setup(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)

        p.loadURDF("plane.urdf")

        self._robot = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0, 0, 0],
            useFixedBase=True
        )

        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=90,
            cameraPitch=-70,
            cameraTargetPosition=[0.43, 0.0, 0.0]
        )

    def _ik(self, x, y, z):
        return p.calculateInverseKinematics(
            self._robot,
            EEF_LINK,
            targetPosition=[x, y, z]
        )

    def _apply(self, joints):
        # ---- Extract first 3 joints only (3DOF) ----
        base_angle     = joints[0]
        shoulder_angle = joints[1]
        elbow_angle    = joints[2]

        # ---- Convert radians → degrees ----
        base_deg     = max(0, min(180, base_angle * 180 / 3.14159))
        shoulder_deg = max(0, min(180, shoulder_angle * 180 / 3.14159))
        elbow_deg    = max(0, min(180, elbow_angle * 180 / 3.14159))

        # ---- Send once to ESP32 ----
        send_to_esp(base_deg, shoulder_deg, elbow_deg)

        # ---- Move simulator arm ----
        for i, a in enumerate(joints[:NUM_JOINTS]):
            p.setJointMotorControl2(
                self._robot,
                i,
                p.POSITION_CONTROL,
                targetPosition=a,
                force=SIM_FORCE
            )

        for _ in range(SETTLE_STEPS):
            p.stepSimulation()
            time.sleep(STEP_DELAY)

    def _home(self):
        self._apply([0, -0.5, 0, 1.0, 0, -0.5, 0])

    def _draw(self, waypoints):
        for x, y, z, pen in waypoints:
            arm_z = z if pen == 1 else z + 0.1
            joints = self._ik(x, y, arm_z)
            self._apply(joints)


_sim = None

def get_simulator():
    global _sim
    if _sim is None:
        _sim = RobotSimulator()
        _sim.start()
        time.sleep(2)
    return _sim