"""
robot_simulator.py — KUKA iiwa handwriting simulator
Improved version with paper surface, pen tool and better camera
"""

import time
import threading
import queue

import pybullet as p
import pybullet_data


SETTLE_STEPS = 40
STEP_DELAY   = 0.004
SIM_FORCE    = 500
NUM_JOINTS   = 7
EEF_LINK     = 6

LINE_COLOR = [0.0, 0.2, 1.0]
LINE_WIDTH = 8


# Paper bounds
PX0, PX1 = 0.28, 0.58
PY0, PY1 = -0.62, 0.62
PZ = 0.012


class RobotSimulator:

    def __init__(self):
        self._wq = queue.Queue()
        self._running = False
        self._thread = None
        self._lock = threading.Lock()

        self._line_ids = []
        self._pen_body = None
        self._robot = None


    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
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

            self._clear_lines()
            self._draw(wps)

        p.disconnect()


    # -------------------------
    # Setup simulation
    # -------------------------

    def _setup(self):

        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)


        # Floor
        p.loadURDF("plane.urdf")


        # Robot
        self._robot = p.loadURDF(
            "kuka_iiwa/model.urdf",
            basePosition=[0,0,0],
            useFixedBase=True
        )


        # Camera (top writing view)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.3,
            cameraYaw=90,
            cameraPitch=-70,
            cameraTargetPosition=[0.43,0,0]
        )


        # Paper surface
        paper_vis = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.20,0.70,0.001],
            rgbaColor=[1,1,1,1]
        )

        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=paper_vis,
            basePosition=[0.43,0,PZ]
        )


        # Paper outline
        corners = [
            [PX0, PY0, PZ+0.002],
            [PX1, PY0, PZ+0.002],
            [PX1, PY1, PZ+0.002],
            [PX0, PY1, PZ+0.002]
        ]

        for i in range(4):
            p.addUserDebugLine(corners[i], corners[(i+1)%4], [0,0,0], 2)


        # Pen tool
        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.01,
            length=0.08,
            rgbaColor=[1,0,0,1]
        )

        self._pen_body = p.createMultiBody(
            baseVisualShapeIndex=vis
        )


    # -------------------------
    # Inverse kinematics
    # -------------------------

    def _ik(self,x,y,z):

        return p.calculateInverseKinematics(
            self._robot,
            EEF_LINK,
            targetPosition=[x,y,z],
            maxNumIterations=200,
            residualThreshold=1e-4
        )


    # -------------------------
    # Move joints
    # -------------------------

    def _apply(self,joints,steps):

        for _ in range(steps):

            for i,a in enumerate(joints[:NUM_JOINTS]):

                p.setJointMotorControl2(
                    self._robot,
                    i,
                    p.POSITION_CONTROL,
                    targetPosition=a,
                    force=SIM_FORCE
                )

            p.stepSimulation()
            time.sleep(STEP_DELAY)


    # -------------------------
    # Home position
    # -------------------------

    def _home(self):

        self._apply([0,-0.5,0,1.0,0,-0.5,0],steps=100)


    # -------------------------
    # Draw text
    # -------------------------

    def _draw(self,waypoints):

        prev=None

        for wp in waypoints:

            x,y,z,pen=wp

            arm_z = z if pen==1 else z+0.1

            joints=self._ik(x,y,arm_z)

            self._apply(joints,SETTLE_STEPS)


            # Pen visual follows
            pen_z=z+0.01 if pen==1 else z+0.1

            p.resetBasePositionAndOrientation(
                self._pen_body,
                [x,y,pen_z],
                [0,0,0,1]
            )


            # Draw stroke
            if pen==1 and prev is not None:

                lid=p.addUserDebugLine(
                    [prev[0],prev[1],PZ+0.005],
                    [x,y,PZ+0.005],
                    LINE_COLOR,
                    LINE_WIDTH
                )

                with self._lock:
                    self._line_ids.append(lid)

            prev=[x,y,z]


        self._home()


    # -------------------------
    # Clear strokes
    # -------------------------

    def _clear_lines(self):

        with self._lock:

            for lid in self._line_ids:

                try:
                    p.removeUserDebugItem(lid)
                except:
                    pass

            self._line_ids.clear()



_sim=None

def get_simulator()->RobotSimulator:

    global _sim

    if _sim is None:

        _sim=RobotSimulator()
        _sim.start()

        time.sleep(2.5)

    return _sim