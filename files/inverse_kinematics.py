"""
inverse_kinematics.py
---------------------
Solves IK for the KUKA iiwa to reach ground-plane targets.

Writing surface is flat on the ground (z ~ 0).
The robot reaches DOWN and FORWARD/SIDEWAYS.

Joint mapping:
  Joint 0 (base yaw)    : rotate to face target in XY plane
  Joint 1 (shoulder)    : pitch arm forward/down
  Joint 2 (elbow)       : bend elbow to reach correct distance + height

Link lengths (kuka_iiwa approximate):
  L1 = 0.4 m  (upper arm)
  L2 = 0.4 m  (forearm)
"""

import math
import numpy as np

L1 = 0.4
L2 = 0.4


def solve_ik(x_target, y_target, z_target):
    """
    Returns (theta0, theta1, theta2) in radians to reach (x, y, z).

    theta0: base yaw  - rotate toward target
    theta1: shoulder  - pitch forward and down
    theta2: elbow     - bend to correct reach/height
    """
    # ── Joint 0: rotate base toward target in XY plane ────────────────────────
    theta0 = math.atan2(y_target, x_target)

    # ── Reach distance in XY plane ────────────────────────────────────────────
    r = math.sqrt(x_target**2 + y_target**2)

    # ── 2-link IK in the (r, z) vertical plane ────────────────────────────────
    # Target height relative to shoulder (shoulder is roughly at z=0.36 on kuka)
    SHOULDER_HEIGHT = 0.36
    dz = z_target - SHOULDER_HEIGHT   # negative means reaching DOWN

    dist_sq = r**2 + dz**2
    dist    = math.sqrt(dist_sq)

    # Clamp to reachable envelope
    max_reach = L1 + L2 - 0.01
    min_reach = abs(L1 - L2) + 0.01
    if dist > max_reach:
        scale = max_reach / dist
        r  *= scale
        dz *= scale
        dist = max_reach
    elif dist < min_reach:
        dist = min_reach

    cos2 = (r**2 + dz**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos2 = max(-1.0, min(1.0, cos2))

    # Elbow DOWN config (arm reaches forward and down to ground)
    theta2_raw = math.acos(cos2)                          # positive = elbow up above arm

    theta1_raw = (math.atan2(dz, r)
                  - math.atan2(L2 * math.sin(theta2_raw),
                               L1 + L2 * math.cos(theta2_raw)))

    # Convert to kuka joint convention:
    # kuka joint 1: 0 = arm vertical up, positive = lean forward
    # kuka joint 2: 0 = arm straight, positive = bend elbow inward
    theta1 = theta1_raw          # already in correct convention (atan2 gives forward angle from horizontal)
    theta2 = -theta2_raw         # negate for elbow-down

    return theta0, theta1, theta2


def ik_home():
    """Safe home: arm pointing up and slightly forward."""
    return (0.0, -0.3, 0.8)
