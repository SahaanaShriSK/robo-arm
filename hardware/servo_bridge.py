import serial
import time
import math
from inverse_kinematics import solve_ik

SERIAL_PORT = "COM8"
BAUD_RATE = 115200

# ===== CALIBRATION =====
WRITE_Z = 2.3
LIFT_Z = 5.5

X_OFFSET = 7.5   # distance of paper from base (cm)
Y_CENTER = 0.0   # center of writing

SCALE = 0.16     # letter scale (0.14–0.20)

METERS_TO_CM = 100.0
# =======================

ser = None


def init_serial():
    global ser
    if ser is None:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("[SERIAL] Connected")


def send_angles(theta0, theta1, theta2):
    global ser

    if ser is None:
        init_serial()

    # radians → servo degrees
    base = int(math.degrees(theta0)) + 90
    shoulder = int(math.degrees(theta1)) + 90
    elbow = int(math.degrees(theta2)) + 90

    # safe limits
    base = max(30, min(150, base))
    shoulder = max(40, min(140, shoulder))
    elbow = max(30, min(150, elbow))

    cmd = f"{base},{shoulder},{elbow}\n"
    ser.write(cmd.encode())
    time.sleep(0.08)


# 🔴 THIS FUNCTION FIXES FORWARD/BACKWARD MOTION
def move_to_xyz(x_cm, y_cm, z_cm):
    t0, t1, t2 = solve_ik(x_cm, y_cm, z_cm)
    send_angles(t0, t1, t2)


def draw_waypoints(waypoints):
    init_serial()
    print(f"[SERIAL] Points: {len(waypoints)}")

    for x, y, z, pen in waypoints:

        # ===== MAP LETTER DATA → REAL ARM SPACE =====
        X = (x * METERS_TO_CM * SCALE) - 5.0 + X_OFFSET
        Y = (y * METERS_TO_CM * SCALE) + Y_CENTER
        Z = WRITE_Z if pen == 1 else LIFT_Z
        # ============================================

        move_to_xyz(X, Y, Z)