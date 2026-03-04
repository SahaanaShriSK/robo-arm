from servo_bridge import init_serial, send_angles
from inverse_kinematics import solve_ik
import time

init_serial()

# ===== REALISTIC ARM POSITION =====
# Tune ONLY these numbers
X = 7.5      # distance to paper (cm)
Y = 0.0      # center
Z_UP = 5.5   # pen up
Z_DOWN = 2.3 # pen touching paper
# ==================================

def move(x, y, z):
    t0, t1, t2 = solve_ik(x, y, z)
    send_angles(t0, t1, t2)
    time.sleep(1)

# 1. move above start
move(X, Y, Z_UP)

# 2. pen down
move(X, Y, Z_DOWN)

# 3. draw vertical stroke
move(X, Y - 2.0, Z_DOWN)

# 4. pen up
move(X, Y - 2.0, Z_UP)