import math

# 🔴 SET YOUR ARM LENGTHS IN CM
L1 = 11.0   # shoulder → elbow
L2 = 11.0   # elbow → pen tip


def solve_ik(x, y, z):
    # base rotation
    theta0 = math.atan2(y, x)

    # planar distance
    r = math.sqrt(x**2 + y**2)

    # height from shoulder
    z_shoulder = z

    d = math.sqrt(r**2 + z_shoulder**2)

    # clamp to reachable workspace
    d = min(max(d, 1.0), L1 + L2 - 0.1)

    # cosine law
    cos_elbow = (L1**2 + L2**2 - d**2) / (2 * L1 * L2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    theta2 = math.pi - math.acos(cos_elbow)

    cos_shoulder = (d**2 + L1**2 - L2**2) / (2 * L1 * d)
    cos_shoulder = max(-1.0, min(1.0, cos_shoulder))
    theta1 = math.atan2(z_shoulder, r) + math.acos(cos_shoulder)

    return theta0, theta1, theta2