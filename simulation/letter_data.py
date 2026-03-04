"""
letter_data.py — Text → flat ground waypoints [x, y, z, pen]

Bird's-eye view (robot at origin, facing +X direction):

   Robot
    (0,0)──────────────────→ +X (forward)
      |
      ↓ +Y (right side)

  Text written on the ground in front of robot:
    - Letters go LEFT → RIGHT along the Y axis
      (left = negative Y, right = positive Y)
    - Letter HEIGHT goes forward along X axis
      (bottom of letter = closer to robot = smaller X,
       top of letter    = further from robot = larger X)

  So from the camera looking down, text reads normally left-to-right.

Letter size: LW=0.18m (Y width), LH=0.22m (X height)
"""
import math

LW  = 0.18    # letter width  along Y axis
LH  = 0.22    # letter height along X axis  
GAP = 0.05    # gap between letters
X0  = 0.32    # X of letter BOTTOM (closest to robot)
PZ  = 0.012   # pen height — raised to be clearly visible above ground


def _make(pts):
    return pts


# Normalised strokes: (nx, ny, pen)
#   nx: 0=left edge of letter, 1=right edge  → maps to Y axis
#   ny: 0=BOTTOM of letter, 1=TOP            → maps to X axis (forward)
#
# IMPORTANT: to read correctly from above (camera top-down),
#   left→right in the letter = increasing Y in world space
#   Letters are NOT mirrored — nx=0 is leftmost stroke

LETTERS = {
  'A': [(0,0,0),(0.5,1,1),(1,0,1),(0.2,0.5,0),(0.8,0.5,1)],

  'B': [(0,0,0),(0,1,1),
        (0,1,0),(0.65,1,1),(1,0.82,1),(0.65,0.5,1),(0,0.5,1),
        (0,0.5,0),(0.65,0.5,1),(1,0.18,1),(0.65,0,1),(0,0,1)],

  'C': [(1,0.82,0)] + [
        (0.5+0.5*math.cos(math.radians(15+330*i/20)),
         0.5+0.5*math.sin(math.radians(15+330*i/20)), 1)
        for i in range(21)],

  'D': [(0,0,0),(0,1,1),(0.6,1,1),(1,0.75,1),(1,0.25,1),(0.6,0,1),(0,0,1)],

  'E': [(1,1,0),(0,1,1),(0,0,1),(1,0,1),(0,0.5,0),(0.75,0.5,1)],

  'F': [(1,1,0),(0,1,1),(0,0,1),(0,0.5,0),(0.75,0.5,1)],

  'G': [(1,0.82,0)] + [
        (0.5+0.5*math.cos(math.radians(15+300*i/18)),
         0.5+0.5*math.sin(math.radians(15+300*i/18)), 1 if i>0 else 0)
        for i in range(19)] + [(0.5,0.5,0),(0.5,0.5,1),(1,0.5,1)],

  'H': [(0,0,0),(0,1,1),
        (0,0.5,0),(1,0.5,1),
        (1,1,0),(1,1,1),(1,0,1)],

  'I': [(0,1,0),(1,1,1),(0.5,1,0),(0.5,0,1),(0,0,0),(1,0,1)],

  'J': [(0,1,0),(1,1,1),(0.5,1,0),(0.5,0.25,1)] + [
        (0.5+0.5*math.cos(math.radians(-90-180*i/10)),
         0.25+0.25*math.sin(math.radians(-90-180*i/10)), 1)
        for i in range(1, 11)],

  'K': [(0,0,0),(0,1,1),(0,0.5,0),(1,1,1),(0,0.5,0),(1,0,1)],

  'L': [(0,1,0),(0,0,1),(1,0,1)],

  'M': [(0,0,0),(0,1,1),(0.5,0.5,1),(1,1,1),(1,0,1)],

  'N': [(0,0,0),(0,1,1),(1,0,1),(1,1,1)],

  'O': [(0.5,1,0)] + [
        (0.5+0.5*math.cos(math.pi/2+2*math.pi*i/24),
         0.5+0.5*math.sin(math.pi/2+2*math.pi*i/24), 1)
        for i in range(25)],

  'P': [(0,0,0),(0,1,1),(0.65,1,1),(1,0.8,1),(1,0.55,1),(0.65,0.5,1),(0,0.5,1)],

  'Q': [(0.5,1,0)] + [
        (0.5+0.5*math.cos(math.pi/2+2*math.pi*i/24),
         0.5+0.5*math.sin(math.pi/2+2*math.pi*i/24), 1)
        for i in range(25)] + [(0.6,0.3,0),(1,0,1)],

  'R': [(0,0,0),(0,1,1),(0.65,1,1),(1,0.8,1),(1,0.55,1),(0.65,0.5,1),(0,0.5,1),
        (0.5,0.5,0),(1,0,1)],

  'S': [(0.5+0.5*math.cos(math.radians(45 + 225*i/12)),
         0.75+0.25*math.sin(math.radians(45 + 225*i/12)), 1 if i>0 else 0)
        for i in range(13)] +
       [(0.5+0.5*math.cos(math.radians(90 - 225*i/12)),
         0.25+0.25*math.sin(math.radians(90 - 225*i/12)), 1)
        for i in range(1, 13)],

  'T': [(0,1,0),(1,1,1),(0.5,1,0),(0.5,0,1)],

  'U': [(0,1,0),(0,0.3,1)] + [
        (0.5+0.5*math.cos(math.radians(180+180*i/10)),
         0.3+0.3*math.sin(math.radians(180+180*i/10)), 1)
        for i in range(11)] + [(1,1,1)],

  'V': [(0,1,0),(0.5,0,1),(1,1,1)],

  'W': [(0,1,0),(0.2,0,1),(0.5,0.4,1),(0.8,0,1),(1,1,1)],

  'X': [(0,1,0),(1,0,1),(0,0,0),(1,1,1)],

  'Y': [(0,1,0),(0.5,0.5,1),(1,1,0),(1,1,1),(0.5,0.5,1),(0.5,0,1)],

  'Z': [(0,1,0),(1,1,1),(0,0,1),(1,0,1)],

  ' ': [],
}


def text_to_waypoints(text):
    """
    Convert text → [x, y, z, pen] world coordinates.

    Mapping so text reads left→right when viewed from above:
      nx (0=left, 1=right in letter) → world_y  (negative=left, positive=right)
      ny (0=bottom, 1=top of letter) → world_x  (X0 = bottom, X0+LH = top)
      z  = PZ  (fixed height above ground, clearly visible)
    """
    text = text.upper()

    # Total Y-span for centering
    total_w = 0.0
    for ch in text:
        total_w += (LW * 0.5 + GAP) if ch == ' ' else (LW + GAP)
    total_w -= GAP

    # Screen RIGHT is +Y, Screen LEFT is -Y.
    # We want text to start on the left and read normally to the right.
    # So y_start should be the most negative Y.
    y_start = -(total_w / 2.0)
    y_cursor = y_start
    waypoints = []

    for ch in text:
        if ch not in LETTERS:
            y_cursor += LW + GAP
            continue
        strokes = LETTERS[ch]
        if not strokes:
            y_cursor += LW * 0.5 + GAP
            continue

        for nx, ny, pen in strokes:
            # We want letters to read normally left-to-right and upright.
            # In the camera view (Yaw=90):
            #  - Screen RIGHT is world +Y
            #  - Screen LEFT is world -Y
            #  - Screen UP is world -X  (Robot at X=0 is above the text at X=0.32)
            #  - Screen DOWN is world +X
            
            # For each letter:
            # Left side of letter (nx=0) should be Screen LEFT (-Y) -> y_cursor
            # Right side of letter (nx=1) should be Screen RIGHT (+Y) -> y_cursor + LW
            wy = y_cursor + nx * LW

            # Bottom of letter (ny=0) should be Screen DOWN (+X) -> X0 + LH
            # Top of letter (ny=1) should be Screen UP (-X) -> X0
            wx = X0 + LH - ny * LH

            waypoints.append([round(wx, 4), round(wy, 4), PZ, int(pen)])

        y_cursor += LW + GAP

    return waypoints