# IoT-Enabled Handwriting Robotic Arm Using AI-Based Stroke Generation and Optimized Inverse Kinematics

An interactive robotics project that simulates a robotic arm capable of
writing user-input text on a virtual surface.\
The system integrates **AI-based handwriting generation, robotics
simulation, and IoT communication** to demonstrate how robotic
manipulators can perform handwriting tasks.

Users enter text through a client interface. The text is transmitted to
a server using the **CoAP protocol**, converted into stroke waypoints,
and executed by a robotic arm inside a **PyBullet simulation**.

------------------------------------------------------------------------

# Project Overview

This project demonstrates how **Artificial Intelligence, Robotics, and
IoT systems** can work together to produce robotic handwriting.

The system pipeline:

Text Input → AI Stroke Generation → Coordinate Mapping → Inverse
Kinematics → Robotic Arm Motion → Handwriting Output

The robotic arm writes the text on a virtual paper surface inside a
physics-based simulation environment.

------------------------------------------------------------------------

# Technologies Used

-   Python
-   PyBullet Physics Simulator
-   CoAP (Constrained Application Protocol)
-   AsyncIO
-   Robotics Inverse Kinematics
-   AI Handwriting Generation
-   GUI-based client interface

The simulator uses the **KUKA iiwa industrial robotic arm**, commonly
used in robotics research and industrial automation.

------------------------------------------------------------------------

# System Architecture

User Interface (Client) │ │ CoAP Request ▼ CoAP Server │ │ Converts text
→ stroke waypoints ▼ Robot Simulator │ ▼ PyBullet Physics Engine │ ▼
Robotic Arm Draws Text

------------------------------------------------------------------------

# 🖊️ How the System Works

The robotic handwriting system converts input text into robotic arm
motion through several processing stages.\
The pipeline combines **AI handwriting generation, stroke processing,
coordinate normalization, and robotic kinematics**.

------------------------------------------------------------------------

## 1️⃣ AI-Based Handwriting Stroke Generation

The system uses the **Graves Handwriting Synthesis Model**.

**Architecture** - LSTM (Long Short-Term Memory Network) - Mixture
Density Network (MDN)

**Training Dataset** - IAM Handwriting Dataset

**Input** - Text sequence

**Output** Continuous handwriting stroke coordinates:

(xₜ, yₜ, pₜ)

Where:

xₜ , yₜ → Cartesian stroke coordinates\
pₜ → Pen state

Pen state meaning:

1 → Pen Down (writing)\
0 → Pen Up (movement without writing)

------------------------------------------------------------------------

## 2️⃣ Stroke Representation & Pen Control Logic

Each handwriting sample is represented as:

(xₜ, yₜ, pₜ)

Pen control logic determines whether the robot writes or moves without
drawing.

Pen lifts occur between:

-   strokes
-   letters
-   words

This ensures **natural handwriting behavior**.

------------------------------------------------------------------------

## 3️⃣ Coordinate Normalization & Workspace Mapping

Generated stroke coordinates are normalized to:

\[0,1\] × \[0,1\]

This ensures:

-   Safe robot motion
-   Collision-free operation
-   Consistent handwriting scale
-   Proper mapping to the robot workspace

Coordinates are then scaled to the drawing surface.

------------------------------------------------------------------------

## 4️⃣ Mathematical Modeling of the Robotic Arm

The robotic arm is modeled as a **3‑DOF planar manipulator**.

Robot structure:

Base → Shoulder → Elbow → Wrist → Pen

Configuration:

-   Three revolute joints
-   Fixed base
-   End-effector holding a pen

Assumptions:

-   Rigid links
-   Planar motion
-   No joint elasticity

------------------------------------------------------------------------

## 5️⃣ Inverse Kinematics Formulation

Inverse Kinematics computes the required joint angles to reach a target
point.

Objective:

Compute joint angles

θ₁, θ₂, θ₃

Given end-effector position:

(x, y, z)

The IK solver calculates the required joint angles so that the pen
reaches the target coordinate.

------------------------------------------------------------------------

## 6️⃣ Robotic Drawing Execution

The drawing process works as follows:

1.  The robotic arm moves to the target coordinate.
2.  If p = 1, the pen touches the surface and draws a stroke.
3.  If p = 0, the pen lifts and moves without drawing.
4.  The process repeats for the entire stroke sequence.

The result is **smooth robotic handwriting** inside the simulation.

------------------------------------------------------------------------

# 🔬 Applications

-   Robotics education
-   AI handwriting generation
-   IoT-based robot control
-   Industrial robot simulation
-   Human--robot interaction research

------------------------------------------------------------------------

# 🔮 Future Improvements

-   Support for cursive handwriting
-   More realistic robotic arm models
-   Smooth trajectory generation
-   Integration with physical robotic hardware
-   AI-generated handwriting styles

------------------------------------------------------------------------

# 📜 License

This project is released under the **MIT License**.
