# IoT Robotic Handwriting Simulator

A PyBullet + CoAP robotic arm that draws text in 3-D.

```
┌──────────────────┐       CoAP PUT        ┌────────────────────┐
│  coap_client_ui  │ ──────────────────▶  │   coap_server.py   │
│  (Tkinter UI)    │   coap://127.0.0.1   │   (aiocoap async)  │
└──────────────────┘       /write          └────────┬───────────┘
                                                    │ queue
                                          ┌─────────▼───────────┐
                                          │  robot_simulator.py  │
                                          │  (PyBullet thread)   │
                                          └─────────────────────┘
```

## Expected behaviour

1. GUI sends text as a CoAP PUT payload to `127.0.0.1:5683/write`.
2. Server decodes text, converts it to 3-D waypoints (letter_data).
3. Waypoints are queued to the simulator thread (non-blocking).
4. Robot arm moves to the writing plane (x=0.55 m in front).
5. Blue lines are drawn for each pen-down segment.
6. Previous drawing is cleared before each new text.
7. Robot returns to home position after writing.

## Architecture notes

- CoAP server is fully async (asyncio).
- PyBullet runs in a background daemon thread with a queue.Queue – server never blocks.
- IK uses a 2-link planar solver. Joint 0 (base) rotates toward writing plane; joints 1 & 2 control reach/height.
- Letters are centred on the robot forward axis (y = 0).
- All 26 letters A–Z + space are supported.
