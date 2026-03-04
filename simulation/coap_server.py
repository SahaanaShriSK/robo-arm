"""
coap_server.py
--------------
Async CoAP server (aiocoap) that:
  1. Listens on coap://127.0.0.1:5683/write  (PUT)
  2. Decodes incoming text payload
  3. Converts text → waypoints  (letter_data)
  4. Submits waypoints to the robot simulator (non-blocking, via queue)

Run this file first, then run coap_client_ui.py in a second terminal.

Usage:
    python coap_server.py
"""

import asyncio
import logging

from aiocoap import resource, Context, Message, CHANGED

from letter_data import text_to_waypoints
from robot_simulator import get_simulator

logging.basicConfig(level=logging.INFO, format="[SERVER] %(message)s")
log = logging.getLogger(__name__)


class WriteResource(resource.Resource):
    """CoAP resource at /write – handles PUT requests."""

    async def render_put(self, request):
        text = request.payload.decode("utf-8", errors="replace").strip()
        if not text:
            return Message(code=CHANGED, payload=b"Empty text ignored")

        log.info("Received text: %r", text)

        # Convert to waypoints (fast, synchronous, CPU-light)
        waypoints = text_to_waypoints(text)
        log.info("Generated %d waypoints", len(waypoints))

        # Submit to simulator (non-blocking – uses an internal queue)
        sim = get_simulator()
        sim.submit_waypoints(waypoints)

        return Message(code=CHANGED, payload=b"Drawing started")


async def main():
    # Make sure the simulator is running before accepting requests
    log.info("Initialising robot simulator …")
    get_simulator()   # starts background thread + PyBullet GUI

    # Build CoAP site
    root = resource.Site()
    root.add_resource(["write"], WriteResource())

    await Context.create_server_context(root, bind=("127.0.0.1", 5683))
    log.info("CoAP server listening on coap://127.0.0.1:5683/write")
    log.info("Send text via the UI client or: aiocoap-client -m PUT coap://127.0.0.1/write --payload 'HELLO'")

    # Run forever
    await asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    asyncio.run(main())
