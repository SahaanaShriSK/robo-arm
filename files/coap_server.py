import asyncio
import logging
from aiocoap import resource, Context, Message, CHANGED

from letter_data import text_to_waypoints
from servo_bridge import draw_waypoints

logging.basicConfig(level=logging.INFO, format="[SERVER] %(message)s")
log = logging.getLogger(__name__)
print("[SERVO_BRIDGE] Module loaded")


SERVER_IP = "172.27.131.184"   # 🔴 CHANGE to your server laptop Wi-Fi IP


class WriteResource(resource.Resource):
    async def render_put(self, request):
        try:
            text = request.payload.decode("utf-8").strip()

            if not text:
                return Message(code=CHANGED, payload=b"Empty text")

            log.info("Received text: %s", text)
            
            waypoints = text_to_waypoints(text)
            log.info("Generated %d waypoints", len(waypoints))
            print("[SERVER] Calling draw_waypoints()")
            draw_waypoints(waypoints)
            print("[SERVER] Finished draw_waypoints()")

            return Message(code=CHANGED, payload=b"Drawing started")

        except Exception as e:
            log.error(str(e))
            return Message(code=CHANGED, payload=b"Error")


async def main():
    root = resource.Site()
    root.add_resource(["write"], WriteResource())

    await Context.create_server_context(root, bind=(SERVER_IP, 5683))

    log.info(f"CoAP server running at coap://{SERVER_IP}:5683/write")

    await asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    asyncio.run(main())