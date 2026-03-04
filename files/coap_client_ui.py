"""
coap_client_ui.py
-----------------
Tkinter desktop GUI that sends text to the CoAP server via PUT.

Run this AFTER coap_server.py is already running.

Usage:
    python coap_client_ui.py
"""

import asyncio
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from aiocoap import Context, Message, PUT

SERVER_URI = "coap://127.0.0.1:5683/write"   # ← server laptop IP


# ── CoAP send (runs in a background thread so Tkinter doesn't freeze) ──────────

def _send_coap(text: str, status_var: tk.StringVar):
    """Run an async CoAP PUT in a fresh event loop (called from thread)."""
    async def _put():
        try:
            protocol = await Context.create_client_context()
            request  = Message(code=PUT, uri=SERVER_URI, payload=text.encode())
            response = await protocol.request(request).response
            status_var.set(f"✔ Sent! Server replied: {response.payload.decode()}")
        except Exception as exc:
            status_var.set(f"✖ Error: {exc}")

    asyncio.run(_put())


def on_send(entry: tk.Entry, status_var: tk.StringVar, btn: tk.Button):
    text = entry.get().strip()
    if not text:
        messagebox.showwarning("Empty input", "Please type some text first.")
        return

    status_var.set("Sending …")
    btn.configure(state="disabled")

    def worker():
        _send_coap(text, status_var)
        btn.configure(state="normal")

    threading.Thread(target=worker, daemon=True).start()


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui():
    root = tk.Tk()
    root.title("🤖 IoT Robotic Handwriting – CoAP Client")
    root.resizable(False, False)
    root.configure(padx=20, pady=20)

    # Title
    tk.Label(root, text="Robotic Handwriting Simulator",
             font=("Helvetica", 16, "bold")).pack(pady=(0, 10))

    tk.Label(root, text="Type text below and click Send.\n"
             "The robot will draw your letters in 3-D.",
             justify="center").pack()

    # Entry
    frame = tk.Frame(root)
    frame.pack(pady=10)
    tk.Label(frame, text="Text:").pack(side="left")
    entry = tk.Entry(frame, width=28, font=("Helvetica", 14))
    entry.pack(side="left", padx=6)
    entry.insert(0, "HELLO")
    entry.focus()

    # Status bar
    status_var = tk.StringVar(value="Ready")
    tk.Label(root, textvariable=status_var, fg="gray").pack()

    # Send button
    btn = ttk.Button(root, text="✉  Send via CoAP")
    btn.configure(command=lambda: on_send(entry, status_var, btn))
    btn.pack(pady=8)

    # Allow Enter key to send
    root.bind("<Return>", lambda _: on_send(entry, status_var, btn))

    # Quick preset buttons
    tk.Label(root, text="Quick presets:").pack()
    presets_frame = tk.Frame(root)
    presets_frame.pack()
    for word in ("HELLO", "WORLD", "IOT", "ROBOT", "ABC"):
        ttk.Button(
            presets_frame, text=word,
            command=lambda w=word: (entry.delete(0, "end"),
                                    entry.insert(0, w),
                                    on_send(entry, status_var, btn))
        ).pack(side="left", padx=2, pady=4)

    # CoAP address info
    tk.Label(root,
             text=f"CoAP endpoint: {SERVER_URI}",
             font=("Courier", 9), fg="gray").pack(pady=(10, 0))

    return root


if __name__ == "__main__":
    ui = build_ui()
    ui.mainloop()
