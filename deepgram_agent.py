#!/usr/bin/env python3
"""
Two-way voice conversation with a Deepgram Voice Agent.

Requirements:
    pip install websockets sounddevice numpy

Usage:
    DEEPGRAM_API_KEY=<your-key> python deepgram_agent.py              # GPT-5.5 (default)
    DEEPGRAM_API_KEY=<your-key> python deepgram_agent.py --llm gpt-5.5
    DEEPGRAM_API_KEY=<your-key> python deepgram_agent.py --llm claude-sonnet-4-6
"""

import argparse
import asyncio
import json
import os
import queue
import signal
import sys
import threading

import numpy as np
import sounddevice as sd
import websockets

# ── audio config ──────────────────────────────────────────────────────────────

SAMPLE_RATE  = 16_000
CHANNELS     = 1
DTYPE        = "int16"
CHUNK_FRAMES = int(SAMPLE_RATE * 0.1)   # 100 ms chunks

AGENT_URL = "wss://agent.deepgram.com/v1/agent"

SYSTEM_PROMPT = "You are a helpful voice assistant. Keep replies brief and conversational."

# Supported LLM aliases → (provider_type, model_id)
LLM_CONFIGS: dict[str, tuple[str, str]] = {
    "gpt-5.5":           ("open_ai",   "gpt-5.5"),
    "claude-sonnet-4-6": ("anthropic", "claude-sonnet-4-6"),
}
DEFAULT_LLM = "gpt-5.5"


def build_settings(llm: str) -> dict:
    provider_type, model_id = LLM_CONFIGS[llm]
    return {
        "type": "SettingsConfiguration",
        "audio": {
            "input":  {"encoding": "linear16", "sample_rate": SAMPLE_RATE},
            "output": {"encoding": "linear16", "sample_rate": SAMPLE_RATE, "container": "none"},
        },
        "agent": {
            "listen": {"model": "nova-3"},
            "think": {
                "provider": {"type": provider_type},
                "model": model_id,
                "instructions": SYSTEM_PROMPT,
            },
            "speak": {"model": "aura-2-thalia-en"},
        },
    }


# ── main class ────────────────────────────────────────────────────────────────

class VoiceAgent:
    def __init__(self, api_key: str, llm: str = DEFAULT_LLM) -> None:
        self.api_key = api_key
        self.llm = llm

        # asyncio side
        self._loop: asyncio.AbstractEventLoop | None = None
        self._mic_q: asyncio.Queue[bytes] | None = None
        self._stop = asyncio.Event()

        # thread-safe output buffer for PortAudio playback callback
        self._out_q: queue.Queue[bytes] = queue.Queue()
        self._out_buf = bytearray()
        self._out_lock = threading.Lock()

    # ── sounddevice callbacks (run in PortAudio thread) ───────────────────────

    def _mic_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[mic status] {status}", file=sys.stderr)
        # hand PCM bytes to the asyncio event loop safely
        self._loop.call_soon_threadsafe(self._mic_q.put_nowait, bytes(indata))

    def _play_callback(self, outdata, frames, time_info, status):
        needed = frames * CHANNELS * 2     # int16 = 2 bytes per sample
        with self._out_lock:
            # drain the thread-safe queue into our bytearray buffer
            while len(self._out_buf) < needed:
                try:
                    self._out_buf.extend(self._out_q.get_nowait())
                except queue.Empty:
                    break
            chunk = bytes(self._out_buf[:needed])
            del self._out_buf[:needed]

        # pad with silence if agent hasn't sent enough audio yet
        if len(chunk) < needed:
            chunk += b"\x00" * (needed - len(chunk))

        outdata[:] = np.frombuffer(chunk, dtype=np.int16).reshape(-1, CHANNELS)

    # ── async tasks ───────────────────────────────────────────────────────────

    async def _send_mic(self, ws) -> None:
        """Forward microphone PCM chunks to the WebSocket."""
        while not self._stop.is_set():
            try:
                data = await asyncio.wait_for(self._mic_q.get(), timeout=0.3)
                await ws.send(data)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                break

    async def _recv_loop(self, ws) -> None:
        """Receive messages from the WebSocket; route audio or print events."""
        while not self._stop.is_set():
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                print("\n[disconnected]")
                self._stop.set()
                break

            if isinstance(msg, bytes):
                self._out_q.put_nowait(msg)
            else:
                self._on_event(msg)

    def _on_event(self, raw: str) -> None:
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            return

        t = ev.get("type", "")

        if t == "Welcome":
            print(f"[session] {ev.get('session_id', '—')}")

        elif t == "ConversationText":
            role  = ev.get("role", "")
            text  = ev.get("content", "")
            label = "You  " if role == "user" else "Agent"
            print(f"  {label}: {text}")

        elif t == "UserStartedSpeaking":
            print("  [listening …]", end="\r")

        elif t == "AgentStartedSpeaking":
            print("  [agent speaking]", end="\r")

        elif t == "AgentAudioDone":
            print()   # newline after the \r updates above

        elif t == "Error":
            print(f"[error] {ev.get('description', ev)}", file=sys.stderr)

    # ── entry point ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._mic_q = asyncio.Queue()

        headers = {"Authorization": f"Token {self.api_key}"}
        print("Connecting to Deepgram Voice Agent …")

        async with websockets.connect(AGENT_URL, additional_headers=headers) as ws:
            await ws.send(json.dumps(build_settings(self.llm)))
            print(f"LLM: {self.llm}  |  Ready — speak into your microphone.  Ctrl+C to quit.\n")

            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
                blocksize=CHUNK_FRAMES, callback=self._mic_callback,
            ):
                with sd.OutputStream(
                    samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE,
                    blocksize=CHUNK_FRAMES, callback=self._play_callback,
                ):
                    await asyncio.gather(
                        self._send_mic(ws),
                        self._recv_loop(ws),
                    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-way voice chat with a Deepgram Voice Agent.")
    parser.add_argument(
        "--llm",
        choices=list(LLM_CONFIGS),
        default=DEFAULT_LLM,
        metavar="MODEL",
        help=f"LLM backend for the agent. Choices: {', '.join(LLM_CONFIGS)}. Default: {DEFAULT_LLM}",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        sys.exit("Error: DEEPGRAM_API_KEY environment variable is not set.")

    agent = VoiceAgent(api_key, llm=args.llm)

    loop = asyncio.get_running_loop()

    def _stop(*_):
        print("\nStopping …")
        loop.call_soon_threadsafe(agent._stop.set)

    try:
        # Unix/Mac: preferred asyncio-native signal handling
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _stop)
    except NotImplementedError:
        # Windows: ProactorEventLoop doesn't support add_signal_handler
        signal.signal(signal.SIGINT, _stop)

    await agent.run()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
