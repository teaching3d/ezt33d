#!/usr/bin/env python3
"""
Two-way voice conversation with a Deepgram Voice Agent.

Requirements:
    pip install websockets sounddevice numpy

Usage:
    DEEPGRAM_API_KEY=<your-key> python deepgram_agent.py              # GPT-5.5 (default)
    DEEPGRAM_API_KEY=<your-key> python deepgram_agent.py --llm gpt-5.5
    DEEPGRAM_API_KEY=<your-key> python deepgram_agent.py --llm claude-sonnet-4-6

Python compatibility: 3.8+
"""

from __future__ import annotations  # defer annotations → X|Y and dict[...] work on 3.8/3.9

import argparse
import asyncio
import importlib.metadata
import json
import os
import queue
import signal
import sys
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
import websockets

# ── Python version flags ──────────────────────────────────────────────────────

_PY311 = sys.version_info >= (3, 11)   # asyncio.TaskGroup

# ── websockets version: header kwarg name changed across major versions ───────

def _ws_connect(url: str, headers: dict) -> object:
    """Return a websockets connection context manager with the correct header kwarg."""
    _ver = tuple(int(x) for x in importlib.metadata.version("websockets").split(".")[:2])
    if _ver >= (10, 0):
        # additional_headers is correct for websockets 10+ (including the v13 asyncio-native client)
        return websockets.connect(url, additional_headers=headers)
    else:
        # websockets < 10 used the name extra_headers
        return websockets.connect(url, extra_headers=headers)

# ── audio config ──────────────────────────────────────────────────────────────

SAMPLE_RATE  = 24_000   # 24 kHz — Deepgram spec default; better quality than 16 kHz
CHANNELS     = 1
DTYPE        = "int16"
CHUNK_FRAMES = int(SAMPLE_RATE * 0.1)   # 100 ms chunks

# GA endpoint (wss://api.deepgram.com, not the old agent.deepgram.com subdomain)
AGENT_URL = "wss://api.deepgram.com/v1/agent/converse"

SYSTEM_PROMPT = "You are a helpful voice assistant. Keep replies brief and conversational."

# Supported LLM aliases → (provider_type, model_id)
# provider_type values are Deepgram enum strings ("open_ai", "anthropic", …)
LLM_CONFIGS: dict[str, tuple[str, str]] = {
    "gpt-5.5":           ("open_ai",   "gpt-5.5"),
    "claude-sonnet-4-6": ("anthropic", "claude-sonnet-4-6"),
}
DEFAULT_LLM = "gpt-5.5"


def build_settings(llm: str) -> dict:
    provider_type, model_id = LLM_CONFIGS[llm]
    return {
        # "SettingsConfiguration" was renamed to "Settings" in the v1 GA release
        "type": "Settings",
        "audio": {
            "input":  {"encoding": "linear16", "sample_rate": SAMPLE_RATE},
            "output": {"encoding": "linear16", "sample_rate": SAMPLE_RATE, "container": "none"},
        },
        "agent": {
            # model now lives inside provider (GA structure)
            "listen": {
                "provider": {"type": "deepgram", "model": "nova-3"},
            },
            "think": {
                "provider": {"type": provider_type, "model": model_id},
                # "instructions" was renamed to "prompt" in the GA release
                "prompt": SYSTEM_PROMPT,
            },
            "speak": {
                "provider": {"type": "deepgram", "model": "aura-2-thalia-en"},
            },
        },
    }


# ── main class ────────────────────────────────────────────────────────────────

class VoiceAgent:
    def __init__(self, api_key: str, llm: str = DEFAULT_LLM) -> None:
        self.api_key = api_key
        self.llm = llm

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._mic_q: Optional[asyncio.Queue[bytes]] = None
        self._stop = asyncio.Event()

        # thread-safe output buffer for PortAudio playback callback
        self._out_q: queue.Queue[bytes] = queue.Queue()
        self._out_buf = bytearray()
        self._out_lock = threading.Lock()

    # ── sounddevice callbacks (run in PortAudio thread) ───────────────────────

    def _mic_callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[mic status] {status}", file=sys.stderr)
        self._loop.call_soon_threadsafe(self._mic_q.put_nowait, bytes(indata))

    def _play_callback(self, outdata, frames, time_info, status) -> None:
        needed = frames * CHANNELS * 2     # int16 = 2 bytes per sample
        with self._out_lock:
            while len(self._out_buf) < needed:
                try:
                    self._out_buf.extend(self._out_q.get_nowait())
                except queue.Empty:
                    break
            chunk = bytes(self._out_buf[:needed])
            del self._out_buf[:needed]

        if len(chunk) < needed:
            chunk += b"\x00" * (needed - len(chunk))

        outdata[:] = np.frombuffer(chunk, dtype=np.int16).reshape(-1, CHANNELS)

    def _clear_output_buffer(self) -> None:
        """Discard buffered agent audio (called on barge-in)."""
        with self._out_lock:
            self._out_buf.clear()
        while True:
            try:
                self._out_q.get_nowait()
            except queue.Empty:
                break

    # ── async tasks ───────────────────────────────────────────────────────────

    async def _send_mic(self, ws) -> None:
        while not self._stop.is_set():
            try:
                data = await asyncio.wait_for(self._mic_q.get(), timeout=0.3)
                await ws.send(data)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                break

    async def _recv_loop(self, ws) -> None:
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
            print(f"[session] {ev.get('request_id', '—')}")
        elif t == "SettingsApplied":
            pass   # settings acknowledged; audio can now flow
        elif t == "ConversationText":
            role  = ev.get("role", "")
            text  = ev.get("content", "")
            label = "You  " if role == "user" else "Agent"
            print(f"  {label}: {text}")
        elif t == "UserStartedSpeaking":
            # Barge-in: drop any agent audio that hasn't played yet
            self._clear_output_buffer()
            print("  [listening …]", end="\r")
        elif t == "AgentThinking":
            print("  [thinking …] ", end="\r")
        elif t == "AgentStartedSpeaking":
            print("  [agent speaking]", end="\r")
        elif t == "AgentAudioDone":
            print()
        elif t == "Warning":
            print(f"[warning] {ev.get('description', ev)}", file=sys.stderr)
        elif t == "Error":
            print(f"[error] {ev.get('description', ev)}", file=sys.stderr)

    # ── entry point ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._mic_q = asyncio.Queue()

        headers = {"Authorization": f"Token {self.api_key}"}
        print("Connecting to Deepgram Voice Agent …")

        async with _ws_connect(AGENT_URL, headers) as ws:
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
                    if _PY311:
                        # TaskGroup gives better error propagation (Python 3.11+)
                        async with asyncio.TaskGroup() as tg:
                            tg.create_task(self._send_mic(ws))
                            tg.create_task(self._recv_loop(ws))
                    else:
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
        help=f"LLM backend. Choices: {', '.join(LLM_CONFIGS)}. Default: {DEFAULT_LLM}",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not api_key:
        sys.exit("Error: DEEPGRAM_API_KEY environment variable is not set.")

    agent = VoiceAgent(api_key, llm=args.llm)

    loop = asyncio.get_running_loop()

    def _stop(*_) -> None:
        print("\nStopping …")
        loop.call_soon_threadsafe(agent._stop.set)

    try:
        # Unix/Mac: asyncio-native signal handling
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _stop)
    except NotImplementedError:
        # Windows: ProactorEventLoop doesn't support add_signal_handler
        signal.signal(signal.SIGINT, _stop)

    await agent.run()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
