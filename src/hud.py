"""Heads-up display: performance, tracking, chord state and visualisers."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence

import cv2
import numpy as np

from effects import glass_panel
from themes import Theme, mix, scale

try:
    import psutil
    _PROCESS: Optional["psutil.Process"] = psutil.Process()
except Exception:
    psutil = None
    _PROCESS = None

FONT = cv2.FONT_HERSHEY_DUPLEX
MONO = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class HudState:
    """Everything the HUD needs for one frame."""

    fps: float = 0.0
    frame_ms: float = 0.0
    capture_fps: float = 0.0
    latency_ms: float = 0.0
    confidence: float = 0.0
    chord: str = "OFF"
    modifier: str = "Major"
    note_label: str = "--"
    volume: float = 0.75
    tone: str = "acoustic"
    theme_label: str = ""
    gesture: str = "none"
    camera_ok: bool = True
    tracking_ok: bool = True
    muted: bool = False
    paused: bool = False
    particles: int = 0


class Meter:
    """A rolling series rendered as a filled sparkline."""

    def __init__(self, length: int = 64):
        self.values: Deque[float] = deque([0.0] * length, maxlen=length)

    def push(self, value: float) -> None:
        """Append a sample, clamped to [0, 1]."""
        self.values.append(float(np.clip(value, 0.0, 1.0)))

    def draw(
        self,
        frame: np.ndarray,
        rect: tuple[int, int, int, int],
        colour: tuple[int, int, int],
        baseline: tuple[int, int, int],
    ) -> None:
        """Render the series inside a rectangle."""
        x, y, width, height = rect
        count = len(self.values)
        if count < 2 or width <= 1:
            return

        step = width / (count - 1)
        points = [
            (int(x + i * step), int(y + height - v * height))
            for i, v in enumerate(self.values)
        ]

        cv2.line(frame, (x, y + height), (x + width, y + height), baseline, 1, cv2.LINE_AA)

        # Blend only inside the meter's own rectangle. Compositing over the
        # whole frame here cost more than every other HUD element combined.
        frame_h, frame_w = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(frame_w, x + width + 1), min(frame_h, y + height + 1)
        if x1 > x0 and y1 > y0:
            region = frame[y0:y1, x0:x1]
            local = np.array(
                [(px - x0, py - y0) for px, py in points]
                + [(x + width - x0, y + height - y0), (x - x0, y + height - y0)],
                np.int32,
            )
            overlay = region.copy()
            cv2.fillPoly(overlay, [local], scale(colour, 0.35))
            cv2.addWeighted(overlay, 0.45, region, 0.55, 0, dst=region)

        cv2.polylines(frame, [np.array(points, np.int32)], False, colour, 1, cv2.LINE_AA)


class Waveform:
    """Decaying bar visualiser driven by string energy."""

    def __init__(self, bars: int = 6):
        self.levels = np.zeros(bars, np.float32)

    def set(self, energies: Sequence[float]) -> None:
        """Track the peak of each string, decaying when it falls."""
        for i, energy in enumerate(energies[: len(self.levels)]):
            self.levels[i] = max(float(energy), float(self.levels[i]))

    def update(self, dt: float) -> None:
        """Decay every bar toward silence."""
        self.levels *= math.exp(-3.5 * dt)

    def draw(
        self,
        frame: np.ndarray,
        rect: tuple[int, int, int, int],
        theme: Theme,
    ) -> None:
        """Render the bars inside a rectangle."""
        x, y, width, height = rect
        count = len(self.levels)
        gap = 4
        bar_w = max(2, (width - gap * (count - 1)) // count)

        for i, level in enumerate(self.levels):
            bx = x + i * (bar_w + gap)
            magnitude = float(np.clip(level, 0.0, 1.0))
            bar_h = max(2, int(magnitude * height))
            top = y + height - bar_h
            colour = mix(theme.secondary, theme.string_hot, magnitude)
            cv2.rectangle(frame, (bx, y), (bx + bar_w, y + height),
                          scale(theme.panel, 1.4), -1)
            cv2.rectangle(frame, (bx, top), (bx + bar_w, y + height), colour, -1)
            if magnitude > 0.15:
                cv2.rectangle(frame, (bx, top), (bx + bar_w, top + 2), theme.highlight, -1)


class Hud:
    """Draws every overlay panel."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.fps_meter = Meter(72)
        self.velocity_meter = Meter(72)
        self.waveform = Waveform(6)
        self._cpu = 0.0
        self._cpu_checked = 0.0
        self._flash = 0.0

    def set_string_count(self, count: int) -> None:
        """Resize the level meter when the instrument changes."""
        self.waveform = Waveform(count)

    def flash(self) -> None:
        """Briefly highlight the chord panel after a strum."""
        self._flash = 1.0

    def update(self, dt: float, state: HudState, energies: Sequence[float]) -> None:
        """Advance meters and sample system load."""
        self.fps_meter.push(state.fps / 90.0)
        self.velocity_meter.push(max(energies) if energies else 0.0)
        self.waveform.set(energies)
        self.waveform.update(dt)
        self._flash = max(0.0, self._flash - dt * 3.0)

        # Sampling CPU is not free, so poll it at 2 Hz.
        now = time.monotonic()
        if _PROCESS is not None and now - self._cpu_checked > 0.5:
            try:
                self._cpu = _PROCESS.cpu_percent()
            except Exception:
                self._cpu = 0.0
            self._cpu_checked = now

    def draw(self, frame: np.ndarray, theme: Theme, state: HudState) -> None:
        """Render the status strip."""
        self._draw_strip(frame, theme, state)
        if state.paused or state.muted:
            self._draw_banner(frame, theme, state)

    # -------------------------------------------------------------- panels

    def _draw_strip(self, frame: np.ndarray, theme: Theme, state: HudState) -> None:
        """One bottom strip carrying every readout.

        Four floating corner panels used to fight the interface for space and
        each cost its own region blur. Everything worth showing fits on a
        single line along the bottom, out of the way of both hands.
        """
        height = 72
        margin = max(24, int(self.width * 0.03))
        rect = (margin, self.height - height - 20, self.width - margin * 2, height)
        glass_panel(frame, rect, theme, radius=14,
                    opacity=0.58 + self._flash * 0.18)

        x, y = rect[0] + 20, rect[1]
        mid = y + 46

        # Chord, the one thing worth reading at a glance.
        name = state.chord
        cv2.putText(frame, name, (x, mid), FONT, 1.15,
                    mix(theme.text, theme.highlight, self._flash), 2, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(name, FONT, 1.15, 2)

        if state.chord != "OFF":
            cv2.putText(frame, state.modifier, (x + tw + 14, y + 30),
                        MONO, 0.46, theme.accent, 1, cv2.LINE_AA)
            cv2.putText(frame, state.note_label, (x + tw + 14, y + 52),
                        MONO, 0.42, theme.text_dim, 1, cv2.LINE_AA)

        # String activity bars, centred.
        bars_x = rect[0] + rect[2] // 2 - 60
        self.waveform.draw(frame, (bars_x, y + 16, 120, 40), theme)

        # Right-hand block: tone, volume, tracking and frame rate.
        right = rect[0] + rect[2] - 20
        fps_text = f"{state.fps:.0f} FPS"
        (fw, _), _ = cv2.getTextSize(fps_text, MONO, 0.5, 1)
        fps_colour = theme.accent if state.fps >= 40 else (
            theme.text_dim if state.fps >= 22 else theme.warn
        )
        cv2.putText(frame, fps_text, (right - fw, y + 28), MONO, 0.5,
                    fps_colour, 1, cv2.LINE_AA)

        status = state.tone if state.tracking_ok and state.camera_ok else (
            "NO CAMERA" if not state.camera_ok else "NO TRACKING"
        )
        (sw, _), _ = cv2.getTextSize(status, MONO, 0.44, 1)
        cv2.putText(frame, status, (right - sw, y + 52), MONO, 0.44,
                    theme.text_dim if state.tracking_ok and state.camera_ok else theme.warn,
                    1, cv2.LINE_AA)

        # Volume bar and a tracking-confidence dot.
        bar_w = 110
        bx = right - fw - bar_w - 30
        cv2.rectangle(frame, (bx, y + 22), (bx + bar_w, y + 30),
                      scale(theme.secondary, 0.45), -1)
        filled = int(bar_w * float(np.clip(state.volume, 0.0, 1.0)))
        cv2.rectangle(frame, (bx, y + 22), (bx + filled, y + 30),
                      theme.warn if state.muted else theme.accent, -1)

        dot = theme.accent if state.confidence > 0.5 else (
            theme.text_dim if state.confidence > 0.2 else theme.warn
        )
        cv2.circle(frame, (bx + 6, y + 50), 5, dot, -1, cv2.LINE_AA)
        cv2.putText(frame, f"{int(state.confidence * 100)}%", (bx + 18, y + 55),
                    MONO, 0.42, theme.text_dim, 1, cv2.LINE_AA)

    def _draw_banner(self, frame: np.ndarray, theme: Theme, state: HudState) -> None:
        """Centre banner shown while muted or paused."""
        label = "PAUSED" if state.paused else "MUTED"
        (tw, th), _ = cv2.getTextSize(label, FONT, 1.4, 3)
        cx = (self.width - tw) // 2
        # Sit in the clear band between the instrument and the status strip;
        # at the top it covered the chord buttons it was drawn over.
        cy = int(self.height * 0.80)
        glass_panel(frame, (cx - 26, cy - th - 18, tw + 52, th + 36), theme, opacity=0.7)
        cv2.putText(frame, label, (cx, cy), FONT, 1.4, theme.warn, 3, cv2.LINE_AA)


HELP_LINES: tuple[tuple[str, str], ...] = (
    ("HOW TO PLAY", ""),
    ("point up", "hold a finger over a chord button to pick it"),
    ("pinch", "pick the chord under your finger instantly"),
    ("sweep down", "strum across the strings with your other hand"),
    ("", ""),
    ("GESTURES", ""),
    ("swipe", "previous / next chord"),
    ("thumbs up", "electric tone"),
    ("peace", "acoustic tone"),
    ("rock horns", "distortion tone"),
    ("ok sign", "fingerstyle tone"),
    ("fist", "mute"),
    ("open hand", "unmute"),
    ("", ""),
    ("KEYBOARD", ""),
    ("ESC / Q", "quit"),
    ("1 - 5", "themes"),
    ("G", "guitar: acoustic / electric / bass / ukulele"),
    ("Z X C V B N", "tone: acoustic electric clean dist mute finger"),
    ("[  ]", "previous / next chord"),
    ("K", "cycle chord quality (7, maj7, sus, power)"),
    ("- =", "volume down / up"),
    ("M / P / R", "mute / pause / reset"),
    ("H", "hide this help"),
    ("D", "debug landmarks"),
)


def draw_help(frame: np.ndarray, theme: Theme, width: int, height: int) -> None:
    """Draw the keyboard and gesture reference overlay."""
    panel_w, panel_h = 480, 25 * len(HELP_LINES) + 34
    x = (width - panel_w) // 2
    y = max(10, (height - panel_h) // 2)
    glass_panel(frame, (x, y, panel_w, panel_h), theme, opacity=0.80)

    ty = y + 34
    for key, description in HELP_LINES:
        if key and not description:
            cv2.putText(frame, key, (x + 22, ty), FONT, 0.5, theme.accent, 1, cv2.LINE_AA)
        elif key:
            cv2.putText(frame, key, (x + 22, ty), MONO, 0.44, theme.text, 1, cv2.LINE_AA)
            cv2.putText(frame, description, (x + 168, ty), MONO, 0.40,
                        theme.text_dim, 1, cv2.LINE_AA)
        ty += 25


def _self_check() -> None:
    """Verify HUD rendering and meter behaviour on a synthetic frame."""
    from themes import THEMES

    theme = THEMES["cyber_blue"]
    width, height = 1280, 720
    frame = np.full((height, width, 3), 30, np.uint8)

    hud = Hud(width, height)
    state = HudState(
        fps=59.4, frame_ms=16.2, latency_ms=11.9, confidence=0.86,
        chord="Am", modifier="Minor", note_label="A2 E3 A3 C4 E4",
        volume=0.75, tone="acoustic", theme_label="CYBER BLUE",
        gesture="pinch", particles=120,
    )

    energies = [0.9, 0.7, 0.5, 0.3, 0.1, 0.0]
    hud.update(0.016, state, energies)
    hud.draw(frame, theme, state)
    assert frame.any(), "HUD should mark the frame"

    # The strip must be drawn, and must stay inside its own band so it never
    # paints over the chord bar or the strings.
    assert (frame[height - 92:height - 20] != 30).any(), "no status strip"
    assert (frame[:height - 100] == 30).all(), "HUD leaked above its strip"

    # Waveform must rise on input and decay toward silence.
    peak = float(hud.waveform.levels.max())
    assert peak > 0.8, peak
    for _ in range(200):
        hud.waveform.update(0.016)
    assert float(hud.waveform.levels.max()) < peak * 0.05, "waveform did not decay"

    # Meters must respect their window length and clamp inputs.
    meter = Meter(8)
    for i in range(50):
        meter.push(i / 50.0)
    assert len(meter.values) == 8
    meter.push(5.0)
    meter.push(-5.0)
    assert max(meter.values) <= 1.0 and min(meter.values) >= 0.0
    meter.draw(frame, (10, 10, 100, 20), theme.accent, theme.secondary)

    # Flash must fade out.
    hud.flash()
    assert hud._flash == 1.0
    for _ in range(60):
        hud.update(0.016, state, energies)
    assert hud._flash == 0.0

    # Banner, help and degraded states must all render.
    hud.draw(frame, theme, HudState(paused=True))
    hud.draw(frame, theme, HudState(muted=True))
    hud.draw(frame, theme, HudState(camera_ok=False, tracking_ok=False, fps=12.0))
    draw_help(frame, theme, width, height)

    for name in THEMES:
        hud.draw(frame, THEMES[name], state)

    # A small window must not raise on any panel.
    small = np.zeros((420, 640, 3), np.uint8)
    Hud(640, 420).draw(small, theme, state)
    draw_help(small, theme, 640, 420)

    print("hud self-check passed")


if __name__ == "__main__":
    _self_check()
