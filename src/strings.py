"""Six horizontal guitar strings with damped-oscillator vibration.

Strings run horizontally across the centre of the frame so that a vertical
sweep of the picking hand crosses them in sequence, matching how a real
strum feels. A pluck is registered when the pick point moves from one side
of a string to the other between frames.
"""


from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from effects import glow_line
from themes import Theme, mix, scale


@dataclass
class GuitarString:
    """One string: a damped sine standing wave driven by plucks.

    Rather than integrating a mass-spring chain, the visible motion is a
    decaying standing wave. It is cheaper, unconditionally stable, and
    matches how a real string looks far better than a coarse chain does.
    """

    index: int
    y: float
    x_start: float
    x_end: float
    frequency: float
    amplitude: float = 0.0
    phase: float = 0.0
    decay: float = 2.4
    thickness: int = 2
    last_velocity: float = 0.0

    #: Idle shimmer keeps strings alive when nothing is being played.
    idle_amplitude: float = 0.55

    def pluck(self, velocity: float, position: float = 0.5) -> None:
        """Excite the string.

        Args:
            velocity: Strike strength in [0, 1].
            position: Normalised strike point along the string.
        """
        strength = float(np.clip(velocity, 0.05, 1.0))
        self.amplitude = min(26.0, self.amplitude * 0.35 + 9.0 + strength * 17.0)
        self.phase = 0.0
        self.last_velocity = strength
        self._node = float(np.clip(position, 0.05, 0.95))

    def update(self, dt: float, time: float) -> None:
        """Advance the vibration envelope."""
        self.phase += dt * self.frequency
        self.amplitude *= math.exp(-self.decay * dt)
        if self.amplitude < 0.05:
            self.amplitude = 0.0
        self._time = time

    def displacement(self, t: float) -> float:
        """Peak vertical displacement in pixels at the current instant."""
        return self.amplitude * math.sin(self.phase * math.tau)

    @property
    def energy(self) -> float:
        """Normalised vibration energy in [0, 1]."""
        return min(1.0, self.amplitude / 26.0)

    def points(self, time: float, samples: int = 34) -> np.ndarray:
        """Sample the string's centre line for rendering.

        Args:
            time: Seconds since start, driving the idle shimmer.
            samples: Number of points along the string.

        Returns:
            An (N, 2) int32 array of pixel coordinates.
        """
        xs = np.linspace(self.x_start, self.x_end, samples, dtype=np.float32)
        u = np.linspace(0.0, 1.0, samples, dtype=np.float32)

        # Fundamental mode, pinned at both ends.
        envelope = np.sin(u * math.pi)
        wave = envelope * self.displacement(time)

        # A quieter second harmonic keeps the motion from looking synthetic.
        wave += np.sin(u * math.tau) * self.displacement(time) * 0.22

        # Idle shimmer so the instrument never looks frozen.
        shimmer = self.idle_amplitude * np.sin(u * 3.0 + time * (1.1 + self.index * 0.23))
        ys = self.y + wave + shimmer * envelope

        return np.stack([xs, ys], axis=1).astype(np.int32)

    def draw(self, frame: np.ndarray, theme: Theme, time: float, glow: float = 1.0) -> None:
        """Render the string, brightening with vibration energy."""
        pts = self.points(time)
        energy = self.energy
        colour = mix(theme.string, theme.string_hot, energy)
        thickness = self.thickness + (1 if energy > 0.45 else 0)

        halo = scale(colour, 0.30 * glow * (0.6 + energy))
        cv2.polylines(frame, [pts], False, halo, thickness + 6, cv2.LINE_AA)
        cv2.polylines(frame, [pts], False, scale(colour, 0.75), thickness + 2, cv2.LINE_AA)
        cv2.polylines(frame, [pts], False, colour, thickness, cv2.LINE_AA)

        if energy > 0.25:
            cv2.polylines(frame, [pts], False, theme.string_hot, 1, cv2.LINE_AA)


class StringSet:
    """An instrument's strings plus pick-crossing detection."""

    #: Relative visual thickness, low string to high, for a six-string.
    THICKNESS = (3, 3, 2, 2, 1, 1)

    def __init__(
        self,
        width: int,
        height: int,
        spacing: float = 46.0,
        span: float = 0.46,
        bounds: Optional[tuple[float, float]] = None,
        centre_y: Optional[float] = None,
        count: int = 6,
        thickness: Optional[tuple[int, ...]] = None,
    ):
        """Create the string set.

        Args:
            width: Render width in pixels.
            height: Render height in pixels.
            spacing: Vertical gap between strings.
            span: Fraction of the width the strings occupy, used when
                ``bounds`` is not given.
            bounds: Explicit (left, right) pixel bounds.
            centre_y: Vertical centre of the string block. Defaults to the
                middle of the frame; the renderer nudges it down to leave
                room for the chord bar.
            count: Number of strings; four for a bass or ukulele.
            thickness: Drawn gauge per string, low to high.
        """
        self.width = width
        self.height = height
        self.spacing = spacing
        self.count = max(1, count)

        if thickness is None:
            thickness = self.THICKNESS
        if bounds is not None:
            left, right = bounds
        else:
            half = width * span * 0.5
            left, right = width * 0.5 - half, width * 0.5 + half
        if centre_y is None:
            centre_y = height * 0.5

        # Centre the block whatever the string count, so a four-string sits
        # in the same place a six-string did rather than drifting upward.
        middle = (self.count - 1) * 0.5

        self.strings: list[GuitarString] = []
        for i in range(self.count):
            self.strings.append(
                GuitarString(
                    index=i,
                    y=centre_y + (i - middle) * spacing,
                    x_start=left,
                    x_end=right,
                    # Higher strings shimmer faster.
                    frequency=7.0 + i * 2.6,
                    thickness=thickness[i % len(thickness)],
                )
            )

        self._previous_pick: Optional[tuple[float, float]] = None
        self._cooldown = np.zeros(self.count, np.float32)
        self._time = 0.0

    @property
    def top(self) -> float:
        """Y coordinate of the highest string."""
        return self.strings[0].y

    @property
    def bottom(self) -> float:
        """Y coordinate of the lowest string."""
        return self.strings[-1].y

    def string_at(self, index: int) -> GuitarString:
        """Return a string by index."""
        return self.strings[index]

    def centre_of(self, index: int, x: Optional[float] = None) -> tuple[int, int]:
        """Return a point on a string, defaulting to its midpoint."""
        s = self.strings[index]
        px = s.x_start + (s.x_end - s.x_start) * 0.5 if x is None else x
        return int(px), int(s.y)

    def update(self, dt: float) -> None:
        """Advance vibration and per-string strike cooldowns."""
        self._time += dt
        for s in self.strings:
            s.update(dt, self._time)
        np.maximum(self._cooldown - dt, 0.0, out=self._cooldown)

    def detect_strum(
        self,
        pick: Optional[tuple[float, float]],
        dt: float,
        min_speed: float = 190.0,
        debounce: float = 0.055,
    ) -> list[tuple[int, float, int]]:
        """Find strings crossed by the pick since the previous frame.

        Args:
            pick: Pick position in pixels, or None when untracked.
            dt: Seconds since the previous frame.
            min_speed: Minimum vertical speed, in pixels per second, that
                counts as a deliberate strum rather than drift.
            debounce: Seconds a string ignores further strikes.

        Returns:
            Tuples of (string index, velocity in [0, 1], direction) where
            direction is +1 downward and -1 upward.
        """
        if pick is None:
            self._previous_pick = None
            return []

        previous = self._previous_pick
        self._previous_pick = pick
        if previous is None or dt <= 0.0:
            return []

        x, y = pick
        prev_x, prev_y = previous

        # Only strum while horizontally over the fretboard.
        first = self.strings[0]
        margin = 70.0
        if not (first.x_start - margin <= x <= first.x_end + margin):
            return []

        speed = abs(y - prev_y) / dt
        if speed < min_speed:
            return []

        direction = 1 if y > prev_y else -1
        low, high = (prev_y, y) if y > prev_y else (y, prev_y)
        velocity = float(np.clip(speed / 1500.0, 0.12, 1.0))

        hits: list[tuple[int, float, int]] = []
        for s in self.strings:
            if low <= s.y <= high and self._cooldown[s.index] <= 0.0:
                self._cooldown[s.index] = debounce
                position = float(np.clip((x - s.x_start) / (s.x_end - s.x_start), 0.05, 0.95))
                s.pluck(velocity, position)
                hits.append((s.index, velocity, direction))

        return hits

    def draw(self, frame: np.ndarray, theme: Theme, glow: float = 1.0) -> None:
        """Render all strings and their end anchors."""
        for s in self.strings:
            s.draw(frame, theme, self._time, glow)

        anchor = scale(theme.secondary, 0.9)
        for s in self.strings:
            cv2.circle(frame, (int(s.x_start), int(s.y)), 4, anchor, -1, cv2.LINE_AA)
            cv2.circle(frame, (int(s.x_end), int(s.y)), 4, anchor, -1, cv2.LINE_AA)

    def reset(self) -> None:
        """Silence all vibration and forget the pick history."""
        for s in self.strings:
            s.amplitude = 0.0
        self._cooldown[:] = 0.0
        self._previous_pick = None


def _self_check() -> None:
    """Verify vibration decay and crossing detection."""
    strings = StringSet(width=1280, height=720, spacing=46.0)
    assert len(strings.strings) == 6
    assert strings.top < strings.bottom, "string 0 must sit above string 5"

    # Strings must be evenly spaced.
    gaps = [strings.strings[i + 1].y - strings.strings[i].y for i in range(5)]
    assert all(abs(g - 46.0) < 1e-3 for g in gaps), gaps

    s = strings.string_at(0)
    assert s.energy == 0.0
    s.pluck(1.0)
    assert s.energy > 0.5, "a hard pluck should carry energy"

    # Vibration must decay to rest.
    for _ in range(200):
        s.update(0.016, 0.0)
    assert s.amplitude == 0.0, f"string never settled: {s.amplitude}"

    centre_x = 1280 * 0.5

    # A slow drift must not trigger a strum.
    strings.reset()
    strings.detect_strum((centre_x, strings.top - 40), 0.016)
    assert strings.detect_strum((centre_x, strings.top - 39), 0.016) == []

    # A fast downward sweep must cross every string exactly once.
    strings.reset()
    strings.detect_strum((centre_x, strings.top - 60), 0.016)
    hits = strings.detect_strum((centre_x, strings.bottom + 60), 0.016)
    assert len(hits) == 6, f"expected 6 strings struck, got {len(hits)}"
    assert [h[0] for h in hits] == [0, 1, 2, 3, 4, 5]
    assert all(h[2] == 1 for h in hits), "downward sweep must report direction +1"
    assert all(0.0 < h[1] <= 1.0 for h in hits)

    # Debounce must suppress an immediate repeat.
    assert strings.detect_strum((centre_x, strings.top - 60), 0.016) == []

    # An upward sweep after the cooldown must report direction -1.
    strings.reset()
    strings.update(0.2)
    strings.detect_strum((centre_x, strings.bottom + 60), 0.016)
    up = strings.detect_strum((centre_x, strings.top - 60), 0.016)
    assert len(up) == 6 and all(h[2] == -1 for h in up), "upstroke direction wrong"

    # Sweeping outside the fretboard must be ignored.
    strings.reset()
    strings.update(0.5)
    strings.detect_strum((10.0, strings.top - 60), 0.016)
    assert strings.detect_strum((10.0, strings.bottom + 60), 0.016) == []

    # Losing the pick must clear history so no phantom strum fires.
    strings.reset()
    strings.update(0.5)
    strings.detect_strum((centre_x, strings.top - 60), 0.016)
    assert strings.detect_strum(None, 0.016) == []
    assert strings.detect_strum((centre_x, strings.bottom + 60), 0.016) == []

    frame = np.zeros((720, 1280, 3), np.uint8)
    from themes import THEMES
    strings.draw(frame, THEMES["cyber_blue"])
    assert frame.any(), "strings should mark the frame"

    print("strings self-check passed")


if __name__ == "__main__":
    _self_check()
