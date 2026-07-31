"""Pooled particle system and motion trails.

Particle state lives in preallocated numpy arrays and every update is a
vectorised operation over the live slice, so a strum costs no allocation.
"""


from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

from themes import scale


class ParticleSystem:
    """Fixed-capacity particle pool updated as numpy array slices.

    Live particles are kept packed at the front of each array; retiring a
    particle swaps it with the last live entry, which keeps updates
    contiguous without ever allocating.
    """

    def __init__(self, capacity: int = 900):
        self.capacity = max(16, capacity)
        self.position = np.zeros((self.capacity, 2), np.float32)
        self.velocity = np.zeros((self.capacity, 2), np.float32)
        self.colour = np.zeros((self.capacity, 3), np.float32)
        self.life = np.zeros(self.capacity, np.float32)
        self.max_life = np.ones(self.capacity, np.float32)
        self.size = np.zeros(self.capacity, np.float32)
        self.count = 0
        self._rng = np.random.default_rng(7)

    def emit(
        self,
        origin: tuple[float, float],
        colour: tuple[int, int, int],
        count: int = 30,
        speed: float = 260.0,
        spread: float = math.pi * 2.0,
        direction: float = 0.0,
        lifetime: float = 0.85,
    ) -> int:
        """Spawn particles, silently truncating at pool capacity.

        Args:
            origin: Spawn point in pixels.
            colour: BGR colour for the burst.
            count: Requested particle count.
            speed: Base speed in pixels per second.
            spread: Angular spread in radians around ``direction``.
            direction: Central emission angle in radians.
            lifetime: Base lifetime in seconds.

        Returns:
            How many particles were actually spawned.
        """
        available = self.capacity - self.count
        spawn = int(min(count, available))
        if spawn <= 0:
            return 0

        end = self.count + spawn
        block = slice(self.count, end)

        angles = direction + self._rng.uniform(-spread / 2, spread / 2, spawn)
        speeds = speed * self._rng.uniform(0.35, 1.0, spawn)

        self.position[block] = origin
        self.velocity[block, 0] = np.cos(angles) * speeds
        self.velocity[block, 1] = np.sin(angles) * speeds
        self.colour[block] = colour
        lifetimes = lifetime * self._rng.uniform(0.6, 1.25, spawn)
        self.life[block] = lifetimes
        self.max_life[block] = lifetimes
        self.size[block] = self._rng.uniform(1.6, 4.2, spawn)

        self.count = end
        return spawn

    def update(self, dt: float, gravity: float = 210.0, drag: float = 1.9) -> None:
        """Integrate motion and retire dead particles."""
        if self.count == 0:
            return

        live = slice(0, self.count)
        self.velocity[live, 1] += gravity * dt
        self.velocity[live] *= max(0.0, 1.0 - drag * dt)
        self.position[live] += self.velocity[live] * dt
        self.life[live] -= dt

        # Compact the pool by swapping dead entries with the live tail.
        index = 0
        while index < self.count:
            if self.life[index] > 0.0:
                index += 1
                continue
            last = self.count - 1
            if index != last:
                for array in (
                    self.position, self.velocity, self.colour,
                    self.life, self.max_life, self.size,
                ):
                    array[index] = array[last]
            self.count = last

    def draw(self, frame: np.ndarray) -> None:
        """Render live particles, fading with remaining lifetime."""
        if self.count == 0:
            return

        height, width = frame.shape[:2]
        positions = self.position[: self.count]
        fades = np.clip(self.life[: self.count] / self.max_life[: self.count], 0.0, 1.0)

        for i in range(self.count):
            x, y = positions[i]
            if not (0 <= x < width and 0 <= y < height):
                continue
            fade = float(fades[i]) ** 0.7
            colour = tuple(int(c * fade) for c in self.colour[i])
            radius = max(1, int(self.size[i] * fade))
            cv2.circle(frame, (int(x), int(y)), radius, colour, -1, cv2.LINE_AA)
            if radius > 1:
                cv2.circle(frame, (int(x), int(y)), radius + 2, scale(colour, 0.35), 1, cv2.LINE_AA)

    def clear(self) -> None:
        """Retire every particle."""
        self.count = 0

    def __len__(self) -> int:
        """Number of live particles."""
        return self.count


class MotionTrail:
    """Fading trail that follows a tracked point."""

    def __init__(self, length: int = 20, min_step: float = 3.0):
        self.points: Deque[tuple[float, float]] = deque(maxlen=max(2, length))
        self.min_step = min_step

    def push(self, point: tuple[float, float]) -> None:
        """Append a point when it has moved far enough to matter."""
        if self.points:
            last = self.points[-1]
            if math.hypot(point[0] - last[0], point[1] - last[1]) < self.min_step:
                return
        self.points.append(point)

    def decay(self) -> None:
        """Drop the oldest point, used when tracking is lost."""
        if self.points:
            self.points.popleft()

    def clear(self) -> None:
        """Remove every point."""
        self.points.clear()

    def draw(
        self,
        frame: np.ndarray,
        colour: tuple[int, int, int],
        thickness: int = 3,
    ) -> None:
        """Render the trail, brightest and thickest at the newest point."""
        if len(self.points) < 2:
            return

        total = len(self.points)
        for i in range(total - 1):
            fade = (i + 1) / total
            start = (int(self.points[i][0]), int(self.points[i][1]))
            end = (int(self.points[i + 1][0]), int(self.points[i + 1][1]))
            width = max(1, int(thickness * fade))
            cv2.line(frame, start, end, scale(colour, 0.30 * fade), width + 4, cv2.LINE_AA)
            cv2.line(frame, start, end, scale(colour, fade), width, cv2.LINE_AA)

    def __len__(self) -> int:
        """Number of stored points."""
        return len(self.points)


def _self_check() -> None:
    """Verify pooling, physics and trail behaviour."""
    system = ParticleSystem(capacity=50)
    assert len(system) == 0

    assert system.emit((10.0, 10.0), (0, 255, 255), count=20) == 20
    assert len(system) == 20

    # Capacity must be respected rather than growing the arrays.
    assert system.emit((10.0, 10.0), (0, 255, 255), count=100) == 30
    assert len(system) == 50
    assert system.emit((10.0, 10.0), (0, 255, 255), count=10) == 0, "full pool must reject"

    # Gravity must pull particles downward over time.
    before = system.position[:, 1].copy()
    for _ in range(5):
        system.update(0.05)
    assert system.position[: system.count, 1].mean() > before[: system.count].mean()

    # Every particle must eventually retire and free its slot.
    for _ in range(200):
        system.update(0.05)
    assert len(system) == 0, f"pool did not drain, {len(system)} left"

    # After draining, the pool must still be usable.
    assert system.emit((5.0, 5.0), (255, 0, 0), count=10) == 10
    system.clear()
    assert len(system) == 0

    frame = np.zeros((120, 160, 3), np.uint8)
    system.emit((80.0, 60.0), (0, 255, 255), count=25)
    system.update(0.016)
    system.draw(frame)
    assert frame.any(), "particles should mark the frame"

    # Off-screen particles must be skipped without raising.
    system.position[: system.count] = (10_000.0, 10_000.0)
    system.draw(frame)

    trail = MotionTrail(length=5, min_step=2.0)
    trail.push((0.0, 0.0))
    trail.push((0.5, 0.5))
    assert len(trail) == 1, "sub-threshold movement must be ignored"
    trail.push((20.0, 20.0))
    assert len(trail) == 2
    for i in range(20):
        trail.push((float(i * 10), 0.0))
    assert len(trail) == 5, "trail must respect its maximum length"

    trail.draw(frame, (0, 255, 255))
    trail.decay()
    assert len(trail) == 4
    trail.clear()
    assert len(trail) == 0
    trail.draw(frame, (0, 255, 255))

    print("particles self-check passed")


if __name__ == "__main__":
    _self_check()
