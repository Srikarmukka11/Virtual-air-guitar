"""Easing curves and frame-rate independent value smoothing."""

from __future__ import annotations

import math
from typing import Callable, Dict


def _clamp01(t: float) -> float:
    """Clamp a progress value to [0, 1]."""
    return 0.0 if t < 0.0 else 1.0 if t > 1.0 else t


def linear(t: float) -> float:
    """No easing."""
    return _clamp01(t)


def ease_out_sine(t: float) -> float:
    """Gentle deceleration."""
    return math.sin(_clamp01(t) * math.pi / 2.0)


def ease_out_cubic(t: float) -> float:
    """Standard UI deceleration."""
    return 1.0 - (1.0 - _clamp01(t)) ** 3


def ease_out_quart(t: float) -> float:
    """Sharper deceleration than cubic."""
    return 1.0 - (1.0 - _clamp01(t)) ** 4


def ease_in_out_cubic(t: float) -> float:
    """Symmetric acceleration then deceleration."""
    t = _clamp01(t)
    return 4.0 * t ** 3 if t < 0.5 else 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


def ease_out_elastic(t: float) -> float:
    """Overshoot with a settling wobble."""
    t = _clamp01(t)
    if t in (0.0, 1.0):
        return t
    return 2.0 ** (-10.0 * t) * math.sin((t * 10.0 - 0.75) * (2.0 * math.pi / 3.0)) + 1.0


def ease_out_back(t: float) -> float:
    """Slight overshoot without oscillation."""
    t = _clamp01(t)
    c1, c3 = 1.70158, 2.70158
    return 1.0 + c3 * (t - 1.0) ** 3 + c1 * (t - 1.0) ** 2


EASINGS: Dict[str, Callable[[float], float]] = {
    "linear": linear,
    "sine": ease_out_sine,
    "cubic": ease_out_cubic,
    "quart": ease_out_quart,
    "in_out_cubic": ease_in_out_cubic,
    "elastic": ease_out_elastic,
    "back": ease_out_back,
}


class Smoothed:
    """A value that eases toward a target at a frame-rate independent rate.

    Uses exponential convergence driven by delta time, so behaviour is
    identical at 30 and 144 FPS.
    """

    def __init__(self, value: float = 0.0, rate: float = 12.0):
        self._value = float(value)
        self._target = float(value)
        self.rate = rate

    @property
    def value(self) -> float:
        """The current eased value."""
        return self._value

    @property
    def target(self) -> float:
        """The value being approached."""
        return self._target

    @target.setter
    def target(self, value: float) -> None:
        self._target = float(value)

    def set_now(self, value: float) -> None:
        """Jump immediately, skipping the transition."""
        self._value = self._target = float(value)

    def update(self, dt: float) -> float:
        """Advance toward the target and return the new value."""
        if dt <= 0.0:
            return self._value
        alpha = 1.0 - math.exp(-self.rate * dt)
        self._value += (self._target - self._value) * alpha
        if abs(self._target - self._value) < 1e-4:
            self._value = self._target
        return self._value

    @property
    def settled(self) -> bool:
        """Whether the value has reached its target."""
        return self._value == self._target


class Pulse:
    """A one-shot 0..1 envelope, used for hit flashes and pops."""

    def __init__(self, duration: float = 0.35, easing: Callable[[float], float] = ease_out_cubic):
        self.duration = max(1e-3, duration)
        self.easing = easing
        self._elapsed = self.duration

    def fire(self) -> None:
        """Restart the envelope from zero."""
        self._elapsed = 0.0

    def update(self, dt: float) -> None:
        """Advance the envelope."""
        if self._elapsed < self.duration:
            self._elapsed = min(self.duration, self._elapsed + dt)

    @property
    def active(self) -> bool:
        """Whether the envelope is still running."""
        return self._elapsed < self.duration

    @property
    def value(self) -> float:
        """Envelope level, 1 at trigger falling to 0."""
        if self._elapsed >= self.duration:
            return 0.0
        return 1.0 - self.easing(self._elapsed / self.duration)


def _self_check() -> None:
    """Verify easing bounds and smoothing convergence."""
    for name, fn in EASINGS.items():
        assert abs(fn(0.0)) < 1e-6, f"{name} must start at 0"
        assert abs(fn(1.0) - 1.0) < 1e-6, f"{name} must end at 1"
        assert abs(fn(-5.0)) < 1e-6, f"{name} must clamp below 0"
        assert abs(fn(5.0) - 1.0) < 1e-6, f"{name} must clamp above 1"

    # Monotone easings must never move backwards.
    for name in ("linear", "sine", "cubic", "quart", "in_out_cubic"):
        fn = EASINGS[name]
        values = [fn(i / 50.0) for i in range(51)]
        assert all(b >= a - 1e-9 for a, b in zip(values, values[1:])), f"{name} not monotonic"

    # Elastic and back are expected to overshoot; that is their purpose.
    assert max(ease_out_back(i / 50.0) for i in range(51)) > 1.0

    s = Smoothed(0.0, rate=12.0)
    s.target = 10.0
    for _ in range(200):
        s.update(0.016)
    assert abs(s.value - 10.0) < 1e-3, s.value
    assert s.settled

    # Convergence must be frame-rate independent.
    slow, fast = Smoothed(0.0, rate=8.0), Smoothed(0.0, rate=8.0)
    slow.target = fast.target = 1.0
    for _ in range(10):
        slow.update(0.1)
    for _ in range(100):
        fast.update(0.01)
    assert abs(slow.value - fast.value) < 0.02, (slow.value, fast.value)

    s.set_now(3.0)
    assert s.value == 3.0 and s.settled
    assert s.update(0.0) == 3.0, "zero dt must not advance"

    p = Pulse(duration=0.2)
    assert not p.active and p.value == 0.0
    p.fire()
    assert p.active and p.value > 0.9
    p.update(0.1)
    mid = p.value
    assert 0.0 < mid < 1.0
    p.update(0.2)
    assert not p.active and p.value == 0.0

    print("animations self-check passed")


if __name__ == "__main__":
    _self_check()
