"""Static pose and swipe recognition from hand landmarks.

Poses are classified from which fingers are extended, a rotation-invariant
signal, rather than from raw landmark coordinates. Recognition is purely
geometric; no classifier is trained or loaded.
"""


from __future__ import annotations

import math
from collections import deque
from enum import Enum
from typing import Deque, Optional

import numpy as np

from camera import HandData

THUMB, INDEX, MIDDLE, RING, PINKY = range(5)


class Gesture(Enum):
    """Recognised hand poses and swipes."""

    NONE = "none"
    PINCH = "pinch"
    OPEN_HAND = "open_hand"
    CLOSED_FIST = "closed_fist"
    THUMBS_UP = "thumbs_up"
    PEACE = "peace"
    ROCK = "rock"
    OK = "ok"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"


#: Finger extension patterns that identify a pose, checked in order.
_POSE_PATTERNS: tuple[tuple[Gesture, tuple[bool, bool, bool, bool, bool]], ...] = (
    (Gesture.PEACE,       (False, True, True, False, False)),
    (Gesture.ROCK,        (False, True, False, False, True)),
    (Gesture.THUMBS_UP,   (True, False, False, False, False)),
    (Gesture.OPEN_HAND,   (True, True, True, True, True)),
    (Gesture.CLOSED_FIST, (False, False, False, False, False)),
)


class GestureRecognizer:
    """Classifies one hand's pose, with hysteresis and swipe tracking.

    A pose must persist for several frames before it is reported, which
    stops tracking noise from firing spurious mode changes.
    """

    def __init__(self, sensitivity: float = 0.7, hold_frames: int = 3):
        self.sensitivity = float(np.clip(sensitivity, 0.1, 1.0))
        self.hold_frames = max(1, hold_frames)

        # Higher sensitivity tolerates a looser pinch and a shorter swipe.
        self.pinch_threshold = 0.30 + 0.25 * self.sensitivity
        self.swipe_distance = 0.30 - 0.12 * self.sensitivity
        self.swipe_window = 0.45

        self._candidate = Gesture.NONE
        self._streak = 0
        self._stable = Gesture.NONE
        self._history: Deque[tuple[float, float]] = deque(maxlen=16)
        self._swipe_lockout = 0.0

    @property
    def stable(self) -> Gesture:
        """The most recently confirmed pose."""
        return self._stable

    def update(self, hand: HandData, now: float) -> Gesture:
        """Classify the current pose.

        Args:
            hand: Smoothed landmarks for one hand.
            now: Monotonic timestamp in seconds.

        Returns:
            The confirmed pose, or ``Gesture.NONE`` while unstable.
        """
        if not hand.tracked:
            self._history.clear()
            self._candidate = Gesture.NONE
            self._streak = 0
            self._stable = Gesture.NONE
            return Gesture.NONE

        swipe = self._detect_swipe(hand, now)
        if swipe is not Gesture.NONE:
            self._stable = swipe
            return swipe

        pose = self._classify_pose(hand)

        if pose is self._candidate:
            self._streak += 1
        else:
            self._candidate = pose
            self._streak = 1

        if self._streak >= self.hold_frames:
            self._stable = pose
        return self._stable

    def _classify_pose(self, hand: HandData) -> Gesture:
        """Map finger extension and pinch distance onto a pose."""
        extended = hand.extended

        # Pinch and OK both close thumb to index; the other fingers decide.
        if hand.pinch < self.pinch_threshold:
            if extended[MIDDLE] and extended[RING]:
                return Gesture.OK
            return Gesture.PINCH

        for gesture, pattern in _POSE_PATTERNS:
            if extended == pattern:
                return gesture

        return Gesture.NONE

    def _detect_swipe(self, hand: HandData, now: float) -> Gesture:
        """Detect a fast horizontal sweep of the palm."""
        palm = hand.points[9]
        self._history.append((now, float(palm[0])))

        if now < self._swipe_lockout or len(self._history) < 4:
            return Gesture.NONE

        recent = [(t, x) for t, x in self._history if now - t <= self.swipe_window]
        if len(recent) < 4:
            return Gesture.NONE

        travel = recent[-1][1] - recent[0][1]
        if abs(travel) < self.swipe_distance:
            return Gesture.NONE

        # A swipe is a sweep of the whole hand, not a curled finger flick.
        if sum(hand.extended) < 2:
            return Gesture.NONE

        self._history.clear()
        self._swipe_lockout = now + 0.5
        return Gesture.SWIPE_RIGHT if travel > 0 else Gesture.SWIPE_LEFT

    def reset(self) -> None:
        """Clear all pose and swipe state."""
        self._history.clear()
        self._candidate = Gesture.NONE
        self._streak = 0
        self._stable = Gesture.NONE
        self._swipe_lockout = 0.0


def _make_hand(extended: tuple[bool, ...], pinch: float = 1.0, palm_x: float = 0.5) -> HandData:
    """Build a synthetic hand with the given features, for testing."""
    hand = HandData()
    hand.tracked = True
    hand.confidence = 0.9
    hand.extended = extended
    hand.pinch = pinch
    hand.openness = sum(extended) / 5.0
    hand.points = np.zeros((21, 2), np.float32)
    hand.points[9] = (palm_x, 0.5)
    return hand


def _self_check() -> None:
    """Verify pose classification, hysteresis and swipe detection."""
    recognizer = GestureRecognizer(hold_frames=3)

    def settle(hand: HandData, frames: int = 3, start: float = 0.0) -> Gesture:
        result = Gesture.NONE
        for i in range(frames):
            result = recognizer.update(hand, start + i * 0.016)
        return result

    assert settle(_make_hand((False, True, True, False, False))) is Gesture.PEACE
    recognizer.reset()
    assert settle(_make_hand((False, True, False, False, True))) is Gesture.ROCK
    recognizer.reset()
    assert settle(_make_hand((True, False, False, False, False))) is Gesture.THUMBS_UP
    recognizer.reset()
    assert settle(_make_hand((True,) * 5)) is Gesture.OPEN_HAND
    recognizer.reset()
    assert settle(_make_hand((False,) * 5)) is Gesture.CLOSED_FIST

    # Pinch beats the extension pattern; OK adds the other fingers.
    recognizer.reset()
    assert settle(_make_hand((False, False, False, False, False), pinch=0.1)) is Gesture.PINCH
    recognizer.reset()
    assert settle(_make_hand((False, False, True, True, True), pinch=0.1)) is Gesture.OK

    # An untracked hand must report nothing and clear state.
    assert recognizer.update(HandData(), 1.0) is Gesture.NONE
    assert recognizer.stable is Gesture.NONE

    # Hysteresis: a single frame must not flip the reported pose.
    recognizer.reset()
    settle(_make_hand((True,) * 5), frames=5)
    assert recognizer.update(_make_hand((False,) * 5), 1.0) is Gesture.OPEN_HAND, \
        "one noisy frame must not change the pose"

    def sweep(extended: tuple[bool, ...], start: float, step: float, interval: float) -> list[Gesture]:
        """Run a horizontal sweep and collect every reported gesture."""
        recognizer.reset()
        return [
            recognizer.update(
                _make_hand(extended, palm_x=start + i * step), i * interval
            )
            for i in range(8)
        ]

    # Swipes need a fast, wide travel by an open hand. The swipe fires once
    # and then locks out, so it appears mid-sequence rather than at the end.
    seen = sweep((True,) * 5, 0.15, 0.09, 0.03)
    assert Gesture.SWIPE_RIGHT in seen, seen
    assert seen.count(Gesture.SWIPE_RIGHT) == 1, f"swipe must fire once, got {seen}"

    seen = sweep((True,) * 5, 0.85, -0.09, 0.03)
    assert Gesture.SWIPE_LEFT in seen, seen

    # A slow drift across the same distance must not count as a swipe.
    seen = sweep((True,) * 5, 0.15, 0.09, 1.0)
    assert Gesture.SWIPE_RIGHT not in seen, "slow drift must not register"

    # A curled hand must not swipe.
    seen = sweep((False,) * 5, 0.15, 0.09, 0.03)
    assert Gesture.SWIPE_RIGHT not in seen, "fist must not swipe"

    # A small jitter must not register either.
    seen = sweep((True,) * 5, 0.50, 0.004, 0.03)
    assert Gesture.SWIPE_RIGHT not in seen, "jitter must not register"

    print("gestures self-check passed")


if __name__ == "__main__":
    _self_check()
