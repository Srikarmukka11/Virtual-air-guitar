"""Selection rows and the holographic pointer.

Chords and instruments are chosen by pointing at a button in a horizontal
row. That is a
one-dimensional target roughly 120 px wide and 90 px tall, which jittery
hand tracking hits reliably; the radial dials this replaced asked the user
to land a fingertip inside a 26-degree arc of a 44 px ring, and missed
constantly. Resting on a button commits it after a short dwell, and a pinch
commits immediately for anyone who wants to move faster than the dwell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np

from animations import Pulse, Smoothed, ease_out_back
from effects import glass_panel
from themes import Theme, mix, scale

#: Chord bar entries: label, root note, quality. Ordered so the chords that
#: turn up in most songs sit toward the middle, under a resting hand.
CHORD_ENTRIES: tuple[tuple[str, Optional[str], str], ...] = (
    ("C", "C", "Major"),
    ("G", "G", "Major"),
    ("Am", "A", "Minor"),
    ("Em", "E", "Minor"),
    ("F", "F", "Major"),
    ("D", "D", "Major"),
    ("Dm", "D", "Minor"),
    ("A", "A", "Major"),
    ("E", "E", "Major"),
    ("Bm", "B", "Minor"),
)

#: Chord qualities cycled by the K key, for the voicings the bar omits.
MODIFIER_ENTRIES: tuple[str, ...] = ("7", "Maj7", "Sus2", "Sus4", "Power")

FONT = cv2.FONT_HERSHEY_DUPLEX


@dataclass
class _Button:
    """One chord button and its animation state."""

    label: str
    index: int
    x: int
    width: int
    hover: Smoothed = field(default_factory=lambda: Smoothed(0.0, rate=16.0))
    pop: Pulse = field(default_factory=lambda: Pulse(0.28, ease_out_back))


class ButtonRow:
    """A row of buttons selected by pointing.

    Selection is driven entirely by where the pointer is: no click gesture
    is required. A button under the pointer fills a progress arc, and
    commits once ``dwell`` seconds have passed. The commit then latches
    until the pointer leaves the button, so resting a hand on a chord does
    not retrigger it every frame.
    """

    def __init__(
        self,
        width: int,
        top: int,
        height: int,
        labels: Sequence[str],
        margin: int = 40,
        dwell: float = 0.30,
        pad: int = 16,
    ):
        """Lay the row out.

        Args:
            width: Frame width in pixels.
            top: Y coordinate of the row's top edge.
            height: Row height in pixels.
            labels: Button captions, left to right.
            margin: Free space at each end of the row.
            dwell: Seconds of hovering needed to commit a button.
            pad: Vertical slack above and below that still counts as aiming
                at this row. Keep it under half the gap to the next row, or
                the two rows will both claim the same pointer position.
        """
        self.top = top
        self.height = height
        self.dwell = dwell
        self.pad = pad

        span = width - margin * 2
        gap = 8
        button_w = (span - gap * (len(labels) - 1)) // len(labels)

        self.buttons = [
            _Button(label=label, index=i, x=margin + i * (button_w + gap), width=button_w)
            for i, label in enumerate(labels)
        ]

        self.selected: int = -1
        self.hovered: int = -1
        self._progress = 0.0
        self._latched = -1

    @property
    def bottom(self) -> int:
        """Y coordinate just below the bar."""
        return self.top + self.height

    def contains(self, point: Optional[tuple[float, float]]) -> bool:
        """Whether a point falls within the bar's vertical band."""
        if point is None:
            return False
        # The band is padded so a hand approaching the row counts as aiming
        # at it, rather than having to land inside the exact rectangle.
        return self.top - self.pad <= point[1] <= self.bottom + self.pad

    def hit_test(self, point: Optional[tuple[float, float]]) -> int:
        """Return the button index under a point, or -1."""
        if not self.contains(point):
            return -1
        x = point[0]
        for button in self.buttons:
            if button.x <= x <= button.x + button.width:
                return button.index
        return -1

    def update(
        self,
        dt: float,
        point: Optional[tuple[float, float]],
        instant: bool = False,
    ) -> int:
        """Advance hover state and dwell timing.

        Args:
            dt: Seconds since the previous frame.
            point: Pointer position in pixels, or None when untracked.
            instant: Commit the hovered button immediately, e.g. on a pinch.

        Returns:
            The index newly committed this frame, or -1 if none was.
        """
        self.hovered = self.hit_test(point)

        if self.hovered == -1:
            self._progress = 0.0
            self._latched = -1
        else:
            if self.hovered != self._latched:
                self._progress += dt
        for button in self.buttons:
            button.hover.target = 1.0 if button.index == self.hovered else 0.0
            button.hover.update(dt)
            button.pop.update(dt)

        committed = -1
        if self.hovered != -1 and self.hovered != self._latched:
            if instant or self._progress >= self.dwell:
                committed = self.hovered
                self._latched = self.hovered
                self._progress = 0.0
                self.select(committed)

        return committed

    def select(self, index: int) -> bool:
        """Select a button directly, e.g. from the keyboard."""
        if not (0 <= index < len(self.buttons)):
            return False
        self.selected = index
        self.buttons[index].pop.fire()
        return True

    def select_offset(self, delta: int) -> int:
        """Step the selection along the bar, wrapping at the ends."""
        count = len(self.buttons)
        current = self.selected if self.selected >= 0 else 0
        self.select((current + delta) % count)
        return self.selected

    @property
    def selected_label(self) -> str:
        """Label of the current selection, or an empty string."""
        if 0 <= self.selected < len(self.buttons):
            return self.buttons[self.selected].label
        return ""

    def draw(self, frame: np.ndarray, theme: Theme) -> None:
        """Render the bar."""
        first, last = self.buttons[0], self.buttons[-1]
        pad = 12
        # One frosted plate behind the whole row: blurring per button would
        # cost ten region blurs a frame for the same look.
        glass_panel(
            frame,
            (first.x - pad, self.top - pad,
             last.x + last.width - first.x + pad * 2, self.height + pad * 2),
            theme,
            radius=14,
            opacity=0.62,
        )

        for button in self.buttons:
            self._draw_button(frame, theme, button)

    def _draw_button(self, frame: np.ndarray, theme: Theme, button: _Button) -> None:
        """Render one chord button with its hover and dwell state."""
        hover = button.hover.value
        chosen = button.index == self.selected
        lift = int(hover * 4 + button.pop.value * 5)

        x0, y0 = button.x, self.top - lift
        x1, y1 = button.x + button.width, self.top + self.height + lift

        if chosen:
            fill = theme.accent
        elif hover > 0.01:
            fill = mix(scale(theme.panel, 1.9), theme.accent, hover * 0.65)
        else:
            fill = scale(theme.panel, 1.9)

        cv2.rectangle(frame, (x0, y0), (x1, y1), fill, -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1),
                      theme.highlight if chosen else scale(theme.secondary, 0.8),
                      2 if chosen else 1, cv2.LINE_AA)

        # Dwell progress fills along the bottom edge of the hovered button.
        if button.index == self.hovered and self._progress > 0.0:
            fraction = min(1.0, self._progress / self.dwell)
            cv2.rectangle(frame, (x0, y1 - 5),
                          (x0 + int(button.width * fraction), y1),
                          theme.string_hot, -1)

        size = self.height / 100.0 + hover * 0.10
        colour = theme.panel if chosen else (theme.text if hover > 0.25 else theme.text_dim)
        (tw, th), _ = cv2.getTextSize(button.label, FONT, size, 2)
        cv2.putText(frame, button.label,
                    (x0 + (button.width - tw) // 2, y0 + (self.height + th) // 2 + lift),
                    FONT, size, colour, 2, cv2.LINE_AA)


class Pointer:
    """Animated holographic cursor for a tracked fingertip."""

    def __init__(self, rate: float = 22.0):
        self.x = Smoothed(0.0, rate=rate)
        self.y = Smoothed(0.0, rate=rate)
        self.visible = False
        self.active = False
        self._spin = 0.0
        self._pulse = Pulse(0.3)

    @property
    def position(self) -> tuple[float, float]:
        """Current smoothed position in pixels."""
        return self.x.value, self.y.value

    def set(self, point: Optional[tuple[float, float]]) -> None:
        """Aim the pointer, or hide it when the hand is lost."""
        if point is None:
            self.visible = False
            return
        if not self.visible:
            self.x.set_now(point[0])
            self.y.set_now(point[1])
        else:
            self.x.target = point[0]
            self.y.target = point[1]
        self.visible = True

    def click(self) -> None:
        """Play the selection flourish."""
        self._pulse.fire()

    def update(self, dt: float) -> None:
        """Advance smoothing and idle rotation."""
        self.x.update(dt)
        self.y.update(dt)
        self._spin += dt * 1.6
        self._pulse.update(dt)

    def draw(self, frame: np.ndarray, theme: Theme) -> None:
        """Render the cursor."""
        if not self.visible:
            return

        cx, cy = int(self.x.value), int(self.y.value)
        colour = theme.highlight if self.active else theme.accent
        burst = self._pulse.value

        outer = int(15 + burst * 12)
        cv2.circle(frame, (cx, cy), outer + 5, scale(colour, 0.25), 2, cv2.LINE_AA)

        # Three rotating arcs read as a holographic reticle.
        for i in range(3):
            start = math.degrees(self._spin + i * math.tau / 3.0)
            cv2.ellipse(frame, (cx, cy), (outer, outer), 0, start, start + 72,
                        colour, 2, cv2.LINE_AA)

        cv2.circle(frame, (cx, cy), 4, colour, -1, cv2.LINE_AA)


def _self_check() -> None:
    """Verify hit testing, dwell selection and pointer behaviour."""
    from themes import THEMES

    theme = THEMES["cyber_blue"]
    bar = ButtonRow(1280, top=24, height=90,
                   labels=[e[0] for e in CHORD_ENTRIES], dwell=0.3)
    assert len(bar.buttons) == len(CHORD_ENTRIES)

    # Buttons must tile left to right without overlapping.
    for a, b in zip(bar.buttons, bar.buttons[1:]):
        assert a.x + a.width <= b.x, "buttons overlap"
    assert bar.buttons[0].x >= 0
    assert bar.buttons[-1].x + bar.buttons[-1].width <= 1280

    # The centre of each button must hit that button.
    mid_y = bar.top + bar.height / 2
    for button in bar.buttons:
        point = (button.x + button.width / 2, mid_y)
        assert bar.hit_test(point) == button.index, f"missed {button.label}"

    # Points outside the band, and no point at all, must miss.
    assert bar.hit_test((640.0, 600.0)) == -1
    assert bar.hit_test(None) == -1
    assert bar.hit_test((5.0, mid_y)) == -1, "the margin is not a button"

    # Dwell: hovering commits once the dwell elapses, and not before.
    target = bar.buttons[2]
    point = (target.x + target.width / 2, mid_y)
    assert bar.update(0.1, point) == -1, "committed before the dwell elapsed"
    assert bar.update(0.1, point) == -1
    assert bar.update(0.15, point) == 2, "dwell did not commit"
    assert bar.selected == 2

    # Holding still must not retrigger.
    for _ in range(30):
        assert bar.update(0.016, point) == -1, "latched button retriggered"

    # Moving away and back must arm it again.
    bar.update(0.016, None)
    assert bar.update(0.35, point) == 2, "leaving and returning must rearm"

    # A pinch must commit without waiting.
    other = bar.buttons[7]
    point = (other.x + other.width / 2, mid_y)
    assert bar.update(0.016, point, instant=True) == 7, "pinch must commit at once"
    assert bar.selected == 7

    # Leaving the bar clears dwell progress rather than banking it.
    bar.update(0.2, (640.0, 600.0))
    assert bar._progress == 0.0

    # Keyboard stepping must wrap in both directions.
    bar.select(0)
    assert bar.select_offset(-1) == len(bar.buttons) - 1
    assert bar.select_offset(1) == 0
    assert bar.selected_label == "C"
    assert not bar.select(99), "out-of-range must be rejected"

    frame = np.zeros((720, 1280, 3), np.uint8)
    bar.draw(frame, theme)
    assert frame.any(), "bar should mark the frame"
    # The bar must stay inside its own band and not paint over the guitar.
    assert not frame[bar.bottom + 40:].any(), "bar leaked below its band"

    pointer = Pointer()
    assert not pointer.visible
    pointer.draw(frame, theme)

    pointer.set((100.0, 100.0))
    assert pointer.visible
    assert pointer.position == (100.0, 100.0), "first aim must snap, not glide"

    pointer.set((200.0, 100.0))
    pointer.update(0.016)
    assert 100.0 < pointer.position[0] < 200.0, "pointer must ease toward the target"

    pointer.click()
    pointer.active = True
    pointer.update(0.016)
    pointer.draw(frame, theme)

    pointer.set(None)
    assert not pointer.visible

    print("ui self-check passed")


if __name__ == "__main__":
    _self_check()
