"""Post-processing and glassmorphism primitives.

Effects are written for speed on CPU: bloom works on a quarter-resolution
buffer, the vignette mask is cached per frame size, and glass panels blur
only the rectangle they occupy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from themes import Theme, mix, scale


class Bloom:
    """Additive bloom built from a downscaled blur.

    Downsampling before blurring is what keeps this real-time: a large
    perceptual blur radius costs a quarter of the pixels, and the blend
    back is a single fused ``addWeighted`` rather than per-pixel maths.
    """

    def __init__(self, threshold: int = 165, spread: int = 21):
        self.threshold = threshold
        self.spread = spread if spread % 2 else spread + 1
        self._glow: Optional[np.ndarray] = None

    def apply(self, frame: np.ndarray, intensity: float = 1.0) -> np.ndarray:
        """Blend a glow derived from the frame's brightest pixels."""
        if intensity <= 0.01:
            return frame

        height, width = frame.shape[:2]
        small = cv2.resize(frame, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
        bright = cv2.threshold(small, self.threshold, 255, cv2.THRESH_TOZERO)[1]
        bright = cv2.GaussianBlur(bright, (self.spread, self.spread), 0)

        if self._glow is None or self._glow.shape != frame.shape:
            self._glow = np.empty_like(frame)
        cv2.resize(bright, (width, height), dst=self._glow, interpolation=cv2.INTER_LINEAR)

        return cv2.addWeighted(frame, 1.0, self._glow, 0.55 * intensity, 0)


class ColorGrade:
    """Vignette and scanlines fused into one cached integer multiply.

    Both effects are static for a given frame size, so they are baked into
    a single uint8 mask and applied with ``cv2.multiply``. Integer SIMD is
    several times faster than the float32 conversion each would need
    separately, and one pass replaces two.
    """

    def __init__(
        self,
        vignette: float = 0.55,
        scanline: float = 0.05,
        period: int = 3,
    ):
        self.vignette = vignette
        self.scanline = scanline
        self.period = max(2, period)
        self._mask: Optional[np.ndarray] = None
        self._size: tuple[int, int] = (0, 0)

    def _build(self, width: int, height: int) -> np.ndarray:
        """Bake the combined darkening mask for a frame size."""
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)

        radius = np.sqrt(grid_x ** 2 + grid_y ** 2) / math.sqrt(2.0)
        mask = 1.0 - np.clip(radius, 0.0, 1.0) ** 1.8 * self.vignette

        if self.scanline > 0.0:
            rows = np.arange(height, dtype=np.float32)
            band = 1.0 - self.scanline * (np.sin(rows * math.pi / self.period) ** 2)
            mask *= band[:, None]

        baked = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        return cv2.merge([baked, baked, baked])

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Darken edges and add scanlines in a single pass."""
        height, width = frame.shape[:2]
        if self._mask is None or self._size != (width, height):
            self._mask = self._build(width, height)
            self._size = (width, height)

        return cv2.multiply(frame, self._mask, scale=1.0 / 255.0)


def glass_panel(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    theme: Theme,
    radius: int = 18,
    opacity: float = 0.55,
    border: bool = True,
) -> None:
    """Draw a frosted glass panel in place.

    Blurs the region beneath the panel, tints it with the theme's panel
    colour, then adds a rounded border and a specular top edge.

    Args:
        frame: Destination image, modified in place.
        rect: Panel bounds as (x, y, width, height).
        theme: Palette supplying tint and border colours.
        radius: Corner radius in pixels.
        opacity: Tint strength from 0 (clear) to 1 (opaque).
        border: Whether to stroke the panel outline.
    """
    x, y, width, height = rect
    frame_h, frame_w = frame.shape[:2]

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame_w, x + width), min(frame_h, y + height)
    if x1 <= x0 or y1 <= y0:
        return

    region = frame[y0:y1, x0:x1]

    # Two box blurs approximate a Gaussian at a fraction of the cost: box
    # blur is O(1) per pixel regardless of radius, whereas the equivalent
    # Gaussian needs a ~55-tap kernel and dominated the whole HUD budget.
    blurred = cv2.blur(region, (17, 17))
    blurred = cv2.blur(blurred, (17, 17))

    frosted = cv2.addWeighted(
        blurred, 1.0 - opacity,
        np.full_like(blurred, theme.panel), opacity, 0,
    )

    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    _rounded_rect(mask, (0, 0, x1 - x0, y1 - y0), radius, 255, -1)

    cv2.copyTo(frosted, mask, region)

    if border:
        edge = np.zeros_like(mask)
        _rounded_rect(edge, (0, 0, x1 - x0, y1 - y0), radius, 255, 1)
        region[edge > 0] = mix(theme.panel, theme.accent, 0.55)

        # Specular highlight along the top edge sells the glass.
        highlight_h = max(1, (y1 - y0) // 14)
        gloss = region[:highlight_h]
        region[:highlight_h] = cv2.addWeighted(
            gloss, 0.82, np.full_like(gloss, theme.highlight), 0.18, 0
        )


def glass_disc(
    frame: np.ndarray,
    centre: tuple[int, int],
    radius: int,
    theme: Theme,
    opacity: float = 0.62,
) -> None:
    """Draw a circular frosted panel, used as a dial's backing plate.

    Args:
        frame: Destination image, modified in place.
        centre: Disc centre in pixels.
        radius: Disc radius in pixels.
        theme: Palette supplying the tint.
        opacity: Tint strength from 0 (clear) to 1 (opaque).
    """
    cx, cy = centre
    frame_h, frame_w = frame.shape[:2]

    x0, y0 = max(0, cx - radius), max(0, cy - radius)
    x1, y1 = min(frame_w, cx + radius), min(frame_h, cy + radius)
    if x1 <= x0 or y1 <= y0:
        return

    region = frame[y0:y1, x0:x1]
    blurred = cv2.blur(cv2.blur(region, (15, 15)), (15, 15))
    frosted = cv2.addWeighted(
        blurred, 1.0 - opacity,
        np.full_like(blurred, theme.panel), opacity, 0,
    )

    mask = np.zeros(region.shape[:2], np.uint8)
    cv2.circle(mask, (cx - x0, cy - y0), radius, 255, -1, cv2.LINE_AA)
    cv2.copyTo(frosted, mask, region)


def _rounded_rect(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    radius: int,
    colour,
    thickness: int,
) -> None:
    """Draw a rounded rectangle onto a canvas."""
    x, y, width, height = rect
    radius = max(0, min(radius, width // 2, height // 2))

    if radius == 0:
        cv2.rectangle(canvas, (x, y), (x + width, y + height), colour, thickness)
        return

    if thickness < 0:
        cv2.rectangle(canvas, (x + radius, y), (x + width - radius, y + height), colour, -1)
        cv2.rectangle(canvas, (x, y + radius), (x + width, y + height - radius), colour, -1)
    else:
        cv2.line(canvas, (x + radius, y), (x + width - radius, y), colour, thickness)
        cv2.line(canvas, (x + radius, y + height), (x + width - radius, y + height), colour, thickness)
        cv2.line(canvas, (x, y + radius), (x, y + height - radius), colour, thickness)
        cv2.line(canvas, (x + width, y + radius), (x + width, y + height - radius), colour, thickness)

    for cx, cy, start in (
        (x + radius, y + radius, 180),
        (x + width - radius, y + radius, 270),
        (x + width - radius, y + height - radius, 0),
        (x + radius, y + height - radius, 90),
    ):
        cv2.ellipse(canvas, (cx, cy), (radius, radius), start, 0, 90, colour, thickness)


def glow_line(
    frame: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    colour: tuple[int, int, int],
    thickness: int = 2,
    intensity: float = 1.0,
) -> None:
    """Draw a line with a soft halo around a bright core."""
    halo = scale(colour, 0.45 * intensity)
    cv2.line(frame, start, end, halo, thickness + 6, cv2.LINE_AA)
    cv2.line(frame, start, end, scale(colour, 0.8), thickness + 2, cv2.LINE_AA)
    cv2.line(frame, start, end, colour, thickness, cv2.LINE_AA)


def glow_circle(
    frame: np.ndarray,
    centre: tuple[int, int],
    radius: int,
    colour: tuple[int, int, int],
    thickness: int = 2,
    intensity: float = 1.0,
) -> None:
    """Draw a circle with a soft halo."""
    cv2.circle(frame, centre, radius + 4, scale(colour, 0.35 * intensity), thickness, cv2.LINE_AA)
    cv2.circle(frame, centre, radius, colour, thickness, cv2.LINE_AA)


@dataclass
class Ripple:
    """An expanding ring emitted when a string is struck."""

    centre: tuple[int, int]
    colour: tuple[int, int, int]
    age: float = 0.0
    lifetime: float = 0.55
    max_radius: float = 190.0

    @property
    def alive(self) -> bool:
        """Whether the ripple still has time left to render."""
        return self.age < self.lifetime

    def update(self, dt: float) -> None:
        """Advance the ripple's age."""
        self.age += dt

    def draw(self, frame: np.ndarray) -> None:
        """Render the ring at its current radius and fade."""
        if not self.alive:
            return
        progress = self.age / self.lifetime
        radius = int(progress * self.max_radius)
        if radius < 1:
            return
        fade = (1.0 - progress) ** 1.6
        cv2.circle(frame, self.centre, radius, scale(self.colour, fade), 2, cv2.LINE_AA)
        if radius > 6:
            cv2.circle(frame, self.centre, radius - 5, scale(self.colour, fade * 0.4), 1, cv2.LINE_AA)


class RippleField:
    """Owns the live ripples for a scene."""

    def __init__(self, limit: int = 24):
        self.limit = limit
        self._ripples: list[Ripple] = []

    def emit(self, centre: tuple[int, int], colour: tuple[int, int, int], max_radius: float = 190.0) -> None:
        """Spawn a ripple, discarding the oldest when at capacity."""
        if len(self._ripples) >= self.limit:
            self._ripples.pop(0)
        self._ripples.append(Ripple(centre=centre, colour=colour, max_radius=max_radius))

    def update(self, dt: float) -> None:
        """Age all ripples and retire the expired ones."""
        for ripple in self._ripples:
            ripple.update(dt)
        self._ripples = [r for r in self._ripples if r.alive]

    def draw(self, frame: np.ndarray) -> None:
        """Render every live ripple."""
        for ripple in self._ripples:
            ripple.draw(frame)

    def __len__(self) -> int:
        """Number of live ripples."""
        return len(self._ripples)


def _self_check() -> None:
    """Verify effects run and behave correctly on a synthetic frame."""
    from themes import THEMES

    theme = THEMES["cyber_blue"]
    frame = np.full((180, 320, 3), 40, np.uint8)

    # Bloom must brighten a frame containing highlights, never shrink it.
    frame[80:100, 140:180] = 255
    bloomed = Bloom().apply(frame.copy(), 1.0)
    assert bloomed.shape == frame.shape
    assert bloomed.dtype == np.uint8
    assert bloomed.mean() > frame.mean(), "bloom should add light"
    assert Bloom().apply(frame.copy(), 0.0).mean() == frame.mean(), "zero intensity is a no-op"

    # The fused grade must darken corners more than the centre.
    flat = np.full((180, 320, 3), 200, np.uint8)
    graded = ColorGrade(vignette=0.8, scanline=0.05).apply(flat)
    assert graded.dtype == np.uint8 and graded.shape == flat.shape
    assert int(graded[0, 0].mean()) < int(graded[90, 160].mean()), "corners must be darker"
    assert graded.mean() < flat.mean(), "grade should darken overall"

    # The mask is cached, so a second call must be identical.
    grade = ColorGrade()
    assert np.array_equal(grade.apply(flat), grade.apply(flat)), "grade not deterministic"

    # A changed frame size must rebuild the mask rather than raise.
    grade.apply(np.full((90, 160, 3), 200, np.uint8))

    # A glass panel must alter only its own rectangle.
    canvas = np.full((180, 320, 3), 40, np.uint8)
    glass_panel(canvas, (20, 20, 120, 60), theme)
    assert not np.array_equal(canvas[20:80, 20:140], np.full((60, 120, 3), 40, np.uint8))
    assert np.array_equal(canvas[150:, 250:], np.full((30, 70, 3), 40, np.uint8)), "leaked outside rect"

    # Panels clipped by the frame edge must not raise.
    glass_panel(canvas, (-40, -30, 90, 70), theme)
    glass_panel(canvas, (300, 160, 200, 200), theme)
    glass_panel(canvas, (400, 400, 50, 50), theme)

    # Ripples must expand, fade and retire.
    field = RippleField(limit=3)
    field.emit((80, 60), theme.accent)
    assert len(field) == 1
    field.update(0.1)
    field.draw(canvas)
    field.update(1.0)
    assert len(field) == 0, "expired ripples must be removed"

    for _ in range(10):
        field.emit((10, 10), theme.accent)
    assert len(field) == 3, "ripple field must respect its limit"

    glow_line(canvas, (5, 5), (100, 100), theme.accent)
    glow_circle(canvas, (60, 60), 20, theme.accent)

    print("effects self-check passed")


if __name__ == "__main__":
    _self_check()
