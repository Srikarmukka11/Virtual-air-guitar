"""Frame compositor.

Builds each frame in a fixed order: camera backdrop, ambient depth cues,
instrument, interactive overlays, post-processing, then HUD. The HUD is
drawn after bloom so that text stays crisp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from audio import DEFAULT_GUITAR, GUITARS, Guitar
from camera import HandData
from effects import Bloom, ColorGrade, RippleField, glow_line
from hud import Hud, HudState, draw_help
from particles import MotionTrail, ParticleSystem
from strings import StringSet
from themes import Theme, ThemeManager, mix, scale
from ui import CHORD_ENTRIES, ButtonRow, Pointer

#: Landmark pairs forming the hand skeleton, for debug rendering.
_BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


class Renderer:
    """Owns every visual element and composites the final image."""

    def __init__(
        self,
        width: int,
        height: int,
        theme_name: str = "cyber_blue",
        glow_intensity: float = 1.0,
        particle_count: int = 900,
        string_spacing: float = 46.0,
        camera_opacity: float = 0.55,
    ):
        self.width = width
        self.height = height
        self.glow_intensity = glow_intensity
        self.camera_opacity = camera_opacity

        self.themes = ThemeManager(theme_name)

        # Three horizontal bands, top to bottom: chord bar, guitar, status.
        # Nothing overlaps, so the instrument gets the full width instead of
        # the narrow slot it used to occupy between two dials.
        margin = max(24, int(width * 0.03))
        self.bar_height = max(64, int(height * 0.125))
        chord_top = max(16, int(height * 0.033))
        self.chord_bar = ButtonRow(
            width,
            top=chord_top,
            height=self.bar_height,
            labels=[entry[0] for entry in CHORD_ENTRIES],
            margin=margin,
            pad=16,
        )

        # Instrument row, directly under the chords. Its aim band and the
        # chord row's must not overlap, or a hand between them would be
        # claimed by both; the gap below is sized to keep them apart.
        instrument_top = self.chord_bar.bottom + max(34, int(height * 0.06))
        instrument_h = max(34, int(height * 0.058))
        self.instrument_bar = ButtonRow(
            width,
            top=instrument_top,
            height=instrument_h,
            labels=[g.label for g in GUITARS.values()],
            margin=margin + int(width * 0.13),
            pad=12,
        )
        assert self.instrument_bar.top - self.instrument_bar.pad > \
            self.chord_bar.bottom + self.chord_bar.pad, "aim bands overlap"

        # The instrument sits in a centred block rather than spanning the
        # whole frame: a shorter strum stroke is quicker to play, and the
        # strings read as an instrument instead of as scan lines.
        # Spacing is given for a 720-tall frame and scaled from there, so the
        # instrument keeps its proportions instead of running into the status
        # strip on a shorter window.
        self._string_spacing = string_spacing * (height / 720.0)
        half = width * 0.31
        self._string_bounds = (width * 0.5 - half, width * 0.5 + half)
        self._string_centre = height * 0.55
        self.guitar = GUITARS[DEFAULT_GUITAR]
        self.strings = self._build_strings()
        self.particles = ParticleSystem(particle_count)
        self.ripples = RippleField()
        self.pointer = Pointer()
        self.pick_trail = MotionTrail(length=22, min_step=4.0)

        self.bloom = Bloom()
        self.grade = ColorGrade(vignette=0.55, scanline=0.05)
        self.hud = Hud(width, height)

        self.show_help = False
        self.show_landmarks = False

        self._time = 0.0
        self._backdrop = np.empty((height, width, 3), np.uint8)

    @property
    def theme(self) -> Theme:
        """The active theme."""
        return self.themes.theme

    def _build_strings(self) -> StringSet:
        """Lay out the strings for the current instrument."""
        return StringSet(
            self.width, self.height,
            spacing=self._string_spacing * self.guitar.spacing,
            centre_y=self._string_centre,
            bounds=self._string_bounds,
            count=self.guitar.strings,
            thickness=self.guitar.thickness,
        )

    def in_pointer_zone(self, point) -> bool:
        """Whether a point is aiming at either selection row."""
        return self.chord_bar.contains(point) or self.instrument_bar.contains(point)

    def set_guitar(self, guitar: Guitar) -> None:
        """Switch instrument, rebuilding the strings and the level meter."""
        self.guitar = guitar
        self.strings = self._build_strings()
        self.hud.set_string_count(guitar.strings)
        self.instrument_bar.select(list(GUITARS).index(guitar.name))

    # ------------------------------------------------------------- update

    def update(
        self,
        dt: float,
        pointer_px: Optional[tuple[float, float]],
        pick_px: Optional[tuple[float, float]],
        pinching: bool = False,
    ) -> tuple[int, int]:
        """Advance every animated element.

        Returns:
            ``(chord index, instrument index)`` committed this frame, each
            -1 when nothing was committed.
        """
        self._time += dt

        self.pointer.set(pointer_px)
        self.pointer.active = self.in_pointer_zone(pointer_px)
        self.pointer.update(dt)

        if pick_px is not None:
            self.pick_trail.push(pick_px)
        elif len(self.pick_trail):
            self.pick_trail.decay()

        # Route the pointer to whichever row it is aiming at, so dwell
        # progress on one row is not fed by time spent over the other.
        on_chords = self.chord_bar.contains(pointer_px)
        on_instruments = self.instrument_bar.contains(pointer_px)
        chord = self.chord_bar.update(
            dt, pointer_px if on_chords else None, instant=pinching
        )
        instrument = self.instrument_bar.update(
            dt, pointer_px if on_instruments else None, instant=pinching
        )
        if chord != -1 or instrument != -1:
            self.pointer.click()

        self.strings.update(dt)
        self.particles.update(dt)
        self.ripples.update(dt)
        return chord, instrument

    def on_strum(self, string_index: int, velocity: float, direction: int, x: float) -> None:
        """Spawn the visual response to a struck string."""
        theme = self.theme
        point = self.strings.centre_of(string_index, x)

        self.ripples.emit(point, theme.string_hot, max_radius=120.0 + velocity * 130.0)
        self.particles.emit(
            origin=point,
            colour=theme.particle,
            count=int(18 + velocity * 34),
            speed=180.0 + velocity * 340.0,
            spread=math.pi * 0.85,
            direction=math.pi / 2 if direction > 0 else -math.pi / 2,
            lifetime=0.55 + velocity * 0.45,
        )
        self.hud.flash()

    # ------------------------------------------------------------- render

    def render(
        self,
        camera_frame: Optional[np.ndarray],
        left: HandData,
        right: HandData,
        state: HudState,
    ) -> np.ndarray:
        """Composite and return the final frame."""
        theme = self.theme
        frame = self._compose_backdrop(camera_frame, theme)

        self._draw_stage(frame, theme)
        self.chord_bar.draw(frame, theme)
        self.instrument_bar.draw(frame, theme)

        self.strings.draw(frame, theme, self.glow_intensity)
        self.ripples.draw(frame)

        self.pick_trail.draw(frame, theme.string_hot, thickness=4)
        self.particles.draw(frame)

        if self.show_landmarks:
            self._draw_skeleton(frame, theme, left)
            self._draw_skeleton(frame, theme, right)

        self.pointer.draw(frame, theme)

        frame = self.bloom.apply(frame, self.glow_intensity)
        frame = self.grade.apply(frame)

        self.hud.draw(frame, theme, state)
        if self.show_help:
            draw_help(frame, theme, self.width, self.height)

        return frame

    def _compose_backdrop(
        self,
        camera_frame: Optional[np.ndarray],
        theme: Theme,
    ) -> np.ndarray:
        """Dim and tint the camera image into a stage backdrop."""
        if camera_frame is None:
            self._backdrop[:] = theme.backdrop
            return self._backdrop.copy()

        if camera_frame.shape[0] != self.height or camera_frame.shape[1] != self.width:
            camera_frame = cv2.resize(
                camera_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR
            )

        # Desaturate so the coloured UI is the only saturated content.
        grey = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2GRAY)
        desaturated = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

        # The backdrop tint is a constant colour, so it is added as a scalar
        # instead of allocating and blending a full-frame array each frame.
        gain = min(1.0, theme.camera_gain * self.camera_opacity * 2.0)
        toned = cv2.addWeighted(camera_frame, 0.45 * gain, desaturated, 0.55 * gain, 0)
        offset = tuple(c * theme.backdrop_mix for c in theme.backdrop) + (0.0,)
        return cv2.add(toned, offset)

    def _draw_stage(self, frame: np.ndarray, theme: Theme) -> None:
        """Draw the fretboard frame behind the strings."""
        top = int(self.strings.top - 42)
        bottom = int(self.strings.bottom + 42)
        left = int(self.strings.strings[0].x_start - 34)
        right = int(self.strings.strings[0].x_end + 34)

        # Shade only the fretboard rectangle rather than the whole frame.
        x0, y0 = max(0, left), max(0, top)
        x1, y1 = min(self.width, right), min(self.height, bottom)
        if x1 > x0 and y1 > y0:
            region = frame[y0:y1, x0:x1]
            tint = np.full_like(region, theme.panel)
            cv2.addWeighted(tint, 0.42, region, 0.58, 0, dst=region)

        edge = scale(theme.secondary, 0.85)
        cv2.rectangle(frame, (left, top), (right, bottom), edge, 1, cv2.LINE_AA)

        # Bridge and nut posts at either end of the string span.
        pulse = 0.6 + 0.4 * math.sin(self._time * 2.0)
        for x in (left + 16, right - 16):
            glow_line(frame, (x, top + 12), (x, bottom - 12),
                      scale(theme.accent, pulse), 2, self.glow_intensity)

    def _draw_skeleton(self, frame: np.ndarray, theme: Theme, hand: HandData) -> None:
        """Draw a hand's landmark skeleton for debugging."""
        if not hand.tracked:
            return

        points = [
            (int(p[0] * self.width), int(p[1] * self.height)) for p in hand.points
        ]
        for a, b in _BONES:
            cv2.line(frame, points[a], points[b], scale(theme.accent, 0.65), 1, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 3, theme.highlight, -1, cv2.LINE_AA)

    # ------------------------------------------------------------ helpers

    def set_theme(self, name: str) -> bool:
        """Switch palette by name."""
        return self.themes.set(name)

    def set_theme_index(self, index: int) -> bool:
        """Switch palette by number-key index."""
        return self.themes.set_index(index)

    def string_energies(self) -> list[float]:
        """Current vibration energy of each string."""
        return [s.energy for s in self.strings.strings]

    def reset(self) -> None:
        """Clear transient visual state."""
        self.strings.reset()
        self.particles.clear()
        self.pick_trail.clear()


def _self_check() -> None:
    """Render frames headlessly and verify the pipeline produces output."""
    width, height = 960, 540
    renderer = Renderer(width, height, particle_count=200)

    camera = np.random.default_rng(3).integers(
        0, 255, (height, width, 3), dtype=np.uint8
    )
    state = HudState(fps=60.0, chord="Am", modifier="Minor", confidence=0.9)

    # A frame with no camera must still render the full interface.
    blank = renderer.render(None, HandData(), HandData(), state)
    assert blank.shape == (height, width, 3), blank.shape
    assert blank.dtype == np.uint8
    assert blank.std() > 5.0, "interface should not be a flat fill"

    # A frame with a camera image must differ from the camera-less one.
    renderer.update(0.016, (200.0, 270.0), (480.0, 250.0))
    withcam = renderer.render(camera, HandData(), HandData(), state)
    assert withcam.shape == blank.shape
    assert not np.array_equal(withcam, blank), "camera frame had no effect"

    # A mismatched camera resolution must be resized, not crash.
    odd = np.zeros((480, 640, 3), np.uint8)
    assert renderer.render(odd, HandData(), HandData(), state).shape == (height, width, 3)

    # The three bands must not overlap: chord bar, then strings, then HUD.
    assert renderer.chord_bar.bottom < renderer.strings.top - 40, "chord bar overlaps the strings"
    assert renderer.strings.bottom + 40 < height - 92, "strings overlap the status strip"
    # The instrument is a centred block: wide enough to play, not so wide the
    # strum stroke becomes a reach.
    first = renderer.strings.strings[0]
    span = first.x_end - first.x_start
    assert width * 0.5 < span < width * 0.75, f"string span off at {span / width:.2f}"
    assert abs((first.x_start + first.x_end) / 2 - width / 2) < 1.0, "not centred"

    # Pointing at a chord button must commit it after the dwell, with no
    # pinch, and must not commit before then.
    def dwell_on(row, index, steps=40):
        """Point at a button until it commits; return (chord, instrument)."""
        button = row.buttons[index]
        aim = (button.x + button.width / 2.0, row.top + row.height / 2.0)
        renderer.update(0.016, None, None)
        for _ in range(steps):
            chord, instrument = renderer.update(0.016, aim, None)
            if chord != -1 or instrument != -1:
                return chord, instrument
        return -1, -1

    button = renderer.chord_bar.buttons[3]
    aim = (button.x + button.width / 2.0, renderer.chord_bar.top + 20.0)
    assert renderer.update(0.05, aim, None) == (-1, -1), "committed before the dwell"
    assert dwell_on(renderer.chord_bar, 3) == (3, -1), "chord dwell did not commit"

    # The instrument row must commit on its own, without touching the chords.
    assert dwell_on(renderer.instrument_bar, 2) == (-1, 2), "instrument dwell failed"

    # The two rows must not both claim the same pointer position. A point in
    # one row's band must leave the other row's dwell progress at zero.
    renderer.update(0.016, None, None)
    mid_chord = (renderer.chord_bar.buttons[0].x + 5.0,
                 renderer.chord_bar.top + renderer.chord_bar.height / 2.0)
    for _ in range(20):
        renderer.update(0.016, mid_chord, None)
    assert renderer.instrument_bar._progress == 0.0, "rows both claimed the pointer"
    assert renderer.in_pointer_zone(mid_chord)
    assert not renderer.in_pointer_zone((640.0, height - 40.0))

    # Strumming must spawn particles and ripples.
    renderer.reset()
    assert len(renderer.particles) == 0
    renderer.on_strum(2, 0.8, 1, 480.0)
    assert len(renderer.particles) > 0, "strum produced no particles"
    assert len(renderer.ripples) == 1

    # Every theme must render, and each must look different.
    fingerprints = []
    for theme in renderer.themes:
        assert renderer.set_theme(theme.name)
        out = renderer.render(camera, HandData(), HandData(), state)
        fingerprints.append(out.mean(axis=(0, 1)).round(1).tobytes())
    assert len(set(fingerprints)) == len(fingerprints), "themes render identically"

    # Overlays must draw without raising.
    renderer.show_help = True
    renderer.show_landmarks = True
    hand = HandData()
    hand.tracked = True
    hand.points = np.random.default_rng(5).random((21, 2)).astype(np.float32)
    renderer.render(camera, hand, hand, state)

    energies = renderer.string_energies()
    assert len(energies) == 6 and all(0.0 <= e <= 1.0 for e in energies)

    # Switching instrument must rebuild the strings and the level meter, and
    # a four-string must stay centred where the six-string was.
    from audio import GUITARS
    centre_before = (renderer.strings.top + renderer.strings.bottom) / 2
    renderer.set_guitar(GUITARS["bass"])
    assert len(renderer.strings.strings) == 4, "bass should have four strings"
    assert len(renderer.string_energies()) == 4
    assert len(renderer.hud.waveform.levels) == 4, "meter did not follow the instrument"
    centre_after = (renderer.strings.top + renderer.strings.bottom) / 2
    assert abs(centre_after - centre_before) < 1.0, "four-string drifted off centre"
    renderer.render(camera, HandData(), HandData(), state)
    renderer.on_strum(3, 0.7, 1, 640.0)

    renderer.set_guitar(GUITARS["acoustic"])
    assert len(renderer.strings.strings) == 6

    renderer.reset()
    assert len(renderer.particles) == 0
    assert all(e == 0.0 for e in renderer.string_energies())

    print("renderer self-check passed")


if __name__ == "__main__":
    _self_check()
