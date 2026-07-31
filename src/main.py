"""Virtual Air Guitar - application entry point."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from audio import (AudioEngine, DEFAULT_GUITAR, GUITARS, TIMBRES,
                   chord_voicing, note_name)
from camera import HandData, HandTracker, INDEX_TIP
from config import Config
from gestures import Gesture, GestureRecognizer
from hud import HudState
from logger import setup_logger
from renderer import Renderer
from ui import CHORD_ENTRIES, MODIFIER_ENTRIES

WINDOW = "Virtual Air Guitar"

#: Keyboard key -> tone pack.
TONE_KEYS = {
    ord("z"): "acoustic",
    ord("x"): "electric",
    ord("c"): "clean",
    ord("v"): "distortion",
    ord("b"): "muted",
    ord("n"): "fingerstyle",
}

#: Hand pose -> tone pack.
GESTURE_TONES = {
    Gesture.THUMBS_UP: "electric",
    Gesture.PEACE: "acoustic",
    Gesture.ROCK: "distortion",
    Gesture.OK: "fingerstyle",
}


def assign_roles(in_zone, candidates):
    """Split tracked hands into a pointing hand and a picking hand.

    Roles follow where a hand is, not which hand MediaPipe labelled it.
    Handedness is frequently mislabelled on a mirrored frame, and pinning
    "left selects, right strums" to that label made the controls silently
    swap mid-session. A hand raised into the chord bar points; a hand below
    it picks.

    Args:
        in_zone: Predicate telling whether a point is aiming at a
            selection row.
        candidates: ``(hand, pixel position)`` for every tracked hand.

    Returns:
        ``(pointer hand, pointer position, pick position)``, each None when
        no hand fills that role.
    """
    pointer = pointer_px = pick_px = None
    for hand, point in candidates:
        if pointer is None and in_zone(point):
            pointer, pointer_px = hand, point
        elif pick_px is None:
            pick_px = point

    # A lone hand that is not up at the bar still has to be able to reach it,
    # so it keeps the pick role and gains the pointer role as well.
    if pointer is None and len(candidates) == 1:
        pointer, pointer_px = candidates[0]

    return pointer, pointer_px, pick_px


class VirtualAirGuitar:
    """Wires tracking, gestures, audio and rendering into a running app."""

    def __init__(self, config: Config, headless: bool = False):
        self.config = config
        self.headless = headless
        self.log = setup_logger("VirtualAirGuitar", config.debug.log_level)

        self.width = config.ui.render_width
        self.height = config.ui.render_height

        self.tracker = HandTracker(config.camera)
        self.audio = AudioEngine(config.audio, Path(__file__).parent.parent / "assets")
        self.renderer = Renderer(
            width=self.width,
            height=self.height,
            theme_name=config.ui.default_theme,
            glow_intensity=config.ui.glow_intensity,
            particle_count=config.ui.particle_count,
            string_spacing=config.ui.string_spacing,
            camera_opacity=config.ui.camera_opacity,
        )

        self.left_gestures = GestureRecognizer(config.ui.gesture_sensitivity)
        self.right_gestures = GestureRecognizer(config.ui.gesture_sensitivity)

        self.chord_root: Optional[str] = None
        self.chord_label = "OFF"
        self.default_quality = "Major"
        self.quality_override: Optional[str] = None
        self.voicing: tuple[int, ...] = (-1,) * 6
        self._last_sample = -1

        self.muted = False
        self.paused = False
        self.running = False
        self._last_gesture = "none"

        self.set_guitar(config.audio.default_guitar)

        self.renderer.show_landmarks = config.debug.show_landmarks
        self.renderer.show_help = False

        self._frame_times: list[float] = []

    # -------------------------------------------------------------- state

    @property
    def quality(self) -> str:
        """The chord quality in force, override winning over the default."""
        return self.quality_override or self.default_quality

    def _refresh_voicing(self) -> None:
        """Recompute the voicing after any chord change."""
        tuning = self.renderer.guitar.tuning
        if self.chord_root is None:
            self.voicing = (-1,) * len(tuning)
        else:
            self.voicing = chord_voicing(self.chord_root, self.quality, tuning)

    def _select_chord(self, index: int) -> None:
        """Arm a chord from its button index."""
        label, root, default_quality = CHORD_ENTRIES[index]
        self.chord_label = label
        self.chord_root = root
        self.default_quality = default_quality
        self.renderer.chord_bar.select(index)
        self._refresh_voicing()
        self.log.info("Chord: %s %s", label, self.quality)

    def _clear_chord(self) -> None:
        """Disarm the current chord."""
        self.chord_label = "OFF"
        self.chord_root = None
        self._refresh_voicing()

    def _note_label(self) -> str:
        """Human-readable list of the armed notes."""
        names = [note_name(n) for n in self.voicing if n >= 0]
        return " ".join(names) if names else "--"

    # --------------------------------------------------------------- loop

    def run(self, max_frames: int = 0, shot_dir: Optional[Path] = None) -> int:
        """Run the main loop.

        Args:
            max_frames: Stop after this many frames; 0 runs until quit.
            shot_dir: When set, save periodic PNG frames here.

        Returns:
            Process exit code.
        """
        if not self.tracker.available:
            self.log.warning("No camera; rendering interface without video")
        self.tracker.start()

        windowed = not self.headless
        if windowed:
            try:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW, self.width, self.height)
                if self.config.ui.fullscreen:
                    cv2.setWindowProperty(
                        WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                    )
            except cv2.error as exc:
                self.log.error("No display available (%s); continuing headless", exc)
                windowed = False

        if shot_dir is not None:
            shot_dir.mkdir(parents=True, exist_ok=True)

        self.running = True
        frames = 0
        previous = time.perf_counter()
        self.log.info("Running at %dx%d", self.width, self.height)

        # Render no faster than the display needs. Left uncapped the loop
        # spins on a core it has to share with the capture thread, which
        # slows the hand tracking it is waiting on.
        budget = 1.0 / max(30, self.config.camera.target_fps)

        try:
            while self.running:
                now = time.perf_counter()
                dt = min(0.1, now - previous)
                previous = now

                frame, left, right = self.tracker.snapshot()
                self._step(dt, now, left, right)

                output = self.renderer.render(frame, left, right, self._hud_state(dt))

                if windowed:
                    cv2.imshow(WINDOW, output)
                    spare = budget - (time.perf_counter() - now)
                    key = cv2.waitKey(max(1, int(spare * 1000))) & 0xFF
                    if key != 255 and not self._handle_key(key):
                        break
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break

                if shot_dir is not None and frames % 30 == 0:
                    cv2.imwrite(str(shot_dir / f"frame_{frames:04d}.png"), output)

                frames += 1
                if max_frames and frames >= max_frames:
                    break

        except KeyboardInterrupt:
            self.log.info("Interrupted")
        finally:
            self.shutdown()

        self.log.info("Rendered %d frames", frames)
        return 0

    def _step(self, dt: float, now: float, left: HandData, right: HandData) -> None:
        """Advance one frame of interaction."""
        left_gesture = self.left_gestures.update(left, now)
        right_gesture = self.right_gestures.update(right, now)

        pointer, pointer_px, pick_px = self._assign_hands(left, right)
        pointer_gesture = (
            left_gesture if pointer is left else
            right_gesture if pointer is right else Gesture.NONE
        )
        self._apply_gestures(pointer_gesture, left_gesture, right_gesture)

        chord, instrument = self.renderer.update(
            dt, pointer_px, pick_px, pointer_gesture is Gesture.PINCH
        )
        if chord != -1:
            self._select_chord(chord)
        if instrument != -1:
            self.set_guitar(list(GUITARS)[instrument])

        # Strum detection must run once per camera sample, timed against the
        # camera. Running it every render frame re-measured the same hand
        # position against a much shorter render dt, which read every strum,
        # however gentle, as maximum velocity and threw the dynamics away.
        sample = self.tracker.sample_id
        if sample == self._last_sample:
            return
        self._last_sample = sample
        hand_dt = self.tracker.sample_dt or dt

        if self.paused:
            return
        if pick_px is None:
            self.renderer.strings.detect_strum(None, hand_dt)
            return

        for index, velocity, direction in self.renderer.strings.detect_strum(
            pick_px, hand_dt
        ):
            self.renderer.on_strum(index, velocity, direction, pick_px[0])
            if not self.muted and self.chord_root is not None:
                note = self.voicing[index]
                if note >= 0:
                    self.audio.play_note(note, velocity)

    def _assign_hands(
        self, left: HandData, right: HandData
    ) -> tuple[Optional[HandData], Optional[tuple[float, float]], Optional[tuple[float, float]]]:
        """Decide which hand points and which one strums."""
        candidates = [
            (hand, self._to_pixels(hand))
            for hand in (left, right)
            if hand.tracked and self._to_pixels(hand) is not None
        ]
        return assign_roles(self.renderer.in_pointer_zone, candidates)

    def _to_pixels(self, hand: HandData) -> Optional[tuple[float, float]]:
        """Convert a hand's index fingertip to pixel coordinates."""
        if not hand.tracked:
            return None
        point = hand.points[INDEX_TIP]
        return float(point[0]) * self.width, float(point[1]) * self.height

    def set_guitar(self, name: str) -> None:
        """Switch instrument, re-voicing the armed chord for its tuning."""
        guitar = GUITARS.get(name)
        if guitar is None or guitar is self.renderer.guitar:
            return
        self.renderer.set_guitar(guitar)
        self.audio.set_pack(guitar.pack)
        self._refresh_voicing()
        self.log.info("Guitar: %s (%d strings)", guitar.label, guitar.strings)

    def _cycle_guitar(self) -> None:
        """Step to the next instrument."""
        names = list(GUITARS)
        index = names.index(self.renderer.guitar.name)
        self.set_guitar(names[(index + 1) % len(names)])

    def _cycle_quality(self) -> None:
        """Step through the extra chord qualities the bar does not show."""
        options = (None,) + MODIFIER_ENTRIES
        index = options.index(self.quality_override) if self.quality_override in options else 0
        self.quality_override = options[(index + 1) % len(options)]
        self._refresh_voicing()
        self.log.info("Quality: %s", self.quality)

    def _apply_gestures(
        self, pointer: Gesture, left: Gesture, right: Gesture
    ) -> None:
        """Map recognised poses onto application actions.

        Poses are only honoured from the raised, pointing hand. Read from the
        picking hand they misfire constantly: a hand resting between strums
        curls into what reads as a closed fist, which used to mute the guitar
        mid-song. Swipes are accepted from either hand, since a sweep is
        deliberate enough not to happen by accident.
        """
        self._last_gesture = (
            left.value if left is not Gesture.NONE else right.value
        )

        for gesture in (left, right):
            if gesture is Gesture.SWIPE_LEFT:
                self._step_chord(-1)
            elif gesture is Gesture.SWIPE_RIGHT:
                self._step_chord(1)

        tone = GESTURE_TONES.get(pointer)
        if tone and tone != self.audio.current_pack:
            self.audio.set_pack(tone)

        if pointer is Gesture.CLOSED_FIST and not self.muted:
            self._set_muted(True)
        elif pointer is Gesture.OPEN_HAND and self.muted:
            self._set_muted(False)

    def _step_chord(self, delta: int) -> None:
        """Move the chord selection one button along the bar."""
        self._select_chord(self.renderer.chord_bar.select_offset(delta))

    def _set_muted(self, muted: bool) -> None:
        """Mute or unmute output."""
        self.muted = muted
        if muted:
            self.audio.stop_all()
        self.log.info("Muted: %s", muted)

    def _hud_state(self, dt: float) -> HudState:
        """Assemble the HUD's per-frame data."""
        self._frame_times.append(dt)
        if len(self._frame_times) > 60:
            self._frame_times.pop(0)
        average = sum(self._frame_times) / len(self._frame_times)

        confidence = max(self.tracker.left.confidence, self.tracker.right.confidence)
        state = HudState(
            fps=1.0 / average if average > 0 else 0.0,
            frame_ms=average * 1000.0,
            capture_fps=self.tracker.capture_fps,
            latency_ms=self.tracker.latency_ms,
            confidence=confidence,
            chord=self.chord_label,
            modifier=self.quality,
            note_label=self._note_label(),
            volume=self.audio.get_master_volume(),
            tone=f"{self.renderer.guitar.label.lower()} / {self.audio.current_pack}",
            theme_label=self.renderer.theme.label,
            gesture=self._last_gesture,
            camera_ok=self.tracker.available,
            tracking_ok=self.tracker.tracking_available,
            muted=self.muted,
            paused=self.paused,
            particles=len(self.renderer.particles),
        )
        self.renderer.hud.update(dt, state, self.renderer.string_energies())
        return state

    # ----------------------------------------------------------- controls

    def _handle_key(self, key: int) -> bool:
        """Handle a keypress; returns False to quit."""
        if key in (27, ord("q")):
            return False

        if ord("1") <= key <= ord("5"):
            self.renderer.set_theme_index(key - ord("1"))
            self.log.info("Theme: %s", self.renderer.theme.label)
        elif key in TONE_KEYS:
            self.audio.set_pack(TONE_KEYS[key])
        elif key == ord("["):
            self._step_chord(-1)
        elif key == ord("]"):
            self._step_chord(1)
        elif key in (ord("-"), ord("_")):
            self.audio.set_master_volume(self.audio.get_master_volume() - 0.05)
        elif key in (ord("="), ord("+")):
            self.audio.set_master_volume(self.audio.get_master_volume() + 0.05)
        elif key == ord("g"):
            self._cycle_guitar()
        elif key == ord("k"):
            self._cycle_quality()
        elif key == ord("m"):
            self._set_muted(not self.muted)
        elif key == ord("p"):
            self.paused = not self.paused
            self.log.info("Paused: %s", self.paused)
        elif key == ord("r"):
            self.renderer.reset()
            self.left_gestures.reset()
            self.right_gestures.reset()
            self.log.info("Reset")
        elif key == ord("h"):
            self.renderer.show_help = not self.renderer.show_help
        elif key == ord("d"):
            self.renderer.show_landmarks = not self.renderer.show_landmarks

        return True

    def shutdown(self) -> None:
        """Release every resource."""
        self.running = False
        self.tracker.stop()
        self.audio.shutdown()
        cv2.destroyAllWindows()
        try:
            self.config.save()
        except OSError as exc:
            self.log.warning("Could not save config: %s", exc)


def _self_check() -> None:
    """Verify hands are given roles by where they are, not by handedness."""
    from ui import CHORD_ENTRIES, ButtonRow

    bar = ButtonRow(1280, top=24, height=90, labels=[e[0] for e in CHORD_ENTRIES])
    in_zone = bar.contains
    up = (400.0, bar.top + 40.0)
    down = (600.0, 500.0)
    left, right = HandData(), HandData()

    # No hands means no roles.
    assert assign_roles(in_zone, []) == (None, None, None)

    # A hand at the bar points; the other one picks, whichever is which.
    pointer, pointer_px, pick_px = assign_roles(in_zone, [(left, up), (right, down)])
    assert pointer is left and pointer_px == up and pick_px == down
    pointer, pointer_px, pick_px = assign_roles(in_zone, [(left, down), (right, up)])
    assert pointer is right and pointer_px == up and pick_px == down, \
        "roles must follow position, not handedness"

    # A lone hand down at the strings picks, and still counts as the pointer
    # so that it can reach up to the bar.
    pointer, pointer_px, pick_px = assign_roles(in_zone, [(right, down)])
    assert pick_px == down and pointer is right

    # Raised, that same lone hand stops strumming so selecting cannot pluck.
    pointer, pointer_px, pick_px = assign_roles(in_zone, [(right, up)])
    assert pointer is right and pointer_px == up
    assert pick_px is None, "a hand at the bar must not also strum"

    # Both hands down: nobody points, and only one of them picks.
    pointer, _, pick_px = assign_roles(in_zone, [(left, down), (right, (700.0, 480.0))])
    assert pointer is None and pick_px == down

    print("main self-check passed")


def run_self_checks() -> int:
    """Run every module's self-check in dependency order."""
    modules = [
        "config", "themes", "animations", "effects", "particles",
        "strings", "audio", "camera", "gestures", "ui", "hud", "renderer",
    ]
    import importlib

    failures = 0
    for name in modules:
        module = importlib.import_module(name)
        check = getattr(module, "_self_check", None)
        if check is None:
            print(f"{name:12s} no self-check")
            continue
        try:
            check()
        except AssertionError as exc:
            failures += 1
            print(f"{name:12s} FAILED: {exc}")

    try:
        _self_check()
    except AssertionError as exc:
        failures += 1
        print(f"{'main':12s} FAILED: {exc}")

    print("all self-checks passed" if not failures else f"{failures} module(s) failed")
    return 1 if failures else 0


def main(argv: Optional[list[str]] = None) -> int:
    """Parse arguments and start the application."""
    parser = argparse.ArgumentParser(description="Virtual Air Guitar")
    parser.add_argument("--headless", action="store_true",
                        help="run without opening a window")
    parser.add_argument("--frames", type=int, default=0,
                        help="stop after N frames (0 = run until quit)")
    parser.add_argument("--shots", type=Path, default=None,
                        help="directory to save periodic PNG frames")
    parser.add_argument("--selftest", action="store_true",
                        help="run every module self-check and exit")
    parser.add_argument("--theme", type=str, default=None,
                        help="override the startup theme")
    parser.add_argument("--fullscreen", action="store_true",
                        help="start fullscreen")
    args = parser.parse_args(argv)

    if args.selftest:
        return run_self_checks()

    config = Config()
    if args.theme:
        config.ui.default_theme = args.theme
    if args.fullscreen:
        config.ui.fullscreen = True

    app = VirtualAirGuitar(config, headless=args.headless)
    return app.run(max_frames=args.frames, shot_dir=args.shots)


if __name__ == "__main__":
    sys.exit(main())
