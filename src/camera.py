"""Threaded camera capture and hand tracking (MediaPipe Tasks API).

MediaPipe 1.x removed the legacy ``mp.solutions.hands`` module; this uses
the current ``mediapipe.tasks.python.vision.HandLandmarker`` API and falls
back gracefully when the model or a camera is unavailable.
"""

from __future__ import annotations

import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import CameraConfig
from logger import get_logger

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

#: Width the frame is downscaled to before hand detection. The landmarker is
#: trained on roughly this scale and gains no accuracy from more pixels.
DETECT_WIDTH = 480

# MediaPipe hand landmark indices.
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
FINGER_TIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_PIPS = (3, 6, 10, 14, 18)
FINGER_MCPS = (2, 5, 9, 13, 17)
PALM = 9


class OneEuro:
    """Speed-adaptive low-pass filter for a landmark array.

    A fixed smoothing factor cannot win: low enough to kill the jitter of a
    resting hand, and it lags badly behind a fast one; high enough to keep up
    with a strum, and a still hand shivers. This varies the cutoff with how
    fast the hand is actually moving, so a still hand is smoothed hard and a
    moving hand is barely smoothed at all.

    See Casiez, Roussel & Vogel, "1e Filter" (CHI 2012).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 3.0,
                 d_cutoff: float = 1.0):
        """Configure the filter.

        Args:
            min_cutoff: Cutoff in Hz for a stationary hand. Lower is
                smoother but adds lag at rest.
            beta: How sharply the cutoff opens up with speed. Higher tracks
                fast motion more closely at the cost of more jitter.
            d_cutoff: Cutoff in Hz for the speed estimate itself.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        """Smoothing factor for a cutoff frequency over a timestep."""
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        """Forget history, so the next sample is taken as-is."""
        self._x = None
        self._dx = None

    def __call__(self, points: np.ndarray, dt: float) -> np.ndarray:
        """Filter one landmark array.

        Args:
            points: Landmark positions, any shape.
            dt: Seconds since the previous sample.

        Returns:
            The filtered positions.
        """
        if self._x is None or dt <= 0.0:
            self._x = points.copy()
            self._dx = np.zeros_like(points)
            return self._x

        speed = (points - self._x) / dt
        self._dx += self._alpha(self.d_cutoff, dt) * (speed - self._dx)

        # The cutoff rises with speed, so fast motion passes through almost
        # unfiltered while a resting hand stays locked down.
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        alpha = 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff) / dt)

        self._x += alpha * (points - self._x)
        return self._x


@dataclass
class HandData:
    """Smoothed landmark set and derived features for one hand.

    Positions are normalised to [0, 1] in frame space, with the frame
    already mirrored so that motion matches the user's own movement.
    """

    points: np.ndarray = field(default_factory=lambda: np.zeros((21, 2), np.float32))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros((21, 2), np.float32))
    handedness: str = "Unknown"
    confidence: float = 0.0
    tracked: bool = False

    pinch: float = 1.0
    openness: float = 0.0
    extended: tuple[bool, ...] = (False,) * 5

    def tip(self, index: int) -> np.ndarray:
        """Return the normalised position of a landmark."""
        return self.points[index]

    def pixel(self, index: int, width: int, height: int) -> tuple[int, int]:
        """Return a landmark position in pixel coordinates."""
        p = self.points[index]
        return int(p[0] * width), int(p[1] * height)

    def compute_features(self) -> None:
        """Derive pinch distance, openness and per-finger extension."""
        palm = self.points[PALM]
        wrist = self.points[WRIST]
        span = float(np.linalg.norm(palm - wrist)) or 1e-6

        self.pinch = float(
            np.linalg.norm(self.points[THUMB_TIP] - self.points[INDEX_TIP])
        ) / span

        # A finger counts as extended when its tip sits further from the
        # wrist than its middle joint, which is rotation independent.
        extended = []
        for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
            tip_d = float(np.linalg.norm(self.points[tip] - wrist))
            pip_d = float(np.linalg.norm(self.points[pip] - wrist))
            extended.append(tip_d > pip_d * 1.08)
        self.extended = tuple(extended)

        self.openness = sum(extended) / 5.0


class HandTracker:
    """Captures frames and tracks two hands on a background thread."""

    def __init__(self, config: CameraConfig, model_path: Optional[Path] = None):
        self.config = config
        self.logger = get_logger("HandTracker")

        self.left = HandData()
        self.right = HandData()
        self.frame: Optional[np.ndarray] = None
        self.frame_size: tuple[int, int] = (
            config.resolution_width,
            config.resolution_height,
        )

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._capture_fps = 0.0
        self._latency_ms = 0.0

        #: Bumped on every captured frame so consumers can spot a repeat.
        self.sample_id = 0
        #: Seconds between the last two captured frames.
        self.sample_dt = 0.0

        # `smoothing` reads 0 (loose) to 1 (heavy), so it maps inversely onto
        # the resting cutoff: more smoothing means a lower cutoff.
        heaviness = float(np.clip(config.smoothing, 0.05, 1.0))
        cutoff = 0.4 + (1.0 - heaviness) * 3.0
        beta = max(0.0, config.responsiveness)
        self._left_filter = OneEuro(min_cutoff=cutoff, beta=beta)
        self._right_filter = OneEuro(min_cutoff=cutoff, beta=beta)

        self._landmarker = self._create_landmarker(model_path)
        self._cap = self._open_camera()

        if self._cap is not None:
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_size = (width, height)
            self.logger.info("Camera open at %dx%d", width, height)

    # ---------------------------------------------------------------- setup

    def _create_landmarker(self, model_path: Optional[Path]):
        """Build a HandLandmarker, downloading the model if needed."""
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            self.logger.error("MediaPipe tasks API unavailable: %s", exc)
            return None

        if model_path is None:
            model_path = Path(__file__).parent.parent / "assets" / "models" / "hand_landmarker.task"

        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info("Downloading hand landmark model...")
            try:
                urllib.request.urlopen(MODEL_URL, timeout=60)
                urllib.request.urlretrieve(MODEL_URL, model_path)
                self.logger.info("Model saved to %s", model_path)
            except Exception as exc:
                self.logger.error("Model download failed: %s", exc)
                return None

        try:
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            landmarker = vision.HandLandmarker.create_from_options(options)
            self.logger.info("Hand landmarker ready")
            return landmarker
        except Exception as exc:
            self.logger.error("Could not create landmarker: %s", exc)
            return None

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        """Open the first working camera index."""
        for index in (self.config.device_index, 0, 1, 2):
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                cap.release()
                continue
            # MJPG before the frame size, and it must come first: most UVC
            # webcams only offer their high resolutions at a usable frame
            # rate in MJPG. Uncompressed YUYV at 720p is capped at 8 fps by
            # USB bandwidth on this class of device, which is what made the
            # whole app feel laggy regardless of how fast rendering was.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution_height)
            cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self.config.limit_exposure:
                self._limit_exposure(cap, index)
            ok, _ = cap.read()
            if ok:
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                self.logger.info(
                    "Camera %d: %s @ %.0f fps",
                    index,
                    fourcc.to_bytes(4, "little").decode(errors="replace"),
                    cap.get(cv2.CAP_PROP_FPS),
                )
                return cap
            cap.release()

        self.logger.error("No usable camera found")
        return None

    def _limit_exposure(self, cap: cv2.VideoCapture, index: int) -> None:
        """Stop auto-exposure from lowering the camera's frame rate.

        Left to itself a UVC webcam in a dim room lengthens its exposure past
        the frame interval to gather more light, dropping delivery to a
        measured 7.5 fps while still reporting 30. That, not rendering, was
        what made the interface feel laggy.

        There are two ways to stop it, and the first is much the better one:

        1. ``exposure_dynamic_framerate`` tells the sensor it may not trade
           frame rate for light. Auto-exposure stays on, so the picture is
           still correctly exposed for whatever the room is doing. V4L2 only,
           and OpenCV does not expose it, so it goes through v4l2-ctl.
        2. Failing that, pin the exposure manually to just inside one frame.
           This works everywhere but fixes the brightness, so the image gets
           dark in a dim room.

        Set ``limit_exposure = False`` in config.ini to hand exposure back to
        the camera, at the cost of frame rate in low light.
        """
        target = max(1, self.config.target_fps)

        try:
            done = subprocess.run(
                ["v4l2-ctl", "-d", f"/dev/video{index}",
                 "--set-ctrl=exposure_dynamic_framerate=0"],
                capture_output=True, timeout=2.0,
            ).returncode == 0
        except (OSError, subprocess.SubprocessError):
            done = False

        if done:
            self.logger.info("Locked frame rate via exposure_dynamic_framerate")
            return

        # V4L2 exposure is in units of 100 microseconds; mode 1 is manual.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, int(10_000 / target * 0.9))
        self.logger.info("Pinned manual exposure for %d fps", target)

    @property
    def available(self) -> bool:
        """True when a camera is delivering frames."""
        return self._cap is not None

    @property
    def tracking_available(self) -> bool:
        """True when hand tracking is active."""
        return self._landmarker is not None

    # ----------------------------------------------------------- threading

    def start(self) -> None:
        """Begin background capture."""
        if self._cap is None or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop capture and release hardware."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass

    def _loop(self) -> None:
        """Capture, mirror and track until stopped."""
        import mediapipe as mp

        frame_times: list[float] = []
        start = time.perf_counter()
        last_frame = start
        last_stamp = -1

        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue

            began = time.perf_counter()

            if self.config.flip_horizontal:
                frame = cv2.flip(frame, 1)

            left, right = HandData(), HandData()
            if self._landmarker is not None:
                # Detect on a downscaled copy. Landmarks come back normalised
                # to [0, 1], so the coordinates are unchanged while the model
                # sees a quarter of the pixels.
                if frame.shape[1] > DETECT_WIDTH:
                    scale = DETECT_WIDTH / frame.shape[1]
                    small = cv2.resize(frame, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
                else:
                    small = frame
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                # Timestamps must increase strictly, or MediaPipe raises.
                timestamp = max(last_stamp + 1, int((began - start) * 1000))
                last_stamp = timestamp
                try:
                    result = self._landmarker.detect_for_video(image, timestamp)
                    left, right = self._parse(result)
                except Exception as exc:
                    self.logger.debug("Detection error: %s", exc)

            sample_dt = began - last_frame

            with self._lock:
                self.frame = frame
                self.frame_size = (frame.shape[1], frame.shape[0])
                self.left = self._blend(self.left, left, self._left_filter, sample_dt)
                self.right = self._blend(self.right, right, self._right_filter, sample_dt)
                # Consumers use this to tell a fresh sample from a repeat, so
                # they can time motion against the camera rather than against
                # their own, much faster, render loop.
                self.sample_id += 1
                self.sample_dt = sample_dt
                self._latency_ms = (time.perf_counter() - began) * 1000.0

            # Measure the whole iteration, blocking read included. Timing
            # only the processing reported the rate the CPU could sustain,
            # not the rate the camera actually delivers, which hid the fact
            # that the capture was the bottleneck all along.
            now = time.perf_counter()
            frame_times.append(now - last_frame)
            last_frame = now
            if len(frame_times) > 30:
                frame_times.pop(0)
            total = sum(frame_times)
            self._capture_fps = len(frame_times) / total if total > 0 else 0.0

    def _parse(self, result) -> tuple[HandData, HandData]:
        """Convert a MediaPipe result into left/right HandData."""
        left, right = HandData(), HandData()
        landmarks = getattr(result, "hand_landmarks", None) or []
        handedness = getattr(result, "handedness", None) or []

        for points, hand in zip(landmarks, handedness):
            data = HandData()
            data.points = np.array(
                [[lm.x, lm.y] for lm in points], dtype=np.float32
            )
            label = hand[0].category_name if hand else "Right"
            score = float(hand[0].score) if hand else 0.0
            data.confidence = score
            data.tracked = True
            data.compute_features()

            # The frame is mirrored, so MediaPipe's label refers to the
            # image side; flip it to match the user's real hand.
            if self.config.flip_horizontal:
                label = "Right" if label == "Left" else "Left"
            data.handedness = label

            if label == "Left":
                left = data
            else:
                right = data

        return left, right

    def _blend(
        self,
        previous: HandData,
        incoming: HandData,
        filt: OneEuro,
        dt: float,
    ) -> HandData:
        """Smooth landmarks with a speed-adaptive filter."""
        if not incoming.tracked:
            filt.reset()
            faded = HandData()
            faded.points = previous.points
            faded.handedness = previous.handedness
            return faded

        smoothed = filt(incoming.points, dt).copy()
        incoming.velocity = smoothed - previous.points
        incoming.points = smoothed
        incoming.compute_features()
        return incoming

    # ------------------------------------------------------------- readers

    def snapshot(self) -> tuple[Optional[np.ndarray], HandData, HandData]:
        """Return the latest frame and both hands atomically.

        The frame is handed out by reference, not copied: the capture thread
        publishes a freshly allocated array each iteration and never writes
        into one it has already published, and every consumer treats it as
        read-only. Copying it was a 2.7 MB memcpy per frame for nothing.
        """
        with self._lock:
            return self.frame, self.left, self.right

    @property
    def capture_fps(self) -> float:
        """Frames per second achieved by the capture thread."""
        return self._capture_fps

    @property
    def latency_ms(self) -> float:
        """Milliseconds spent capturing and tracking the last frame."""
        return self._latency_ms


def _self_check() -> None:
    """Verify feature extraction against synthetic landmark sets."""
    hand = HandData()
    # Flat open hand: fingers extend downward from a wrist at the origin.
    points = np.zeros((21, 2), np.float32)
    points[WRIST] = (0.5, 0.9)
    points[PALM] = (0.5, 0.7)
    for i, (tip, pip, mcp) in enumerate(zip(FINGER_TIPS, FINGER_PIPS, FINGER_MCPS)):
        x = 0.4 + i * 0.05
        points[mcp] = (x, 0.72)
        points[pip] = (x, 0.60)
        points[tip] = (x, 0.40)
    hand.points = points
    hand.compute_features()
    assert hand.openness == 1.0, f"open hand should read fully open, got {hand.openness}"
    assert all(hand.extended), hand.extended

    # Closed fist: tips curl back toward the wrist, inside the joints.
    fist = HandData()
    closed = points.copy()
    for tip, pip in zip(FINGER_TIPS, FINGER_PIPS):
        closed[tip] = closed[pip] + np.array([0.0, 0.06], np.float32)
    fist.points = closed
    fist.compute_features()
    assert fist.openness == 0.0, f"fist should read closed, got {fist.openness}"

    # Pinch: thumb and index tips together, scaled by palm span.
    pinched = HandData()
    touching = points.copy()
    touching[THUMB_TIP] = (0.5, 0.5)
    touching[INDEX_TIP] = (0.503, 0.5)
    pinched.points = touching
    pinched.compute_features()
    assert pinched.pinch < 0.1, f"pinch not detected: {pinched.pinch}"
    assert hand.pinch > pinched.pinch, "open hand must exceed pinched hand"

    # The adaptive filter must beat a fixed one at both ends: steadier on a
    # resting hand, and closer behind a fast one.
    rng = np.random.default_rng(4)
    dt = 1.0 / 20.0

    still = np.full((21, 2), 0.5, np.float32)
    euro = OneEuro()
    fixed = still.copy()
    euro_out = fixed.copy()
    euro_err = fixed_err = 0.0
    for _ in range(60):
        noisy = still + rng.normal(0.0, 0.004, still.shape).astype(np.float32)
        euro_out = euro(noisy, dt)
        fixed = 0.45 * noisy + 0.55 * fixed
        euro_err += float(np.abs(euro_out - still).mean())
        fixed_err += float(np.abs(fixed - still).mean())
    assert euro_err < fixed_err, "adaptive filter should be steadier at rest"

    # A fast sweep: the adaptive filter must lag less than the fixed one.
    euro = OneEuro()
    moving = np.full((21, 2), 0.2, np.float32)
    euro(moving, dt)
    fixed = moving.copy()
    for step in range(1, 12):
        truth = moving + np.float32(step * 0.06)
        euro_out = euro(truth, dt)
        fixed = 0.45 * truth + 0.55 * fixed
    euro_lag = float(np.abs(euro_out - truth).mean())
    fixed_lag = float(np.abs(fixed - truth).mean())
    assert euro_lag < fixed_lag, \
        f"adaptive filter should keep up better: {euro_lag:.4f} vs {fixed_lag:.4f}"

    # Losing the hand must clear history so the next sample is taken as-is.
    euro.reset()
    fresh = np.full((21, 2), 0.9, np.float32)
    assert np.allclose(euro(fresh, dt), fresh), "reset must not carry lag over"
    # A zero or negative timestep must not divide by zero.
    euro(fresh, 0.0)

    print("camera self-check passed")


if __name__ == "__main__":
    _self_check()
