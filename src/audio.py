"""Audio engine: procedural guitar synthesis (Karplus-Strong) + sample playback.

Generates physically-modelled plucked-string tones at startup so the
application produces real guitar sound with no asset files present. If
.wav samples exist under assets/audio/<pack>/ they take precedence.
"""


from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pygame

from config import AudioConfig
from logger import get_logger

# Standard tuning, low->high (MIDI note numbers): E2 A2 D3 G3 B3 E4
STANDARD_TUNING: tuple[int, ...] = (40, 45, 50, 55, 59, 64)

# Root pitch classes for the chord wheel.
ROOT_PITCH_CLASS: Dict[str, int] = {
    "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11,
}

# Interval structure per modifier, in semitones from the root.
CHORD_INTERVALS: Dict[str, tuple[int, ...]] = {
    "Major": (0, 4, 7),
    "Minor": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "Maj7": (0, 4, 7, 11),
    "Sus2": (0, 2, 7),
    "Sus4": (0, 5, 7),
    "Dim": (0, 3, 6),
    "Aug": (0, 4, 8),
    "Power": (0, 7),
    "Barre": (0, 4, 7),
}

NOTE_NAMES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)


@dataclass(frozen=True)
class Guitar:
    """An instrument: how many strings, tuned to what, and how it sounds."""

    name: str
    label: str
    tuning: tuple[int, ...]     # Open MIDI pitch of each string, low to high.
    pack: str                   # Timbre this instrument defaults to.
    thickness: tuple[int, ...]  # Drawn gauge of each string, low to high.
    spacing: float = 1.0        # String spacing multiplier.

    @property
    def strings(self) -> int:
        """Number of strings."""
        return len(self.tuning)


#: Selectable instruments. Bass and ukulele change the string count and the
#: octave, not just the timbre, so they play as genuinely different
#: instruments rather than as the same guitar through another amp.
GUITARS: Dict[str, Guitar] = {
    "acoustic": Guitar(
        "acoustic", "ACOUSTIC", STANDARD_TUNING, "acoustic", (3, 3, 2, 2, 1, 1),
    ),
    "electric": Guitar(
        "electric", "ELECTRIC", STANDARD_TUNING, "electric", (2, 2, 2, 1, 1, 1),
    ),
    "bass": Guitar(
        # E1 A1 D2 G2: four heavy strings an octave below the guitar.
        "bass", "BASS", (28, 33, 38, 43), "clean", (5, 4, 4, 3), spacing=1.35,
    ),
    "ukulele": Guitar(
        # Linear low-G tuning (G3 C4 E4 A4) so the strings still run low to
        # high, which is what the strum direction assumes.
        "ukulele", "UKULELE", (55, 60, 64, 69), "fingerstyle", (2, 2, 1, 1),
        spacing=0.85,
    ),
}

DEFAULT_GUITAR = "acoustic"


@dataclass(frozen=True)
class Timbre:
    """Synthesis parameters defining a guitar tone."""

    name: str
    decay: float            # Energy retained per wavetable pass (sustain).
    brightness: float       # Lowpass on the excitation; 1.0 = bright.
    drive: float            # tanh saturation amount; 0 = clean.
    attack_noise: float     # Pick-attack transient level.
    duration: float         # Seconds of audio to render.
    body: float             # Sympathetic body resonance mix.


TIMBRES: Dict[str, Timbre] = {
    "acoustic":   Timbre("acoustic",   0.9960, 0.65, 0.00, 0.30, 2.6, 0.25),
    "electric":   Timbre("electric",   0.9975, 0.85, 0.12, 0.20, 3.0, 0.10),
    "clean":      Timbre("clean",      0.9968, 0.75, 0.00, 0.18, 2.8, 0.15),
    "distortion": Timbre("distortion", 0.9980, 0.95, 0.85, 0.35, 3.2, 0.05),
    "muted":      Timbre("muted",      0.9820, 0.40, 0.10, 0.45, 0.7, 0.05),
    "fingerstyle": Timbre("fingerstyle", 0.9955, 0.50, 0.00, 0.08, 2.4, 0.30),
}


def midi_to_freq(midi_note: int) -> float:
    """Convert a MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def note_name(midi_note: int) -> str:
    """Human-readable name for a MIDI note, e.g. 'E2'."""
    return f"{NOTE_NAMES[midi_note % 12]}{midi_note // 12 - 1}"


def chord_voicing(
    root: str,
    quality: str,
    tuning: tuple[int, ...] = STANDARD_TUNING,
) -> tuple[int, ...]:
    """Build a playable voicing for a root/quality pair on a given tuning.

    For each open string, the lowest fret within a five-semitone window
    landing on a chord tone is chosen. Strings with no chord tone in reach
    are muted (-1). This generates musically valid voicings for every
    combination of chord, quality and instrument, with no lookup table.

    Args:
        root: Root note letter, e.g. "C" or "F".
        quality: Modifier name, e.g. "Major", "Sus4".
        tuning: Open MIDI pitch of each string, low to high.

    Returns:
        One MIDI note per string, low to high; -1 marks a muted string.
    """
    if root not in ROOT_PITCH_CLASS:
        return (-1,) * len(tuning)

    root_pc = ROOT_PITCH_CLASS[root]
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS["Major"])
    chord_pcs = {(root_pc + i) % 12 for i in intervals}

    voicing: list[int] = []
    for open_pitch in tuning:
        for fret in range(6):
            if (open_pitch + fret) % 12 in chord_pcs:
                voicing.append(open_pitch + fret)
                break
        else:
            voicing.append(-1)

    # Power chords use only the lowest strings, as a guitarist would. On a
    # four-string instrument that is the lowest two.
    if quality == "Power":
        keep = 2 if len(tuning) <= 4 else 3
        voicing = voicing[:keep] + [-1] * (len(tuning) - keep)

    return tuple(voicing)


def _karplus_strong(
    freq: float,
    sample_rate: int,
    timbre: Timbre,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesise one plucked-string note.

    Uses the block-wise Karplus-Strong formulation: each pass over the
    wavetable is a single vectorised numpy operation rather than a
    per-sample Python loop, which keeps startup synthesis fast.

    Args:
        freq: Fundamental frequency in Hz.
        sample_rate: Output sample rate.
        timbre: Synthesis parameters.
        rng: Random generator for the excitation burst.

    Returns:
        Mono float32 waveform normalised to roughly [-1, 1].
    """
    period = max(2, int(round(sample_rate / freq)))
    total = int(sample_rate * timbre.duration)
    passes = total // period + 1

    # Excitation: noise burst, lowpassed to control brightness.
    wavetable = rng.uniform(-1.0, 1.0, period).astype(np.float32)
    smooth = 1.0 - timbre.brightness
    if smooth > 0.0:
        kernel = max(1, int(smooth * period * 0.25))
        if kernel > 1:
            wavetable = np.convolve(
                wavetable, np.ones(kernel, dtype=np.float32) / kernel, mode="same"
            )

    blocks = np.empty((passes, period), dtype=np.float32)
    current = wavetable
    for i in range(passes):
        blocks[i] = current
        # Averaging adjacent samples is the string's lowpass loss filter.
        current = timbre.decay * 0.5 * (current + np.roll(current, -1))

    signal = blocks.reshape(-1)[:total]

    # Pick attack: a short bright transient at note onset.
    if timbre.attack_noise > 0.0:
        attack_len = min(int(sample_rate * 0.006), total)
        envelope = np.linspace(1.0, 0.0, attack_len, dtype=np.float32) ** 2
        burst = rng.uniform(-1.0, 1.0, attack_len).astype(np.float32)
        signal[:attack_len] += burst * envelope * timbre.attack_noise

    # Body resonance: a quiet octave-down copy fills out the low end.
    if timbre.body > 0.0:
        octave = signal[::2]
        padded = np.zeros_like(signal)
        length = min(len(octave) * 2, total)
        padded[:length] = np.repeat(octave, 2)[:length]
        signal += padded * timbre.body * 0.5

    if timbre.drive > 0.0:
        signal = np.tanh(signal * (1.0 + timbre.drive * 8.0))

    # Fade the tail so notes never end on a click.
    fade = min(int(sample_rate * 0.05), total)
    signal[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)

    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / peak).astype(np.float32)


class AudioEngine:
    """Polyphonic guitar playback with procedural synthesis."""

    #: MIDI range synthesised at startup. Reaches down to E1 for the bass
    #: and up to the 12th fret of a guitar's high E; the extra low octave
    #: costs 0.04 s of startup render.
    MIN_NOTE = 28
    MAX_NOTE = 76

    def __init__(self, config: AudioConfig, asset_dir: Path):
        self.config = config
        self.asset_dir = asset_dir
        self.logger = get_logger("AudioEngine")

        buffer_size = 1 << max(6, int(np.log2(
            max(64, config.latency_ms * config.sample_rate / 1000.0)
        )))
        pygame.mixer.init(
            frequency=config.sample_rate,
            size=-16,
            channels=2,
            buffer=buffer_size,
        )
        pygame.mixer.set_num_channels(32)

        self._cache: Dict[tuple[str, int], pygame.mixer.Sound] = {}
        self._samples: Dict[str, Dict[int, pygame.mixer.Sound]] = {}
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(0xA1B2C3)

        self.current_pack = config.default_pack if config.default_pack in TIMBRES else "acoustic"

        self._load_samples()
        self._prerender(self.current_pack)
        self.logger.info(
            "Audio ready: pack=%s buffer=%d cached=%d",
            self.current_pack, buffer_size, len(self._cache),
        )

    def _load_samples(self) -> None:
        """Load any .wav files shipped under assets/audio/<pack>/."""
        audio_dir = self.asset_dir / "audio"
        if not audio_dir.is_dir():
            return

        for pack_dir in sorted(audio_dir.iterdir()):
            if not pack_dir.is_dir():
                continue
            found: Dict[int, pygame.mixer.Sound] = {}
            for wav in pack_dir.glob("*.wav"):
                stem = wav.stem.split("_")[-1]
                if stem.isdigit():
                    try:
                        found[int(stem)] = pygame.mixer.Sound(str(wav))
                    except pygame.error as exc:
                        self.logger.warning("Could not load %s: %s", wav.name, exc)
            if found:
                self._samples[pack_dir.name] = found
                self.logger.info("Loaded %d samples for pack '%s'", len(found), pack_dir.name)

    def _prerender(self, pack: str) -> None:
        """Synthesise and cache every note for a pack."""
        timbre = TIMBRES[pack]
        for note in range(self.MIN_NOTE, self.MAX_NOTE + 1):
            key = (pack, note)
            if key in self._cache:
                continue
            mono = _karplus_strong(
                midi_to_freq(note), self.config.sample_rate, timbre, self._rng
            )
            # Pan higher strings slightly right for a natural stereo image.
            pan = (note - self.MIN_NOTE) / (self.MAX_NOTE - self.MIN_NOTE) - 0.5
            left = mono * (1.0 - pan * 0.35)
            right = mono * (1.0 + pan * 0.35)
            stereo = np.clip(np.stack([left, right], axis=1), -1.0, 1.0)
            self._cache[key] = pygame.mixer.Sound(
                buffer=(stereo * 32767).astype(np.int16).tobytes()
            )

    def set_pack(self, pack: str) -> bool:
        """Switch tone packs, synthesising the new one on first use."""
        if pack not in TIMBRES:
            return False
        with self._lock:
            self.current_pack = pack
            self._prerender(pack)
        self.logger.info("Tone pack: %s", pack)
        return True

    def play_note(self, midi_note: int, velocity: float = 1.0) -> None:
        """Play a single note.

        Args:
            midi_note: MIDI note number; values outside the synthesised
                range are clamped.
            velocity: Strike strength in [0, 1], scaling output level.
        """
        if midi_note < 0:
            return

        note = int(np.clip(midi_note, self.MIN_NOTE, self.MAX_NOTE))
        pack = self.current_pack

        sound: Optional[pygame.mixer.Sound] = None
        pack_samples = self._samples.get(pack)
        if pack_samples:
            sound = pack_samples.get(note % len(pack_samples))
        if sound is None:
            with self._lock:
                sound = self._cache.get((pack, note))
        if sound is None:
            return

        volume = float(np.clip(velocity, 0.05, 1.0) * self.config.master_volume)
        channel = pygame.mixer.find_channel(True)
        if channel is not None:
            channel.set_volume(volume)
            channel.play(sound)

    def play_chord(
        self,
        voicing: Sequence[int],
        velocity: float = 1.0,
        strum_delay_ms: float = 0.0,
    ) -> None:
        """Play every non-muted string of a voicing.

        Args:
            voicing: Six MIDI notes; -1 entries are skipped.
            velocity: Strike strength in [0, 1].
            strum_delay_ms: Unused spacing hint retained for callers that
                schedule their own per-string timing.
        """
        for note in voicing:
            if note >= 0:
                self.play_note(note, velocity)

    def stop_all(self) -> None:
        """Silence all channels immediately."""
        pygame.mixer.stop()

    def set_master_volume(self, volume: float) -> None:
        """Set the master output level, clamped to [0, 1]."""
        self.config.master_volume = float(np.clip(volume, 0.0, 1.0))

    def get_master_volume(self) -> float:
        """Return the master output level."""
        return self.config.master_volume

    def available_packs(self) -> list[str]:
        """Return the selectable tone pack names."""
        return list(TIMBRES)

    def shutdown(self) -> None:
        """Stop playback and release the mixer."""
        try:
            pygame.mixer.stop()
            pygame.mixer.quit()
        except pygame.error:
            pass


def _self_check() -> None:
    """Verify voicing generation and synthesis without opening an audio device."""
    e_major = chord_voicing("E", "Major")
    assert e_major[0] == 40, f"E major should start on open low E, got {e_major}"
    assert all(n % 12 in {4, 8, 11} for n in e_major if n >= 0), e_major

    a_minor = chord_voicing("A", "Minor")
    assert all(n % 12 in {9, 0, 4} for n in a_minor if n >= 0), a_minor

    power = chord_voicing("G", "Power")
    assert power[3:] == (-1, -1, -1), f"power chord must mute high strings: {power}"

    assert chord_voicing("H", "Major") == (-1,) * 6, "unknown root must mute everything"

    # Voicings must follow the instrument's tuning and string count.
    for guitar in GUITARS.values():
        voiced = chord_voicing("C", "Major", guitar.tuning)
        assert len(voiced) == guitar.strings, f"{guitar.name} voicing width"
        assert any(n >= 0 for n in voiced), f"{guitar.name} C major is silent"
        assert all(n < 0 or n >= min(guitar.tuning) for n in voiced)
    bass = chord_voicing("C", "Major", GUITARS["bass"].tuning)
    assert len(bass) == 4 and max(bass) < 60, f"bass should stay low, got {bass}"
    assert chord_voicing("H", "Major", GUITARS["bass"].tuning) == (-1,) * 4
    # Power chords keep only the lowest strings, scaled to the instrument.
    assert sum(n >= 0 for n in chord_voicing("E", "Power", GUITARS["bass"].tuning)) == 2
    assert sum(n >= 0 for n in chord_voicing("E", "Power", STANDARD_TUNING)) == 3
    # Every instrument's open strings must be inside the synthesised range.
    for guitar in GUITARS.values():
        assert min(guitar.tuning) >= AudioEngine.MIN_NOTE, \
            f"{guitar.name} goes below the synthesised range"

    for quality in CHORD_INTERVALS:
        voicing = chord_voicing("C", quality)
        assert len(voicing) == 6, quality
        assert any(n >= 0 for n in voicing), f"{quality} produced silence"

    assert abs(midi_to_freq(69) - 440.0) < 1e-9
    assert abs(midi_to_freq(40) - 82.41) < 0.01, midi_to_freq(40)
    assert note_name(40) == "E2", note_name(40)
    assert note_name(64) == "E4", note_name(64)

    rng = np.random.default_rng(1)
    wave = _karplus_strong(midi_to_freq(40), 44100, TIMBRES["acoustic"], rng)
    assert wave.dtype == np.float32
    assert len(wave) == int(44100 * TIMBRES["acoustic"].duration)
    assert np.isfinite(wave).all(), "synthesis produced NaN/inf"
    assert 0.99 <= float(np.max(np.abs(wave))) <= 1.0, "output not normalised"
    # A plucked string must decay: the tail is quieter than the onset.
    head = float(np.mean(np.abs(wave[:4410])))
    tail = float(np.mean(np.abs(wave[-4410:])))
    assert tail < head * 0.5, f"note does not decay (head={head:.4f} tail={tail:.4f})"

    muted = _karplus_strong(midi_to_freq(40), 44100, TIMBRES["muted"], rng)
    assert len(muted) < len(wave), "muted tone should be shorter than acoustic"

    print("audio self-check passed")


if __name__ == "__main__":
    _self_check()
