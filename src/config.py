"""Typed configuration backed by config.ini."""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class CameraConfig:
    """Capture device and tracking settings."""

    device_index: int = 0
    resolution_width: int = 1280
    resolution_height: int = 720
    target_fps: int = 30      # Most webcams cap here; exposure is sized from it.
    flip_horizontal: bool = True
    smoothing: float = 0.45
    responsiveness: float = 3.0
    limit_exposure: bool = True


@dataclass
class AudioConfig:
    """Mixer and synthesis settings."""

    sample_rate: int = 44100
    latency_ms: float = 12.0
    master_volume: float = 0.75
    default_pack: str = "acoustic"
    default_guitar: str = "acoustic"


@dataclass
class UIConfig:
    """Presentation and interaction settings."""

    render_width: int = 1280
    render_height: int = 720
    fullscreen: bool = False
    glow_intensity: float = 1.0
    particle_count: int = 450
    string_spacing: float = 34.0   # At 720p; scaled by frame height.
    ui_scale: float = 1.0
    default_theme: str = "cyber_blue"
    gesture_sensitivity: float = 0.7
    camera_opacity: float = 0.55


@dataclass
class DebugConfig:
    """Diagnostics and logging settings."""

    show_fps: bool = True
    show_landmarks: bool = False
    log_level: str = "INFO"


class Config:
    """Loads, validates and persists application settings."""

    _SECTIONS = {
        "camera": "camera",
        "audio": "audio",
        "ui": "ui",
        "debug": "debug",
    }

    def __init__(self, config_path: Optional[Path] = None):
        self.camera = CameraConfig()
        self.audio = AudioConfig()
        self.ui = UIConfig()
        self.debug = DebugConfig()

        self.config_path = config_path or (Path(__file__).parent.parent / "config.ini")
        if self.config_path.exists():
            self.load()
        else:
            self.save()

    def load(self) -> None:
        """Read settings from disk, ignoring unknown or malformed keys."""
        parser = ConfigParser()
        parser.read(self.config_path)

        for section, attr in self._SECTIONS.items():
            if section not in parser:
                continue
            target = getattr(self, attr)
            for field_def in fields(target):
                if field_def.name not in parser[section]:
                    continue
                raw = parser[section][field_def.name]
                try:
                    setattr(target, field_def.name, _coerce(raw, field_def.type))
                except (ValueError, AttributeError):
                    continue

    def save(self) -> None:
        """Write current settings to disk."""
        parser = ConfigParser()
        for section, attr in self._SECTIONS.items():
            target = getattr(self, attr)
            parser[section] = {
                f.name: str(getattr(target, f.name)) for f in fields(target)
            }

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as handle:
            parser.write(handle)


def _coerce(raw: str, annotation: Any) -> Any:
    """Convert a config string to the annotated field type."""
    name = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "str")
    if name == "bool":
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if name == "int":
        return int(float(raw))
    if name == "float":
        return float(raw)
    return raw.strip()


def _self_check() -> None:
    """Verify round-tripping and type coercion."""
    import tempfile

    assert _coerce("True", "bool") is True
    assert _coerce("off", "bool") is False
    assert _coerce("60", "int") == 60
    assert _coerce("1.5", "float") == 1.5
    assert _coerce(" dark ", "str") == "dark"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.ini"
        first = Config(path)
        assert path.exists(), "missing config must be created"

        first.ui.default_theme = "synthwave"
        first.camera.target_fps = 30
        first.audio.master_volume = 0.5
        first.save()

        second = Config(path)
        assert second.ui.default_theme == "synthwave"
        assert second.camera.target_fps == 30
        assert second.audio.master_volume == 0.5
        assert isinstance(second.camera.flip_horizontal, bool)

        # A corrupt value must fall back to the default, not crash.
        path.write_text(path.read_text().replace("target_fps = 30", "target_fps = abc"))
        third = Config(path)
        assert third.camera.target_fps == CameraConfig.target_fps

    print("config self-check passed")


if __name__ == "__main__":
    _self_check()
