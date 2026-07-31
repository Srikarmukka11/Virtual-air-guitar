"""Colour themes.

All colours are stored as BGR triples because every consumer is an OpenCV
drawing call. Storing them in the renderer's native order removes the
channel-swap bugs that come from converting at each call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator


def bgr(r: int, g: int, b: int) -> tuple[int, int, int]:
    """Build a BGR tuple from human-readable RGB components."""
    return (b, g, r)


def scale(colour: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    """Scale a colour's brightness, clamped to valid channel range."""
    return tuple(int(max(0, min(255, c * factor))) for c in colour)


def mix(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    """Blend two colours; ``amount`` of 0 returns the first colour."""
    t = max(0.0, min(1.0, amount))
    return tuple(int(a + (b - a) * t) for a, b in zip(first, second))


@dataclass(frozen=True)
class Theme:
    """A complete colour palette, in BGR."""

    name: str
    label: str
    accent: tuple[int, int, int]
    secondary: tuple[int, int, int]
    highlight: tuple[int, int, int]
    string: tuple[int, int, int]
    string_hot: tuple[int, int, int]
    text: tuple[int, int, int]
    text_dim: tuple[int, int, int]
    panel: tuple[int, int, int]
    backdrop: tuple[int, int, int]
    particle: tuple[int, int, int]
    warn: tuple[int, int, int]
    camera_gain: float = 0.34
    backdrop_mix: float = 0.55


THEMES: Dict[str, Theme] = {
    "dark": Theme(
        name="dark",
        label="DARK",
        accent=bgr(120, 190, 255),
        secondary=bgr(80, 120, 170),
        highlight=bgr(255, 255, 255),
        string=bgr(150, 200, 245),
        string_hot=bgr(255, 255, 255),
        text=bgr(232, 240, 250),
        text_dim=bgr(130, 150, 175),
        panel=bgr(14, 18, 26),
        backdrop=bgr(8, 10, 15),
        particle=bgr(150, 205, 255),
        warn=bgr(255, 140, 90),
    ),
    "cyber_blue": Theme(
        name="cyber_blue",
        label="CYBER BLUE",
        accent=bgr(0, 224, 255),
        secondary=bgr(0, 128, 200),
        highlight=bgr(190, 255, 255),
        string=bgr(90, 230, 255),
        string_hot=bgr(235, 255, 255),
        text=bgr(215, 245, 255),
        text_dim=bgr(95, 150, 180),
        panel=bgr(6, 20, 34),
        backdrop=bgr(3, 9, 18),
        particle=bgr(0, 224, 255),
        warn=bgr(255, 90, 140),
    ),
    "neon_purple": Theme(
        name="neon_purple",
        label="NEON PURPLE",
        accent=bgr(200, 90, 255),
        secondary=bgr(130, 40, 200),
        highlight=bgr(245, 210, 255),
        string=bgr(225, 130, 255),
        string_hot=bgr(255, 240, 255),
        text=bgr(238, 220, 255),
        text_dim=bgr(150, 110, 180),
        panel=bgr(20, 8, 32),
        backdrop=bgr(12, 4, 20),
        particle=bgr(215, 110, 255),
        warn=bgr(120, 255, 190),
    ),
    "synthwave": Theme(
        name="synthwave",
        label="SYNTHWAVE",
        accent=bgr(255, 70, 150),
        secondary=bgr(120, 60, 220),
        highlight=bgr(255, 215, 130),
        string=bgr(255, 110, 175),
        string_hot=bgr(255, 240, 200),
        text=bgr(255, 225, 240),
        text_dim=bgr(170, 110, 150),
        panel=bgr(26, 10, 40),
        backdrop=bgr(16, 6, 28),
        particle=bgr(255, 120, 190),
        warn=bgr(120, 255, 220),
    ),
    "minimal_white": Theme(
        name="minimal_white",
        label="MINIMAL",
        accent=bgr(20, 110, 220),
        secondary=bgr(120, 150, 190),
        highlight=bgr(0, 0, 0),
        string=bgr(40, 60, 90),
        string_hot=bgr(20, 110, 220),
        text=bgr(18, 22, 30),
        text_dim=bgr(110, 120, 135),
        panel=bgr(242, 244, 248),
        backdrop=bgr(232, 236, 242),
        particle=bgr(20, 110, 220),
        warn=bgr(200, 60, 40),
        camera_gain=0.26,
        backdrop_mix=0.72,
    ),
}

#: Presentation order used by the number-key shortcuts.
THEME_ORDER: tuple[str, ...] = (
    "dark", "cyber_blue", "neon_purple", "synthwave", "minimal_white",
)


class ThemeManager:
    """Holds the active theme and cycles between available palettes."""

    def __init__(self, name: str = "cyber_blue"):
        self._name = name if name in THEMES else "cyber_blue"

    @property
    def theme(self) -> Theme:
        """The currently active theme."""
        return THEMES[self._name]

    def set(self, name: str) -> bool:
        """Activate a theme by name, returning False if unknown."""
        if name not in THEMES:
            return False
        self._name = name
        return True

    def set_index(self, index: int) -> bool:
        """Activate a theme by its position in the presentation order."""
        if 0 <= index < len(THEME_ORDER):
            return self.set(THEME_ORDER[index])
        return False

    def cycle(self) -> Theme:
        """Advance to the next theme and return it."""
        position = THEME_ORDER.index(self._name)
        self.set(THEME_ORDER[(position + 1) % len(THEME_ORDER)])
        return self.theme

    def __iter__(self) -> Iterator[Theme]:
        """Iterate themes in presentation order."""
        return (THEMES[n] for n in THEME_ORDER)


def _self_check() -> None:
    """Verify colour helpers and theme integrity."""
    assert bgr(255, 0, 0) == (0, 0, 255), "RGB red must become BGR (0,0,255)"
    assert scale((100, 100, 100), 2.0) == (200, 200, 200)
    assert scale((200, 200, 200), 4.0) == (255, 255, 255), "must clamp at 255"
    assert scale((100, 100, 100), -1.0) == (0, 0, 0), "must clamp at 0"
    assert mix((0, 0, 0), (100, 200, 50), 0.0) == (0, 0, 0)
    assert mix((0, 0, 0), (100, 200, 50), 1.0) == (100, 200, 50)
    assert mix((0, 0, 0), (100, 200, 100), 0.5) == (50, 100, 50)

    assert set(THEME_ORDER) == set(THEMES), "THEME_ORDER must cover every theme"

    for theme in THEMES.values():
        for field_name, value in vars(theme).items():
            if isinstance(value, tuple):
                assert len(value) == 3, f"{theme.name}.{field_name} malformed"
                assert all(0 <= c <= 255 for c in value), f"{theme.name}.{field_name} out of range"

    manager = ThemeManager("dark")
    assert manager.theme.name == "dark"
    assert manager.set("synthwave") and manager.theme.name == "synthwave"
    assert not manager.set("nonexistent"), "unknown theme must be rejected"
    assert manager.theme.name == "synthwave", "failed set must not change state"
    assert manager.set_index(0) and manager.theme.name == "dark"
    assert not manager.set_index(99)

    # Cycling must visit every theme exactly once before repeating.
    manager.set("dark")
    visited = [manager.cycle().name for _ in THEME_ORDER]
    assert len(set(visited)) == len(THEME_ORDER), visited

    print("themes self-check passed")


if __name__ == "__main__":
    _self_check()
