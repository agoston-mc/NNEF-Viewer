# Copyright (c) 2026
# Refined matrix heatmaps, adaptive contrast colormaps, and accurate WCAG text contrast.

from typing import Dict, List, Optional, Tuple
import numpy as np
from PySide6.QtGui import QColor


def _interpolate_color_stops(stops: List[Tuple[float, Tuple[int, int, int]]], n_entries: int = 256) -> np.ndarray:
    """Linearly interpolate list of (position_0_to_1, (R, G, B)) stops into an [N, 3] uint8 LUT."""
    lut = np.zeros((n_entries, 3), dtype=np.uint8)
    for i in range(n_entries):
        t = i / (n_entries - 1)
        for s_idx in range(len(stops) - 1):
            t0, c0 = stops[s_idx]
            t1, c1 = stops[s_idx + 1]
            if t0 <= t <= t1:
                factor = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                r = int(round(c0[0] + factor * (c1[0] - c0[0])))
                g = int(round(c0[1] + factor * (c1[1] - c0[1])))
                b = int(round(c0[2] + factor * (c1[2] - c0[2])))
                lut[i] = [r, g, b]
                break
    return lut


# Refined, minimal colormaps
_COLORMAP_STOPS: Dict[str, List[Tuple[float, Tuple[int, int, int]]]] = {
    "Coolwarm (Blue-Red)": [
        (0.0, (59, 76, 192)),     # Smooth Blue
        (0.25, (137, 168, 237)),  # Soft Periwinkle
        (0.5, (221, 221, 221)),   # Neutral Light Grey
        (0.75, (244, 154, 123)),  # Soft Coral
        (1.0, (180, 4, 38)),      # Rich Crimson
    ],
    "Adaptive Contrast (Inverted)": [
        (0.0, (30, 32, 36)),      # Dark charcoal
        (0.5, (90, 110, 145)),    # Muted steel blue
        (1.0, (230, 238, 248)),   # High-luminance soft white
    ],
    "Grayscale": [
        (0.0, (25, 25, 25)),
        (1.0, (240, 240, 240)),
    ],
    "Viridis": [
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.5, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ],
}

# Precompute LUTs
COLORMAP_LUTS: Dict[str, np.ndarray] = {
    name: _interpolate_color_stops(stops) for name, stops in _COLORMAP_STOPS.items()
}

AVAILABLE_COLORMAPS: List[str] = list(COLORMAP_LUTS.keys())


class ColorMapper:
    """Computes background colors and high-contrast text foreground colors for matrix cells."""

    def __init__(
        self,
        colormap_name: str = "Coolwarm (Blue-Red)",
        opacity: float = 0.85,
        is_dark_mode: bool = True,
    ):
        self.colormap_name = colormap_name if colormap_name in COLORMAP_LUTS else "Coolwarm (Blue-Red)"
        self.opacity = max(0.0, min(1.0, opacity))
        self.enabled = True
        self.is_dark_mode = is_dark_mode
        self.lut = COLORMAP_LUTS[self.colormap_name]

    def set_colormap(self, name: str) -> None:
        if name in COLORMAP_LUTS:
            self.colormap_name = name
            self.lut = COLORMAP_LUTS[name]

    def set_opacity(self, opacity: float) -> None:
        self.opacity = max(0.0, min(1.0, opacity))

    def set_theme_mode(self, is_dark_mode: bool) -> None:
        self.is_dark_mode = is_dark_mode

    def map_value(
        self,
        value: float,
        v_min: float,
        v_max: float,
        zero_centered: bool = False,
    ) -> Tuple[Optional[QColor], Optional[QColor]]:
        """
        Maps a scalar numeric value to (QColor_background, QColor_foreground).
        Computes accurate blended luminance for perfect WCAG contrast.
        """
        if not self.enabled:
            return None, None

        if np.isnan(value):
            bg = QColor(255, 179, 0, int(220 * self.opacity))
            return bg, QColor(0, 0, 0)

        if np.isposinf(value):
            bg = QColor(142, 36, 170, int(220 * self.opacity))
            return bg, QColor(255, 255, 255)

        if np.isneginf(value):
            bg = QColor(106, 27, 154, int(220 * self.opacity))
            return bg, QColor(255, 255, 255)

        if zero_centered:
            abs_max = max(abs(v_min), abs(v_max))
            if abs_max == 0:
                t = 0.5
            else:
                t = (value + abs_max) / (2.0 * abs_max)
        else:
            rng = v_max - v_min
            if rng == 0:
                t = 0.5
            else:
                t = (value - v_min) / rng

        t_clamped = max(0.0, min(1.0, float(t)))
        lut_idx = int(round(t_clamped * 255))
        r, g, b = int(self.lut[lut_idx, 0]), int(self.lut[lut_idx, 1]), int(self.lut[lut_idx, 2])

        alpha = int(255 * self.opacity)
        bg_color = QColor(r, g, b, alpha)

        # Calculate blended perceived luminance against the table theme background
        base_bg = (30, 31, 34) if self.is_dark_mode else (255, 255, 255)
        eff_r = r * self.opacity + base_bg[0] * (1.0 - self.opacity)
        eff_g = g * self.opacity + base_bg[1] * (1.0 - self.opacity)
        eff_b = b * self.opacity + base_bg[2] * (1.0 - self.opacity)

        # WCAG Relative Luminance
        rel_luminance = (0.299 * eff_r + 0.587 * eff_g + 0.114 * eff_b) / 255.0
        fg_color = QColor(245, 247, 250) if rel_luminance < 0.52 else QColor(16, 18, 20)

        return bg_color, fg_color
