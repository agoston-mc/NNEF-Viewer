# Copyright (c) 2026
# JetBrains-style matrix heatmaps, colormaps, normalization, and automatic text contrast.

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


# JetBrains-inspired Colormaps (Diverging & Sequential)
_COLORMAP_STOPS: Dict[str, List[Tuple[float, Tuple[int, int, int]]]] = {
    "Coolwarm (Blue-Red)": [
        (0.0, (59, 76, 192)),     # Blue
        (0.25, (137, 168, 237)),  # Light Blue
        (0.5, (221, 221, 221)),   # Neutral Light Grey
        (0.75, (244, 154, 123)),  # Light Coral
        (1.0, (180, 4, 38)),      # Deep Red
    ],
    "Seismic": [
        (0.0, (0, 0, 180)),
        (0.5, (255, 255, 255)),
        (1.0, (180, 0, 0)),
    ],
    "Viridis": [
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.5, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.0, (253, 231, 37)),
    ],
    "Plasma": [
        (0.0, (13, 8, 135)),
        (0.25, (126, 3, 168)),
        (0.5, (204, 71, 120)),
        (0.75, (248, 149, 64)),
        (1.0, (240, 249, 33)),
    ],
    "Magma": [
        (0.0, (0, 0, 4)),
        (0.25, (81, 18, 124)),
        (0.5, (182, 54, 121)),
        (0.75, (251, 136, 97)),
        (1.0, (252, 253, 191)),
    ],
    "Jet": [
        (0.0, (0, 0, 143)),
        (0.15, (0, 0, 255)),
        (0.4, (0, 255, 255)),
        (0.65, (255, 255, 0)),
        (0.85, (255, 0, 0)),
        (1.0, (128, 0, 0)),
    ],
    "Blues": [
        (0.0, (247, 251, 255)),
        (0.5, (107, 174, 214)),
        (1.0, (8, 48, 107)),
    ],
    "Reds": [
        (0.0, (255, 245, 240)),
        (0.5, (251, 106, 74)),
        (1.0, (103, 0, 13)),
    ],
    "Grayscale": [
        (0.0, (30, 30, 30)),
        (1.0, (240, 240, 240)),
    ],
    "Diff Highlight": [
        (0.0, (46, 125, 50)),     # Green (Match)
        (0.1, (245, 124, 0)),    # Orange (Minor Diff)
        (1.0, (198, 40, 40)),     # Red (Major Diff)
    ],
}

# Precompute LUTs
COLORMAP_LUTS: Dict[str, np.ndarray] = {
    name: _interpolate_color_stops(stops) for name, stops in _COLORMAP_STOPS.items()
}

AVAILABLE_COLORMAPS: List[str] = list(COLORMAP_LUTS.keys())


class ColorMapper:
    """Computes background colors and high-contrast text foreground colors for matrix cells."""

    def __init__(self, colormap_name: str = "Coolwarm (Blue-Red)", opacity: float = 0.85):
        self.colormap_name = colormap_name if colormap_name in COLORMAP_LUTS else "Coolwarm (Blue-Red)"
        self.opacity = max(0.0, min(1.0, opacity))
        self.enabled = True
        self.lut = COLORMAP_LUTS[self.colormap_name]

    def set_colormap(self, name: str) -> None:
        if name in COLORMAP_LUTS:
            self.colormap_name = name
            self.lut = COLORMAP_LUTS[name]

    def set_opacity(self, opacity: float) -> None:
        self.opacity = max(0.0, min(1.0, opacity))

    def map_value(
        self,
        value: float,
        v_min: float,
        v_max: float,
        zero_centered: bool = False,
    ) -> Tuple[Optional[QColor], Optional[QColor]]:
        """
        Maps a scalar numeric value to (QColor_background, QColor_foreground).
        Handles NaNs, Infs, and zero-centered diverging ranges.
        """
        if not self.enabled:
            return None, None

        if np.isnan(value):
            # Amber warning for NaN
            bg = QColor(255, 179, 0, int(200 * self.opacity))
            return bg, QColor(0, 0, 0)

        if np.isposinf(value):
            bg = QColor(156, 39, 176, int(200 * self.opacity))
            return bg, QColor(255, 255, 255)

        if np.isneginf(value):
            bg = QColor(106, 27, 154, int(200 * self.opacity))
            return bg, QColor(255, 255, 255)

        if zero_centered:
            abs_max = max(abs(v_min), abs(v_max))
            if abs_max == 0:
                t = 0.5
            else:
                # Map [-abs_max, +abs_max] to [0.0, 1.0] with 0 at 0.5
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

        # WCAG Relative Luminance calculation for crisp readability
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        fg_color = QColor(255, 255, 255) if luminance < 135 else QColor(15, 15, 15)

        return bg_color, fg_color
