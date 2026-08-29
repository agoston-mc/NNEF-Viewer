# Copyright (c) 2026
# Real-time statistics summary, interactive QPainter distribution histogram, and heatmap controls.

from typing import Any, Dict, Optional, Tuple
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ...core.stats import TensorStats
from ..colormap import AVAILABLE_COLORMAPS, ColorMapper


class DistributionHistogramWidget(QWidget):
    """Zero-latency vector histogram rendered using native QPainter."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.counts = np.zeros(0, dtype=np.int64)
        self.edges = np.zeros(0, dtype=np.float64)
        self.min_val: float = 0.0
        self.max_val: float = 1.0

    def set_data(self, counts: np.ndarray, edges: np.ndarray, min_val: float, max_val: float) -> None:
        self.counts = counts
        self.edges = edges
        self.min_val = min_val
        self.max_val = max_val
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        padding_x = 10
        padding_top = 10
        padding_bottom = 24
        chart_w = w - 2 * padding_x
        chart_h = h - padding_top - padding_bottom

        # Background
        painter.fillRect(self.rect(), QColor(26, 27, 30))

        if len(self.counts) == 0 or np.sum(self.counts) == 0:
            painter.setPen(QColor(130, 130, 130))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No finite distribution data")
            return

        max_count = int(np.max(self.counts))
        if max_count == 0:
            max_count = 1

        n_bins = len(self.counts)
        bar_w = chart_w / n_bins

        # Draw bars with gradient
        for i, count in enumerate(self.counts):
            if count == 0:
                continue
            bar_h = (count / max_count) * chart_h
            x = padding_x + i * bar_w
            y = padding_top + chart_h - bar_h

            gradient = QLinearGradient(x, y, x, y + bar_h)
            gradient.setColorAt(0.0, QColor(53, 116, 240, 220))
            gradient.setColorAt(1.0, QColor(30, 80, 180, 180))

            painter.setBrush(QBrush(gradient))
            painter.setPen(QPen(QColor(53, 116, 240, 100), 1))
            painter.drawRect(QRectF(x, y, max(1.0, bar_w - 1.0), bar_h))

        # Bottom Axis & Labels
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.drawLine(padding_x, padding_top + chart_h, padding_x + chart_w, padding_top + chart_h)

        font = QFont("JetBrains Mono, monospace", 9)
        painter.setFont(font)
        painter.setPen(QColor(160, 160, 160))

        min_str = f"{self.min_val:.4g}"
        max_str = f"{self.max_val:.4g}"
        painter.drawText(padding_x, h - 6, min_str)
        painter.drawText(w - padding_x - 60, h - 6, max_str)


class StatsPanel(QWidget):
    """Stats & Distribution sidebar dock."""

    heatmapSettingsChanged = Signal()

    def __init__(self, color_mapper: ColorMapper, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.color_mapper = color_mapper
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Scroll area for compact sidebar
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget(scroll)
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(10)

        # 1. Distribution Chart Group
        hist_group = QGroupBox("Value Distribution", container)
        h_layout = QVBoxLayout(hist_group)
        self.histogram_widget = DistributionHistogramWidget(hist_group)
        h_layout.addWidget(self.histogram_widget)
        c_layout.addWidget(hist_group)

        # 2. Heatmap Controls Group
        heat_group = QGroupBox("Heatmap & Colors", container)
        heat_layout = QFormLayout(heat_group)
        heat_layout.setSpacing(6)

        self.heatmap_enable_chk = QCheckBox("Enable Background Colors", heat_group)
        self.heatmap_enable_chk.setChecked(True)
        self.heatmap_enable_chk.toggled.connect(self._on_heatmap_toggle)
        heat_layout.addRow(self.heatmap_enable_chk)

        self.colormap_combo = QComboBox(heat_group)
        self.colormap_combo.addItems(AVAILABLE_COLORMAPS)
        self.colormap_combo.setCurrentText(self.color_mapper.colormap_name)
        self.colormap_combo.currentTextChanged.connect(self._on_colormap_changed)
        heat_layout.addRow("Colormap:", self.colormap_combo)

        self.norm_mode_combo = QComboBox(heat_group)
        self.norm_mode_combo.addItems(["Slice Min-Max", "Global Min-Max"])
        self.norm_mode_combo.currentTextChanged.connect(self._on_norm_changed)
        heat_layout.addRow("Normalize:", self.norm_mode_combo)

        self.zero_centered_chk = QCheckBox("Zero-Centered Diverging", heat_group)
        self.zero_centered_chk.setChecked(False)
        self.zero_centered_chk.toggled.connect(self._on_zero_centered_toggle)
        heat_layout.addRow(self.zero_centered_chk)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal, heat_group)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(int(self.color_mapper.opacity * 100))
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        heat_layout.addRow("Opacity:", self.opacity_slider)

        c_layout.addWidget(heat_group)

        # 3. Statistics Table Group
        stats_group = QGroupBox("Tensor Statistics", container)
        self.stats_form = QFormLayout(stats_group)
        self.stats_form.setSpacing(4)
        self.stat_labels: Dict[str, QLabel] = {}

        for key in [
            "Shape", "Rank", "Dtype", "Total Elements", "Memory Size",
            "Min", "Max", "Mean", "Std Dev", "Median", "Variance",
            "Sparsity", "NaN Count", "Inf Count"
        ]:
            lbl = QLabel("-", stats_group)
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.stat_labels[key] = lbl
            self.stats_form.addRow(f"<b>{key}:</b>", lbl)

        c_layout.addWidget(stats_group)

        # Progress indicator for background computations
        self.calc_progress = QProgressBar(container)
        self.calc_progress.setRange(0, 0)  # Indeterminate
        self.calc_progress.setVisible(False)
        self.calc_progress.setFixedHeight(12)
        c_layout.addWidget(self.calc_progress)

        c_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

    def show_computation_loading(self, loading: bool) -> None:
        self.calc_progress.setVisible(loading)

    def update_stats(self, stats: TensorStats) -> None:
        stats_dict = stats.to_dict()
        for key, val in stats_dict.items():
            if key in self.stat_labels:
                self.stat_labels[key].setText(str(val))

    def update_histogram(self, counts: np.ndarray, edges: np.ndarray, min_val: float, max_val: float) -> None:
        self.histogram_widget.set_data(counts, edges, min_val, max_val)

    def _on_heatmap_toggle(self, checked: bool) -> None:
        self.color_mapper.enabled = checked
        self.heatmapSettingsChanged.emit()

    def _on_colormap_changed(self, name: str) -> None:
        self.color_mapper.set_colormap(name)
        self.heatmapSettingsChanged.emit()

    def _on_norm_changed(self, mode: str) -> None:
        self.heatmapSettingsChanged.emit()

    def _on_zero_centered_toggle(self, checked: bool) -> None:
        self.heatmapSettingsChanged.emit()

    def _on_opacity_changed(self, val: int) -> None:
        self.color_mapper.set_opacity(val / 100.0)
        self.heatmapSettingsChanged.emit()
