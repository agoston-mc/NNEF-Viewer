# Copyright (c) 2026
# Dynamic Axis Mapping & Dimension Scrubbing Controls for N-D Tensors.

from typing import List, Optional, Sequence, Tuple
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class DimensionSliderWidget(QWidget):
    """
    Dynamic Axis Selector (Row / Col) and interactive slice sliders for all other N-2 dimensions.
    Includes auto-play / slice animation support.
    """

    slicingChanged = Signal(int, int, list)  # (row_axis, col_axis, slice_indices)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._shape: Tuple[int, ...] = ()
        self._row_axis: int = 0
        self._col_axis: int = 1
        self._slice_indices: List[int] = []
        self._dim_controls: List[Tuple[int, QSlider, QSpinBox]] = []

        # Auto-play animation timer
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._playing_axis_idx: Optional[int] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(6, 4, 6, 4)
        self.main_layout.setSpacing(4)

        # Top Bar: Axis Selectors
        axis_bar = QFrame(self)
        axis_layout = QHBoxLayout(axis_bar)
        axis_layout.setContentsMargins(0, 0, 0, 0)
        axis_layout.setSpacing(8)

        axis_layout.addWidget(QLabel("<b>Row Axis:</b>"))
        self.row_combo = QComboBox(self)
        self.row_combo.currentIndexChanged.connect(self._on_axis_selection_changed)
        axis_layout.addWidget(self.row_combo)

        axis_layout.addWidget(QLabel("<b>Column Axis:</b>"))
        self.col_combo = QComboBox(self)
        self.col_combo.currentIndexChanged.connect(self._on_axis_selection_changed)
        axis_layout.addWidget(self.col_combo)

        axis_layout.addStretch()
        self.main_layout.addWidget(axis_bar)

        # Sliders Container
        self.sliders_container = QWidget(self)
        self.sliders_layout = QVBoxLayout(self.sliders_container)
        self.sliders_layout.setContentsMargins(0, 0, 0, 0)
        self.sliders_layout.setSpacing(4)
        self.main_layout.addWidget(self.sliders_container)

    def set_shape(self, shape: Sequence[int], preferred_row: int = 0, preferred_col: int = 1) -> None:
        """Update tensor shape and regenerate dimension controls."""
        self._play_timer.stop()
        self._shape = tuple(int(x) for x in shape)
        ndim = len(self._shape)

        self.row_combo.blockSignals(True)
        self.col_combo.blockSignals(True)
        self.row_combo.clear()
        self.col_combo.clear()

        if ndim == 0:
            self.row_combo.addItem("Scalar [0D]")
            self.col_combo.addItem("Scalar [0D]")
            self._row_axis = 0
            self._col_axis = 0
        elif ndim == 1:
            self.row_combo.addItem(f"Axis 0 (Len: {self._shape[0]})")
            self.col_combo.addItem("None (1D Vector)")
            self._row_axis = 0
            self._col_axis = 0
        else:
            for axis in range(ndim):
                label = f"Axis {axis} (Dim: {self._shape[axis]})"
                self.row_combo.addItem(label)
                self.col_combo.addItem(label)

            self._row_axis = min(preferred_row, ndim - 1)
            self._col_axis = min(preferred_col if preferred_col != self._row_axis else (self._row_axis + 1) % ndim, ndim - 1)
            self.row_combo.setCurrentIndex(self._row_axis)
            self.col_combo.setCurrentIndex(self._col_axis)

        self.row_combo.blockSignals(False)
        self.col_combo.blockSignals(False)

        self._rebuild_sliders()

    def _rebuild_sliders(self) -> None:
        # Clear existing slider widgets
        while self.sliders_layout.count():
            item = self.sliders_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self._dim_controls.clear()
        ndim = len(self._shape)
        if ndim <= 2:
            self._slice_indices = []
            self.sliders_container.setVisible(False)
            self._emit_change()
            return

        self.sliders_container.setVisible(True)
        other_axes = [ax for ax in range(ndim) if ax != self._row_axis and ax != self._col_axis]
        self._slice_indices = [0] * len(other_axes)

        for idx_in_others, axis in enumerate(other_axes):
            dim_size = self._shape[axis]
            max_idx = max(0, dim_size - 1)

            row_widget = QFrame(self.sliders_container)
            h_layout = QHBoxLayout(row_widget)
            h_layout.setContentsMargins(0, 2, 0, 2)
            h_layout.setSpacing(6)

            lbl = QLabel(f"<b>Axis {axis}</b> [0..{max_idx}]:", row_widget)
            lbl.setMinimumWidth(110)
            h_layout.addWidget(lbl)

            prev_btn = QPushButton("◀", row_widget)
            prev_btn.setFixedWidth(28)
            h_layout.addWidget(prev_btn)

            slider = QSlider(Qt.Orientation.Horizontal, row_widget)
            slider.setRange(0, max_idx)
            slider.setValue(0)
            h_layout.addWidget(slider)

            next_btn = QPushButton("▶", row_widget)
            next_btn.setFixedWidth(28)
            h_layout.addWidget(next_btn)

            spin = QSpinBox(row_widget)
            spin.setRange(0, max_idx)
            spin.setValue(0)
            spin.setFixedWidth(65)
            h_layout.addWidget(spin)

            play_btn = QPushButton("Play ⏯", row_widget)
            play_btn.setFixedWidth(60)
            h_layout.addWidget(play_btn)

            # Connect controls
            def make_callbacks(s=slider, sp=spin, o_idx=idx_in_others, p_btn=play_btn, m_idx=max_idx):
                def on_val_change(val):
                    s.blockSignals(True)
                    sp.blockSignals(True)
                    s.setValue(val)
                    sp.setValue(val)
                    s.blockSignals(False)
                    sp.blockSignals(False)
                    self._slice_indices[o_idx] = val
                    self._emit_change()

                s.valueChanged.connect(on_val_change)
                sp.valueChanged.connect(on_val_change)

                def on_prev():
                    on_val_change(max(0, s.value() - 1))

                def on_next():
                    on_val_change(min(m_idx, s.value() + 1))

                def on_play():
                    if self._play_timer.isActive() and self._playing_axis_idx == o_idx:
                        self._play_timer.stop()
                        p_btn.setText("Play ⏯")
                        self._playing_axis_idx = None
                    else:
                        self._playing_axis_idx = o_idx
                        p_btn.setText("Stop ⏹")
                        self._play_timer.start(100)

                return on_prev, on_next, on_play

            cb_prev, cb_next, cb_play = make_callbacks()
            prev_btn.clicked.connect(cb_prev)
            next_btn.clicked.connect(cb_next)
            play_btn.clicked.connect(cb_play)

            self._dim_controls.append((axis, slider, spin))
            self.sliders_layout.addWidget(row_widget)

        self._emit_change()

    def _on_play_tick(self) -> None:
        if self._playing_axis_idx is None or self._playing_axis_idx >= len(self._dim_controls):
            self._play_timer.stop()
            return
        _, slider, _ = self._dim_controls[self._playing_axis_idx]
        val = slider.value() + 1
        if val > slider.maximum():
            val = 0
        slider.setValue(val)

    def _on_axis_selection_changed(self) -> None:
        ndim = len(self._shape)
        if ndim <= 1:
            return
        r_idx = self.row_combo.currentIndex()
        c_idx = self.col_combo.currentIndex()
        if r_idx == c_idx:
            # Shift column index to keep them distinct
            c_idx = (r_idx + 1) % ndim
            self.col_combo.blockSignals(True)
            self.col_combo.setCurrentIndex(c_idx)
            self.col_combo.blockSignals(False)

        self._row_axis = r_idx
        self._col_axis = c_idx
        self._rebuild_sliders()

    def _emit_change(self) -> None:
        self.slicingChanged.emit(self._row_axis, self._col_axis, list(self._slice_indices))
