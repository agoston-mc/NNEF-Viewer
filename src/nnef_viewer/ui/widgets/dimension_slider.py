# Copyright (c) 2026
# Dynamic Axis Mapping & Dimension Scrubbing Controls for N-D Tensors (Sleek & Compact).

from typing import List, Optional, Sequence, Tuple
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class DimensionSliderWidget(QWidget):
    """
    Sleek, compact, and collapsible Dimension Scrubbing & Axis Mapping Widget.
    Hugs its contents tightly without wasting vertical space.
    """

    slicingChanged = Signal(int, int, list)  # (row_axis, col_axis, slice_indices)
    collapseToggled = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self._shape: Tuple[int, ...] = ()
        self._row_axis: int = 0
        self._col_axis: int = 1
        self._slice_indices: List[int] = []
        self._axis_values: dict = {}
        self._dim_controls: List[Tuple[int, QSlider, QSpinBox, QPushButton]] = []
        self._is_collapsed = False

        # Auto-play animation timer
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_play_tick)
        self._playing_axis_idx: Optional[int] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(6, 4, 6, 4)
        self.main_layout.setSpacing(4)

        # Top Bar: Integrated Axis & Shape Control Frame
        self.axis_bar = QFrame(self)
        self.axis_bar.setObjectName("axisBar")
        self.axis_bar.setStyleSheet("""
            #axisBar {
                background-color: #25272a;
                border: 1px solid #393b40;
                border-radius: 6px;
                padding: 2px 6px;
            }
        """)
        axis_layout = QHBoxLayout(self.axis_bar)
        axis_layout.setContentsMargins(4, 2, 4, 2)
        axis_layout.setSpacing(8)

        # Shape Badge
        self.shape_badge = QLabel("Shape: -", self.axis_bar)
        self.shape_badge.setStyleSheet("""
            background-color: #1e1f22;
            color: #79a8ff;
            font-family: "JetBrains Mono", monospace;
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            border: 1px solid #3574f0;
        """)
        axis_layout.addWidget(self.shape_badge)

        # Row Axis Selector
        axis_layout.addWidget(QLabel("Row Axis:", self.axis_bar))
        self.row_combo = QComboBox(self.axis_bar)
        self.row_combo.setMinimumWidth(130)
        self.row_combo.currentIndexChanged.connect(self._on_axis_selection_changed)
        axis_layout.addWidget(self.row_combo)

        # Column Axis Selector
        axis_layout.addWidget(QLabel("Column Axis:", self.axis_bar))
        self.col_combo = QComboBox(self.axis_bar)
        self.col_combo.setMinimumWidth(130)
        self.col_combo.currentIndexChanged.connect(self._on_axis_selection_changed)
        axis_layout.addWidget(self.col_combo)

        # Swap Button
        self.swap_btn = QPushButton("Swap", self.axis_bar)
        self.swap_btn.setToolTip("Swap Row and Column axes (Transpose slice)")
        self.swap_btn.setFixedWidth(50)
        self.swap_btn.clicked.connect(self._on_swap_axes)
        axis_layout.addWidget(self.swap_btn)

        axis_layout.addStretch()

        # Collapse / Expand Slicing Panel Button
        self.collapse_btn = QPushButton("Collapse Sliders", self.axis_bar)
        self.collapse_btn.setToolTip("Show / Hide slice sliders panel")
        self.collapse_btn.setCheckable(True)
        self.collapse_btn.setChecked(False)
        self.collapse_btn.clicked.connect(self._toggle_collapse)
        axis_layout.addWidget(self.collapse_btn)

        self.main_layout.addWidget(self.axis_bar)

        # Sliders Container
        self.sliders_container = QFrame(self)
        self.sliders_container.setObjectName("slidersContainer")
        self.sliders_container.setStyleSheet("""
            #slidersContainer {
                background-color: #212326;
                border: 1px solid #33363a;
                border-radius: 6px;
                padding: 4px 6px;
            }
        """)
        self.sliders_layout = QVBoxLayout(self.sliders_container)
        self.sliders_layout.setContentsMargins(4, 4, 4, 4)
        self.sliders_layout.setSpacing(4)
        self.main_layout.addWidget(self.sliders_container)

    def set_shape(self, shape: Sequence[int], preferred_row: Optional[int] = None, preferred_col: Optional[int] = None) -> None:
        """Update tensor shape and regenerate dimension controls, defaulting to last 2 axes (H, W)."""
        self._play_timer.stop()
        self._shape = tuple(int(x) for x in shape)
        ndim = len(self._shape)

        # Format shape badge
        if ndim == 0:
            shape_str = "0D Scalar"
        else:
            shape_str = " x ".join(str(x) for x in self._shape)
        self.shape_badge.setText(f"Shape: [{shape_str}]")

        self.row_combo.blockSignals(True)
        self.col_combo.blockSignals(True)
        self.row_combo.clear()
        self.col_combo.clear()

        if ndim == 0:
            self.row_combo.addItem("Scalar")
            self.col_combo.addItem("Scalar")
            self._row_axis = 0
            self._col_axis = 0
        elif ndim == 1:
            self.row_combo.addItem(f"Axis 0 ({self._shape[0]})")
            self.col_combo.addItem("None (1D Vector)")
            self._row_axis = 0
            self._col_axis = 0
        else:
            if preferred_row is None:
                preferred_row = max(0, ndim - 2)
            if preferred_col is None:
                preferred_col = max(0, ndim - 1)

            for axis in range(ndim):
                label = f"Axis {axis} ({self._shape[axis]})"
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
            self.collapse_btn.setVisible(False)
            self._emit_change()
            return

        self.collapse_btn.setVisible(True)
        self.sliders_container.setVisible(not self._is_collapsed)
        other_axes = [ax for ax in range(ndim) if ax != self._row_axis and ax != self._col_axis]
        self._slice_indices = [0] * len(other_axes)

        for idx_in_others, axis in enumerate(other_axes):
            dim_size = self._shape[axis]
            max_idx = max(0, dim_size - 1)
            # Restore previous coordinate for this axis
            saved_val = max(0, min(self._axis_values.get(axis, 0), max_idx))
            self._slice_indices[idx_in_others] = saved_val
            self._axis_values[axis] = saved_val

            row_widget = QWidget(self.sliders_container)
            h_layout = QHBoxLayout(row_widget)
            h_layout.setContentsMargins(0, 1, 0, 1)
            h_layout.setSpacing(6)

            lbl = QLabel(f"<b>Axis {axis}</b> (dim: {dim_size}):", row_widget)
            lbl.setMinimumWidth(110)
            lbl.setStyleSheet("color: #bcbec4; font-size: 11px;")
            h_layout.addWidget(lbl)

            prev_btn = QPushButton("<", row_widget)
            prev_btn.setFixedWidth(24)
            prev_btn.setFixedHeight(22)
            h_layout.addWidget(prev_btn)

            slider = QSlider(Qt.Orientation.Horizontal, row_widget)
            slider.setRange(0, max_idx)
            slider.setValue(saved_val)
            slider.setFixedHeight(20)
            h_layout.addWidget(slider)

            next_btn = QPushButton(">", row_widget)
            next_btn.setFixedWidth(24)
            next_btn.setFixedHeight(22)
            h_layout.addWidget(next_btn)

            spin = QSpinBox(row_widget)
            spin.setRange(0, max_idx)
            spin.setValue(saved_val)
            spin.setFixedWidth(60)
            spin.setFixedHeight(22)
            h_layout.addWidget(spin)

            play_btn = QPushButton("Play", row_widget)
            play_btn.setFixedWidth(46)
            play_btn.setFixedHeight(22)
            h_layout.addWidget(play_btn)

            def make_callbacks(s=slider, sp=spin, o_idx=idx_in_others, p_btn=play_btn, m_idx=max_idx, ax=axis):
                def on_val_change(val):
                    s.blockSignals(True)
                    sp.blockSignals(True)
                    s.setValue(val)
                    sp.setValue(val)
                    s.blockSignals(False)
                    sp.blockSignals(False)
                    self._slice_indices[o_idx] = val
                    self._axis_values[ax] = val
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
                        p_btn.setText("Play")
                        self._playing_axis_idx = None
                    else:
                        self._playing_axis_idx = o_idx
                        p_btn.setText("Stop")
                        self._play_timer.start(100)

                return on_prev, on_next, on_play

            cb_prev, cb_next, cb_play = make_callbacks()
            prev_btn.clicked.connect(cb_prev)
            next_btn.clicked.connect(cb_next)
            play_btn.clicked.connect(cb_play)

            self._dim_controls.append((axis, slider, spin, play_btn))
            self.sliders_layout.addWidget(row_widget)

        self._emit_change()

    def _toggle_collapse(self) -> None:
        self._is_collapsed = not self._is_collapsed
        self.sliders_container.setVisible(not self._is_collapsed and len(self._shape) > 2)
        self.collapse_btn.setText("Expand Sliders" if self._is_collapsed else "Collapse Sliders")
        self.collapseToggled.emit(self._is_collapsed)

    def _on_swap_axes(self) -> None:
        if len(self._shape) < 2:
            return
        r = self.row_combo.currentIndex()
        c = self.col_combo.currentIndex()
        self.row_combo.blockSignals(True)
        self.col_combo.blockSignals(True)
        self.row_combo.setCurrentIndex(c)
        self.col_combo.setCurrentIndex(r)
        self.row_combo.blockSignals(False)
        self.col_combo.blockSignals(False)
        self._row_axis = c
        self._col_axis = r
        self._rebuild_sliders()

    def _on_play_tick(self) -> None:
        if self._playing_axis_idx is None or self._playing_axis_idx >= len(self._dim_controls):
            self._play_timer.stop()
            return
        _, slider, _, _ = self._dim_controls[self._playing_axis_idx]
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
            c_idx = (r_idx + 1) % ndim
            self.col_combo.blockSignals(True)
            self.col_combo.setCurrentIndex(c_idx)
            self.col_combo.blockSignals(False)

        self._row_axis = r_idx
        self._col_axis = c_idx
        self._rebuild_sliders()

    def _emit_change(self) -> None:
        self.slicingChanged.emit(self._row_axis, self._col_axis, list(self._slice_indices))
