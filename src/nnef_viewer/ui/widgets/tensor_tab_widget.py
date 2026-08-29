# Copyright (c) 2026
# Embeddable NNEFTensorViewerWidget combining Matrix View, Slicing Sliders, and Stats.

from typing import Optional, Sequence
import numpy as np
from PySide6.QtCore import Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ...core.tensor_model import NNEFTensorDocument
from ...core.stats import compute_tensor_stats, compute_histogram
from ..colormap import ColorMapper
from ..models.tensor_table_model import TensorTableModel
from ..workers.async_workers import StatsComputeWorker
from .dimension_slider import DimensionSliderWidget
from .matrix_view import MatrixTableView
from .stats_panel import StatsPanel


class NNEFTensorViewerWidget(QWidget):
    """
    Modular, embeddable PySide6 widget for viewing and editing a multi-dimensional NNEF tensor.
    Can be instantiated standalone or embedded in any external Qt application.
    """

    dataModified = Signal()

    def __init__(
        self,
        document: NNEFTensorDocument,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.doc = document
        self.color_mapper = ColorMapper()

        self._setup_ui()
        self._connect_signals()

        # Perform initial slice setup and background stats computation
        self.dimension_bar.set_shape(self.doc.shape)
        self._trigger_async_stats()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        # 1. Mini Top Toolbar
        self.toolbar = QToolBar(self)
        self.toolbar.setIconSize(self.toolbar.iconSize())

        # Undo / Redo
        self.undo_btn = QPushButton("↶ Undo", self)
        self.undo_btn.setToolTip("Undo last change (Ctrl+Z)")
        self.undo_btn.clicked.connect(self._on_undo)
        self.toolbar.addWidget(self.undo_btn)

        self.redo_btn = QPushButton("↷ Redo", self)
        self.redo_btn.setToolTip("Redo last undone change (Ctrl+Y)")
        self.redo_btn.clicked.connect(self._on_redo)
        self.toolbar.addWidget(self.redo_btn)

        self.toolbar.addSeparator()

        # Jump to Coordinate
        self.jump_btn = QPushButton("🔍 Jump to (r, c)...", self)
        self.jump_btn.clicked.connect(self._on_jump_to_coord)
        self.toolbar.addWidget(self.jump_btn)

        self.toolbar.addSeparator()

        # Display Format / Precision
        self.toolbar.addWidget(QLabel(" Format: "))
        self.format_combo = QComboBox(self)
        self.format_combo.addItems(["General (%.5g)", "Scientific (%.4e)", "Fixed (%.2f)", "Fixed (%.6f)"])
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        self.toolbar.addWidget(self.format_combo)

        self.toolbar.addSeparator()

        # Toggle Stats Sidebar
        self.toggle_stats_btn = QPushButton("📊 Toggle Stats", self)
        self.toggle_stats_btn.setCheckable(True)
        self.toggle_stats_btn.setChecked(True)
        self.toggle_stats_btn.toggled.connect(self._on_toggle_stats)
        self.toolbar.addWidget(self.toggle_stats_btn)

        main_layout.addWidget(self.toolbar)

        # 2. Dimension Scrubbing Bar
        self.dimension_bar = DimensionSliderWidget(self)
        main_layout.addWidget(self.dimension_bar)

        # 3. Central Splitter: Matrix Table (Left) + Stats Panel (Right)
        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # Table Model & View
        self.table_model = TensorTableModel(self.doc, self.color_mapper, self)
        self.table_view = MatrixTableView(self.splitter)
        self.table_view.setModel(self.table_model)
        self.splitter.addWidget(self.table_view)

        # Stats Panel
        self.stats_panel = StatsPanel(self.color_mapper, self.splitter)
        self.splitter.addWidget(self.stats_panel)

        # Set initial splitter proportions (75% table, 25% stats)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self.splitter)

        self._update_undo_redo_state()

    def _connect_signals(self) -> None:
        self.dimension_bar.slicingChanged.connect(self._on_slicing_changed)
        self.stats_panel.heatmapSettingsChanged.connect(self._on_heatmap_settings_changed)

        self.doc.add_data_change_listener(self._on_doc_data_changed)
        self.doc.add_structure_change_listener(self._on_doc_structure_changed)

    # ---------------- Slicing & Visual Updates ----------------

    def _on_slicing_changed(self, row_axis: int, col_axis: int, slice_indices: list) -> None:
        self.table_model.set_slicing(row_axis, col_axis, slice_indices)

    def _on_heatmap_settings_changed(self) -> None:
        norm_mode = self.stats_panel.norm_mode_combo.currentText()
        zero_centered = self.stats_panel.zero_centered_chk.isChecked()
        self.table_model.set_normalization_mode(norm_mode, zero_centered=zero_centered)

    def _on_format_changed(self, index: int) -> None:
        formats = ["%.5g", "%.4e", "%.2f", "%.6f"]
        fmt = formats[index] if 0 <= index < len(formats) else "%.5g"
        self.table_model.float_format = fmt
        self.table_model.layoutChanged.emit()

    def _on_toggle_stats(self, checked: bool) -> None:
        self.stats_panel.setVisible(checked)

    def _on_jump_to_coord(self) -> None:
        text, ok = QInputDialog.getText(self, "Jump to Cell", "Enter coordinates (row, col) e.g. '10, 25':")
        if not ok or not text:
            return
        parts = [p.strip() for p in text.replace("(", "").replace(")", "").split(",")]
        if len(parts) >= 2:
            try:
                r, c = int(parts[0]), int(parts[1])
                idx = self.table_model.index(r, c)
                if idx.isValid():
                    self.table_view.scrollTo(idx)
                    self.table_view.selectRow(r)
                    self.table_view.setCurrentIndex(idx)
            except ValueError:
                pass

    # ---------------- Undo / Redo ----------------

    def _on_undo(self) -> None:
        self.doc.undo()
        self._update_undo_redo_state()

    def _on_redo(self) -> None:
        self.doc.redo()
        self._update_undo_redo_state()

    def _update_undo_redo_state(self) -> None:
        self.undo_btn.setEnabled(self.doc.can_undo())
        self.redo_btn.setEnabled(self.doc.can_redo())

    # ---------------- Background Concurrency for Stats ----------------

    def _on_doc_data_changed(self) -> None:
        self._update_undo_redo_state()
        self.dataModified.emit()
        self._trigger_async_stats()

    def _on_doc_structure_changed(self) -> None:
        self.dimension_bar.set_shape(self.doc.shape)
        self._update_undo_redo_state()
        self.dataModified.emit()
        self._trigger_async_stats()

    def _trigger_async_stats(self) -> None:
        """Calculate global stats & histogram asynchronously in background worker."""
        # For instant UI response, calculate local slice stats immediately
        if self.doc.size < 50_000:
            stats = compute_tensor_stats(self.doc.data)
            counts, edges, min_v, max_v = compute_histogram(self.doc.data)
            self.stats_panel.update_stats(stats)
            self.stats_panel.update_histogram(counts, edges, min_v, max_v)
            self.table_model.set_global_bounds(stats.min_val, stats.max_val)
        else:
            self.stats_panel.show_computation_loading(True)
            worker = StatsComputeWorker(self.doc.data.copy())
            worker.signals.finished.connect(self._on_async_stats_finished)
            worker.signals.error.connect(lambda err: self.stats_panel.show_computation_loading(False))
            QThreadPool.globalInstance().start(worker)

    @Slot(object)
    def _on_async_stats_finished(self, results) -> None:
        self.stats_panel.show_computation_loading(False)
        stats, (counts, edges, min_v, max_v) = results
        self.stats_panel.update_stats(stats)
        self.stats_panel.update_histogram(counts, edges, min_v, max_v)
        self.table_model.set_global_bounds(stats.min_val, stats.max_val)
