# Copyright (c) 2026
# Synchronized Side-by-Side Tensor Comparison, Difference Matrix, and Mismatch Navigator.

from typing import Optional, Sequence, Tuple
import numpy as np
from PySide6.QtCore import Qt, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...core.comparison import DiffResult, MismatchIterator, compare_tensors
from ...core.tensor_model import NNEFTensorDocument
from ..colormap import ColorMapper
from ..models.tensor_table_model import TensorTableModel
from ..workers.async_workers import DiffComputeWorker
from .dimension_slider import DimensionSliderWidget
from .matrix_view import MatrixTableView


class NNEFDiffViewerWidget(QWidget):
    """
    Synchronized Dual-View Tensor Comparison Tool:
    - Side-by-side synchronized view (Tensor A and Tensor B)
    - Third computed Difference Matrix tab
    - Tolerance threshold configuration (atol, rtol)
    - Comprehensive error metrics (MSE, MAE, Max Diff, Cosine Similarity)
    - Jump to next/prev mismatch navigation with automated slice scrubbing.
    """

    def __init__(
        self,
        doc_a: NNEFTensorDocument,
        doc_b: NNEFTensorDocument,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.doc_a = doc_a
        self.doc_b = doc_b

        # Diff document for computed difference tab
        self.doc_diff = NNEFTensorDocument(
            np.zeros_like(doc_a.data, dtype=np.float32) if doc_a.shape == doc_b.shape else np.empty(0),
            display_name=f"Diff: {doc_a.display_name} - {doc_b.display_name}",
        )

        self.diff_result: Optional[DiffResult] = None
        self.mismatch_iterator = MismatchIterator([])

        self._color_mapper_a = ColorMapper("Coolwarm (Blue-Red)")
        self._color_mapper_b = ColorMapper("Coolwarm (Blue-Red)")
        self._color_mapper_diff = ColorMapper("Coolwarm (Blue-Red)")

        self._setup_ui()
        self._sync_scrollbars()
        self._trigger_recompute_diff()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # 1. Top Container: Tolerance Controls, Mismatch Navigation & Slicing Bar
        self.top_container = QWidget(self)
        self.top_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        top_c_layout = QVBoxLayout(self.top_container)
        top_c_layout.setContentsMargins(0, 0, 0, 0)
        top_c_layout.setSpacing(4)

        top_bar = QFrame(self.top_container)
        top_bar.setObjectName("diffTopBar")
        top_bar.setStyleSheet("""
            #diffTopBar {
                background-color: #2b2d30;
                border: 1px solid #393b40;
                border-radius: 6px;
                padding: 2px 6px;
            }
        """)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(4, 2, 4, 2)
        top_layout.setSpacing(8)

        top_layout.addWidget(QLabel("Absolute Tol (atol):", top_bar))
        self.atol_spin = QDoubleSpinBox(top_bar)
        self.atol_spin.setDecimals(8)
        self.atol_spin.setRange(0.0, 1e6)
        self.atol_spin.setValue(1e-5)
        top_layout.addWidget(self.atol_spin)

        top_layout.addWidget(QLabel("Relative Tol (rtol):", top_bar))
        self.rtol_spin = QDoubleSpinBox(top_bar)
        self.rtol_spin.setDecimals(8)
        self.rtol_spin.setRange(0.0, 1e6)
        self.rtol_spin.setValue(1e-5)
        top_layout.addWidget(self.rtol_spin)

        self.recompute_btn = QPushButton("Recompute Diff", top_bar)
        self.recompute_btn.clicked.connect(self._trigger_recompute_diff)
        top_layout.addWidget(self.recompute_btn)

        sep = QFrame(top_bar)
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        top_layout.addWidget(sep)

        # Mismatch Navigation
        self.prev_mismatch_btn = QPushButton("Prev Mismatch", top_bar)
        self.prev_mismatch_btn.clicked.connect(self._on_prev_mismatch)
        top_layout.addWidget(self.prev_mismatch_btn)

        self.mismatch_status_lbl = QLabel("No diff computed", top_bar)
        self.mismatch_status_lbl.setStyleSheet("font-weight: 600; color: #79a8ff; font-family: monospace;")
        top_layout.addWidget(self.mismatch_status_lbl)

        self.next_mismatch_btn = QPushButton("Next Mismatch", top_bar)
        self.next_mismatch_btn.clicked.connect(self._on_next_mismatch)
        top_layout.addWidget(self.next_mismatch_btn)

        top_layout.addStretch()
        top_c_layout.addWidget(top_bar)

        # Progress bar
        self.progress_bar = QProgressBar(self.top_container)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setVisible(False)
        top_c_layout.addWidget(self.progress_bar)

        # 2. Shared Dimension Slider Bar
        self.dimension_bar = DimensionSliderWidget(self.top_container)
        self.dimension_bar.set_shape(self.doc_a.shape)
        self.dimension_bar.slicingChanged.connect(self._on_slicing_changed)
        top_c_layout.addWidget(self.dimension_bar)

        main_layout.addWidget(self.top_container, stretch=0)

        # 3. Main Display Tabs: Side-by-Side vs Difference Matrix vs Metrics
        self.tabs = QTabWidget(self)

        # Tab 1: Side-by-Side Synchronized View
        side_by_side_widget = QWidget(self.tabs)
        sbs_layout = QHBoxLayout(side_by_side_widget)
        sbs_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal, side_by_side_widget)

        # View A Container
        group_a = QGroupBox(f"Tensor A: {self.doc_a.display_name} ({self.doc_a.dtype})", splitter)
        layout_a = QVBoxLayout(group_a)
        layout_a.setContentsMargins(2, 6, 2, 2)
        self.model_a = TensorTableModel(self.doc_a, self._color_mapper_a, group_a)
        self.view_a = MatrixTableView(group_a)
        self.view_a.setModel(self.model_a)
        layout_a.addWidget(self.view_a)
        splitter.addWidget(group_a)

        # View B Container
        group_b = QGroupBox(f"Tensor B: {self.doc_b.display_name} ({self.doc_b.dtype})", splitter)
        layout_b = QVBoxLayout(group_b)
        layout_b.setContentsMargins(2, 6, 2, 2)
        self.model_b = TensorTableModel(self.doc_b, self._color_mapper_b, group_b)
        self.view_b = MatrixTableView(group_b)
        self.view_b.setModel(self.model_b)
        layout_b.addWidget(self.view_b)
        splitter.addWidget(group_b)

        splitter.setSizes([500, 500])
        sbs_layout.addWidget(splitter)
        self.tabs.addTab(side_by_side_widget, "Synchronized Side-by-Side View")

        # Tab 2: Computed Difference Matrix View
        diff_matrix_widget = QWidget(self.tabs)
        diff_layout = QVBoxLayout(diff_matrix_widget)
        diff_layout.setContentsMargins(0, 0, 0, 0)
        self.model_diff = TensorTableModel(self.doc_diff, self._color_mapper_diff, diff_matrix_widget)
        self.view_diff = MatrixTableView(diff_matrix_widget)
        self.view_diff.setModel(self.model_diff)
        diff_layout.addWidget(self.view_diff)
        self.tabs.addTab(diff_matrix_widget, "Difference Matrix (|A - B|)")

        # Tab 3: Detailed Metrics & Statistical Report
        metrics_widget = QWidget(self.tabs)
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_layout.setContentsMargins(16, 16, 16, 16)

        self.metrics_grid = QGridLayout()
        self.metrics_grid.setSpacing(10)
        self.metrics_labels = {}

        metric_keys = [
            "Shapes Match", "Dtypes Match", "Total Elements",
            "Mismatched Elements", "Exact Matches", "Max Absolute Diff",
            "Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root MSE (RMSE)",
            "Cosine Similarity", "L1 Norm (Sum |diff|)", "L2 Norm (Euclidean)", "Tolerance"
        ]

        for i, key in enumerate(metric_keys):
            row = i // 2
            col = (i % 2) * 2

            k_lbl = QLabel(f"<b>{key}:</b>", metrics_widget)
            v_lbl = QLabel("-", metrics_widget)
            v_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            v_lbl.setStyleSheet("font-size: 13px; font-family: monospace;")

            self.metrics_grid.addWidget(k_lbl, row, col)
            self.metrics_grid.addWidget(v_lbl, row, col + 1)
            self.metrics_labels[key] = v_lbl

        metrics_layout.addLayout(self.metrics_grid)
        metrics_layout.addStretch()
        self.tabs.addTab(metrics_widget, "Comparison Metrics & Report")

        main_layout.addWidget(self.tabs)

    def _sync_scrollbars(self) -> None:
        """Lock scrollbars between View A and View B for synchronized panning without infinite recursion."""
        self._is_syncing_scroll = False

        def sync_v_a_to_b(val):
            if not self._is_syncing_scroll:
                self._is_syncing_scroll = True
                self.view_b.verticalScrollBar().setValue(val)
                self._is_syncing_scroll = False

        def sync_v_b_to_a(val):
            if not self._is_syncing_scroll:
                self._is_syncing_scroll = True
                self.view_a.verticalScrollBar().setValue(val)
                self._is_syncing_scroll = False

        def sync_h_a_to_b(val):
            if not self._is_syncing_scroll:
                self._is_syncing_scroll = True
                self.view_b.horizontalScrollBar().setValue(val)
                self._is_syncing_scroll = False

        def sync_h_b_to_a(val):
            if not self._is_syncing_scroll:
                self._is_syncing_scroll = True
                self.view_a.horizontalScrollBar().setValue(val)
                self._is_syncing_scroll = False

        self.view_a.verticalScrollBar().valueChanged.connect(sync_v_a_to_b)
        self.view_b.verticalScrollBar().valueChanged.connect(sync_v_b_to_a)
        self.view_a.horizontalScrollBar().valueChanged.connect(sync_h_a_to_b)
        self.view_b.horizontalScrollBar().valueChanged.connect(sync_h_b_to_a)

    def _on_slicing_changed(self, row_axis: int, col_axis: int, slice_indices: list) -> None:
        self.model_a.set_slicing(row_axis, col_axis, slice_indices)
        self.model_b.set_slicing(row_axis, col_axis, slice_indices)
        if self.doc_diff.ndim > 0:
            self.model_diff.set_slicing(row_axis, col_axis, slice_indices)
        self._update_slice_mismatch_highlights(row_axis, col_axis, slice_indices)

    def _update_slice_mismatch_highlights(self, row_axis: int, col_axis: int, slice_indices: list) -> None:
        if self.diff_result is None or not self.diff_result.shape_match:
            return

        # Extract 2D slice of mismatch mask
        ndim = self.doc_a.ndim
        if ndim <= 1:
            mask_slice = self.diff_result.mismatch_mask.reshape(-1, 1) if ndim == 1 else self.diff_result.mismatch_mask
        else:
            slice_spec = []
            other_idx = 0
            for axis in range(ndim):
                if axis == row_axis or axis == col_axis:
                    slice_spec.append(slice(None))
                else:
                    idx = slice_indices[other_idx] if other_idx < len(slice_indices) else 0
                    slice_spec.append(idx)
                    other_idx += 1
            mask_slice = self.diff_result.mismatch_mask[tuple(slice_spec)]
            if row_axis > col_axis:
                mask_slice = np.transpose(mask_slice)

        self.model_a.set_highlight_mask(mask_slice)
        self.model_b.set_highlight_mask(mask_slice)

    # ---------------- Background Concurrency for Diff ----------------

    def _trigger_recompute_diff(self) -> None:
        atol = float(self.atol_spin.value())
        rtol = float(self.rtol_spin.value())

        if self.doc_a.size < 50_000:
            # Instant synchronous computation for small/medium arrays
            diff_res = compare_tensors(self.doc_a.data, self.doc_b.data, atol=atol, rtol=rtol)
            self._on_diff_finished(diff_res)
        else:
            self.progress_bar.setVisible(True)
            self.recompute_btn.setEnabled(False)
            worker = DiffComputeWorker(self.doc_a.data.copy(), self.doc_b.data.copy(), atol=atol, rtol=rtol)
            worker.signals.finished.connect(self._on_diff_finished)
            worker.signals.error.connect(lambda err: self.progress_bar.setVisible(False))
            QThreadPool.globalInstance().start(worker)

    @Slot(object)
    def _on_diff_finished(self, result: DiffResult) -> None:
        self.progress_bar.setVisible(False)
        self.recompute_btn.setEnabled(True)
        self.diff_result = result
        self.mismatch_iterator = MismatchIterator(result.mismatch_coordinates)

        if result.shape_match:
            self.doc_diff.data = np.abs(result.diff_tensor)
            self.model_diff.set_global_bounds(0.0, max(1e-5, result.max_abs_diff))

        # Update metrics labels
        summary = result.summary_dict()
        for key, val in summary.items():
            if key in self.metrics_labels:
                self.metrics_labels[key].setText(str(val))

        # Update status
        if result.mismatch_count == 0:
            self.mismatch_status_lbl.setText("Match (within tolerance)")
            self.mismatch_status_lbl.setStyleSheet("font-weight: 600; color: #4caf50; font-family: monospace;")
        else:
            self.mismatch_status_lbl.setText(f"{result.mismatch_count:,} Mismatches ({result.mismatch_pct:.2f}%)")
            self.mismatch_status_lbl.setStyleSheet("font-weight: 600; color: #f44336; font-family: monospace;")

        # Update highlights
        self._on_slicing_changed(
            self.dimension_bar._row_axis,
            self.dimension_bar._col_axis,
            self.dimension_bar._slice_indices,
        )

    # ---------------- Mismatch Navigation ----------------

    def _on_next_mismatch(self) -> None:
        coord = self.mismatch_iterator.next()
        if coord is not None:
            self._jump_to_mismatch_coord(coord)

    def _on_prev_mismatch(self) -> None:
        coord = self.mismatch_iterator.prev()
        if coord is not None:
            self._jump_to_mismatch_coord(coord)

    def _jump_to_mismatch_coord(self, coord: Tuple[int, ...]) -> None:
        """Jump sliders and table views directly to the mismatched cell coordinate."""
        ndim = len(coord)
        curr_idx = self.mismatch_iterator.current_idx + 1
        total = self.mismatch_iterator.total
        self.mismatch_status_lbl.setText(f"Mismatch {curr_idx} of {total}: {coord}")

        if ndim == 0:
            return
        if ndim == 1:
            r = coord[0]
            idx = self.model_a.index(r, 0)
            self.view_a.scrollTo(idx)
            self.view_b.scrollTo(idx)
            self.view_a.selectRow(r)
            self.view_b.selectRow(r)
            return

        # N-D coordinate: update slider indices for outer axes
        row_axis = self.dimension_bar._row_axis
        col_axis = self.dimension_bar._col_axis

        other_axes = [ax for ax in range(ndim) if ax != row_axis and ax != col_axis]
        for idx_in_others, ax in enumerate(other_axes):
            val = coord[ax]
            if idx_in_others < len(self.dimension_bar._dim_controls):
                _, s, sp, *_ = self.dimension_bar._dim_controls[idx_in_others]
                s.setValue(val)
                sp.setValue(val)

        r = coord[row_axis]
        c = coord[col_axis]
        idx = self.model_a.index(r, c)
        if idx.isValid():
            self.view_a.scrollTo(idx)
            self.view_b.scrollTo(idx)
            self.view_a.setCurrentIndex(idx)
            self.view_b.setCurrentIndex(idx)
