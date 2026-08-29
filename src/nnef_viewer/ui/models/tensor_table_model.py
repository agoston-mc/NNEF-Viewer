# Copyright (c) 2026
# High-performance virtualized QAbstractTableModel for 2D tensor slices with heatmap colors.

from typing import Any, List, Optional, Sequence, Tuple
import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor

from ...core.tensor_model import NNEFTensorDocument
from ..colormap import ColorMapper


class TensorTableModel(QAbstractTableModel):
    """
    Virtualized Qt Table Model for a 2D slice of an N-D NNEFTensorDocument.
    Provides dynamic JetBrains-style background heatmaps and auto-contrast foreground text.
    """

    def __init__(
        self,
        document: NNEFTensorDocument,
        color_mapper: Optional[ColorMapper] = None,
        parent: Optional[Any] = None,
    ):
        super().__init__(parent)
        self.doc = document
        self.color_mapper = color_mapper or ColorMapper()

        # Slicing state
        self.row_axis: int = 0
        self.col_axis: int = 1 if document.ndim > 1 else 0
        self.slice_indices: List[int] = [0] * max(0, document.ndim - 2)

        # Active cached slice
        self._cached_slice: np.ndarray = np.empty((0, 0))
        self._slice_min: float = 0.0
        self._slice_max: float = 1.0

        # Visualization settings
        self.normalization_mode: str = "Slice Min-Max"  # 'Slice Min-Max', 'Global Min-Max', 'Zero-Centered'
        self.zero_centered: bool = False
        self.global_min: float = 0.0
        self.global_max: float = 1.0
        self.float_format: str = "%.5g"
        self.highlight_mask: Optional[np.ndarray] = None  # Optional 2D boolean mask for diff mismatches

        # Subscribe to document changes
        self.doc.add_data_change_listener(self._on_doc_data_changed)
        self.doc.add_structure_change_listener(self._on_doc_structure_changed)

        self._refresh_slice_cache()

    def set_slicing(self, row_axis: int, col_axis: int, slice_indices: Sequence[int]) -> None:
        """Update active row/col axes and outer dimension slice indices."""
        self.row_axis = row_axis
        self.col_axis = col_axis
        self.slice_indices = list(slice_indices)
        self._refresh_slice_cache()

    def set_normalization_mode(self, mode: str, zero_centered: bool = False) -> None:
        self.normalization_mode = mode
        self.zero_centered = zero_centered
        self._refresh_min_max()
        self.layoutChanged.emit()

    def set_global_bounds(self, g_min: float, g_max: float) -> None:
        self.global_min = g_min
        self.global_max = g_max
        if self.normalization_mode == "Global Min-Max":
            self.layoutChanged.emit()

    def set_highlight_mask(self, mask: Optional[np.ndarray]) -> None:
        self.highlight_mask = mask
        self.layoutChanged.emit()

    def _refresh_slice_cache(self) -> None:
        self.beginResetModel()
        try:
            self._cached_slice = self.doc.get_2d_slice(
                self.row_axis, self.col_axis, self.slice_indices
            )
        except Exception:
            self._cached_slice = np.empty((0, 0), dtype=self.doc.dtype)

        self._refresh_min_max()
        self.endResetModel()

    def _refresh_min_max(self) -> None:
        if self._cached_slice.size > 0:
            if np.issubdtype(self._cached_slice.dtype, np.floating):
                finite = self._cached_slice[np.isfinite(self._cached_slice)]
                if finite.size > 0:
                    self._slice_min = float(np.min(finite))
                    self._slice_max = float(np.max(finite))
                else:
                    self._slice_min, self._slice_max = 0.0, 1.0
            else:
                self._slice_min = float(np.min(self._cached_slice))
                self._slice_max = float(np.max(self._cached_slice))
        else:
            self._slice_min, self._slice_max = 0.0, 1.0

    def _on_doc_data_changed(self) -> None:
        self._refresh_slice_cache()

    def _on_doc_structure_changed(self) -> None:
        ndim = self.doc.ndim
        self.row_axis = 0 if ndim > 0 else 0
        self.col_axis = 1 if ndim > 1 else 0
        self.slice_indices = [0] * max(0, ndim - 2)
        self._refresh_slice_cache()

    # ---------------- QAbstractTableModel Interface ----------------

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return self._cached_slice.shape[0] if self._cached_slice.ndim >= 2 else 0

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return self._cached_slice.shape[1] if self._cached_slice.ndim >= 2 else 0

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return f"[{section}]"
            else:
                return f"[{section}]"
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        row, col = index.row(), index.column()
        if row >= self._cached_slice.shape[0] or col >= self._cached_slice.shape[1]:
            return None

        val = self._cached_slice[row, col]

        if role == Qt.ItemDataRole.DisplayRole:
            return self._format_value(val)

        elif role == Qt.ItemDataRole.EditRole:
            return str(val)

        elif role == Qt.ItemDataRole.ToolTipRole:
            return f"Coord: ({row}, {col})\nValue: {val}\nDtype: {self.doc.dtype}"

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        elif role == Qt.ItemDataRole.BackgroundRole:
            # Check diff highlight mask first
            if self.highlight_mask is not None and self.highlight_mask.shape == self._cached_slice.shape:
                if self.highlight_mask[row, col]:
                    # Highlight mismatch cell with amber/red
                    return QColor(220, 53, 69, 180)

            if not self.color_mapper.enabled:
                return None

            v_min = self.global_min if self.normalization_mode == "Global Min-Max" else self._slice_min
            v_max = self.global_max if self.normalization_mode == "Global Min-Max" else self._slice_max
            bg_color, _ = self.color_mapper.map_value(
                float(val), v_min, v_max, zero_centered=self.zero_centered
            )
            return bg_color

        elif role == Qt.ItemDataRole.ForegroundRole:
            if not self.color_mapper.enabled:
                return None

            v_min = self.global_min if self.normalization_mode == "Global Min-Max" else self._slice_min
            v_max = self.global_max if self.normalization_mode == "Global Min-Max" else self._slice_max
            _, fg_color = self.color_mapper.map_value(
                float(val), v_min, v_max, zero_centered=self.zero_centered
            )
            return fg_color

        return None

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False

        row, col = index.row(), index.column()
        try:
            # Parse entered string value according to document dtype
            dt = self.doc.dtype
            if np.issubdtype(dt, np.bool_):
                parsed = str(value).lower() in ("true", "1", "t", "yes", "y")
            elif np.issubdtype(dt, np.integer):
                parsed = int(value, 0) if str(value).startswith("0x") else int(value)
            elif np.issubdtype(dt, np.floating):
                parsed = float(value)
            else:
                parsed = value

            self.doc.set_2d_slice_cell_value(
                self.row_axis,
                self.col_axis,
                self.slice_indices,
                row,
                col,
                parsed,
            )
            return True
        except Exception:
            return False

    def _format_value(self, val: Any) -> str:
        if np.issubdtype(self.doc.dtype, np.floating):
            if np.isnan(val):
                return "NaN"
            if np.isposinf(val):
                return "+Inf"
            if np.isneginf(val):
                return "-Inf"
            return self.float_format % val
        elif np.issubdtype(self.doc.dtype, np.bool_):
            return "True" if val else "False"
        else:
            return str(val)
