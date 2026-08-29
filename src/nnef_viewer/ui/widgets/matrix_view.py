# Copyright (c) 2026
# High-performance QTableView with matrix features, zoom, and clipboard operations.

import io
import csv
from typing import Optional
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QKeySequence, QShortcut, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QHeaderView,
    QInputDialog,
    QMenu,
    QTableView,
    QWidget,
)

from ..models.tensor_table_model import TensorTableModel


class MatrixTableView(QTableView):
    """
    Optimized Matrix View with virtualized scrolling, copy to clipboard,
    custom font zoom, and context actions.
    """

    cellHovered = Signal(int, int, str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Performance optimizations
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.horizontalHeader().setDefaultSectionSize(75)
        self.verticalHeader().setDefaultSectionSize(26)
        self.setAlternatingRowColors(False)
        self.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.setSelectionBehavior(QTableView.SelectionBehavior.SelectItems)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # Base font
        self._font_size = 11
        self._update_font()

        # Keyboard shortcuts
        copy_shortcut = QShortcut(QKeySequence.StandardKey.Copy, self)
        copy_shortcut.activated.connect(self.copy_selection_to_clipboard)

    def _update_font(self) -> None:
        font = QFont("JetBrains Mono, Menlo, Monaco, Consolas, monospace", self._font_size)
        self.setFont(font)

    def wheelEvent(self, event: QWheelEvent) -> None:
        # Ctrl + Wheel to Zoom Font
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self._font_size = min(24, self._font_size + 1)
            elif delta < 0:
                self._font_size = max(7, self._font_size - 1)
            self._update_font()
            event.accept()
        else:
            super().wheelEvent(event)

    def copy_selection_to_clipboard(self) -> None:
        """Copy selected matrix cells to system clipboard as TSV / CSV."""
        selection = self.selectionModel().selectedIndexes()
        if not selection:
            return

        rows = sorted(list(set(idx.row() for idx in selection)))
        cols = sorted(list(set(idx.column() for idx in selection)))

        model = self.model()
        if not model:
            return

        output = io.StringIO()
        writer = csv.writer(output, delimiter="\t")

        for r in rows:
            row_data = []
            for c in cols:
                idx = model.index(r, c)
                val = model.data(idx, Qt.ItemDataRole.DisplayRole)
                row_data.append(val if val is not None else "")
            writer.writerow(row_data)

        QApplication.clipboard().setText(output.getvalue())

    def _show_context_menu(self, pos) -> None:
        menu = QMenu(self)

        copy_action = menu.addAction("Copy Selection (TSV)")
        copy_action.triggered.connect(self.copy_selection_to_clipboard)

        copy_py_action = menu.addAction("Copy as NumPy Python Array")
        copy_py_action.triggered.connect(self._copy_as_python_array)

        menu.addSeparator()

        fill_action = menu.addAction("Fill Selected Cells with Value...")
        fill_action.triggered.connect(self._fill_selection_dialog)

        menu.addSeparator()

        fit_action = menu.addAction("Auto-fit Column Widths")
        fit_action.triggered.connect(self.resizeColumnsToContents)

        reset_zoom_action = menu.addAction("Reset Font Zoom")
        reset_zoom_action.triggered.connect(self._reset_zoom)

        menu.exec(self.viewport().mapToGlobal(pos))

    def _copy_as_python_array(self) -> None:
        selection = self.selectionModel().selectedIndexes()
        if not selection:
            return
        rows = sorted(list(set(idx.row() for idx in selection)))
        cols = sorted(list(set(idx.column() for idx in selection)))
        model = self.model()
        if not model:
            return

        matrix = []
        for r in rows:
            row_vals = []
            for c in cols:
                idx = model.index(r, c)
                val_str = model.data(idx, Qt.ItemDataRole.EditRole)
                row_vals.append(val_str)
            matrix.append("[" + ", ".join(row_vals) + "]")
        py_str = "np.array([\n  " + ",\n  ".join(matrix) + "\n])"
        QApplication.clipboard().setText(py_str)

    def _fill_selection_dialog(self) -> None:
        selection = self.selectionModel().selectedIndexes()
        if not selection:
            return
        val_str, ok = QInputDialog.getText(self, "Fill Selection", "Enter value to fill selected cells:")
        if not ok or not val_str:
            return
        model = self.model()
        if not isinstance(model, TensorTableModel):
            return

        for idx in selection:
            model.setData(idx, val_str, Qt.ItemDataRole.EditRole)

    def _reset_zoom(self) -> None:
        self._font_size = 11
        self._update_font()
        self.horizontalHeader().setDefaultSectionSize(75)
        self.verticalHeader().setDefaultSectionSize(26)
