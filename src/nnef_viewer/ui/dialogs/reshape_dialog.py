# Copyright (c) 2026
# Dialog for reshaping and dtype casting an NNEF tensor.

from typing import Optional, Tuple
import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from ...core.operations import reshape_tensor, cast_dtype
from ...core.tensor_model import NNEFTensorDocument


class ReshapeDialog(QDialog):
    """Dialog for reshaping tensor dimensions and casting data type."""

    def __init__(self, doc: NNEFTensorDocument, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.doc = doc
        self.setWindowTitle("Reshape & Cast Tensor")
        self.setMinimumWidth(380)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        group = QGroupBox("Reshape Dimensions", self)
        form = QFormLayout(group)
        form.setSpacing(8)

        cur_shape_str = ", ".join(str(d) for d in self.doc.shape)
        form.addRow("Current Shape:", QLabel(f"<b>{cur_shape_str}</b> (Total: {self.doc.size:,} items)"))

        self.shape_edit = QLineEdit(cur_shape_str, self)
        self.shape_edit.textChanged.connect(self._validate_shape)
        form.addRow("New Shape:", self.shape_edit)

        self.validation_lbl = QLabel("✓ Valid element count match", self)
        self.validation_lbl.setStyleSheet("color: #2e7d32; font-size: 11px;")
        form.addRow("", self.validation_lbl)

        layout.addWidget(group)

        # Cast Dtype Group
        cast_group = QGroupBox("Cast Data Type", self)
        cast_form = QFormLayout(cast_group)
        cast_form.setSpacing(8)

        cast_form.addRow("Current Dtype:", QLabel(f"<b>{self.doc.dtype}</b>"))

        self.dtype_combo = QComboBox(self)
        self.dtype_combo.addItems([
            "Keep Current", "float32", "float16", "float64",
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64", "bool"
        ])
        cast_form.addRow("Cast to:", self.dtype_combo)

        layout.addWidget(cast_group)

        # Dialog Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self._apply_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate_shape(self) -> None:
        try:
            new_shape = self.get_new_shape()
            new_count = int(np.prod(new_shape)) if new_shape else 1
            if new_count == self.doc.size:
                self.validation_lbl.setText(f"✓ Valid: {new_count:,} elements match")
                self.validation_lbl.setStyleSheet("color: #2e7d32; font-size: 11px;")
            else:
                self.validation_lbl.setText(f"✗ Mismatch: {new_count:,} vs current {self.doc.size:,}")
                self.validation_lbl.setStyleSheet("color: #c62828; font-size: 11px;")
        except Exception:
            self.validation_lbl.setText("✗ Invalid syntax")
            self.validation_lbl.setStyleSheet("color: #c62828; font-size: 11px;")

    def get_new_shape(self) -> Tuple[int, ...]:
        raw = self.shape_edit.text().strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        if not raw:
            return ()
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return tuple(int(x) for x in parts)

    def _apply_and_accept(self) -> None:
        try:
            new_shape = self.get_new_shape()
            if new_shape != self.doc.shape:
                reshape_tensor(self.doc, new_shape)

            selected_dt = self.dtype_combo.currentText()
            if selected_dt != "Keep Current" and selected_dt != str(self.doc.dtype):
                cast_dtype(self.doc, np.dtype(selected_dt))

            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error Reshaping Tensor", str(e))
