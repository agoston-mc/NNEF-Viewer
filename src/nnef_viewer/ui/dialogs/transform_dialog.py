# Copyright (c) 2026
# Dialog for applying batch math operations, expressions, and normalization to a tensor.

from typing import Any, Dict, Optional
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ...core.operations import apply_math_transform, apply_expression
from ...core.tensor_model import NNEFTensorDocument


class TransformDialog(QDialog):
    """Dialog for applying mathematical transforms and expressions."""

    def __init__(self, doc: NNEFTensorDocument, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.doc = doc
        self.setWindowTitle("Batch Math Transformation")
        self.setMinimumWidth(380)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        group = QGroupBox("Transformation", self)
        form = QFormLayout(group)
        form.setSpacing(8)

        self.op_combo = QComboBox(self)
        self.op_combo.addItems([
            "Add / Shift (+ value)",
            "Multiply / Scale (* value)",
            "Negate (-x)",
            "Clamp [min, max]",
            "Zero Threshold (|x| < eps -> 0)",
            "Normalize Min-Max [0, 1]",
            "Normalize Standard (Z-Score μ=0, σ=1)",
            "Custom NumPy Expression (e.g. sin(x) + 0.5)",
        ])
        self.op_combo.currentIndexChanged.connect(self._on_op_changed)
        form.addRow("Operation:", self.op_combo)

        # Value 1
        self.val1_spin = QDoubleSpinBox(self)
        self.val1_spin.setRange(-1e9, 1e9)
        self.val1_spin.setValue(1.0)
        self.val1_lbl = QLabel("Value:")
        form.addRow(self.val1_lbl, self.val1_spin)

        # Value 2
        self.val2_spin = QDoubleSpinBox(self)
        self.val2_spin.setRange(-1e9, 1e9)
        self.val2_spin.setValue(1.0)
        self.val2_lbl = QLabel("Max:")
        form.addRow(self.val2_lbl, self.val2_spin)

        # Expression
        self.expr_edit = QLineEdit("np.sin(x) * 2.0", self)
        self.expr_lbl = QLabel("Expression:")
        form.addRow(self.expr_lbl, self.expr_edit)

        layout.addWidget(group)
        self._on_op_changed(0)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self._apply_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_op_changed(self, idx: int) -> None:
        # Hide all by default
        self.val1_spin.setVisible(False)
        self.val1_lbl.setVisible(False)
        self.val2_spin.setVisible(False)
        self.val2_lbl.setVisible(False)
        self.expr_edit.setVisible(False)
        self.expr_lbl.setVisible(False)

        if idx in (0, 1):  # Add, Multiply
            self.val1_spin.setVisible(True)
            self.val1_lbl.setVisible(True)
            self.val1_lbl.setText("Value:")
        elif idx == 3:  # Clamp
            self.val1_spin.setVisible(True)
            self.val1_lbl.setVisible(True)
            self.val1_lbl.setText("Min Value:")
            self.val2_spin.setVisible(True)
            self.val2_lbl.setVisible(True)
            self.val2_lbl.setText("Max Value:")
        elif idx == 4:  # Zero Threshold
            self.val1_spin.setVisible(True)
            self.val1_lbl.setVisible(True)
            self.val1_lbl.setText("Epsilon Threshold:")
            self.val1_spin.setValue(1e-4)
        elif idx == 7:  # Expression
            self.expr_edit.setVisible(True)
            self.expr_lbl.setVisible(True)

    def _apply_and_accept(self) -> None:
        idx = self.op_combo.currentIndex()
        if idx == 0:  # Add
            apply_math_transform(self.doc, "add", {"value": self.val1_spin.value()})
        elif idx == 1:  # Multiply
            apply_math_transform(self.doc, "scale", {"value": self.val1_spin.value()})
        elif idx == 2:  # Negate
            apply_math_transform(self.doc, "negate", {})
        elif idx == 3:  # Clamp
            apply_math_transform(self.doc, "clamp", {"min": self.val1_spin.value(), "max": self.val2_spin.value()})
        elif idx == 4:  # Zero threshold
            apply_math_transform(self.doc, "zero_threshold", {"threshold": self.val1_spin.value()})
        elif idx == 5:  # Normalize min-max
            apply_math_transform(self.doc, "normalize", {"mode": "min_max"})
        elif idx == 6:  # Normalize z-score
            apply_math_transform(self.doc, "normalize", {"mode": "z_score"})
        elif idx == 7:  # Expression
            expr = self.expr_edit.text().strip()
            apply_expression(self.doc, expr)

        self.accept()
