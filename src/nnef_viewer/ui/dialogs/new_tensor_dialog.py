# Copyright (c) 2026
# Dialog wizard for creating new NNEF tensors with customizable shape, dtype, and generator.

from typing import Any, Dict, Optional, Tuple
import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core.operations import generate_initial_data
from ...core.tensor_model import NNEFTensorDocument


class NewTensorDialog(QDialog):
    """Wizard dialog to create a new NNEF Tensor."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Create New NNEF Tensor")
        self.setMinimumWidth(380)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form_group = QGroupBox("Tensor Specification", self)
        form = QFormLayout(form_group)
        form.setSpacing(8)

        # Name
        self.name_edit = QLineEdit("untitled_tensor", self)
        form.addRow("Document Name:", self.name_edit)

        # Shape
        self.shape_edit = QLineEdit("1, 3, 224, 224", self)
        self.shape_edit.setPlaceholderText("e.g. 64, 128 or 1, 3, 224, 224")
        self.shape_edit.textChanged.connect(self._update_total_elements)
        form.addRow("Shape Dimensions:", self.shape_edit)

        self.elements_lbl = QLabel("Total Elements: 150,528", self)
        self.elements_lbl.setStyleSheet("color: #3574f0; font-size: 11px;")
        form.addRow("", self.elements_lbl)

        # Dtype
        self.dtype_combo = QComboBox(self)
        self.dtype_combo.addItems([
            "float32", "float16", "float64",
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64",
            "bool", "qint8 (quantized)", "quint8 (quantized)"
        ])
        form.addRow("Data Type (Dtype):", self.dtype_combo)

        # Quantized checkbox
        self.quantized_chk = QCheckBox("Mark as Quantized (NNEF QINT/QUINT)", self)
        form.addRow("", self.quantized_chk)

        layout.addWidget(form_group)

        # Initialization Method Group
        init_group = QGroupBox("Initialization Method", self)
        init_form = QFormLayout(init_group)
        init_form.setSpacing(8)

        self.init_combo = QComboBox(self)
        self.init_combo.addItems([
            "Zeros (0.0)",
            "Ones (1.0)",
            "Constant Value",
            "Random Uniform [low, high]",
            "Random Normal (μ, σ)",
            "Linspace / Range",
            "Identity / Diagonal",
        ])
        self.init_combo.currentIndexChanged.connect(self._on_init_type_changed)
        init_form.addRow("Initialize with:", self.init_combo)

        # Param 1 (Low / Mean / Const)
        self.param1_spin = QDoubleSpinBox(self)
        self.param1_spin.setRange(-1e9, 1e9)
        self.param1_spin.setValue(0.0)
        self.param1_lbl = QLabel("Value / Low:")
        init_form.addRow(self.param1_lbl, self.param1_spin)

        # Param 2 (High / Std)
        self.param2_spin = QDoubleSpinBox(self)
        self.param2_spin.setRange(-1e9, 1e9)
        self.param2_spin.setValue(1.0)
        self.param2_lbl = QLabel("High / Std:")
        init_form.addRow(self.param2_lbl, self.param2_spin)

        layout.addWidget(init_group)
        self._on_init_type_changed(0)

        # Dialog Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_init_type_changed(self, idx: int) -> None:
        if idx in (0, 1, 6):  # Zeros, Ones, Identity
            self.param1_spin.setVisible(False)
            self.param1_lbl.setVisible(False)
            self.param2_spin.setVisible(False)
            self.param2_lbl.setVisible(False)
        elif idx == 2:  # Constant
            self.param1_spin.setVisible(True)
            self.param1_lbl.setVisible(True)
            self.param1_lbl.setText("Constant Value:")
            self.param2_spin.setVisible(False)
            self.param2_lbl.setVisible(False)
        elif idx == 3:  # Uniform
            self.param1_spin.setVisible(True)
            self.param1_lbl.setVisible(True)
            self.param1_lbl.setText("Low (min):")
            self.param2_spin.setVisible(True)
            self.param2_lbl.setVisible(True)
            self.param2_lbl.setText("High (max):")
        elif idx == 4:  # Normal
            self.param1_spin.setVisible(True)
            self.param1_lbl.setVisible(True)
            self.param1_lbl.setText("Mean (μ):")
            self.param2_spin.setVisible(True)
            self.param2_lbl.setVisible(True)
            self.param2_lbl.setText("Std Dev (σ):")
        elif idx == 5:  # Linspace
            self.param1_spin.setVisible(True)
            self.param1_lbl.setVisible(True)
            self.param1_lbl.setText("Start:")
            self.param2_spin.setVisible(True)
            self.param2_lbl.setVisible(True)
            self.param2_lbl.setText("Stop:")

    def _update_total_elements(self) -> None:
        try:
            shape = self.get_shape()
            count = int(np.prod(shape)) if shape else 1
            self.elements_lbl.setText(f"Total Elements: {count:,}")
        except Exception:
            self.elements_lbl.setText("Invalid shape syntax")

    def get_shape(self) -> Tuple[int, ...]:
        raw = self.shape_edit.text().strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        if not raw:
            return ()
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return tuple(int(x) for x in parts)

    def create_document(self) -> NNEFTensorDocument:
        shape = self.get_shape()
        dtype_str = self.dtype_combo.currentText().split()[0]
        is_quant = self.quantized_chk.isChecked() or "quantized" in self.dtype_combo.currentText()

        if dtype_str.startswith("qint"):
            np_dtype = np.int8
        elif dtype_str.startswith("quint"):
            np_dtype = np.uint8
        else:
            np_dtype = np.dtype(dtype_str)

        init_idx = self.init_combo.currentIndex()
        init_types = ["zeros", "ones", "constant", "uniform", "normal", "linspace", "eye"]
        init_type = init_types[init_idx]

        params = {
            "value": self.param1_spin.value(),
            "low": self.param1_spin.value(),
            "high": self.param2_spin.value(),
            "mean": self.param1_spin.value(),
            "std": self.param2_spin.value(),
            "start": self.param1_spin.value(),
            "stop": self.param2_spin.value(),
        }

        data = generate_initial_data(shape, np_dtype, init_type=init_type, params=params)
        name = self.name_edit.text().strip() or "untitled_tensor"

        doc = NNEFTensorDocument(data, display_name=name, is_quantized=is_quant)
        doc.is_dirty = True
        return doc
