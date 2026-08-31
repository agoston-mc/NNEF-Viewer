# Copyright (c) 2026
# Modal dialog for creating and initializing new NNEF tensors with customizable shapes and distributions.

from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ...core.operations import generate_initial_data
from ...core.tensor_model import NNEFTensorDocument


class NewTensorDialog(QDialog):
    """Dialog for creating a new NNEF tensor with full rank, dtype, and distribution control."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("New NNEF Tensor")
        self.setMinimumWidth(420)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Basic Info Group
        form_group = QGroupBox("Tensor Metadata & Geometry", self)
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
            "scalar (float32)",
            "integer (int32)",
            "logical (bool)",
            "quantized (qint8)",
            "quantized (quint8)",
            "scalar (float16)",
            "scalar (float64)",
            "integer (int8)",
            "integer (int16)",
            "integer (uint8)",
            "integer (uint32)",
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
            "Random Normal (Mean, Std)",
            "Linspace / Range",
            "Identity / Diagonal",
            "Target Statistics (Min, Max, Mean, Std Dev)",
        ])
        self.init_combo.currentIndexChanged.connect(self._on_init_type_changed)
        init_form.addRow("Initialize with:", self.init_combo)

        # Param 1
        self.param1_spin = QDoubleSpinBox(self)
        self.param1_spin.setRange(-1e9, 1e9)
        self.param1_spin.setDecimals(4)
        self.param1_spin.setValue(0.0)
        self.param1_lbl = QLabel("Value / Low:")
        init_form.addRow(self.param1_lbl, self.param1_spin)

        # Param 2
        self.param2_spin = QDoubleSpinBox(self)
        self.param2_spin.setRange(-1e9, 1e9)
        self.param2_spin.setDecimals(4)
        self.param2_spin.setValue(1.0)
        self.param2_lbl = QLabel("High / Std:")
        init_form.addRow(self.param2_lbl, self.param2_spin)

        # Param 3 (Mean for Target Stats)
        self.param3_spin = QDoubleSpinBox(self)
        self.param3_spin.setRange(-1e9, 1e9)
        self.param3_spin.setDecimals(4)
        self.param3_spin.setValue(0.0)
        self.param3_lbl = QLabel("Mean (Target):")
        init_form.addRow(self.param3_lbl, self.param3_spin)

        # Param 4 (Std Dev for Target Stats)
        self.param4_spin = QDoubleSpinBox(self)
        self.param4_spin.setRange(0.0, 1e9)
        self.param4_spin.setDecimals(4)
        self.param4_spin.setValue(0.5)
        self.param4_lbl = QLabel("Std Dev (Target):")
        init_form.addRow(self.param4_lbl, self.param4_spin)

        self.stats_guide_lbl = QLabel("", self)
        self.stats_guide_lbl.setStyleSheet("color: #8c909a; font-size: 11px;")
        init_form.addRow("", self.stats_guide_lbl)

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
        self.param3_spin.setVisible(False)
        self.param3_lbl.setVisible(False)
        self.param4_spin.setVisible(False)
        self.param4_lbl.setVisible(False)
        self.stats_guide_lbl.setVisible(False)

        if idx in (0, 1, 6):  # Zeros, Ones, Identity
            self.param1_spin.setVisible(False)
            self.param1_lbl.setVisible(False)
            self.param2_spin.setVisible(False)
            self.param2_lbl.setVisible(False)
        elif idx == 2:  # Constant
            self.param1_spin.setVisible(True)
            self.param1_lbl.setVisible(True)
            self.param1_lbl.setText("Constant Value:")
            self.param1_spin.setValue(0.0)
            self.param2_spin.setVisible(False)
            self.param2_lbl.setVisible(False)
        elif idx == 3:  # Uniform
            self.param1_spin.setVisible(True)
            self.param1_lbl.setVisible(True)
            self.param1_lbl.setText("Min (Low):")
            self.param1_spin.setValue(-1.0)
            self.param2_spin.setVisible(True)
            self.param2_lbl.setVisible(True)
            self.param2_lbl.setText("Max (High):")
            self.param2_spin.setValue(1.0)
        elif idx == 4:  # Normal
            self.param1_spin.setVisible(True)
            self.param1_lbl.setVisible(True)
            self.param1_lbl.setText("Mean:")
            self.param1_spin.setValue(0.0)
            self.param2_spin.setVisible(True)
            self.param2_lbl.setVisible(True)
            self.param2_lbl.setText("Std Dev:")
            self.param2_spin.setValue(1.0)
        elif idx == 5:  # Linspace
            self.param1_spin.setVisible(True)
            self.param1_lbl.setVisible(True)
            self.param1_lbl.setText("Start:")
            self.param1_spin.setValue(0.0)
            self.param2_spin.setVisible(True)
            self.param2_lbl.setVisible(True)
            self.param2_lbl.setText("Stop:")
            self.param2_spin.setValue(1.0)
        elif idx == 7:  # Target Statistics (Min, Max, Mean, Std Dev)
            self.param1_spin.setVisible(True)
            self.param1_lbl.setVisible(True)
            self.param1_lbl.setText("Min Value:")
            self.param1_spin.setValue(-1.0)

            self.param2_spin.setVisible(True)
            self.param2_lbl.setVisible(True)
            self.param2_lbl.setText("Max Value:")
            self.param2_spin.setValue(1.0)

            self.param3_spin.setVisible(True)
            self.param3_lbl.setVisible(True)
            self.param3_spin.setValue(0.0)

            self.param4_spin.setVisible(True)
            self.param4_lbl.setVisible(True)
            self.param4_spin.setValue(0.5)

            self.stats_guide_lbl.setVisible(True)
            self.stats_guide_lbl.setText("Generates exact min, max, mean, and std dev.")

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
        text = self.dtype_combo.currentText().lower()
        is_quant = self.quantized_chk.isChecked() or "quantized" in text or "qint" in text or "quint" in text

        if "float16" in text:
            np_dtype = np.dtype(np.float16)
        elif "float64" in text:
            np_dtype = np.dtype(np.float64)
        elif "float32" in text or "scalar" in text:
            np_dtype = np.dtype(np.float32)
        elif "int8" in text or "qint8" in text:
            np_dtype = np.dtype(np.int8)
        elif "quint8" in text or "uint8" in text:
            np_dtype = np.dtype(np.uint8)
        elif "int16" in text:
            np_dtype = np.dtype(np.int16)
        elif "uint16" in text:
            np_dtype = np.dtype(np.uint16)
        elif "uint32" in text:
            np_dtype = np.dtype(np.uint32)
        elif "int32" in text or "integer" in text:
            np_dtype = np.dtype(np.int32)
        elif "bool" in text or "logical" in text:
            np_dtype = np.dtype(np.bool_)
        else:
            np_dtype = np.dtype(np.float32)

        init_idx = self.init_combo.currentIndex()
        init_types = ["zeros", "ones", "constant", "uniform", "normal", "linspace", "eye", "quant_stats"]
        init_type = init_types[init_idx] if 0 <= init_idx < len(init_types) else "zeros"

        params = {
            "value": self.param1_spin.value(),
            "low": self.param1_spin.value(),
            "high": self.param2_spin.value(),
            "mean": self.param1_spin.value() if init_type != "quant_stats" else self.param3_spin.value(),
            "std": self.param2_spin.value() if init_type != "quant_stats" else self.param4_spin.value(),
            "start": self.param1_spin.value(),
            "stop": self.param2_spin.value(),
            "min": self.param1_spin.value(),
            "max": self.param2_spin.value(),
        }

        data = generate_initial_data(shape, np_dtype, init_type=init_type, params=params)
        name = self.name_edit.text().strip() or "untitled_tensor"

        doc = NNEFTensorDocument(data, display_name=name, is_quantized=is_quant)
        doc.is_dirty = True
        return doc
