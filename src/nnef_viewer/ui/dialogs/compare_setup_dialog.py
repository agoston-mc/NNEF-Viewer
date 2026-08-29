# Copyright (c) 2026
# Dialog for selecting two tensors to compare side-by-side.

from typing import List, Optional, Tuple
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.nnef_io import read_tensor_from_location
from ...core.tensor_model import NNEFTensorDocument


class CompareSetupDialog(QDialog):
    """Dialog allowing user to choose Tensor A and Tensor B for comparison."""

    def __init__(self, open_docs: List[NNEFTensorDocument], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.open_docs = open_docs
        self.setWindowTitle("Compare Tensors")
        self.setMinimumWidth(440)

        self.doc_a: Optional[NNEFTensorDocument] = None
        self.doc_b: Optional[NNEFTensorDocument] = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Tensor A
        group_a = QGroupBox("Select Tensor A (Baseline)", self)
        form_a = QFormLayout(group_a)
        self.combo_a = QComboBox(group_a)
        for i, doc in enumerate(self.open_docs):
            self.combo_a.addItem(f"Tab {i+1}: {doc.display_name} (shape {doc.shape}, {doc.dtype})", doc)
        form_a.addRow("Open Tensor:", self.combo_a)

        h_a = QHBoxLayout()
        self.browse_a_btn = QPushButton("📁 Browse File A...", group_a)
        self.browse_a_btn.clicked.connect(self._browse_a)
        h_a.addWidget(self.browse_a_btn)
        self.file_a_lbl = QLabel("", group_a)
        h_a.addWidget(self.file_a_lbl)
        h_a.addStretch()
        form_a.addRow("", h_a)
        layout.addWidget(group_a)

        # Tensor B
        group_b = QGroupBox("Select Tensor B (Comparison / Candidate)", self)
        form_b = QFormLayout(group_b)
        self.combo_b = QComboBox(group_b)
        for i, doc in enumerate(self.open_docs):
            self.combo_b.addItem(f"Tab {i+1}: {doc.display_name} (shape {doc.shape}, {doc.dtype})", doc)
        if len(self.open_docs) > 1:
            self.combo_b.setCurrentIndex(1)
        form_b.addRow("Open Tensor:", self.combo_b)

        h_b = QHBoxLayout()
        self.browse_b_btn = QPushButton("📁 Browse File B...", group_b)
        self.browse_b_btn.clicked.connect(self._browse_b)
        h_b.addWidget(self.browse_b_btn)
        self.file_b_lbl = QLabel("", group_b)
        h_b.addWidget(self.file_b_lbl)
        h_b.addStretch()
        form_b.addRow("", h_b)
        layout.addWidget(group_b)

        # Dialog Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse_a(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Tensor A", "", "NNEF & NumPy Tensors (*.dat *.npy *.npz *.csv)")
        if path:
            arr, is_quant = read_tensor_from_location(path)
            doc = NNEFTensorDocument(arr, file_path=path, display_name=path.split("/")[-1], is_quantized=is_quant)
            self.doc_a = doc
            self.file_a_lbl.setText(f"Loaded: {doc.display_name}")

    def _browse_b(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open Tensor B", "", "NNEF & NumPy Tensors (*.dat *.npy *.npz *.csv)")
        if path:
            arr, is_quant = read_tensor_from_location(path)
            doc = NNEFTensorDocument(arr, file_path=path, display_name=path.split("/")[-1], is_quantized=is_quant)
            self.doc_b = doc
            self.file_b_lbl.setText(f"Loaded: {doc.display_name}")

    def _on_accept(self) -> None:
        if self.doc_a is None and self.combo_a.count() > 0:
            self.doc_a = self.combo_a.currentData()
        if self.doc_b is None and self.combo_b.count() > 0:
            self.doc_b = self.combo_b.currentData()

        if self.doc_a and self.doc_b:
            self.accept()
        else:
            self.reject()

    def get_selected_documents(self) -> Tuple[NNEFTensorDocument, NNEFTensorDocument]:
        return self.doc_a, self.doc_b
