# Copyright (c) 2026
# Top-level Standalone Application Window with Multi-Tab Management and Menus.

import os
from typing import List, Optional
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.nnef_io import (
    export_tensor,
    list_tensors_in_archive_or_dir,
    read_nnef_tensor,
    read_tensor_from_location,
    write_nnef_tensor,
)
from ..core.tensor_model import NNEFTensorDocument
from .dialogs.compare_setup_dialog import CompareSetupDialog
from .dialogs.new_tensor_dialog import NewTensorDialog
from .dialogs.reshape_dialog import ReshapeDialog
from .dialogs.transform_dialog import TransformDialog
from .styles.themes import DARK_THEME_QSS, LIGHT_THEME_QSS
from .widgets.diff_view_widget import NNEFDiffViewerWidget
from .widgets.tensor_tab_widget import NNEFTensorViewerWidget


class MainWindow(QMainWindow):
    """
    Main desktop window hosting multiple tensor editor tabs, comparison tabs,
    drag-and-drop file loading, themes, and toolbars.
    """

    def __init__(self, initial_files: Optional[List[str]] = None):
        super().__init__()
        self.setWindowTitle("NNEF Tensor Editor & Matrix Viewer")
        self.resize(1200, 800)
        self.setAcceptDrops(True)

        self._is_dark_theme = True
        self._setup_ui()
        self._setup_menus()
        self._apply_theme()

        # Open initial files or sample demo tensor if none provided
        if initial_files:
            for f in initial_files:
                self.open_file(f)
        if self.tab_widget.count() == 0:
            self._create_sample_demo_tensor()

    def _setup_ui(self) -> None:
        self.tab_widget = QTabWidget(self)
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.setMovable(True)
        self.tab_widget.tabCloseRequested.connect(self._on_tab_close_requested)
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        self.setCentralWidget(self.tab_widget)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _setup_menus(self) -> None:
        menubar = self.menuBar()

        # 1. File Menu
        file_menu = menubar.addMenu("&File")

        new_action = QAction("&New Tensor...", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._action_new_tensor)
        file_menu.addAction(new_action)

        open_action = QAction("&Open Tensor File...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._action_open_file)
        file_menu.addAction(open_action)

        open_folder_action = QAction("Open NNEF &Folder / Archive...", self)
        open_folder_action.triggered.connect(self._action_open_folder)
        file_menu.addAction(open_folder_action)

        file_menu.addSeparator()

        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._action_save)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self._action_save_as)
        file_menu.addAction(save_as_action)

        export_action = QAction("&Export (NumPy / CSV)...", self)
        export_action.triggered.connect(self._action_export)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        close_tab_action = QAction("&Close Tab", self)
        close_tab_action.setShortcut(QKeySequence.StandardKey.Close)
        close_tab_action.triggered.connect(self._action_close_current_tab)
        file_menu.addAction(close_tab_action)

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 2. Edit Menu
        edit_menu = menubar.addMenu("&Edit")

        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        self.undo_action.triggered.connect(self._action_undo)
        edit_menu.addAction(self.undo_action)

        self.redo_action = QAction("&Redo", self)
        self.redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        self.redo_action.triggered.connect(self._action_redo)
        edit_menu.addAction(self.redo_action)

        edit_menu.addSeparator()

        transform_action = QAction("Batch &Math Transform...", self)
        transform_action.setShortcut("Ctrl+T")
        transform_action.triggered.connect(self._action_transform)
        edit_menu.addAction(transform_action)

        reshape_action = QAction("&Reshape && Cast...", self)
        reshape_action.setShortcut("Ctrl+R")
        reshape_action.triggered.connect(self._action_reshape)
        edit_menu.addAction(reshape_action)

        # 3. View Menu
        view_menu = menubar.addMenu("&View")

        theme_action = QAction("Toggle Dark / &Light Theme", self)
        theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(theme_action)

        # 4. Tools Menu
        tools_menu = menubar.addMenu("&Tools")

        compare_action = QAction("&Compare Two Tensors (Diff)...", self)
        compare_action.setShortcut("Ctrl+D")
        compare_action.triggered.connect(self._action_compare_tensors)
        tools_menu.addAction(compare_action)

        tools_menu.addSeparator()

        demo_action = QAction("Open &Sample Demo Tensors", self)
        demo_action.triggered.connect(self._create_sample_demo_tensor)
        tools_menu.addAction(demo_action)

    def _apply_theme(self) -> None:
        if self._is_dark_theme:
            self.setStyleSheet(DARK_THEME_QSS)
        else:
            self.setStyleSheet(LIGHT_THEME_QSS)

    def _toggle_theme(self) -> None:
        self._is_dark_theme = not self._is_dark_theme
        self._apply_theme()

    # ---------------- Tab Management ----------------

    def add_tensor_document_tab(self, doc: NNEFTensorDocument) -> NNEFTensorViewerWidget:
        widget = NNEFTensorViewerWidget(doc, self.tab_widget)
        doc.add_dirty_change_listener(lambda is_dirty: self._update_tab_title_for_doc(doc))
        doc.add_structure_change_listener(lambda: self._update_status_bar())

        tab_idx = self.tab_widget.addTab(widget, doc.display_name)
        self.tab_widget.setCurrentIndex(tab_idx)
        self._update_status_bar()
        return widget

    def add_diff_tab(self, doc_a: NNEFTensorDocument, doc_b: NNEFTensorDocument) -> NNEFDiffViewerWidget:
        diff_widget = NNEFDiffViewerWidget(doc_a, doc_b, self.tab_widget)
        title = f"Diff: {doc_a.display_name} vs {doc_b.display_name}"
        tab_idx = self.tab_widget.addTab(diff_widget, title)
        self.tab_widget.setCurrentIndex(tab_idx)
        return diff_widget

    def _get_current_tensor_viewer(self) -> Optional[NNEFTensorViewerWidget]:
        curr = self.tab_widget.currentWidget()
        if isinstance(curr, NNEFTensorViewerWidget):
            return curr
        return None

    def _update_tab_title_for_doc(self, doc: NNEFTensorDocument) -> None:
        for idx in range(self.tab_widget.count()):
            w = self.tab_widget.widget(idx)
            if isinstance(w, NNEFTensorViewerWidget) and w.doc is doc:
                prefix = "* " if doc.is_dirty else ""
                self.tab_widget.setTabText(idx, f"{prefix}{doc.display_name}")
                break

    def _on_tab_changed(self, index: int) -> None:
        self._update_status_bar()

    def _update_status_bar(self) -> None:
        viewer = self._get_current_tensor_viewer()
        if viewer:
            doc = viewer.doc
            shape_str = " × ".join(str(x) for x in doc.shape) if doc.shape else "0D Scalar"
            path_str = doc.file_path or "Unsaved"
            self.status_bar.showMessage(
                f"{doc.display_name} | Shape: [{shape_str}] | Dtype: {doc.dtype} | Elements: {doc.size:,} | File: {path_str}"
            )
        else:
            self.status_bar.showMessage("Ready")

    def _on_tab_close_requested(self, index: int) -> None:
        w = self.tab_widget.widget(index)
        if isinstance(w, NNEFTensorViewerWidget) and w.doc.is_dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                f"Document '{w.doc.display_name}' has unsaved changes. Close anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        self.tab_widget.removeTab(index)
        if hasattr(w, "deleteLater"):
            w.deleteLater()

    # ---------------- File Actions ----------------

    def _action_new_tensor(self) -> None:
        dlg = NewTensorDialog(self)
        if dlg.exec() == NewTensorDialog.DialogCode.Accepted:
            doc = dlg.create_document()
            self.add_tensor_document_tab(doc)

    def _action_open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Tensor File",
            "",
            "NNEF & NumPy Tensors (*.dat *.npy *.npz *.csv *.tsv);;All Files (*)",
        )
        if path:
            self.open_file(path)

    def _action_open_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open NNEF Model Directory")
        if path:
            tensors = list_tensors_in_archive_or_dir(path)
            if not tensors:
                QMessageBox.information(self, "No Tensors Found", f"No .dat binary tensor files found in {path}")
                return
            for name, full_path in tensors:
                self.open_file(full_path, display_name=name)

    def open_file(self, file_path: str, display_name: Optional[str] = None) -> None:
        try:
            arr, is_quant = read_tensor_from_location(file_path)
            name = display_name or os.path.basename(file_path)
            doc = NNEFTensorDocument(arr, file_path=file_path, display_name=name, is_quantized=is_quant)
            self.add_tensor_document_tab(doc)
            self.status_bar.showMessage(f"Opened {file_path}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Error Opening File", f"Failed to open '{file_path}':\n{str(e)}")

    def _action_save(self) -> None:
        viewer = self._get_current_tensor_viewer()
        if not viewer:
            return
        doc = viewer.doc
        if doc.file_path and doc.file_path.endswith(".dat"):
            try:
                write_nnef_tensor(doc.file_path, doc.data, quantized=doc.is_quantized)
                doc.mark_saved()
                self._update_tab_title_for_doc(doc)
                self.status_bar.showMessage(f"Saved {doc.file_path}", 4000)
            except Exception as e:
                QMessageBox.critical(self, "Error Saving File", str(e))
        else:
            self._action_save_as()

    def _action_save_as(self) -> None:
        viewer = self._get_current_tensor_viewer()
        if not viewer:
            return
        doc = viewer.doc
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save NNEF Binary Tensor",
            doc.file_path or f"{doc.display_name}.dat",
            "NNEF Binary Tensor (*.dat);;NumPy Array (*.npy);;CSV (*.csv)",
        )
        if path:
            try:
                if path.endswith(".dat"):
                    write_nnef_tensor(path, doc.data, quantized=doc.is_quantized)
                else:
                    export_tensor(doc.data, path)
                doc.mark_saved(path)
                self._update_tab_title_for_doc(doc)
                self.status_bar.showMessage(f"Saved {path}", 4000)
            except Exception as e:
                QMessageBox.critical(self, "Error Saving File", str(e))

    def _action_export(self) -> None:
        viewer = self._get_current_tensor_viewer()
        if not viewer:
            return
        doc = viewer.doc
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Tensor Data",
            f"{doc.display_name}.npy",
            "NumPy Array (*.npy);;Compressed NumPy (*.npz);;CSV File (*.csv);;TSV File (*.tsv)",
        )
        if path:
            try:
                export_tensor(doc.data, path)
                self.status_bar.showMessage(f"Exported to {path}", 4000)
            except Exception as e:
                QMessageBox.critical(self, "Error Exporting File", str(e))

    def _action_close_current_tab(self) -> None:
        idx = self.tab_widget.currentIndex()
        if idx >= 0:
            self._on_tab_close_requested(idx)

    # ---------------- Edit & Transform Actions ----------------

    def _action_undo(self) -> None:
        viewer = self._get_current_tensor_viewer()
        if viewer:
            viewer._on_undo()

    def _action_redo(self) -> None:
        viewer = self._get_current_tensor_viewer()
        if viewer:
            viewer._on_redo()

    def _action_transform(self) -> None:
        viewer = self._get_current_tensor_viewer()
        if not viewer:
            return
        dlg = TransformDialog(viewer.doc, self)
        dlg.exec()

    def _action_reshape(self) -> None:
        viewer = self._get_current_tensor_viewer()
        if not viewer:
            return
        dlg = ReshapeDialog(viewer.doc, self)
        dlg.exec()

    # ---------------- Compare Action ----------------

    def _action_compare_tensors(self) -> None:
        open_docs = []
        for idx in range(self.tab_widget.count()):
            w = self.tab_widget.widget(idx)
            if isinstance(w, NNEFTensorViewerWidget):
                open_docs.append(w.doc)

        dlg = CompareSetupDialog(open_docs, self)
        if dlg.exec() == CompareSetupDialog.DialogCode.Accepted:
            doc_a, doc_b = dlg.get_selected_documents()
            if doc_a and doc_b:
                self.add_diff_tab(doc_a, doc_b)

    # ---------------- Sample Demo Generator ----------------

    def _create_sample_demo_tensor(self) -> None:
        """Create sample multi-dimensional tensors for demonstration."""
        # 1. 4D Convolution Weights [16, 3, 24, 24] with Gaussian distribution
        w_data = np.random.normal(0.0, 0.5, size=(16, 3, 24, 24)).astype(np.float32)
        doc1 = NNEFTensorDocument(w_data, display_name="conv2d_weights.dat")
        self.add_tensor_document_tab(doc1)

        # 2. 2D Embedding Matrix [64, 32]
        emb_data = np.random.uniform(-1.0, 1.0, size=(64, 32)).astype(np.float32)
        doc2 = NNEFTensorDocument(emb_data, display_name="embedding_table.dat")
        self.add_tensor_document_tab(doc2)

        # 3. Quantized int8 Tensor [32, 32]
        q_data = np.random.randint(-128, 127, size=(32, 32), dtype=np.int8)
        doc3 = NNEFTensorDocument(q_data, display_name="quantized_bias.dat", is_quantized=True)
        self.add_tensor_document_tab(doc3)

    # ---------------- Drag and Drop Support ----------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if os.path.isdir(file_path):
                tensors = list_tensors_in_archive_or_dir(file_path)
                for name, full_path in tensors:
                    self.open_file(full_path, display_name=name)
            elif os.path.isfile(file_path):
                self.open_file(file_path)
