# Copyright (c) 2026
# Tests for UI models, widgets, colormaps, and window initialization.

import os
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from nnef_viewer.core.tensor_model import NNEFTensorDocument
from nnef_viewer.ui.colormap import ColorMapper, AVAILABLE_COLORMAPS
from nnef_viewer.ui.models.tensor_table_model import TensorTableModel
from nnef_viewer.ui.widgets.dimension_slider import DimensionSliderWidget
from nnef_viewer.ui.widgets.matrix_view import MatrixTableView
from nnef_viewer.ui.widgets.stats_panel import StatsPanel
from nnef_viewer.ui.widgets.tensor_tab_widget import NNEFTensorViewerWidget
from nnef_viewer.ui.widgets.diff_view_widget import NNEFDiffViewerWidget
from nnef_viewer.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        # Offscreen platform for headless CI / tests
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QApplication([])
    return app


def test_colormap_lut_and_contrast(qapp):
    mapper = ColorMapper("Coolwarm (Blue-Red)", opacity=0.8)
    assert mapper.colormap_name == "Coolwarm (Blue-Red)"

    # Min value should map to blue
    bg_min, fg_min = mapper.map_value(-10.0, -10.0, 10.0, zero_centered=True)
    assert bg_min is not None
    assert fg_min is not None

    # Max value should map to red
    bg_max, fg_max = mapper.map_value(10.0, -10.0, 10.0, zero_centered=True)
    assert bg_max is not None
    assert fg_max is not None

    # NaN handling
    bg_nan, fg_nan = mapper.map_value(float("nan"), 0.0, 1.0)
    assert bg_nan is not None


def test_tensor_table_model(qapp):
    arr = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], dtype=np.float32)
    doc = NNEFTensorDocument(arr)
    model = TensorTableModel(doc)

    # Initial shape is 2x2 slice (axis 0 and axis 1 with axis 2=0)
    assert model.rowCount() == 2
    assert model.columnCount() == 2

    # Verify data display role
    val = model.data(model.index(0, 1), Qt.ItemDataRole.DisplayRole)
    assert val == "3"

    # Edit cell via table model
    success = model.setData(model.index(0, 1), "99.0", Qt.ItemDataRole.EditRole)
    assert success is True
    assert doc.data[0, 1, 0] == 99.0


def test_dimension_slider_widget(qapp):
    slider_w = DimensionSliderWidget()
    # 4D tensor shape [4, 8, 16, 32] -> Defaults to H=Axis 2, W=Axis 3
    slider_w.set_shape([4, 8, 16, 32])
    assert slider_w._row_axis == 2
    assert slider_w._col_axis == 3
    assert len(slider_w._dim_controls) == 2  # axes 0 and 1
    assert slider_w.sliders_container.isHidden() is False

    # Test collapse
    slider_w._toggle_collapse()
    assert slider_w._is_collapsed is True
    assert slider_w.sliders_container.isHidden() is True

    # Test expand
    slider_w._toggle_collapse()
    assert slider_w._is_collapsed is False
    assert slider_w.sliders_container.isHidden() is False

    # Test Swap
    slider_w._on_swap_axes()
    assert slider_w._row_axis == 3
    assert slider_w._col_axis == 2


def test_linspace_stats_and_dialog(qapp):
    from nnef_viewer.core.operations import generate_initial_data
    from nnef_viewer.core.stats import compute_tensor_stats, compute_histogram

    data = generate_initial_data((1, 3, 224, 224), np.float32, "linspace", {"start": 0.0, "stop": 1.0})
    assert data.shape == (1, 3, 224, 224)
    assert np.isclose(data.min(), 0.0)
    assert np.isclose(data.max(), 1.0)

    stats = compute_tensor_stats(data)
    assert np.isclose(stats.min_val, 0.0)
    assert np.isclose(stats.max_val, 1.0)
    assert stats.max_val <= 1.0 + 1e-6

    counts, edges, min_v, max_v = compute_histogram(data, num_bins=40)
    assert np.isclose(min_v, 0.0)
    assert np.isclose(max_v, 1.0)
    assert np.all(edges >= 0.0 - 1e-6)
    assert np.all(edges <= 1.0 + 1e-6)


def test_embeddable_tensor_viewer_widget(qapp):
    # Shape [2, 3, 4, 5] -> Last two dimensions H=4, W=5
    arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
    doc = NNEFTensorDocument(arr, display_name="test_4d.dat")
    viewer = NNEFTensorViewerWidget(doc)

    assert viewer.table_model.rowCount() == 4
    assert viewer.table_model.columnCount() == 5

    # Toggle slicing bar
    viewer.toggle_slicing_btn.setChecked(False)
    assert viewer.dimension_bar.isHidden() is True
    viewer.toggle_slicing_btn.setChecked(True)
    assert viewer.dimension_bar.isHidden() is False


def test_diff_viewer_widget(qapp):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[1.0, 2.5], [3.0, 4.0]], dtype=np.float32)
    doc_a = NNEFTensorDocument(a, display_name="A.dat")
    doc_b = NNEFTensorDocument(b, display_name="B.dat")

    diff_widget = NNEFDiffViewerWidget(doc_a, doc_b)
    assert diff_widget.tabs.count() == 3


def test_main_window_lifecycle_and_duplication(qapp):
    window = MainWindow()
    # Startup opens to blank welcome page
    assert window.tab_widget.count() == 0
    assert window.stack.currentIndex() == 0

    # Add a new document tab
    doc = NNEFTensorDocument(np.zeros((10, 10)), file_path="/fake/path/tensor.dat", display_name="tensor.dat")
    window.add_tensor_document_tab(doc)
    assert window.tab_widget.count() == 1
    assert window.stack.currentIndex() == 1

    # Prevent duplicate open: try opening same path again
    window.open_file("/fake/path/tensor.dat")
    assert window.tab_widget.count() == 1

    # Duplicate tensor into new tab
    window.duplicate_current_tensor()
    assert window.tab_widget.count() == 2
    assert "tensor_copy" in window.tab_widget.tabText(1)

    # Toggle theme
    window._toggle_theme()
    assert window._is_dark_theme is False
    window._toggle_theme()
    assert window._is_dark_theme is True


def test_unsaved_changes_and_settings_dialogs(qapp):
    from nnef_viewer.ui.dialogs.unsaved_changes_dialog import UnsavedChangesDialog, UnsavedChoice
    from nnef_viewer.ui.dialogs.settings_dialog import SettingsDialog
    from nnef_viewer.core.settings import AppSettings

    settings = AppSettings.get_instance()
    settings.reset_defaults()

    # Test UnsavedChangesDialog
    dlg = UnsavedChangesDialog("my_weights.dat")
    assert dlg.doc_name == "my_weights.dat"
    dlg.remember_chk.setChecked(True)
    dlg._select_choice(UnsavedChoice.SAVE_AND_CLOSE)
    assert dlg.choice == UnsavedChoice.SAVE_AND_CLOSE
    assert settings.get("silence_unsaved_dialog") is True
    assert settings.get("unsaved_close_action") == "save_and_close"

    # Test SettingsDialog
    settings_dlg = SettingsDialog()
    settings_dlg.unsaved_action_combo.setCurrentIndex(settings_dlg.unsaved_action_combo.findData("always_discard"))
    settings_dlg.theme_combo.setCurrentIndex(settings_dlg.theme_combo.findData("light"))
    settings_dlg._save_and_accept()

    assert settings.get("default_theme") == "light"
    settings.reset_defaults()


def test_axis_swapping_preserves_outer_dimensions(qapp):
    slider_w = DimensionSliderWidget()
    # 4D tensor shape [4, 10, 32, 64] -> defaults to H=Axis 2, W=Axis 3; outer: Axis 0 (size 4), Axis 1 (size 10)
    slider_w.set_shape([4, 10, 32, 64])
    assert slider_w._row_axis == 2
    assert slider_w._col_axis == 3

    # Set outer dimension slider for Axis 1 (Channel) to 3, and Axis 0 (Batch) to 2
    controls = slider_w._dim_controls
    assert len(controls) == 2
    # controls[0] is Axis 0, controls[1] is Axis 1
    axis0, slider0, spin0, _ = controls[0]
    axis1, slider1, spin1, _ = controls[1]
    assert axis0 == 0 and axis1 == 1

    slider0.setValue(2)
    slider1.setValue(3)
    assert slider_w._slice_indices == [2, 3]

    # Swap H and W (Axis 2 <-> Axis 3)
    slider_w._on_swap_axes()
    assert slider_w._row_axis == 3
    assert slider_w._col_axis == 2

    # Outer dimensions (Axis 0 and Axis 1) must be preserved
    assert slider_w._slice_indices == [2, 3]
    new_controls = slider_w._dim_controls
    assert new_controls[0][1].value() == 2
    assert new_controls[1][1].value() == 3


def test_status_bar_restores_tensor_info(qapp):
    window = MainWindow()
    doc = NNEFTensorDocument(np.ones((2, 3, 4), dtype=np.float32), display_name="weights.dat")
    window.add_tensor_document_tab(doc)

    assert "weights.dat" in window.status_bar.currentMessage()
    assert "float32" in window.status_bar.currentMessage()

    # Simulate temporary message
    window.status_bar.showMessage("Temporary alert", 2000)
    assert window.status_bar.currentMessage() == "Temporary alert"

    # Clear temporary message (simulating timeout)
    window.status_bar.clearMessage()
    assert "weights.dat" in window.status_bar.currentMessage()
    assert "float32" in window.status_bar.currentMessage()


def test_generate_quant_stats_data_accuracy():
    from nnef_viewer.core.operations import generate_quant_stats_data, generate_initial_data

    shape = (1, 3, 64, 64)
    min_v, max_v, mean_v, std_v = -4.0, 8.0, 1.5, 2.5
    data = generate_quant_stats_data(shape, np.float32, min_v, max_v, mean_v, std_v)

    assert data.shape == shape
    assert np.isclose(data.min(), min_v, atol=1e-5)
    assert np.isclose(data.max(), max_v, atol=1e-5)
    assert np.isclose(data.mean(), mean_v, atol=1e-4)
    assert np.isclose(data.std(), std_v, atol=1e-3)

    # Test through generate_initial_data
    params = {"min": min_v, "max": max_v, "mean": mean_v, "std": std_v}
    data2 = generate_initial_data(shape, np.float32, init_type="quant_stats", params=params)
    assert np.isclose(data2.min(), min_v, atol=1e-5)
    assert np.isclose(data2.max(), max_v, atol=1e-5)
    assert np.isclose(data2.mean(), mean_v, atol=1e-4)
    assert np.isclose(data2.std(), std_v, atol=1e-3)
