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
    # 4D tensor shape
    slider_w.set_shape([4, 8, 16, 32])
    assert slider_w._row_axis == 0
    assert slider_w._col_axis == 1
    assert len(slider_w._dim_controls) == 2  # axes 2 and 3
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
    assert slider_w._row_axis == 1
    assert slider_w._col_axis == 0


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
    arr = np.random.randn(2, 3, 4, 5).astype(np.float32)
    doc = NNEFTensorDocument(arr, display_name="test_4d.dat")
    viewer = NNEFTensorViewerWidget(doc)

    assert viewer.table_model.rowCount() == 2
    assert viewer.table_model.columnCount() == 3

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


def test_main_window_lifecycle(qapp):
    window = MainWindow()
    assert window.tab_widget.count() >= 1

    # Add a new document tab
    doc = NNEFTensorDocument(np.zeros((10, 10)), display_name="sample.dat")
    window.add_tensor_document_tab(doc)
    assert window.tab_widget.count() >= 2

    # Toggle theme
    window._toggle_theme()
    assert window._is_dark_theme is False
    window._toggle_theme()
    assert window._is_dark_theme is True
