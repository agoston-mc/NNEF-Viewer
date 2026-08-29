# Copyright (c) 2026
# End-to-end integration tests for NNEF-Viewer.

import os
import tempfile
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from nnef_viewer.core.nnef_io import write_nnef_tensor, read_nnef_tensor
from nnef_viewer.core.tensor_model import NNEFTensorDocument
from nnef_viewer.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QApplication([])
    return app


def test_e2e_open_modify_save_compare(qapp):
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create a 4D tensor file [4, 8, 16, 16]
        file_a = os.path.join(tmpdir, "tensor_a.dat")
        file_b = os.path.join(tmpdir, "tensor_b.dat")

        data_a = np.random.normal(0.0, 1.0, size=(4, 8, 16, 16)).astype(np.float32)
        data_b = data_a.copy()
        # Introduce deliberate differences at specific coordinates
        data_b[0, 0, 5, 5] += 5.0
        data_b[2, 3, 10, 10] -= 10.0

        write_nnef_tensor(file_a, data_a)
        write_nnef_tensor(file_b, data_b)

        # 2. Launch MainWindow and open files
        window = MainWindow(initial_files=[file_a, file_b])
        assert window.tab_widget.count() == 2

        # 3. Add Diff Tab
        viewer_a = window.tab_widget.widget(0)
        viewer_b = window.tab_widget.widget(1)
        diff_widget = window.add_diff_tab(viewer_a.doc, viewer_b.doc)

        assert diff_widget.diff_result is not None
        assert diff_widget.diff_result.mismatch_count == 2
        assert diff_widget.mismatch_iterator.total == 2

        # 4. Step through mismatch navigation
        first_coord = diff_widget.mismatch_iterator.next()
        assert first_coord == (0, 0, 5, 5)
        diff_widget._jump_to_mismatch_coord(first_coord)

        second_coord = diff_widget.mismatch_iterator.next()
        assert second_coord == (2, 3, 10, 10)
        diff_widget._jump_to_mismatch_coord(second_coord)

        # 5. Modify cell in Viewer A
        viewer_a.doc.set_cell_value((0, 0, 5, 5), data_b[0, 0, 5, 5])
        assert viewer_a.doc.is_dirty

        # 6. Save modified tensor
        window.tab_widget.setCurrentIndex(0)
        window._action_save()
        assert not viewer_a.doc.is_dirty

        # Verify saved binary on disk
        reloaded = read_nnef_tensor(file_a)
        assert reloaded[0, 0, 5, 5] == pytest.approx(data_b[0, 0, 5, 5])
