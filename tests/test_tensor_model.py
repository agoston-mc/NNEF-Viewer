# Copyright (c) 2026
# Tests for NNEFTensorDocument, delta-based undo/redo, slicing, and memory efficiency.

import numpy as np
import pytest

from nnef_viewer.core.tensor_model import (
    NNEFTensorDocument,
    CellEditCommand,
    SliceEditCommand,
    InvertibleTransformCommand,
    MaskedTransformCommand,
    ReshapeCommand,
)
from nnef_viewer.core.operations import (
    apply_math_transform,
    apply_expression,
    reshape_tensor,
    cast_dtype,
    generate_initial_data,
)


def test_tensor_document_cell_edit_undo_redo():
    arr = np.zeros((3, 4, 5), dtype=np.float32)
    doc = NNEFTensorDocument(arr)

    assert doc.data[1, 2, 3] == 0.0
    assert not doc.can_undo()
    assert not doc.can_redo()
    assert not doc.is_dirty

    doc.set_cell_value((1, 2, 3), 42.0)
    assert doc.data[1, 2, 3] == 42.0
    assert doc.can_undo()
    assert doc.is_dirty

    # Undo
    doc.undo()
    assert doc.data[1, 2, 3] == 0.0
    assert not doc.can_undo()
    assert doc.can_redo()

    # Redo
    doc.redo()
    assert doc.data[1, 2, 3] == 42.0
    assert doc.can_undo()


def test_2d_slice_extraction_and_cell_edit():
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    doc = NNEFTensorDocument(arr)

    # Slice with row_axis=1 (size 3), col_axis=2 (size 4), batch idx=0
    slice_2d = doc.get_2d_slice(row_axis=1, col_axis=2, slice_indices=[0])
    assert slice_2d.shape == (3, 4)
    np.testing.assert_array_equal(slice_2d, arr[0, :, :])

    # Edit cell via 2D slice method
    doc.set_2d_slice_cell_value(row_axis=1, col_axis=2, slice_indices=[0], row=1, col=2, new_val=999.0)
    assert doc.data[0, 1, 2] == 999.0

    # Undo
    doc.undo()
    assert doc.data[0, 1, 2] == 6.0


def test_invertible_batch_math_undo_redo():
    arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    doc = NNEFTensorDocument(arr)

    # Offset (+5)
    apply_math_transform(doc, "add", {"value": 5.0})
    np.testing.assert_allclose(doc.data, [15.0, 25.0, 35.0])

    # Scale (* 2)
    apply_math_transform(doc, "scale", {"value": 2.0})
    np.testing.assert_allclose(doc.data, [30.0, 50.0, 70.0])

    # Undo scale
    doc.undo()
    np.testing.assert_allclose(doc.data, [15.0, 25.0, 35.0])

    # Undo add
    doc.undo()
    np.testing.assert_allclose(doc.data, [10.0, 20.0, 30.0])


def test_clamp_and_expression_transforms():
    arr = np.array([-5.0, 2.0, 15.0], dtype=np.float32)
    doc = NNEFTensorDocument(arr)

    apply_math_transform(doc, "clamp", {"min": 0.0, "max": 10.0})
    np.testing.assert_allclose(doc.data, [0.0, 2.0, 10.0])

    doc.undo()
    np.testing.assert_allclose(doc.data, [-5.0, 2.0, 15.0])

    # Expression
    apply_expression(doc, "x * 2.0 + 1.0")
    np.testing.assert_allclose(doc.data, [-9.0, 5.0, 31.0])

    doc.undo()
    np.testing.assert_allclose(doc.data, [-5.0, 2.0, 15.0])


def test_reshape_and_cast():
    arr = np.zeros((2, 6), dtype=np.float32)
    doc = NNEFTensorDocument(arr)

    reshape_tensor(doc, (3, 4))
    assert doc.shape == (3, 4)

    doc.undo()
    assert doc.shape == (2, 6)

    cast_dtype(doc, np.int32)
    assert doc.dtype == np.int32

    doc.undo()
    assert doc.dtype == np.float32
