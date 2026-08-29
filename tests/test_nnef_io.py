# Copyright (c) 2026
# Tests for NNEF binary I/O, converters, and quantization.

import io
import os
import tempfile
import numpy as np
import pytest

from nnef_viewer.core.nnef_io import (
    read_nnef_tensor,
    write_nnef_tensor,
    export_tensor,
    import_tensor,
    ItemType,
)


@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.bool_,
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (),             # 0D scalar
        (7,),           # 1D vector
        (4, 5),         # 2D matrix
        (2, 3, 4),      # 3D tensor
        (1, 3, 16, 16), # 4D tensor
        (2, 2, 2, 2, 2, 2, 2, 2), # 8D maximum rank
    ],
)
def test_nnef_io_roundtrip_all_dtypes_and_ranks(dtype, shape):
    if np.issubdtype(dtype, np.floating):
        arr = np.asarray(np.random.randn(*shape) * 10).astype(dtype)
    elif np.issubdtype(dtype, np.bool_):
        arr = np.asarray(np.random.randint(0, 2, size=shape) == 1).astype(dtype)
    else:
        info = np.iinfo(dtype)
        low = max(-100, info.min)
        high = min(100, info.max)
        arr = np.asarray(np.random.randint(low, high + 1, size=shape, dtype=dtype))

    buf = io.BytesIO()
    write_nnef_tensor(buf, arr)
    buf.seek(0)
    loaded = read_nnef_tensor(buf)

    assert loaded.shape == arr.shape
    assert loaded.dtype == arr.dtype
    np.testing.assert_array_equal(loaded, arr)


def test_quantized_nnef_io():
    arr = np.random.randint(-128, 127, size=(4, 4), dtype=np.int8)
    buf = io.BytesIO()
    write_nnef_tensor(buf, arr, quantized=True)
    buf.seek(0)

    loaded, is_quantized = read_nnef_tensor(buf, return_quantization=True)
    assert is_quantized is True
    assert loaded.shape == arr.shape
    assert loaded.dtype == arr.dtype
    np.testing.assert_array_equal(loaded, arr)


def test_export_and_import():
    arr = np.array([[1.0, 2.5], [3.0, 4.5]], dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        npy_path = os.path.join(tmpdir, "test.npy")
        csv_path = os.path.join(tmpdir, "test.csv")
        dat_path = os.path.join(tmpdir, "test.dat")

        export_tensor(arr, npy_path)
        export_tensor(arr, csv_path)
        export_tensor(arr, dat_path)

        loaded_npy = import_tensor(npy_path)
        loaded_csv = import_tensor(csv_path)
        loaded_dat = import_tensor(dat_path)

        np.testing.assert_allclose(loaded_npy, arr)
        np.testing.assert_allclose(loaded_csv, arr)
        np.testing.assert_allclose(loaded_dat, arr)
