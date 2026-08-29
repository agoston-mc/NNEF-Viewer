# Copyright (c) 2026
# Tests for tensor comparison, diff metrics, tolerance matching, and mismatch navigation.

import numpy as np
import pytest

from nnef_viewer.core.comparison import compare_tensors, MismatchIterator
from nnef_viewer.core.stats import compute_tensor_stats, compute_histogram


def test_compare_identical_tensors():
    arr = np.random.randn(4, 5, 6).astype(np.float32)
    res = compare_tensors(arr, arr.copy())

    assert res.shape_match is True
    assert res.dtype_match is True
    assert res.mismatch_count == 0
    assert res.mismatch_pct == 0.0
    assert res.max_abs_diff == 0.0
    assert res.mean_abs_error == 0.0
    assert res.mean_squared_error == 0.0
    assert res.cosine_similarity == pytest.approx(1.0, rel=1e-5)


def test_compare_tolerance_and_metrics():
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([1.0, 2.0001, 3.5, 4.0], dtype=np.float32)

    # With atol=1e-3, index 1 (diff 0.0001) should match, index 2 (diff 0.5) should mismatch
    res = compare_tensors(a, b, atol=1e-3, rtol=1e-3)
    assert res.mismatch_count == 1
    assert res.max_abs_diff == pytest.approx(0.5, rel=1e-4)
    assert len(res.mismatch_coordinates) == 1
    assert res.mismatch_coordinates[0] == (2,)

    # Mismatch iterator
    it = MismatchIterator(res.mismatch_coordinates)
    assert it.total == 1
    assert it.next() == (2,)


def test_stats_and_histogram():
    arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0], dtype=np.float32)
    stats = compute_tensor_stats(arr)

    assert stats.total_elements == 7
    assert stats.min_val == 0.0
    assert stats.max_val == 5.0
    assert stats.zero_count == 2
    assert stats.sparsity_pct == pytest.approx((2 / 7) * 100.0)

    counts, edges, min_v, max_v = compute_histogram(arr, num_bins=10)
    assert len(counts) == 10
    assert len(edges) == 11
    assert np.sum(counts) == 7
