# Copyright (c) 2026
# High-performance tensor comparison engine with difference metrics and mismatch navigation.

from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import numpy as np


@dataclass
class DiffResult:
    tensor_a: np.ndarray
    tensor_b: np.ndarray
    shape_match: bool
    dtype_match: bool
    diff_tensor: np.ndarray
    mismatch_mask: np.ndarray
    atol: float
    rtol: float
    total_elements: int
    mismatch_count: int
    mismatch_pct: float
    max_abs_diff: float
    mean_abs_error: float
    mean_squared_error: float
    root_mean_squared_error: float
    cosine_similarity: float
    l1_norm: float
    l2_norm: float
    mismatch_coordinates: List[Tuple[int, ...]]

    def summary_dict(self) -> Dict[str, str]:
        return {
            "Shape A": " × ".join(str(d) for d in self.tensor_a.shape),
            "Shape B": " × ".join(str(d) for d in self.tensor_b.shape),
            "Shapes Match": "✓ Yes" if self.shape_match else "✗ No (Mismatch)",
            "Dtypes Match": f"✓ {self.tensor_a.dtype}" if self.dtype_match else f"✗ A: {self.tensor_a.dtype} vs B: {self.tensor_b.dtype}",
            "Total Elements": f"{self.total_elements:,}",
            "Mismatched Elements": f"{self.mismatch_count:,} ({self.mismatch_pct:.4f}%)",
            "Exact Matches": f"{self.total_elements - self.mismatch_count:,} ({100.0 - self.mismatch_pct:.4f}%)",
            "Max Absolute Diff": f"{self.max_abs_diff:.6g}",
            "Mean Absolute Error (MAE)": f"{self.mean_abs_error:.6g}",
            "Mean Squared Error (MSE)": f"{self.mean_squared_error:.6g}",
            "Root MSE (RMSE)": f"{self.root_mean_squared_error:.6g}",
            "Cosine Similarity": f"{self.cosine_similarity:.6f}" if not np.isnan(self.cosine_similarity) else "N/A",
            "L1 Norm (Sum |diff|)": f"{self.l1_norm:.6g}",
            "L2 Norm (Euclidean)": f"{self.l2_norm:.6g}",
            "Tolerance": f"atol={self.atol:.1e}, rtol={self.rtol:.1e}",
        }


def compare_tensors(
    a: np.ndarray,
    b: np.ndarray,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    max_stored_coords: int = 10_000,
) -> DiffResult:
    """
    Compare two NumPy tensors A and B, calculating diff arrays, tolerance mask, and summary error metrics.
    """
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)

    shape_match = (arr_a.shape == arr_b.shape)
    dtype_match = (arr_a.dtype == arr_b.dtype)

    if not shape_match:
        # Cannot directly compute element-wise diff if shapes do not match
        return DiffResult(
            tensor_a=arr_a,
            tensor_b=arr_b,
            shape_match=False,
            dtype_match=dtype_match,
            diff_tensor=np.empty(0),
            mismatch_mask=np.empty(0, dtype=bool),
            atol=atol,
            rtol=rtol,
            total_elements=arr_a.size,
            mismatch_count=arr_a.size,
            mismatch_pct=100.0,
            max_abs_diff=float("nan"),
            mean_abs_error=float("nan"),
            mean_squared_error=float("nan"),
            root_mean_squared_error=float("nan"),
            cosine_similarity=float("nan"),
            l1_norm=float("nan"),
            l2_norm=float("nan"),
            mismatch_coordinates=[],
        )

    # Cast to float64 for high precision diff calculation
    a_f = arr_a.astype(np.float64)
    b_f = arr_b.astype(np.float64)

    # Compute raw diff and absolute diff
    raw_diff = a_f - b_f
    abs_diff = np.abs(raw_diff)

    # Tolerance threshold condition: |A - B| > atol + rtol * |B|
    tolerance_threshold = atol + (rtol * np.abs(b_f))
    mismatch_mask = abs_diff > tolerance_threshold

    # NaN / Inf handling: where both are equal NaNs, consider matched
    both_nan = np.isnan(a_f) & np.isnan(b_f)
    if np.any(both_nan):
        mismatch_mask[both_nan] = False
        abs_diff[both_nan] = 0.0

    one_nan = np.isnan(a_f) ^ np.isnan(b_f)
    if np.any(one_nan):
        mismatch_mask[one_nan] = True

    total_elements = int(arr_a.size)
    mismatch_count = int(np.sum(mismatch_mask))
    mismatch_pct = (mismatch_count / total_elements * 100.0) if total_elements > 0 else 0.0

    finite_abs_diff = abs_diff[np.isfinite(abs_diff)]
    if finite_abs_diff.size > 0:
        max_abs_diff = float(np.max(finite_abs_diff))
        mean_abs_error = float(np.mean(finite_abs_diff))
        squared_diff = raw_diff[np.isfinite(raw_diff)] ** 2
        mean_squared_error = float(np.mean(squared_diff))
        root_mean_squared_error = float(np.sqrt(mean_squared_error))
        l1_norm = float(np.sum(finite_abs_diff))
        l2_norm = float(np.sqrt(np.sum(squared_diff)))
    else:
        max_abs_diff = mean_abs_error = mean_squared_error = root_mean_squared_error = l1_norm = l2_norm = 0.0

    # Cosine similarity
    norm_a = np.linalg.norm(a_f.ravel())
    norm_b = np.linalg.norm(b_f.ravel())
    if norm_a > 0 and norm_b > 0:
        dot_product = np.dot(a_f.ravel(), b_f.ravel())
        cosine_similarity = float(dot_product / (norm_a * norm_b))
    else:
        cosine_similarity = 1.0 if np.array_equal(a_f, b_f) else 0.0

    # Extract first N mismatch coordinates for fast navigation
    mismatch_coords: List[Tuple[int, ...]] = []
    if mismatch_count > 0:
        indices = np.where(mismatch_mask)
        # indices is tuple of arrays per dimension
        num_coords = min(len(indices[0]), max_stored_coords)
        for i in range(num_coords):
            coord = tuple(int(indices[dim][i]) for dim in range(arr_a.ndim))
            mismatch_coords.append(coord)

    return DiffResult(
        tensor_a=arr_a,
        tensor_b=arr_b,
        shape_match=True,
        dtype_match=dtype_match,
        diff_tensor=raw_diff,
        mismatch_mask=mismatch_mask,
        atol=atol,
        rtol=rtol,
        total_elements=total_elements,
        mismatch_count=mismatch_count,
        mismatch_pct=mismatch_pct,
        max_abs_diff=max_abs_diff,
        mean_abs_error=mean_abs_error,
        mean_squared_error=mean_squared_error,
        root_mean_squared_error=root_mean_squared_error,
        cosine_similarity=cosine_similarity,
        l1_norm=l1_norm,
        l2_norm=l2_norm,
        mismatch_coordinates=mismatch_coords,
    )


class MismatchIterator:
    """Helper for stepping through mismatch coordinates in a DiffResult."""

    def __init__(self, coordinates: List[Tuple[int, ...]]):
        self.coordinates = coordinates
        self.current_idx = -1

    @property
    def total(self) -> int:
        return len(self.coordinates)

    def has_next(self) -> bool:
        return self.total > 0 and self.current_idx < self.total - 1

    def has_prev(self) -> bool:
        return self.total > 0 and self.current_idx > 0

    def next(self) -> Optional[Tuple[int, ...]]:
        if self.total == 0:
            return None
        self.current_idx = (self.current_idx + 1) % self.total
        return self.coordinates[self.current_idx]

    def prev(self) -> Optional[Tuple[int, ...]]:
        if self.total == 0:
            return None
        self.current_idx = (self.current_idx - 1 + self.total) % self.total
        return self.coordinates[self.current_idx]

    def current(self) -> Optional[Tuple[int, ...]]:
        if 0 <= self.current_idx < self.total:
            return self.coordinates[self.current_idx]
        return None
