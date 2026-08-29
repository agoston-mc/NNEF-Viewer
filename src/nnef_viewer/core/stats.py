# Copyright (c) 2026
# Fast vectorized summary statistics and histogram computations for NNEF tensors.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class TensorStats:
    shape: Tuple[int, ...]
    ndim: int
    dtype_name: str
    total_elements: int
    memory_bytes: int
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    median_val: float
    variance_val: float
    zero_count: int
    nonzero_count: int
    sparsity_pct: float
    nan_count: int
    pos_inf_count: int
    neg_inf_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Shape": " × ".join(str(d) for d in self.shape) if self.shape else "Scalar (0D)",
            "Rank": str(self.ndim),
            "Dtype": self.dtype_name,
            "Total Elements": f"{self.total_elements:,}",
            "Memory Size": format_bytes(self.memory_bytes),
            "Min": f"{self.min_val:.6g}" if not np.isnan(self.min_val) else "NaN",
            "Max": f"{self.max_val:.6g}" if not np.isnan(self.max_val) else "NaN",
            "Mean": f"{self.mean_val:.6g}" if not np.isnan(self.mean_val) else "NaN",
            "Std Dev": f"{self.std_val:.6g}" if not np.isnan(self.std_val) else "NaN",
            "Median": f"{self.median_val:.6g}" if not np.isnan(self.median_val) else "NaN",
            "Variance": f"{self.variance_val:.6g}" if not np.isnan(self.variance_val) else "NaN",
            "Sparsity": f"{self.sparsity_pct:.2f}% ({self.zero_count:,} zeros)",
            "NaN Count": f"{self.nan_count:,}",
            "Inf Count": f"{self.pos_inf_count + self.neg_inf_count:,}",
        }


def format_bytes(n_bytes: int) -> str:
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.2f} KB"
    elif n_bytes < 1024 * 1024 * 1024:
        return f"{n_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{n_bytes / (1024 * 1024 * 1024):.2f} GB"


def compute_tensor_stats(arr: np.ndarray, max_sample_elements: Optional[int] = None) -> TensorStats:
    """
    Compute comprehensive statistics for a tensor or slice.
    If max_sample_elements is set, sub-samples for ultra-fast median estimation on huge tensors.
    """
    total_elements = int(arr.size)
    if total_elements == 0:
        return TensorStats(
            shape=arr.shape,
            ndim=arr.ndim,
            dtype_name=str(arr.dtype),
            total_elements=0,
            memory_bytes=0,
            min_val=0.0,
            max_val=0.0,
            mean_val=0.0,
            std_val=0.0,
            median_val=0.0,
            variance_val=0.0,
            zero_count=0,
            nonzero_count=0,
            sparsity_pct=0.0,
            nan_count=0,
            pos_inf_count=0,
            neg_inf_count=0,
        )

    memory_bytes = int(arr.nbytes)
    dtype_name = str(arr.dtype)

    # Boolean or Integer or Float handling
    if np.issubdtype(arr.dtype, np.floating):
        nan_mask = np.isnan(arr)
        pos_inf_mask = np.isposinf(arr)
        neg_inf_mask = np.isneginf(arr)
        nan_count = int(np.sum(nan_mask))
        pos_inf_count = int(np.sum(pos_inf_mask))
        neg_inf_count = int(np.sum(neg_inf_mask))

        finite_mask = np.isfinite(arr)
        finite_count = int(np.sum(finite_mask))

        if finite_count > 0:
            finite_data = arr[finite_mask]
            min_val = float(np.min(finite_data))
            max_val = float(np.max(finite_data))
            mean_val = float(np.mean(finite_data))
            var_val = float(np.var(finite_data))
            std_val = float(np.sqrt(var_val))

            if max_sample_elements and finite_count > max_sample_elements:
                sample_idx = np.random.choice(finite_count, size=max_sample_elements, replace=False)
                median_val = float(np.median(finite_data[sample_idx]))
            else:
                median_val = float(np.median(finite_data))
        else:
            min_val = max_val = mean_val = std_val = median_val = var_val = float("nan")

        zero_count = int(np.sum(arr == 0))

    elif np.issubdtype(arr.dtype, np.bool_):
        nan_count = pos_inf_count = neg_inf_count = 0
        true_count = int(np.sum(arr))
        zero_count = total_elements - true_count
        min_val = 0.0 if zero_count > 0 else 1.0
        max_val = 1.0 if true_count > 0 else 0.0
        mean_val = float(true_count / total_elements)
        var_val = mean_val * (1.0 - mean_val)
        std_val = float(np.sqrt(var_val))
        median_val = 1.0 if true_count > (total_elements // 2) else 0.0

    else:
        # Integer
        nan_count = pos_inf_count = neg_inf_count = 0
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        mean_val = float(np.mean(arr))
        var_val = float(np.var(arr))
        std_val = float(np.sqrt(var_val))

        if max_sample_elements and total_elements > max_sample_elements:
            flat = arr.ravel()
            sample_idx = np.random.choice(total_elements, size=max_sample_elements, replace=False)
            median_val = float(np.median(flat[sample_idx]))
        else:
            median_val = float(np.median(arr))

        zero_count = int(np.sum(arr == 0))

    nonzero_count = total_elements - zero_count
    sparsity_pct = (zero_count / total_elements) * 100.0

    return TensorStats(
        shape=arr.shape,
        ndim=arr.ndim,
        dtype_name=dtype_name,
        total_elements=total_elements,
        memory_bytes=memory_bytes,
        min_val=min_val,
        max_val=max_val,
        mean_val=mean_val,
        std_val=std_val,
        median_val=median_val,
        variance_val=var_val,
        zero_count=zero_count,
        nonzero_count=nonzero_count,
        sparsity_pct=sparsity_pct,
        nan_count=nan_count,
        pos_inf_count=pos_inf_count,
        neg_inf_count=neg_inf_count,
    )


def compute_histogram(
    arr: np.ndarray,
    num_bins: int = 40,
    max_sample_elements: int = 500_000,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute histogram counts and bin edges for values in arr.
    Filters out non-finite (NaN, Inf) values.
    Returns (counts, bin_edges, min_finite, max_finite).
    """
    total = arr.size
    if total == 0:
        return np.zeros(num_bins, dtype=np.int64), np.linspace(0, 1, num_bins + 1), 0.0, 1.0

    # Handle floats vs ints
    if np.issubdtype(arr.dtype, np.floating):
        finite_data = arr[np.isfinite(arr)]
    else:
        finite_data = arr.ravel()

    if finite_data.size == 0:
        return np.zeros(num_bins, dtype=np.int64), np.linspace(0, 1, num_bins + 1), 0.0, 1.0

    min_val = float(np.min(finite_data))
    max_val = float(np.max(finite_data))

    if min_val == max_val:
        counts = np.zeros(num_bins, dtype=np.int64)
        counts[num_bins // 2] = len(finite_data)
        edges = np.linspace(min_val, max_val, num_bins + 1)
        return counts, edges, min_val, max_val

    # Subsample if massive to keep GUI responsive
    if finite_data.size > max_sample_elements:
        idx = np.random.choice(finite_data.size, size=max_sample_elements, replace=False)
        data_to_bin = finite_data[idx]
    else:
        data_to_bin = finite_data

    counts, edges = np.histogram(data_to_bin, bins=num_bins, range=(min_val, max_val))
    return counts, edges, min_val, max_val
