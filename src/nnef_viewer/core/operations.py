# Copyright (c) 2026
# Mathematical operations, transformations, dtype casting, and generators for NNEF tensors.

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import numpy as np

from .tensor_model import (
    NNEFTensorDocument,
    InvertibleTransformCommand,
    MaskedTransformCommand,
    ReshapeCommand,
    DtypeCastCommand,
)


def generate_initial_data(
    shape: Sequence[int],
    dtype: Union[np.dtype, str],
    init_type: str = "zeros",
    params: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Generate initial tensor data based on shape, dtype, and generator type."""
    dt = np.dtype(dtype)
    params = params or {}
    shape_tuple = tuple(int(x) for x in shape)

    is_int = np.issubdtype(dt, np.integer)
    is_bool = np.issubdtype(dt, np.bool_)

    if init_type == "zeros":
        return np.zeros(shape_tuple, dtype=dt)
    elif init_type == "ones":
        return np.ones(shape_tuple, dtype=dt)
    elif init_type == "constant":
        val = params.get("value", 0)
        return np.full(shape_tuple, val, dtype=dt)
    elif init_type == "uniform":
        low = float(params.get("low", -1.0 if not (is_int or is_bool) else 0))
        high = float(params.get("high", 1.0 if not (is_int or is_bool) else 100))
        if is_bool:
            return np.random.choice([True, False], size=shape_tuple)
        elif is_int:
            return np.random.randint(int(low), int(high) + 1, size=shape_tuple).astype(dt)
        else:
            return np.random.uniform(low, high, size=shape_tuple).astype(dt)
    elif init_type == "normal":
        mean = float(params.get("mean", 0.0))
        std = float(params.get("std", 1.0))
        if is_bool or is_int:
            data = np.random.normal(mean, std, size=shape_tuple)
            return np.round(data).astype(dt)
        else:
            return np.random.normal(mean, std, size=shape_tuple).astype(dt)
    elif init_type == "linspace":
        start = float(params.get("start", 0.0))
        stop = float(params.get("stop", 1.0))
        count = int(np.prod(shape_tuple))
        if count == 0:
            return np.empty(shape_tuple, dtype=dt)
        arr = np.linspace(start, stop, count)
        return arr.reshape(shape_tuple).astype(dt)
    elif init_type == "eye":
        if len(shape_tuple) == 2:
            return np.eye(shape_tuple[0], shape_tuple[1], dtype=dt)
        else:
            # Identity on last two dimensions
            arr = np.zeros(shape_tuple, dtype=dt)
            if len(shape_tuple) >= 2:
                min_dim = min(shape_tuple[-2], shape_tuple[-1])
                for i in range(min_dim):
                    arr[..., i, i] = 1
            return arr
    else:
        return np.zeros(shape_tuple, dtype=dt)


def apply_math_transform(doc: NNEFTensorDocument, op_type: str, params: Dict[str, Any]) -> None:
    """Apply batch mathematical transform to document with memory-efficient delta/invertible undo."""
    if op_type == "offset" or op_type == "add":
        delta = params.get("value", 0.0)
        if delta == 0:
            return

        def forward(arr: np.ndarray) -> None:
            arr += delta

        def inverse(arr: np.ndarray) -> None:
            arr -= delta

        cmd = InvertibleTransformCommand(
            op_name="add",
            forward_fn=forward,
            inverse_fn=inverse,
            desc=f"Add {delta}",
        )
        doc.apply_command(cmd)

    elif op_type == "scale" or op_type == "multiply":
        factor = params.get("value", 1.0)
        if factor == 1.0:
            return
        if factor != 0.0:
            inv_factor = 1.0 / factor

            def forward(arr: np.ndarray) -> None:
                arr *= factor

            def inverse(arr: np.ndarray) -> None:
                arr *= inv_factor

            cmd = InvertibleTransformCommand(
                op_name="scale",
                forward_fn=forward,
                inverse_fn=inverse,
                desc=f"Multiply by {factor}",
            )
            doc.apply_command(cmd)
        else:
            # Multiplying by zero is not invertible, store mask of non-zero elements
            non_zero_mask = np.where(doc.data != 0)
            old_values = doc.data[non_zero_mask].copy()
            cmd_masked = MaskedTransformCommand(
                modified_indices=non_zero_mask,
                old_values=old_values,
                new_values=0,
                desc="Multiply by 0",
            )
            doc.apply_command(cmd_masked)

    elif op_type == "negate":
        def forward(arr: np.ndarray) -> None:
            np.negative(arr, out=arr)

        cmd = InvertibleTransformCommand(
            op_name="negate",
            forward_fn=forward,
            inverse_fn=forward,
            desc="Negate values",
        )
        doc.apply_command(cmd)

    elif op_type == "clamp":
        c_min = params.get("min", None)
        c_max = params.get("max", None)
        conditions = []
        if c_min is not None:
            conditions.append(doc.data < c_min)
        if c_max is not None:
            conditions.append(doc.data > c_max)

        if not conditions:
            return

        combined_mask = conditions[0] if len(conditions) == 1 else (conditions[0] | conditions[1])
        modified_indices = np.where(combined_mask)
        old_values = doc.data[modified_indices].copy()

        clamped_data = doc.data.copy()
        if c_min is not None and c_max is not None:
            np.clip(clamped_data, c_min, c_max, out=clamped_data)
        elif c_min is not None:
            clamped_data = np.maximum(clamped_data, c_min)
        elif c_max is not None:
            clamped_data = np.minimum(clamped_data, c_max)

        new_values = clamped_data[modified_indices]
        cmd_masked = MaskedTransformCommand(
            modified_indices=modified_indices,
            old_values=old_values,
            new_values=new_values,
            desc=f"Clamp [{c_min}, {c_max}]",
        )
        doc.apply_command(cmd_masked)

    elif op_type == "zero_threshold":
        threshold = float(params.get("threshold", 1e-6))
        mask = np.where((doc.data != 0) & (np.abs(doc.data) < threshold))
        old_values = doc.data[mask].copy()
        cmd_masked = MaskedTransformCommand(
            modified_indices=mask,
            old_values=old_values,
            new_values=0,
            desc=f"Zero-out values < {threshold}",
        )
        doc.apply_command(cmd_masked)

    elif op_type == "normalize":
        mode = params.get("mode", "min_max")  # 'min_max' or 'z_score'
        old_backup = doc.data.copy()
        if mode == "min_max":
            v_min = float(np.min(doc.data))
            v_max = float(np.max(doc.data))
            rng = v_max - v_min
            if rng > 0:
                doc.data = (doc.data - v_min) / rng
        else:
            mean = float(np.mean(doc.data))
            std = float(np.std(doc.data))
            if std > 0:
                doc.data = (doc.data - mean) / std


def apply_expression(doc: NNEFTensorDocument, expr_str: str) -> None:
    """
    Safely evaluate a mathematical expression against tensor x.
    Supported functions: sin, cos, tan, exp, log, sqrt, abs, clip, sinh, cosh, tanh, etc.
    """
    # Safe namespace
    safe_dict = {
        "x": doc.data,
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "log10": np.log10,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "clip": np.clip,
        "maximum": np.maximum,
        "minimum": np.minimum,
        "pi": np.pi,
        "e": np.e,
    }
    old_copy = doc.data.copy()
    try:
        res = eval(expr_str, {"__builtins__": {}}, safe_dict)
        res_arr = np.asarray(res, dtype=doc.dtype)
        if res_arr.shape != doc.shape:
            raise ValueError(f"Expression output shape {res_arr.shape} does not match tensor shape {doc.shape}")

        modified_mask = np.where(old_copy != res_arr)
        cmd = MaskedTransformCommand(
            modified_indices=modified_mask,
            old_values=old_copy[modified_mask],
            new_values=res_arr[modified_mask],
            desc=f"Expression: {expr_str}",
        )
        doc.apply_command(cmd)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{expr_str}': {str(e)}")


def reshape_tensor(doc: NNEFTensorDocument, new_shape: Sequence[int]) -> None:
    """Reshape tensor with validation."""
    old_shape = doc.shape
    new_shape_tuple = tuple(int(x) for x in new_shape)
    old_count = int(np.prod(old_shape)) if len(old_shape) > 0 else 1
    new_count = int(np.prod(new_shape_tuple)) if len(new_shape_tuple) > 0 else 1
    if old_count != new_count:
        raise ValueError(f"Cannot reshape tensor with {old_count} elements into shape {new_shape_tuple} ({new_count} elements)")

    cmd = ReshapeCommand(old_shape=old_shape, new_shape=new_shape_tuple)
    doc.apply_command(cmd)


def cast_dtype(doc: NNEFTensorDocument, new_dtype: Union[np.dtype, str]) -> None:
    """Cast tensor dtype with backup."""
    target_dt = np.dtype(new_dtype)
    if doc.dtype == target_dt:
        return
    old_backup = doc.data.copy()
    cmd = DtypeCastCommand(old_dtype=doc.dtype, new_dtype=target_dt, old_data_backup=old_backup)
    doc.apply_command(cmd)
