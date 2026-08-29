# Copyright (c) 2026
# High-performance, memory-efficient NNEF Tensor Document Model with Delta-based Undo/Redo

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class Command(ABC):
    """Abstract base command for non-destructive, memory-efficient undo/redo."""

    @abstractmethod
    def execute(self, doc: "NNEFTensorDocument") -> None:
        """Apply modification to the tensor document."""
        pass

    @abstractmethod
    def undo(self, doc: "NNEFTensorDocument") -> None:
        """Revert modification in the tensor document."""
        pass

    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the operation."""
        pass


class CellEditCommand(Command):
    """Stores a single element delta: coordinates, old value, new value."""

    def __init__(self, indices: Tuple[int, ...], old_value: Any, new_value: Any):
        self.indices = tuple(int(i) for i in indices)
        self.old_value = old_value
        self.new_value = new_value

    def execute(self, doc: "NNEFTensorDocument") -> None:
        doc.data[self.indices] = self.new_value

    def undo(self, doc: "NNEFTensorDocument") -> None:
        doc.data[self.indices] = self.old_value

    def description(self) -> str:
        return f"Edit cell {self.indices} to {self.new_value}"


class SliceEditCommand(Command):
    """Stores delta only for the modified sub-region slice (not whole tensor)."""

    def __init__(self, slice_tuple: Tuple[Union[int, slice], ...], old_slice_copy: np.ndarray, new_slice_data: np.ndarray):
        self.slice_tuple = slice_tuple
        self.old_slice_copy = old_slice_copy.copy()
        self.new_slice_data = new_slice_data.copy()

    def execute(self, doc: "NNEFTensorDocument") -> None:
        doc.data[self.slice_tuple] = self.new_slice_data

    def undo(self, doc: "NNEFTensorDocument") -> None:
        doc.data[self.slice_tuple] = self.old_slice_copy

    def description(self) -> str:
        return "Edit slice region"


class InvertibleTransformCommand(Command):
    """
    Applies an algebraically invertible transform (e.g. + delta, * factor, negate)
    without allocating extra memory for history.
    """

    def __init__(self, op_name: str, forward_fn: Callable[[np.ndarray], None], inverse_fn: Callable[[np.ndarray], None], desc: str):
        self.op_name = op_name
        self.forward_fn = forward_fn
        self.inverse_fn = inverse_fn
        self.desc = desc

    def execute(self, doc: "NNEFTensorDocument") -> None:
        self.forward_fn(doc.data)

    def undo(self, doc: "NNEFTensorDocument") -> None:
        self.inverse_fn(doc.data)

    def description(self) -> str:
        return self.desc


class MaskedTransformCommand(Command):
    """
    For non-invertible batch operations (like clamp, threshold, or selective assignment),
    only stores the indices and original values of changed elements.
    """

    def __init__(self, modified_indices: Tuple[np.ndarray, ...], old_values: np.ndarray, new_values: Union[np.ndarray, Any], desc: str):
        self.modified_indices = modified_indices
        self.old_values = old_values
        self.new_values = new_values
        self.desc = desc

    def execute(self, doc: "NNEFTensorDocument") -> None:
        doc.data[self.modified_indices] = self.new_values

    def undo(self, doc: "NNEFTensorDocument") -> None:
        doc.data[self.modified_indices] = self.old_values

    def description(self) -> str:
        return self.desc


class ReshapeCommand(Command):
    """Changes tensor shape without modifying data elements."""

    def __init__(self, old_shape: Tuple[int, ...], new_shape: Tuple[int, ...]):
        self.old_shape = old_shape
        self.new_shape = new_shape

    def execute(self, doc: "NNEFTensorDocument") -> None:
        doc.data = doc.data.reshape(self.new_shape)

    def undo(self, doc: "NNEFTensorDocument") -> None:
        doc.data = doc.data.reshape(self.old_shape)

    def description(self) -> str:
        return f"Reshape from {self.old_shape} to {self.new_shape}"


class DtypeCastCommand(Command):
    """Casts tensor dtype."""

    def __init__(self, old_dtype: np.dtype, new_dtype: np.dtype, old_data_backup: Optional[np.ndarray] = None):
        self.old_dtype = old_dtype
        self.new_dtype = new_dtype
        # If casting might be lossy (e.g. float to int), keep old copy
        self.old_data_backup = old_data_backup

    def execute(self, doc: "NNEFTensorDocument") -> None:
        doc.data = doc.data.astype(self.new_dtype)

    def undo(self, doc: "NNEFTensorDocument") -> None:
        if self.old_data_backup is not None:
            doc.data = self.old_data_backup.copy()
        else:
            doc.data = doc.data.astype(self.old_dtype)

    def description(self) -> str:
        return f"Cast dtype from {self.old_dtype} to {self.new_dtype}"


class NNEFTensorDocument:
    """
    Encapsulates a multi-dimensional tensor, its metadata, and an undo/redo stack.
    Pure Python & NumPy with zero UI dependencies.
    """

    def __init__(
        self,
        data: np.ndarray,
        file_path: Optional[str] = None,
        display_name: str = "Untitled",
        is_quantized: bool = False,
        max_history: int = 50,
    ):
        self._data = np.asarray(data)
        self.file_path = file_path
        self.display_name = display_name
        self.is_quantized = is_quantized
        self.max_history = max_history

        self._is_dirty = False
        self._undo_stack: List[Command] = []
        self._redo_stack: List[Command] = []
        self._saved_undo_index: Optional[int] = 0

        # Change listeners
        self._data_change_callbacks: List[Callable[[], None]] = []
        self._structure_change_callbacks: List[Callable[[], None]] = []
        self._dirty_change_callbacks: List[Callable[[bool], None]] = []

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        shape_changed = self._data.shape != new_data.shape or self._data.dtype != new_data.dtype
        self._data = new_data
        if shape_changed:
            self._notify_structure_changed()
        self._notify_data_changed()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    @property
    def is_dirty(self) -> bool:
        return self._is_dirty

    @is_dirty.setter
    def is_dirty(self, value: bool) -> None:
        if self._is_dirty != value:
            self._is_dirty = value
            self._notify_dirty_changed(value)

    # ---------------- Undo / Redo Management ----------------

    def apply_command(self, cmd: Command) -> None:
        """Execute a command, add to undo history, and clear redo stack."""
        cmd.execute(self)
        self._undo_stack.append(cmd)
        if len(self._undo_stack) > self.max_history:
            self._undo_stack.pop(0)
            if self._saved_undo_index is not None:
                self._saved_undo_index -= 1
        self._redo_stack.clear()
        self._update_dirty_state()
        self._notify_data_changed()

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def undo(self) -> Optional[Command]:
        if not self.can_undo():
            return None
        cmd = self._undo_stack.pop()
        cmd.undo(self)
        self._redo_stack.append(cmd)
        self._update_dirty_state()
        self._notify_data_changed()
        return cmd

    def redo(self) -> Optional[Command]:
        if not self.can_redo():
            return None
        cmd = self._redo_stack.pop()
        cmd.execute(self)
        self._undo_stack.append(cmd)
        self._update_dirty_state()
        self._notify_data_changed()
        return cmd

    def mark_saved(self, new_file_path: Optional[str] = None) -> None:
        if new_file_path:
            self.file_path = new_file_path
            self.display_name = new_file_path.split("/")[-1]
        self._saved_undo_index = len(self._undo_stack)
        self.is_dirty = False

    def _update_dirty_state(self) -> None:
        current_index = len(self._undo_stack)
        self.is_dirty = (self._saved_undo_index != current_index)

    # ---------------- Data Access & Slicing ----------------

    def get_2d_slice(
        self,
        row_axis: int,
        col_axis: int,
        slice_indices: Sequence[int],
    ) -> np.ndarray:
        """
        Extract a 2D slice from the N-D tensor.
        - row_axis: which tensor axis maps to matrix rows.
        - col_axis: which tensor axis maps to matrix columns.
        - slice_indices: list/tuple of fixed index values for all other N-2 axes.
        """
        ndim = self.ndim
        if ndim == 0:
            return self._data.reshape(1, 1)
        if ndim == 1:
            if row_axis == 0:
                return self._data.reshape(-1, 1)
            else:
                return self._data.reshape(1, -1)

        # Normalize axes
        row_axis = row_axis % ndim
        col_axis = col_axis % ndim
        if row_axis == col_axis:
            raise ValueError("Row axis and Column axis must be distinct")

        # Build full slicing spec
        slice_spec: List[Union[slice, int]] = []
        other_idx = 0
        for axis in range(ndim):
            if axis == row_axis or axis == col_axis:
                slice_spec.append(slice(None))
            else:
                idx = slice_indices[other_idx] if other_idx < len(slice_indices) else 0
                max_len = self.shape[axis]
                idx = max(0, min(idx, max_len - 1)) if max_len > 0 else 0
                slice_spec.append(idx)
                other_idx += 1

        sub_array = self._data[tuple(slice_spec)]

        # Transpose so that row_axis is first (dim 0) and col_axis is second (dim 1)
        if row_axis > col_axis:
            sub_array = np.transpose(sub_array)

        return sub_array

    def set_cell_value(self, full_indices: Tuple[int, ...], new_val: Any) -> None:
        """Set a single element with memory-efficient delta undo."""
        indices = tuple(int(i) for i in full_indices)
        old_val = self._data[indices]
        try:
            casted_val = np.dtype(self.dtype).type(new_val)
        except Exception:
            casted_val = new_val

        cmd = CellEditCommand(indices, old_val, casted_val)
        self.apply_command(cmd)

    def set_2d_slice_cell_value(
        self,
        row_axis: int,
        col_axis: int,
        slice_indices: Sequence[int],
        row: int,
        col: int,
        new_val: Any,
    ) -> None:
        """Compute the N-D full index from (row_axis, col_axis, slice_indices, row, col) and edit cell."""
        ndim = self.ndim
        if ndim == 0:
            full_indices: Tuple[int, ...] = ()
        elif ndim == 1:
            full_indices = (row if row_axis == 0 else col,)
        else:
            row_axis = row_axis % ndim
            col_axis = col_axis % ndim
            full_idx_list = [0] * ndim
            full_idx_list[row_axis] = row
            full_idx_list[col_axis] = col
            other_idx = 0
            for axis in range(ndim):
                if axis != row_axis and axis != col_axis:
                    idx = slice_indices[other_idx] if other_idx < len(slice_indices) else 0
                    full_idx_list[axis] = max(0, min(idx, self.shape[axis] - 1))
                    other_idx += 1
            full_indices = tuple(full_idx_list)

        self.set_cell_value(full_indices, new_val)

    # ---------------- Change Listeners ----------------

    def add_data_change_listener(self, cb: Callable[[], None]) -> None:
        self._data_change_callbacks.append(cb)

    def add_structure_change_listener(self, cb: Callable[[], None]) -> None:
        self._structure_change_callbacks.append(cb)

    def add_dirty_change_listener(self, cb: Callable[[bool], None]) -> None:
        self._dirty_change_callbacks.append(cb)

    def _notify_data_changed(self) -> None:
        for cb in self._data_change_callbacks:
            try:
                cb()
            except Exception:
                pass

    def _notify_structure_changed(self) -> None:
        for cb in self._structure_change_callbacks:
            try:
                cb()
            except Exception:
                pass

    def _notify_dirty_changed(self, dirty: bool) -> None:
        for cb in self._dirty_change_callbacks:
            try:
                cb(dirty)
            except Exception:
                pass
