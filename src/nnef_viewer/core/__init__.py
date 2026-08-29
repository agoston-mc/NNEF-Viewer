"""Core headless data and computation layer for NNEF tensors (Zero Qt dependencies)."""

from .nnef_io import read_nnef_tensor, write_nnef_tensor, export_tensor, import_tensor, ItemType
from .tensor_model import NNEFTensorDocument, CellEditCommand, SliceEditCommand, InvertibleTransformCommand, MaskedTransformCommand, ReshapeCommand
from .operations import apply_math_transform, apply_expression, cast_dtype, reshape_tensor, generate_initial_data
from .stats import compute_tensor_stats, compute_histogram
from .comparison import compare_tensors, DiffResult, MismatchIterator

__all__ = [
    "read_nnef_tensor",
    "write_nnef_tensor",
    "export_tensor",
    "import_tensor",
    "ItemType",
    "NNEFTensorDocument",
    "CellEditCommand",
    "SliceEditCommand",
    "InvertibleTransformCommand",
    "MaskedTransformCommand",
    "ReshapeCommand",
    "apply_math_transform",
    "apply_expression",
    "cast_dtype",
    "reshape_tensor",
    "generate_initial_data",
    "compute_tensor_stats",
    "compute_histogram",
    "compare_tensors",
    "DiffResult",
    "MismatchIterator",
]
