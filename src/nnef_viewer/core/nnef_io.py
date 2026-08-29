# Copyright (c) 2026
# Fully compliant NNEF binary tensor I/O reader/writer and converter.

import io
import os
import sys
import tarfile
import zipfile
from typing import BinaryIO, List, Optional, Tuple, Union

import numpy as np


class ItemType:
    FLOAT = 0
    UINT = 1
    QUINT = 2
    QINT = 3
    INT = 4
    BOOL = 5


MaxTensorRank = 8
_is_little_endian = sys.byteorder == "little"


def _numpy_dtype_split(dtype: np.dtype) -> Tuple[int, int]:
    dt = np.dtype(dtype)
    splits = {
        np.dtype(np.float16): (ItemType.FLOAT, 16),
        np.dtype(np.float32): (ItemType.FLOAT, 32),
        np.dtype(np.float64): (ItemType.FLOAT, 64),
        np.dtype(np.int8): (ItemType.INT, 8),
        np.dtype(np.uint8): (ItemType.UINT, 8),
        np.dtype(np.int16): (ItemType.INT, 16),
        np.dtype(np.uint16): (ItemType.UINT, 16),
        np.dtype(np.int32): (ItemType.INT, 32),
        np.dtype(np.uint32): (ItemType.UINT, 32),
        np.dtype(np.int64): (ItemType.INT, 64),
        np.dtype(np.uint64): (ItemType.UINT, 64),
        np.dtype(np.bool_): (ItemType.BOOL, 1),
    }
    split = splits.get(dt)
    if split is None:
        raise TypeError(f"Unsupported tensor dtype: {dtype}")
    return split


def _numpy_dtype_make(item_type: int, bits: int) -> np.dtype:
    dtypes = {
        (ItemType.FLOAT, 16): np.float16,
        (ItemType.FLOAT, 32): np.float32,
        (ItemType.FLOAT, 64): np.float64,
        (ItemType.INT, 8): np.int8,
        (ItemType.INT, 16): np.int16,
        (ItemType.INT, 32): np.int32,
        (ItemType.INT, 64): np.int64,
        (ItemType.UINT, 8): np.uint8,
        (ItemType.UINT, 16): np.uint16,
        (ItemType.UINT, 32): np.uint32,
        (ItemType.UINT, 64): np.uint64,
        (ItemType.QINT, 8): np.int8,
        (ItemType.QINT, 16): np.int16,
        (ItemType.QINT, 32): np.int32,
        (ItemType.QINT, 64): np.int64,
        (ItemType.QUINT, 8): np.uint8,
        (ItemType.QUINT, 16): np.uint16,
        (ItemType.QUINT, 32): np.uint32,
        (ItemType.QUINT, 64): np.uint64,
        (ItemType.BOOL, 1): np.bool_,
    }
    dtype = dtypes.get((item_type, bits))
    if dtype is None:
        raise ValueError(f"Unsupported combination of item type ({item_type}) and bits ({bits})")
    return np.dtype(dtype)


def _tofile(data: np.ndarray, file: BinaryIO) -> None:
    if not _is_little_endian and data.dtype not in (np.uint8, np.int8, np.bool_):
        data = data.byteswap()
    # If it's a real OS file with a fileno, data.tofile works, otherwise use file.write(data.tobytes())
    try:
        if hasattr(file, "fileno"):
            file.fileno()  # Check if fileno is actually supported
            data.tofile(file)
            return
    except (io.UnsupportedOperation, AttributeError):
        pass
    file.write(data.tobytes())


def _fromfile(file: BinaryIO, dtype: np.dtype, count: int) -> np.ndarray:
    dt = np.dtype(dtype)
    if count == 0:
        return np.empty(0, dtype=dt)
    try:
        if hasattr(file, "fileno"):
            file.fileno()
            data = np.fromfile(file, dt, count)
            if not _is_little_endian and dt not in (np.uint8, np.int8, np.bool_):
                data = data.byteswap()
            return data
    except (io.UnsupportedOperation, AttributeError):
        pass

    raw_bytes = file.read(count * dt.itemsize)
    data = np.frombuffer(raw_bytes, dt, count)
    if not _is_little_endian and dt not in (np.uint8, np.int8, np.bool_):
        data = data.byteswap()
    return data


def write_nnef_tensor(
    file_or_path: Union[str, BinaryIO],
    tensor: np.ndarray,
    quantized: bool = False,
    version: Tuple[int, int] = (1, 0),
) -> None:
    """Write a NumPy array to an NNEF binary file according to Khronos NNEF specification."""
    if isinstance(file_or_path, str):
        with open(file_or_path, "wb") as f:
            _write_tensor_to_stream(f, tensor, quantized=quantized, version=version)
    else:
        _write_tensor_to_stream(file_or_path, tensor, quantized=quantized, version=version)


def _write_tensor_to_stream(
    file: BinaryIO,
    tensor: np.ndarray,
    quantized: bool = False,
    version: Tuple[int, int] = (1, 0),
) -> None:
    # 4 bytes magic: 0x4E, 0xEF, major, minor
    _tofile(np.asarray([0x4E, 0xEF, version[0], version[1]], dtype=np.uint8), file)

    item_type, bits = _numpy_dtype_split(tensor.dtype)
    if quantized:
        if item_type == ItemType.INT:
            item_type = ItemType.QINT
        elif item_type == ItemType.UINT:
            item_type = ItemType.QUINT
        else:
            raise ValueError(f"Invalid tensor dtype '{tensor.dtype}' for quantized tensor")

    count = int(np.prod(tensor.shape)) if tensor.ndim > 0 else 1
    data_length = (count + 7) // 8 if bits == 1 else count * (bits // 8)
    _tofile(np.asarray([data_length, tensor.ndim], dtype=np.uint32), file)

    if tensor.ndim > MaxTensorRank:
        raise ValueError(f"Tensor rank {tensor.ndim} exceeds maximum possible value of {MaxTensorRank}")

    # Write shape padded to MaxTensorRank (8)
    shape_pad = list(tensor.shape) + [0] * (MaxTensorRank - tensor.ndim)
    _tofile(np.asarray(shape_pad, dtype=np.uint32), file)

    # Write bits and item_type
    _tofile(np.asarray([bits, item_type], dtype=np.uint32), file)
    # Reserved 19 uint32 fields
    _tofile(np.asarray([0] * 19, dtype=np.uint32), file)

    # Write raw data
    data = np.packbits(tensor) if bits == 1 else tensor
    _tofile(data, file)


def read_nnef_tensor(
    file_or_path: Union[str, BinaryIO],
    return_quantization: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, bool]]:
    """Read an NNEF binary file and return the NumPy tensor."""
    if isinstance(file_or_path, str):
        with open(file_or_path, "rb") as f:
            return _read_tensor_from_stream(f, return_quantization=return_quantization)
    else:
        return _read_tensor_from_stream(file_or_path, return_quantization=return_quantization)


def _read_tensor_from_stream(
    file: BinaryIO,
    return_quantization: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, bool]]:
    magic = _fromfile(file, dtype=np.uint8, count=4)
    if len(magic) < 4 or magic[0] != 0x4E or magic[1] != 0xEF:
        raise ValueError("Not a valid NNEF binary file (magic header mismatch)")

    major, minor = int(magic[2]), int(magic[3])
    if major > 1 or minor > 0:
        raise ValueError(f"Unsupported NNEF file version: {major}.{minor}")

    meta = _fromfile(file, dtype=np.uint32, count=2)
    if len(meta) < 2:
        raise ValueError("Unexpected EOF while reading NNEF header metadata")
    data_length, rank = int(meta[0]), int(meta[1])

    try:
        if hasattr(file, "fileno"):
            header_size = 128
            file_size = os.fstat(file.fileno()).st_size
            if file_size < header_size + data_length:
                raise ValueError("Corrupted NNEF file: file size smaller than header specified data length")
    except (io.UnsupportedOperation, AttributeError):
        pass

    if rank > MaxTensorRank:
        raise ValueError(f"Tensor rank {rank} exceeds maximum possible value of {MaxTensorRank}")

    shape_raw = _fromfile(file, dtype=np.uint32, count=MaxTensorRank)
    shape = tuple(int(x) for x in shape_raw[:rank])

    type_info = _fromfile(file, dtype=np.uint32, count=2)
    if len(type_info) < 2:
        raise ValueError("Unexpected EOF while reading NNEF type info")
    bits, item_type = int(type_info[0]), int(type_info[1])

    reserved = _fromfile(file, dtype=np.uint32, count=19)
    if item_type == ItemType.UINT and len(reserved) > 0 and reserved[0] != 0:
        item_type = ItemType.INT

    quantized = item_type in (ItemType.QINT, ItemType.QUINT)
    count = int(np.prod(shape)) if len(shape) > 0 else 1

    if bits == 1:
        byte_count = int((count + 7) // 8)
        raw_bytes = _fromfile(file, dtype=np.uint8, count=byte_count)
        if len(raw_bytes) != byte_count:
            raise ValueError(f"Incomplete NNEF boolean tensor data: read {len(raw_bytes)} of {byte_count} bytes")
        data = np.unpackbits(raw_bytes).astype(bool)[:count]
    else:
        target_dtype = _numpy_dtype_make(item_type, bits)
        data = _fromfile(file, dtype=target_dtype, count=count)
        if len(data) != count:
            raise ValueError(f"Incomplete NNEF tensor data: read {len(data)} of {count} items")

    tensor = data.reshape(shape)
    return (tensor, quantized) if return_quantization else tensor


def list_tensors_in_archive_or_dir(path: str) -> List[Tuple[str, str]]:
    """
    List all tensor files (.dat) inside an NNEF directory or archive (.tar, .tgz, .zip, .nnef).
    Returns a list of (display_name, internal_or_absolute_path).
    """
    tensors = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in sorted(files):
                if file.endswith(".dat"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, path)
                    tensors.append((rel_path, full_path))
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as tar:
            for member in tar.getmembers():
                if member.name.endswith(".dat") and member.isfile():
                    tensors.append((member.name, f"tar://{path}#{member.name}"))
    elif zipfile.is_zipfile(path):
        with zipfile.open(path, "r") as zf:
            for name in zf.namelist():
                if name.endswith(".dat"):
                    tensors.append((name, f"zip://{path}#{name}"))
    return tensors


def read_tensor_from_location(location: str) -> Tuple[np.ndarray, bool]:
    """Read a tensor from a direct path or an archive URI (tar:// or zip://)."""
    if location.startswith("tar://"):
        tar_path, member_name = location[6:].split("#", 1)
        with tarfile.open(tar_path, "r:*") as tar:
            member = tar.getmember(member_name)
            f = tar.extractfile(member)
            if f is None:
                raise IOError(f"Could not extract {member_name} from {tar_path}")
            return read_nnef_tensor(f, return_quantization=True)
    elif location.startswith("zip://"):
        zip_path, member_name = location[6:].split("#", 1)
        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(member_name, "r") as f:
                # Read all bytes into BytesIO so it has seek/buffer behavior
                buf = io.BytesIO(f.read())
                return read_nnef_tensor(buf, return_quantization=True)
    else:
        return read_nnef_tensor(location, return_quantization=True)


def export_tensor(tensor: np.ndarray, export_path: str, delimiter: str = ",") -> None:
    """Export a tensor to various formats: .npy, .npz, .csv, .tsv, .txt, or NNEF .dat."""
    ext = os.path.splitext(export_path)[1].lower()
    if ext == ".npy":
        np.save(export_path, tensor)
    elif ext == ".npz":
        np.savez_compressed(export_path, tensor=tensor)
    elif ext in (".csv", ".tsv", ".txt"):
        sep = "\t" if ext == ".tsv" else delimiter
        # If > 2D, flatten to 2D for CSV export
        arr = tensor
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim == 0:
            arr = arr.reshape(1, 1)

        fmt = "%d" if np.issubdtype(tensor.dtype, np.integer) else "%.8g"
        np.savetxt(export_path, arr, delimiter=sep, fmt=fmt)
    elif ext == ".dat":
        write_nnef_tensor(export_path, tensor)
    else:
        raise ValueError(f"Unsupported export format: {ext}")


def import_tensor(import_path: str, delimiter: str = ",") -> np.ndarray:
    """Import a tensor from .npy, .npz, .csv, .tsv, or .dat."""
    ext = os.path.splitext(import_path)[1].lower()
    if ext == ".npy":
        return np.load(import_path)
    elif ext == ".npz":
        data = np.load(import_path)
        # return first array
        key = list(data.keys())[0]
        return data[key]
    elif ext in (".csv", ".tsv", ".txt"):
        sep = "\t" if ext == ".tsv" else delimiter
        return np.loadtxt(import_path, delimiter=sep)
    elif ext == ".dat":
        return read_nnef_tensor(import_path, return_quantization=False)
    else:
        raise ValueError(f"Unsupported import format: {ext}")
