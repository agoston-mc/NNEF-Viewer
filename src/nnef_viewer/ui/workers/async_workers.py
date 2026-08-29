# Copyright (c) 2026
# Concurrency & background workers using QThreadPool to prevent UI thread blocking.

from typing import Any, Callable, Optional, Tuple
import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from ...core.stats import compute_tensor_stats, compute_histogram, TensorStats
from ...core.comparison import compare_tensors, DiffResult


class WorkerSignals(QObject):
    """Signals emitted by background computation workers."""
    started = Signal()
    progress = Signal(int)
    finished = Signal(object)
    error = Signal(str)


class AsyncWorker(QRunnable):
    """General-purpose QRunnable that dispatches arbitrary functions to QThreadPool."""

    def __init__(self, fn: Callable[..., Any], *args: Any, **kwargs: Any):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        self.signals.started.emit()
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))


class StatsComputeWorker(QRunnable):
    """Asynchronous worker for whole-tensor statistics and histogram binning."""

    def __init__(self, arr: np.ndarray, num_bins: int = 40):
        super().__init__()
        self.arr = arr
        self.num_bins = num_bins
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        self.signals.started.emit()
        try:
            stats = compute_tensor_stats(self.arr)
            hist = compute_histogram(self.arr, num_bins=self.num_bins)
            self.signals.finished.emit((stats, hist))
        except Exception as e:
            self.signals.error.emit(str(e))


class DiffComputeWorker(QRunnable):
    """Asynchronous worker for comparing two massive tensors and computing diff metrics."""

    def __init__(self, arr_a: np.ndarray, arr_b: np.ndarray, atol: float = 1e-5, rtol: float = 1e-5):
        super().__init__()
        self.arr_a = arr_a
        self.arr_b = arr_b
        self.atol = atol
        self.rtol = rtol
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        self.signals.started.emit()
        try:
            diff_res = compare_tensors(self.arr_a, self.arr_b, atol=self.atol, rtol=self.rtol)
            self.signals.finished.emit(diff_res)
        except Exception as e:
            self.signals.error.emit(str(e))


def run_in_background(
    fn: Callable[..., Any],
    *args: Any,
    on_finished: Optional[Callable[[Any], None]] = None,
    on_error: Optional[Callable[[str], None]] = None,
    on_started: Optional[Callable[[], None]] = None,
    **kwargs: Any,
) -> AsyncWorker:
    """Convenience helper to dispatch a task to global QThreadPool with connected signal callbacks."""
    worker = AsyncWorker(fn, *args, **kwargs)
    if on_started:
        worker.signals.started.connect(on_started)
    if on_finished:
        worker.signals.finished.connect(on_finished)
    if on_error:
        worker.signals.error.connect(on_error)
    QThreadPool.globalInstance().start(worker)
    return worker
