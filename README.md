# NNEF Tensor Editor & Matrix Viewer

A high-performance, modular NNEF Binary Tensor Editor & Matrix Viewer application built with Python, PySide6, and NumPy.

## Features
- **Headless Core & Embeddable Qt Widgets**: Decoupled `nnef_viewer.core` (zero Qt dependencies) and embeddable `NNEFTensorViewerWidget` / `NNEFDiffViewerWidget`.
- **NNEF Binary Tensor Support**: Complete support for standard and quantized NNEF binary (`.dat`) tensors from Rank 0 to Rank 8.
- **JetBrains-Style Matrix Viewer**: Dynamic background color cell shading based on value distribution (Coolwarm, Viridis, Plasma, Jet, etc.), with automatic text contrast.
- **N-Dimensional Dynamic Slicing**: Dynamic axis mapping (assign any 2 axes to Row and Column) with smooth scrubbing for remaining dimensions.
- **Memory-Efficient Undo/Redo**: Delta-based command pattern prevents memory bloat even when editing multi-gigabyte tensors.
- **Non-Blocking Background Computations**: `QThreadPool` worker concurrency for statistics, histograms, and diff calculations.
- **Tensor Comparison & Diff**: Synchronized side-by-side view, tolerance thresholding (`atol`/`rtol`), metric calculations (MSE, MAE, Max Diff, Cosine Similarity), and "Jump to next mismatch" navigation.
- **Multi-Tab Workspace**: Tabbed interface with dirty indicators, drag-and-drop file opening, and modern Dark & Light IDE themes.
