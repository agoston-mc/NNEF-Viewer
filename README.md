# NNEF Tensor Viewer & Editor

A desktop GUI and widget library for viewing, editing, and comparing NNEF binary tensors (`.dat`) and NumPy arrays (`.npy`), built with PySide6 and NumPy.

> **Note**: This project is vibecoded. While it includes solid test coverage (94+ passing tests) and handles standard inspection/editing workflows, you may run into occasional AI-generated architectural quirks or edge cases.

## Features

- **2D Matrix Viewer**: Dynamic cell shading using standard colormaps (Viridis, Coolwarm, Plasma, Jet, Grayscale) with automatic high-contrast text rendering.
- **N-Dimensional Slicing**: Dynamic axis mapping (assign any 2 dimensions to rows/columns) and sliders to scrub through remaining dimensions (Rank 0 to Rank 8+).
- **Editing & Transforms**: In-place cell edits and batch transforms (Scale, Offset, Clamp, Normalize, Abs, Negate, and custom Python expressions).
- **Delta-Based Undo/Redo**: Memory-efficient command stack that records changes without cloning entire tensors.
- **Side-by-Side Diff**: Synchronized comparison view with configurable tolerances (`atol`, `rtol`), mismatch metrics (MSE, MAE, Max Diff, Cosine Similarity), and mismatch navigation.
- **Statistics & Histograms**: Computes min, max, mean, std, zero fraction, NaN/Inf checks, and live histograms in a background thread.
- **Decoupled Architecture**: Headless core (`nnef_viewer.core`) with zero Qt dependencies for scripting, alongside embeddable PySide6 widgets (`NNEFTensorViewerWidget`, `NNEFDiffViewerWidget`).
- **Multi-Tab Interface**: Tabbed workspace with dirty state tracking, drag-and-drop support, and dark/light themes.

## Installation

Requirements: Python >= 3.9

### Using `uv`

```bash
git clone https://github.com/your-org/NNEF-Viewer.git
cd NNEF-Viewer
uv sync
uv run nnef-viewer
```

### Using `pip`

```bash
pip install -e .
nnef-viewer
```

## Usage

### Command Line Interface

```bash
# Launch an empty workspace
nnef-viewer

# Open specific files
nnef-viewer tensor1.dat weights.npy

# Open in side-by-side comparison mode
nnef-viewer --diff baseline.dat candidate.dat
```

### Python API

#### Headless Core (Scripting)

```python
from nnef_viewer.core import (
    read_nnef_tensor,
    write_nnef_tensor,
    NNEFTensorDocument,
    compute_tensor_stats,
    compare_tensors,
)

# Load tensor
array, is_quantized = read_nnef_tensor("weights.dat")
doc = NNEFTensorDocument(array, file_path="weights.dat")

# Edit and undo
doc.edit_cell(coord=(0, 2, 4), new_value=0.42)
doc.apply_transform("scale", factor=2.0)
doc.undo()

# Compute stats
stats = compute_tensor_stats(doc.array)
print(f"Mean: {stats.mean:.4f}, Sparsity: {stats.zero_fraction * 100:.2f}%")

# Compare against another array
diff = compare_tensors(doc.array, array, atol=1e-4, rtol=1e-3)
print(f"MSE: {diff.mse}, Mismatches: {diff.mismatch_count}")

# Save
write_nnef_tensor("modified.dat", doc.array, is_quantized=is_quantized)
```

#### Embeddable Qt Widget

```python
import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow
from nnef_viewer.core import NNEFTensorDocument
from nnef_viewer.ui import NNEFTensorViewerWidget

app = QApplication(sys.argv)
window = QMainWindow()

data = np.random.randn(4, 16, 16).astype(np.float32)
doc = NNEFTensorDocument(data, display_name="Layer1_Weights")

viewer = NNEFTensorViewerWidget(doc)
window.setCentralWidget(viewer)
window.resize(1000, 700)
window.show()

sys.exit(app.exec())
```

## Running Tests

```bash
uv run pytest
# or
pytest
```

## Project Structure

```
NNEF-Viewer/
├── src/
│   └── nnef_viewer/
│       ├── core/                      # Headless logic (no Qt dependency)
│       │   ├── nnef_io.py             # File I/O (.dat, .npy)
│       │   ├── tensor_model.py        # Document model & undo/redo commands
│       │   ├── operations.py          # Math transforms & expressions
│       │   ├── stats.py               # Statistics & histograms
│       │   ├── comparison.py          # Tensor diff utilities
│       │   └── settings.py            # User settings
│       ├── ui/                        # PySide6 interface
│       │   ├── colormap.py            # Colormaps & text contrast
│       │   ├── main_window.py         # Main window
│       │   ├── widgets/               # Slicer, matrix view, diff view, stats
│       │   ├── dialogs/               # Transform & settings dialogs
│       │   ├── models/                # QAbstractTableModel implementation
│       │   └── styles/                # Themes
│       └── app.py                     # CLI entry point
└── tests/                             # Unit and UI tests
```
