# Copyright (c) 2026
# Entry point for NNEF Tensor Editor and Matrix Viewer.

import argparse
import sys
from typing import List, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from .core.nnef_io import read_tensor_from_location
from .core.tensor_model import NNEFTensorDocument
from .ui.main_window import MainWindow


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="NNEF Binary Tensor Editor & Matrix Viewer")
    parser.add_argument("files", nargs="*", help="NNEF binary tensor files (.dat), numpy files (.npy), or directories to open")
    parser.add_argument("--diff", nargs=2, metavar=("FILE_A", "FILE_B"), help="Compare two tensor files in diff mode")
    args = parser.parse_args(argv)

    # Enable High DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("NNEF Tensor Viewer")
    app.setOrganizationName("Khronos NNEF")

    window = MainWindow(initial_files=args.files)

    if args.diff:
        file_a, file_b = args.diff
        try:
            arr_a, is_q_a = read_tensor_from_location(file_a)
            arr_b, is_q_b = read_tensor_from_location(file_b)
            doc_a = NNEFTensorDocument(arr_a, file_path=file_a, display_name=file_a.split("/")[-1], is_quantized=is_q_a)
            doc_b = NNEFTensorDocument(arr_b, file_path=file_b, display_name=file_b.split("/")[-1], is_quantized=is_q_b)
            window.add_diff_tab(doc_a, doc_b)
        except Exception as e:
            print(f"Error loading diff files: {e}", file=sys.stderr)

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
