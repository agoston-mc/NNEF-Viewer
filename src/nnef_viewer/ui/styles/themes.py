# Copyright (c) 2026
# Modern Dark and Light stylesheets for PySide6 application (Sleek & Professional).

DARK_THEME_QSS = """
/* Modern JetBrains / CLion / VS Code inspired Dark Theme */
QWidget {
    background-color: #1e1f22;
    color: #bcbec4;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 12px;
    selection-background-color: #2e436e;
    selection-color: #ffffff;
}

QMainWindow, QDialog {
    background-color: #1e1f22;
}

QMenuBar {
    background-color: #2b2d30;
    color: #dfe1e5;
    padding: 2px 6px;
    border-bottom: 1px solid #393b40;
}

QMenuBar::item {
    padding: 4px 8px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #393b40;
}

QMenu {
    background-color: #2b2d30;
    color: #dfe1e5;
    border: 1px solid #393b40;
    padding: 4px;
    border-radius: 6px;
}

QMenu::item {
    padding: 5px 24px 5px 12px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #3574f0;
    color: #ffffff;
}

QMenu::separator {
    height: 1px;
    background-color: #393b40;
    margin: 4px 6px;
}

QToolBar {
    background-color: #2b2d30;
    border: 1px solid #393b40;
    border-radius: 6px;
    spacing: 4px;
    padding: 2px 6px;
}

QToolButton {
    background-color: transparent;
    color: #dfe1e5;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 3px 8px;
    font-weight: 500;
}

QToolButton:hover {
    background-color: #393b40;
    border-color: #4e5157;
}

QToolButton:pressed {
    background-color: #2e436e;
}

QTabWidget::pane {
    border: 1px solid #393b40;
    background-color: #1e1f22;
}

QTabBar::tab {
    background-color: #2b2d30;
    color: #9da0a8;
    padding: 5px 14px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border: 1px solid #393b40;
    border-bottom: none;
}

QTabBar::tab:selected {
    background-color: #1e1f22;
    color: #dfe1e5;
    font-weight: 600;
    border-top: 2px solid #3574f0;
}

QTabBar::tab:hover:!selected {
    background-color: #393b40;
    color: #dfe1e5;
}

QTableView {
    background-color: #1e1f22;
    alternate-background-color: #232528;
    gridline-color: #32353a;
    border: 1px solid #393b40;
    selection-background-color: rgba(53, 116, 240, 0.45);
    selection-color: #ffffff;
    font-family: "JetBrains Mono", Menlo, Monaco, Consolas, "Courier New", monospace;
    font-size: 11px;
}

QHeaderView::section {
    background-color: #26282b;
    color: #8c909a;
    padding: 4px 6px;
    border: 1px solid #32353a;
    font-weight: 600;
    font-size: 10.5px;
    font-family: "JetBrains Mono", monospace;
}

QHeaderView::section:checked {
    background-color: #3574f0;
    color: #ffffff;
}

QScrollBar:vertical {
    background-color: #1e1f22;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #4e5157;
    min-height: 20px;
    border-radius: 5px;
    margin: 2px;
}

QScrollBar::handle:vertical:hover {
    background-color: #6b6e77;
}

QScrollBar:horizontal {
    background-color: #1e1f22;
    height: 10px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background-color: #4e5157;
    min-width: 20px;
    border-radius: 5px;
    margin: 2px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #6b6e77;
}

QScrollBar::add-line, QScrollBar::sub-line {
    width: 0px;
    height: 0px;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #26282b;
    color: #dfe1e5;
    border: 1px solid #43454b;
    border-radius: 4px;
    padding: 3px 6px;
    min-height: 20px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #3574f0;
}

QComboBox::drop-down {
    border: none;
    padding-right: 4px;
}

QSlider::groove:horizontal {
    border: 1px solid #33363a;
    height: 4px;
    background: #25272a;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #3574f0;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #dfe1e5;
    border: 1px solid #4e5157;
    width: 12px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 6px;
}

QSlider::handle:horizontal:hover {
    background: #ffffff;
    border-color: #3574f0;
}

QStatusBar {
    background-color: #26282b;
    color: #8c909a;
    border-top: 1px solid #33363a;
    font-size: 11px;
}

QGroupBox {
    border: 1px solid #393b40;
    border-radius: 6px;
    margin-top: 8px;
    padding-top: 10px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 4px;
    color: #dfe1e5;
    font-size: 11px;
}

QPushButton {
    background-color: #2b2d30;
    color: #dfe1e5;
    border: 1px solid #43454b;
    border-radius: 4px;
    padding: 3px 10px;
    font-weight: 500;
    min-height: 20px;
}

QPushButton:hover {
    background-color: #35383d;
    border-color: #555861;
}

QPushButton:pressed {
    background-color: #25272a;
}

QPushButton:checked {
    background-color: #3574f0;
    color: #ffffff;
    border-color: #3574f0;
}

QPushButton:disabled {
    background-color: #232528;
    color: #5a5d66;
    border-color: #33363a;
}

QSplitter::handle {
    background-color: #2b2d30;
}

QSplitter::handle:horizontal {
    width: 3px;
}

QSplitter::handle:vertical {
    height: 3px;
}

QSplitter::handle:hover {
    background-color: #3574f0;
}
"""

LIGHT_THEME_QSS = """
/* Clean Modern Light Theme */
QWidget {
    background-color: #ffffff;
    color: #1f2328;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 12px;
    selection-background-color: #d0e2ff;
    selection-color: #000000;
}

QMainWindow, QDialog {
    background-color: #f6f8fa;
}

QMenuBar {
    background-color: #ffffff;
    color: #1f2328;
    padding: 2px 6px;
    border-bottom: 1px solid #d0d7de;
}

QMenuBar::item:selected {
    background-color: #eaeef2;
    border-radius: 4px;
}

QMenu {
    background-color: #ffffff;
    color: #1f2328;
    border: 1px solid #d0d7de;
    padding: 4px;
    border-radius: 6px;
}

QMenu::item:selected {
    background-color: #0969da;
    color: #ffffff;
}

QToolBar {
    background-color: #ffffff;
    border-bottom: 1px solid #d0d7de;
    spacing: 4px;
    padding: 2px 6px;
}

QTabWidget::pane {
    border: 1px solid #d0d7de;
    background-color: #ffffff;
}

QTabBar::tab {
    background-color: #f6f8fa;
    color: #656d76;
    padding: 5px 14px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border: 1px solid #d0d7de;
    border-bottom: none;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    color: #1f2328;
    font-weight: 600;
    border-top: 2px solid #0969da;
}

QTableView {
    background-color: #ffffff;
    alternate-background-color: #f6f8fa;
    gridline-color: #e1e4e8;
    border: 1px solid #d0d7de;
    selection-background-color: rgba(9, 105, 218, 0.25);
    selection-color: #000000;
    font-family: "JetBrains Mono", Menlo, Monaco, Consolas, monospace;
    font-size: 11px;
}

QHeaderView::section {
    background-color: #f6f8fa;
    color: #656d76;
    padding: 4px 6px;
    border: 1px solid #d0d7de;
    font-weight: 600;
    font-size: 10.5px;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #ffffff;
    color: #1f2328;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    padding: 3px 6px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #0969da;
}

QPushButton {
    background-color: #f6f8fa;
    color: #1f2328;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    padding: 3px 10px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #eaeef2;
}

QPushButton:checked {
    background-color: #0969da;
    color: #ffffff;
    border-color: #0969da;
}

QStatusBar {
    background-color: #f6f8fa;
    color: #656d76;
    border-top: 1px solid #d0d7de;
}

QSplitter::handle {
    background-color: #e1e4e8;
}

QSplitter::handle:hover {
    background-color: #0969da;
}
"""
