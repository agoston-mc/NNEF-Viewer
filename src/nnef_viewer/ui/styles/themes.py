# Copyright (c) 2026
# Modern Dark and Light stylesheets for PySide6 application.

DARK_THEME_QSS = """
/* Modern JetBrains / VS Code inspired Dark Theme */
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

QMenuBar::item:selected {
    background-color: #393b40;
    border-radius: 4px;
}

QMenu {
    background-color: #2b2d30;
    color: #dfe1e5;
    border: 1px solid #393b40;
    padding: 4px;
    border-radius: 6px;
}

QMenu::item {
    padding: 6px 24px 6px 12px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #3574f0;
    color: #ffffff;
}

QToolBar {
    background-color: #2b2d30;
    border-bottom: 1px solid #393b40;
    spacing: 6px;
    padding: 4px 8px;
}

QToolButton {
    background-color: transparent;
    color: #dfe1e5;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px 8px;
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
    padding: 6px 14px;
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
    alternate-background-color: #25272a;
    gridline-color: #393b40;
    border: 1px solid #393b40;
    selection-background-color: rgba(53, 116, 240, 0.45);
    selection-color: #ffffff;
    font-family: "JetBrains Mono", Menlo, Monaco, Consolas, "Courier New", monospace;
    font-size: 11px;
}

QHeaderView::section {
    background-color: #2b2d30;
    color: #9da0a8;
    padding: 4px 8px;
    border: 1px solid #393b40;
    font-weight: 600;
    font-size: 11px;
}

QHeaderView::section:checked {
    background-color: #393b40;
    color: #3574f0;
}

QScrollBar:vertical {
    background-color: #1e1f22;
    width: 12px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #4e5157;
    min-height: 20px;
    border-radius: 6px;
    margin: 2px;
}

QScrollBar::handle:vertical:hover {
    background-color: #6b6e77;
}

QScrollBar:horizontal {
    background-color: #1e1f22;
    height: 12px;
    margin: 0px;
}

QScrollBar::handle:horizontal {
    background-color: #4e5157;
    min-width: 20px;
    border-radius: 6px;
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
    background-color: #2b2d30;
    color: #dfe1e5;
    border: 1px solid #4e5157;
    border-radius: 4px;
    padding: 4px 8px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #3574f0;
}

QComboBox::drop-down {
    border: none;
    padding-right: 4px;
}

QSlider::groove:horizontal {
    border: 1px solid #393b40;
    height: 6px;
    background: #2b2d30;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #3574f0;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #dfe1e5;
    border: 1px solid #4e5157;
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #ffffff;
    border-color: #3574f0;
}

QStatusBar {
    background-color: #2b2d30;
    color: #9da0a8;
    border-top: 1px solid #393b40;
}

QGroupBox {
    border: 1px solid #393b40;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: #dfe1e5;
}

QPushButton {
    background-color: #3574f0;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #3065cf;
}

QPushButton:pressed {
    background-color: #2651a8;
}

QPushButton:disabled {
    background-color: #393b40;
    color: #6b6e77;
}

QDockWidget {
    color: #dfe1e5;
    titlebar-close-icon: url(close.png);
    titlebar-normal-icon: url(float.png);
}

QDockWidget::title {
    background-color: #2b2d30;
    padding: 6px;
    border-bottom: 1px solid #393b40;
    font-weight: 600;
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
    spacing: 6px;
    padding: 4px 8px;
}

QToolButton {
    background-color: transparent;
    color: #1f2328;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px 8px;
    font-weight: 500;
}

QToolButton:hover {
    background-color: #eaeef2;
    border-color: #d0d7de;
}

QTabWidget::pane {
    border: 1px solid #d0d7de;
    background-color: #ffffff;
}

QTabBar::tab {
    background-color: #f6f8fa;
    color: #656d76;
    padding: 6px 14px;
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
    padding: 4px 8px;
    border: 1px solid #d0d7de;
    font-weight: 600;
    font-size: 11px;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #ffffff;
    color: #1f2328;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    padding: 4px 8px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #0969da;
}

QPushButton {
    background-color: #0969da;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #085cc0;
}

QStatusBar {
    background-color: #f6f8fa;
    color: #656d76;
    border-top: 1px solid #d0d7de;
}
"""
