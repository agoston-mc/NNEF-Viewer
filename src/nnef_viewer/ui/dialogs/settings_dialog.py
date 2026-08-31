# Copyright (c) 2026
# Application Settings & Preferences Dialog.

from typing import Optional
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.settings import AppSettings
from ..colormap import AVAILABLE_COLORMAPS


class SettingsDialog(QDialog):
    """Dialog for configuring and viewing persistent application preferences."""

    settingsChanged = Signal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(440)
        self.settings = AppSettings.get_instance()

        self._setup_ui()
        self._load_values()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(14, 14, 14, 14)

        # 1. Unsaved Changes & File Behavior
        close_group = QGroupBox("File & Tab Close Behavior", self)
        close_form = QFormLayout(close_group)
        close_form.setSpacing(8)

        self.unsaved_action_combo = QComboBox(close_group)
        self.unsaved_action_combo.addItem("Prompt / Ask Every Time", "ask")
        self.unsaved_action_combo.addItem("Always Save & Close", "save_and_close")
        self.unsaved_action_combo.addItem("Always Discard & Close", "discard_and_close")
        self.unsaved_action_combo.addItem("Always Save without Closing", "save_without_closing")
        close_form.addRow("When closing modified tab:", self.unsaved_action_combo)

        self.silence_chk = QCheckBox("Silence confirmation dialog (use default action directly)", close_group)
        close_form.addRow("", self.silence_chk)
        layout.addWidget(close_group)

        # 2. Appearance & Heatmap
        app_group = QGroupBox("Appearance & Display", self)
        app_form = QFormLayout(app_group)
        app_form.setSpacing(8)

        self.theme_combo = QComboBox(app_group)
        self.theme_combo.addItem("Dark Theme", "dark")
        self.theme_combo.addItem("Light Theme", "light")
        app_form.addRow("Application Theme:", self.theme_combo)

        self.colormap_combo = QComboBox(app_group)
        self.colormap_combo.addItems(AVAILABLE_COLORMAPS)
        app_form.addRow("Default Heatmap:", self.colormap_combo)

        self.format_combo = QComboBox(app_group)
        self.format_combo.addItem("General (%.5g)", "%.5g")
        self.format_combo.addItem("Scientific (%.4e)", "%.4e")
        self.format_combo.addItem("Fixed (%.2f)", "%.2f")
        self.format_combo.addItem("Fixed (%.6f)", "%.6f")
        app_form.addRow("Default Number Format:", self.format_combo)
        layout.addWidget(app_group)

        # 3. Diff Comparison Defaults
        diff_group = QGroupBox("Tensor Comparison Defaults", self)
        diff_form = QFormLayout(diff_group)
        diff_form.setSpacing(8)

        self.atol_spin = QDoubleSpinBox(diff_group)
        self.atol_spin.setDecimals(8)
        self.atol_spin.setRange(0.0, 1e6)
        diff_form.addRow("Default Absolute Tol (atol):", self.atol_spin)

        self.rtol_spin = QDoubleSpinBox(diff_group)
        self.rtol_spin.setDecimals(8)
        self.rtol_spin.setRange(0.0, 1e6)
        diff_form.addRow("Default Relative Tol (rtol):", self.rtol_spin)
        layout.addWidget(diff_group)

        # Bottom Buttons
        btn_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset to Defaults", self)
        self.reset_btn.clicked.connect(self._reset_defaults)
        btn_layout.addWidget(self.reset_btn)
        btn_layout.addStretch()

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.button_box.accepted.connect(self._save_and_accept)
        self.button_box.rejected.connect(self.reject)
        btn_layout.addWidget(self.button_box)
        layout.addLayout(btn_layout)

    def _load_values(self) -> None:
        action = self.settings.get("unsaved_close_action", "ask")
        idx = self.unsaved_action_combo.findData(action)
        if idx >= 0:
            self.unsaved_action_combo.setCurrentIndex(idx)

        self.silence_chk.setChecked(self.settings.get("silence_unsaved_dialog", False))

        theme = self.settings.get("default_theme", "dark")
        t_idx = self.theme_combo.findData(theme)
        if t_idx >= 0:
            self.theme_combo.setCurrentIndex(t_idx)

        cmap = self.settings.get("default_colormap", "Coolwarm (Blue-Red)")
        c_idx = self.colormap_combo.findText(cmap)
        if c_idx >= 0:
            self.colormap_combo.setCurrentIndex(c_idx)

        fmt = self.settings.get("default_float_format", "%.5g")
        f_idx = self.format_combo.findData(fmt)
        if f_idx >= 0:
            self.format_combo.setCurrentIndex(f_idx)

        self.atol_spin.setValue(self.settings.get("default_atol", 1e-5))
        self.rtol_spin.setValue(self.settings.get("default_rtol", 1e-5))

    def _save_and_accept(self) -> None:
        self.settings.set("unsaved_close_action", self.unsaved_action_combo.currentData())
        self.settings.set("silence_unsaved_dialog", self.silence_chk.isChecked())
        self.settings.set("default_theme", self.theme_combo.currentData())
        self.settings.set("default_colormap", self.colormap_combo.currentText())
        self.settings.set("default_float_format", self.format_combo.currentData())
        self.settings.set("default_atol", self.atol_spin.value())
        self.settings.set("default_rtol", self.rtol_spin.value())
        self.settingsChanged.emit()
        self.accept()

    def _reset_defaults(self) -> None:
        self.settings.reset_defaults()
        self._load_values()
