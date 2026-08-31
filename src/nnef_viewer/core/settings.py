# Copyright (c) 2026
# Persistent application settings management via QSettings.

from typing import Any, Optional
from PySide6.QtCore import QSettings


class AppSettings:
    """Singleton-like manager for application preferences."""

    ORGANIZATION = "NNEF"
    APPLICATION = "NNEFTensorViewer"

    # Default values
    DEFAULTS = {
        "unsaved_close_action": "ask",  # "ask", "save_and_close", "discard_and_close", "save_without_closing"
        "silence_unsaved_dialog": False,
        "default_theme": "dark",  # "dark", "light"
        "default_colormap": "Coolwarm (Blue-Red)",
        "default_float_format": "%.5g",
        "default_atol": 1e-5,
        "default_rtol": 1e-5,
    }

    _instance: Optional["AppSettings"] = None

    def __init__(self):
        self.qsettings = QSettings(self.ORGANIZATION, self.APPLICATION)

    @classmethod
    def get_instance(cls) -> "AppSettings":
        if cls._instance is None:
            cls._instance = AppSettings()
        return cls._instance

    def get(self, key: str, default: Any = None) -> Any:
        if default is None:
            default = self.DEFAULTS.get(key)
        val = self.qsettings.value(key, default)
        # Type conversion helper
        if isinstance(default, bool):
            if isinstance(val, str):
                return val.lower() in ("true", "1", "yes")
            return bool(val)
        elif isinstance(default, float):
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        elif isinstance(default, int):
            try:
                return int(val)
            except (ValueError, TypeError):
                return default
        return val

    def set(self, key: str, value: Any) -> None:
        self.qsettings.setValue(key, value)
        self.qsettings.sync()

    def reset_defaults(self) -> None:
        for k, v in self.DEFAULTS.items():
            self.set(k, v)
