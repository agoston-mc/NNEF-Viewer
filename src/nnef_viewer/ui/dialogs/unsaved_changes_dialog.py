# Copyright (c) 2026
# Custom dialog for handling unsaved tensor modifications upon tab/window closure.

from enum import IntEnum
from typing import Optional
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core.settings import AppSettings


class UnsavedChoice(IntEnum):
    CANCEL = 0
    SAVE_AND_CLOSE = 1
    DISCARD_AND_CLOSE = 2
    SAVE_WITHOUT_CLOSING = 3


class UnsavedChangesDialog(QDialog):
    """
    Prompt dialog when attempting to close an unsaved tensor.
    Provides options for Save & Close, Discard & Close, Save without Closing, and Silence/Default setting.
    """

    def __init__(self, doc_name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.doc_name = doc_name
        self.choice = UnsavedChoice.CANCEL

        self.setWindowTitle("Unsaved Modifications")
        self.setMinimumWidth(440)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(16, 16, 16, 16)

        msg_lbl = QLabel(
            f"The tensor <b>'{self.doc_name}'</b> has unsaved modifications.\n"
            "What action would you like to take?",
            self,
        )
        msg_lbl.setWordWrap(True)
        layout.addWidget(msg_lbl)

        # Action Buttons Layout
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.save_close_btn = QPushButton("Save & Close", self)
        self.save_close_btn.setStyleSheet("background-color: #3574f0; color: #ffffff; font-weight: 600;")
        self.save_close_btn.clicked.connect(lambda: self._select_choice(UnsavedChoice.SAVE_AND_CLOSE))
        btn_layout.addWidget(self.save_close_btn)

        self.discard_close_btn = QPushButton("Discard & Close", self)
        self.discard_close_btn.clicked.connect(lambda: self._select_choice(UnsavedChoice.DISCARD_AND_CLOSE))
        btn_layout.addWidget(self.discard_close_btn)

        self.save_stay_btn = QPushButton("Save (Keep Open)", self)
        self.save_stay_btn.clicked.connect(lambda: self._select_choice(UnsavedChoice.SAVE_WITHOUT_CLOSING))
        btn_layout.addWidget(self.save_stay_btn)

        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(lambda: self._select_choice(UnsavedChoice.CANCEL))
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

        # Silence / Remember checkbox
        self.remember_chk = QCheckBox(
            "Remember my choice and don't ask again (can be modified in Preferences)",
            self,
        )
        self.remember_chk.setStyleSheet("color: #8c909a; font-size: 11px;")
        layout.addWidget(self.remember_chk)

    def _select_choice(self, choice: UnsavedChoice) -> None:
        self.choice = choice
        if self.remember_chk.isChecked() and choice != UnsavedChoice.CANCEL:
            settings = AppSettings.get_instance()
            settings.set("silence_unsaved_dialog", True)
            action_map = {
                UnsavedChoice.SAVE_AND_CLOSE: "save_and_close",
                UnsavedChoice.DISCARD_AND_CLOSE: "discard_and_close",
                UnsavedChoice.SAVE_WITHOUT_CLOSING: "save_without_closing",
            }
            if choice in action_map:
                settings.set("unsaved_close_action", action_map[choice])

        if choice == UnsavedChoice.CANCEL:
            self.reject()
        else:
            self.accept()
