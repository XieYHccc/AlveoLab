"""Small modal dialogs used by `MainWindow`."""

from __future__ import annotations

import traceback

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


class ArchTypeDialog(QDialog):
    """Ask the user whether a mesh is an upper (U) or lower (L) arch.

    Triggered when `AlveoLab.utils.infer_arch_type` fails to parse the
    filename. `selected_arch_type()` returns "U" or "L"; `None` after Cancel.
    """

    def __init__(self, filename: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select arch type")
        self.setModal(True)
        self._choice: str | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"Could not infer the arch type from filename:\n  {filename}\n\n"
                "Please pick one:"
            )
        )

        self._upper = QRadioButton("Upper (maxillary)")
        self._lower = QRadioButton("Lower (mandibular)")
        self._upper.setChecked(True)

        group = QButtonGroup(self)
        group.addButton(self._upper)
        group.addButton(self._lower)
        layout.addWidget(self._upper)
        layout.addWidget(self._lower)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        self._choice = "U" if self._upper.isChecked() else "L"
        self.accept()

    def selected_arch_type(self) -> str | None:
        return self._choice


def show_error(parent: QWidget | None, title: str, exc: BaseException) -> None:
    """Show an error dialog with the exception type, message, and traceback."""
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle(title)
    msg.setText(f"{type(exc).__name__}: {exc}")
    msg.setDetailedText("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    msg.setStandardButtons(QMessageBox.Ok)
    msg.setTextInteractionFlags(Qt.TextSelectableByMouse)
    msg.exec()
