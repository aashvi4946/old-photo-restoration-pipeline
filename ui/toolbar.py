from PySide6.QtGui import QAction
from PySide6.QtWidgets import QToolBar


class AppToolbar(QToolBar):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setObjectName("appToolbar")

        self.open_action = QAction("Open", self)
        self.save_action = QAction("Save", self)
        self.undo_action = QAction("Undo", self)
        self.redo_action = QAction("Redo", self)
        self.reset_action = QAction("Reset", self)

        self.addAction(self.open_action)
        self.addAction(self.save_action)
        self.addSeparator()
        self.addAction(self.undo_action)
        self.addAction(self.redo_action)
        self.addAction(self.reset_action)

        # Phase 1: functionality not implemented yet
        self.open_action.setEnabled(False)
        self.save_action.setEnabled(False)
        self.undo_action.setEnabled(False)
        self.redo_action.setEnabled(False)
        self.reset_action.setEnabled(False)
