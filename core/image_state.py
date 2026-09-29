from pathlib import Path
from typing import List, Optional

import numpy as np

from utils.image_utils import load_image


class ImageState:
    """Manage original image, current image, and undo/redo history."""

    def __init__(self):
        self.original_image: Optional[np.ndarray] = None
        self.current_image: Optional[np.ndarray] = None

        self._undo_history: List[np.ndarray] = []
        self._redo_history: List[np.ndarray] = []

    @property
    def has_image(self) -> bool:
        """Return True when an image is currently loaded."""

        return self.current_image is not None

    @property
    def can_undo(self) -> bool:
        """Return True when an undo operation is available."""

        return len(self._undo_history) > 0

    @property
    def can_redo(self) -> bool:
        """Return True when a redo operation is available."""

        return len(self._redo_history) > 0

    def set_image(self, image: np.ndarray) -> None:
        """Set a new image and start a fresh history."""

        self._validate_image(image)

        self.original_image = image.copy()
        self.current_image = image.copy()

        self._undo_history.clear()
        self._redo_history.clear()

    def load_image(self, path: str | Path) -> None:
        """Load an image from disk and replace the current image state."""

        image = load_image(path)

        self.set_image(image)

    def apply_change(self, image: np.ndarray) -> None:
        """Replace the current image and create an undo point."""

        self._validate_image(image)

        if self.current_image is None:
            raise ValueError("No image is currently loaded.")

        self._undo_history.append(self.current_image.copy())
        self.current_image = image.copy()

        self._redo_history.clear()

    def undo(self) -> bool:
        """Undo the most recent image change."""

        if not self.can_undo:
            return False

        self._redo_history.append(self.current_image.copy())
        self.current_image = self._undo_history.pop()

        return True

    def redo(self) -> bool:
        """Redo the most recently undone image change."""

        if not self.can_redo:
            return False

        self._undo_history.append(self.current_image.copy())
        self.current_image = self._redo_history.pop()

        return True

    def reset(self) -> bool:
        """Restore the original image."""

        if self.original_image is None:
            return False

        if not np.array_equal(self.current_image, self.original_image):
            self._undo_history.append(self.current_image.copy())

        self.current_image = self.original_image.copy()
        self._redo_history.clear()

        return True

    def clear(self) -> None:
        """Clear the loaded image and all history."""

        self.original_image = None
        self.current_image = None

        self._undo_history.clear()
        self._redo_history.clear()

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Validate that the supplied image is a usable NumPy array."""

        if image is None:
            raise ValueError("Image cannot be None.")

        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a NumPy array.")

        if image.size == 0:
            raise ValueError("Image cannot be empty.")

        if image.ndim not in (2, 3):
            raise ValueError("Unsupported image dimensions.")

        if image.ndim == 3 and image.shape[2] not in (3, 4):
            raise ValueError("Unsupported channel count.")
