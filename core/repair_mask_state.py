"""Non-destructive damage-mask state for guided photo repair."""

from __future__ import annotations

import cv2
import numpy as np


class RepairMaskState:
    """Manage a binary repair mask and independent stroke undo/redo history."""

    def __init__(self) -> None:
        self.mask: np.ndarray | None = None
        self._undo_history: list[np.ndarray] = []
        self._redo_history: list[np.ndarray] = []
        self._stroke_start: np.ndarray | None = None

    @property
    def has_mask(self) -> bool:
        return self.mask is not None

    @property
    def has_damage(self) -> bool:
        return self.mask is not None and bool(np.any(self.mask))

    @property
    def can_undo(self) -> bool:
        return bool(self._undo_history)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo_history)

    def set_image_shape(self, height: int, width: int) -> None:
        """Create a clear mask for a new size, preserving same-size masks."""

        if height <= 0 or width <= 0:
            raise ValueError("Mask dimensions must be positive.")
        if self.mask is not None and self.mask.shape == (height, width):
            return
        self.mask = np.zeros((height, width), dtype=np.uint8)
        self._undo_history.clear()
        self._redo_history.clear()
        self._stroke_start = None

    def clear_all(self) -> bool:
        """Clear the mask as one undoable action."""

        self._require_mask()
        if not self.has_damage:
            return False
        self._undo_history.append(self.mask.copy())
        self.mask.fill(0)
        self._redo_history.clear()
        return True

    def begin_stroke(self) -> None:
        """Start an atomic brush or eraser stroke."""

        self._require_mask()
        self._stroke_start = self.mask.copy()

    def paint_line(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        brush_size: int,
        erase: bool = False,
    ) -> None:
        """Paint a mask segment without committing history until stroke end."""

        self._require_mask()
        if self._stroke_start is None:
            self.begin_stroke()
        if isinstance(brush_size, bool) or not isinstance(brush_size, int) or brush_size <= 0:
            raise ValueError("brush_size must be a positive integer.")
        color = 0 if erase else 255
        thickness = max(1, brush_size)
        cv2.line(self.mask, start, end, color, thickness=thickness, lineType=cv2.LINE_8)
        cv2.circle(self.mask, end, max(1, thickness // 2), color, thickness=-1, lineType=cv2.LINE_8)

    def end_stroke(self) -> bool:
        """Commit a changed stroke as one independent undoable mask action."""

        if self._stroke_start is None:
            return False
        changed = not np.array_equal(self.mask, self._stroke_start)
        if changed:
            self._undo_history.append(self._stroke_start)
            self._redo_history.clear()
        self._stroke_start = None
        return changed

    def undo(self) -> bool:
        self._require_mask()
        if not self.can_undo:
            return False
        self._redo_history.append(self.mask.copy())
        self.mask = self._undo_history.pop()
        return True

    def redo(self) -> bool:
        self._require_mask()
        if not self.can_redo:
            return False
        self._undo_history.append(self.mask.copy())
        self.mask = self._redo_history.pop()
        return True

    def reset(self) -> None:
        if self.mask is not None:
            self.mask.fill(0)
        self._undo_history.clear()
        self._redo_history.clear()
        self._stroke_start = None

    def _require_mask(self) -> None:
        if self.mask is None:
            raise ValueError("No image is available for repair.")
