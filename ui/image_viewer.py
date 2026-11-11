import cv2
import numpy as np

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
)


class ImageViewer(QFrame):
    repair_stroke_started = Signal(object)
    repair_stroke_extended = Signal(object, object)
    repair_stroke_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setObjectName("imageViewer")

        self._image: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self._repair_mask: np.ndarray | None = None
        self._repair_enabled = False
        self._show_mask_overlay = True
        self._last_repair_point: tuple[int, int] | None = None

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title = QLabel("No Image Loaded")
        self.title.setObjectName("emptyStateTitle")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.description = QLabel(
            "Open an old photograph to begin restoration."
        )
        self.description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.description.setWordWrap(True)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.image_label.setMinimumSize(0, 0)
        self.image_label.installEventFilter(self)

        layout.addWidget(self.title)
        layout.addWidget(self.description)
        layout.addWidget(self.image_label)

    def set_image(self, image: np.ndarray) -> None:
        if image is None or image.size == 0:
            raise ValueError("Image cannot be empty.")

        self._image = image
        if self._repair_mask is not None and self._repair_mask.shape != image.shape[:2]:
            self._repair_mask = None
        self._refresh_pixmap()

        self.title.hide()
        self.description.hide()
        self.image_label.show()

        self._update_display()

    def clear(self) -> None:
        self._image = None
        self._pixmap = None
        self._repair_mask = None
        self._last_repair_point = None

        self.image_label.clear()
        self.image_label.hide()

        self.title.show()
        self.description.show()

    def set_repair_mask(self, mask: np.ndarray | None) -> None:
        """Set the separate binary repair-mask overlay for the displayed image."""

        if mask is not None and (self._image is None or mask.shape != self._image.shape[:2]):
            raise ValueError("Repair mask dimensions must match the displayed image.")
        self._repair_mask = mask
        self._refresh_pixmap()

    def set_repair_mode(self, enabled: bool, show_mask_overlay: bool = True) -> None:
        """Enable painting events and optional mask preview without changing pixels."""

        self._repair_enabled = enabled
        self._show_mask_overlay = show_mask_overlay
        self._last_repair_point = None
        self.image_label.setCursor(
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor
        )
        self._refresh_pixmap()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.image_label and self._repair_enabled:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                point = self._event_to_image_point(event.position().toPoint())
                if point is not None:
                    self._last_repair_point = point
                    self.repair_stroke_started.emit(point)
                    self.repair_stroke_extended.emit(point, point)
                    return True
            elif event.type() == QEvent.Type.MouseMove and event.buttons() & Qt.MouseButton.LeftButton:
                point = self._event_to_image_point(event.position().toPoint())
                if point is not None and self._last_repair_point is not None:
                    self.repair_stroke_extended.emit(self._last_repair_point, point)
                    self._last_repair_point = point
                    return True
            elif event.type() == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                if self._last_repair_point is not None:
                    self.repair_stroke_finished.emit()
                    self._last_repair_point = None
                    return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_display()

    def _update_display(self) -> None:
        if self._pixmap is None:
            return

        available_size = self.image_label.size()

        if available_size.width() <= 0 or available_size.height() <= 0:
            return

        scaled_pixmap = self._pixmap.scaled(
            available_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.image_label.setPixmap(scaled_pixmap)

    def _refresh_pixmap(self) -> None:
        if self._image is None:
            return
        self._pixmap = self._numpy_to_pixmap(self._image_with_mask_overlay())
        self._update_display()

    def _image_with_mask_overlay(self) -> np.ndarray:
        """Build a red repair-mask preview without modifying the current image."""

        if (
            self._repair_mask is None
            or not self._show_mask_overlay
            or not np.any(self._repair_mask)
        ):
            return self._image

        mask = self._repair_mask > 0
        if self._image.ndim == 2:
            overlay = cv2.cvtColor(self._image, cv2.COLOR_GRAY2BGR)
        elif self._image.shape[2] == 4:
            overlay = self._image[:, :, :3].copy()
        else:
            overlay = self._image.copy()

        red = np.array([0, 0, 255], dtype=np.float64)
        overlay[mask] = np.clip(
            np.rint(0.45 * overlay[mask].astype(np.float64) + 0.55 * red),
            0,
            255,
        ).astype(np.uint8)
        return overlay

    def _event_to_image_point(self, position) -> tuple[int, int] | None:
        """Map a displayed-pixmap coordinate to an original image coordinate."""

        if self._image is None or self.image_label.pixmap() is None:
            return None
        pixmap = self.image_label.pixmap()
        offset_x = (self.image_label.width() - pixmap.width()) // 2
        offset_y = (self.image_label.height() - pixmap.height()) // 2
        x = position.x() - offset_x
        y = position.y() - offset_y
        if not 0 <= x < pixmap.width() or not 0 <= y < pixmap.height():
            return None
        image_height, image_width = self._image.shape[:2]
        return (
            min(image_width - 1, int(x * image_width / pixmap.width())),
            min(image_height - 1, int(y * image_height / pixmap.height())),
        )

    @staticmethod
    def _numpy_to_pixmap(image: np.ndarray) -> QPixmap:
        if image.ndim == 2:
            height, width = image.shape

            q_image = QImage(
                image.data,
                width,
                height,
                image.strides[0],
                QImage.Format.Format_Grayscale8,
            ).copy()

        elif image.ndim == 3 and image.shape[2] == 3:
            height, width, _ = image.shape

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            q_image = QImage(
                rgb.data,
                width,
                height,
                rgb.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()

        elif image.ndim == 3 and image.shape[2] == 4:
            height, width, _ = image.shape

            rgba = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

            q_image = QImage(
                rgba.data,
                width,
                height,
                rgba.strides[0],
                QImage.Format.Format_RGBA8888,
            ).copy()

        else:
            raise ValueError("Unsupported image format.")

        return QPixmap.fromImage(q_image)
