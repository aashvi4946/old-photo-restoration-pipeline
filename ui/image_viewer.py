import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
)


class ImageViewer(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setObjectName("imageViewer")

        self._image: np.ndarray | None = None
        self._pixmap: QPixmap | None = None

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

        layout.addWidget(self.title)
        layout.addWidget(self.description)
        layout.addWidget(self.image_label)

    def set_image(self, image: np.ndarray) -> None:
        if image is None or image.size == 0:
            raise ValueError("Image cannot be empty.")

        self._image = image

        self._pixmap = self._numpy_to_pixmap(image)

        self.title.hide()
        self.description.hide()
        self.image_label.show()

        self._update_display()

    def clear(self) -> None:
        self._image = None
        self._pixmap = None

        self.image_label.clear()
        self.image_label.hide()

        self.title.show()
        self.description.show()

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
