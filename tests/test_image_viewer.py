import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
from PySide6.QtWidgets import QApplication

from ui.image_viewer import ImageViewer


def test_grayscale_image_can_be_displayed():
    app = QApplication.instance() or QApplication([])

    viewer = ImageViewer()

    image = np.full((100, 100), 128, dtype=np.uint8)

    viewer.set_image(image)

    assert viewer._image is image
    assert not viewer.image_label.pixmap().isNull()
    assert viewer.title.isHidden()
    assert viewer.description.isHidden()

    viewer.close()
    app.quit()


def test_color_image_can_be_displayed():
    app = QApplication.instance() or QApplication([])

    viewer = ImageViewer()

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    viewer.set_image(image)

    assert viewer._image is image
    assert not viewer.image_label.pixmap().isNull()

    viewer.close()
    app.quit()


def test_clear_returns_to_empty_state():
    app = QApplication.instance() or QApplication([])

    viewer = ImageViewer()

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    viewer.set_image(image)
    viewer.clear()

    assert viewer._image is None
    assert viewer.image_label.pixmap().isNull()
    assert not viewer.title.isHidden()
    assert not viewer.description.isHidden()

    viewer.close()
    app.quit()


def test_empty_image_is_rejected():
    app = QApplication.instance() or QApplication([])

    viewer = ImageViewer()

    try:
        viewer.set_image(np.array([]))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

    viewer.close()
    app.quit()


def test_display_does_not_modify_stored_image_dimensions():
    app = QApplication.instance() or QApplication([])

    viewer = ImageViewer()

    image = np.zeros((1200, 1600, 3), dtype=np.uint8)
    original_shape = image.shape

    viewer.resize(400, 300)
    viewer.set_image(image)

    assert image.shape == original_shape
    assert viewer._image.shape == original_shape

    viewer.close()
    app.quit()


def test_large_image_fits_viewer_without_changing_source_dimensions():
    app = QApplication.instance() or QApplication([])

    viewer = ImageViewer()
    viewer.resize(600, 400)

    image = np.zeros((3000, 4000, 3), dtype=np.uint8)

    viewer.set_image(image)

    pixmap = viewer.image_label.pixmap()

    assert pixmap is not None
    assert not pixmap.isNull()

    assert pixmap.width() <= viewer.image_label.width()
    assert pixmap.height() <= viewer.image_label.height()

    assert viewer._image.shape == (3000, 4000, 3)

    viewer.close()
    app.quit()
