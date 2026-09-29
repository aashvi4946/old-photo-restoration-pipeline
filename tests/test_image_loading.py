import cv2
import numpy as np
import pytest

from utils.image_utils import load_image


def test_load_grayscale_png(tmp_path):
    image = np.full((10, 10), 128, dtype=np.uint8)
    path = tmp_path / "test.png"

    assert cv2.imwrite(str(path), image)

    loaded = load_image(path)

    assert loaded.shape == (10, 10)
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, image)


def test_load_color_png(tmp_path):
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    path = tmp_path / "test.png"

    assert cv2.imwrite(str(path), image)

    loaded = load_image(path)

    assert loaded.shape == (10, 10, 3)
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, image)


def test_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_image("does_not_exist.png")


def test_directory_path(tmp_path):
    with pytest.raises(ValueError):
        load_image(tmp_path)


def test_unsupported_extension(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("not an image")

    with pytest.raises(ValueError):
        load_image(path)


def test_corrupt_image(tmp_path):
    path = tmp_path / "corrupt.png"
    path.write_text("this is not a real image")

    with pytest.raises(ValueError):
        load_image(path)


def test_jpeg_is_supported(tmp_path):
    image = np.full((10, 10, 3), 100, dtype=np.uint8)
    path = tmp_path / "test.jpg"

    assert cv2.imwrite(str(path), image)

    loaded = load_image(path)

    assert loaded.shape == (10, 10, 3)


def test_tiff_is_supported(tmp_path):
    image = np.full((10, 10), 100, dtype=np.uint8)
    path = tmp_path / "test.tiff"

    assert cv2.imwrite(str(path), image)

    loaded = load_image(path)

    assert loaded.shape == (10, 10)


def test_bmp_is_supported(tmp_path):
    image = np.full((10, 10, 3), 100, dtype=np.uint8)
    path = tmp_path / "test.bmp"

    assert cv2.imwrite(str(path), image)

    loaded = load_image(path)

    assert loaded.shape == (10, 10, 3)


def test_webp_is_supported(tmp_path):
    image = np.full((10, 10, 3), 100, dtype=np.uint8)
    path = tmp_path / "test.webp"

    assert cv2.imwrite(str(path), image)

    loaded = load_image(path)

    assert loaded.shape == (10, 10, 3)
