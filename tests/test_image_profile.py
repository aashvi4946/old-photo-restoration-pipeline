from models.image_profile import ImageProfile


def test_image_profile_defaults():
    profile = ImageProfile()

    assert profile.width is None
    assert profile.height is None
    assert profile.channels is None
    assert profile.is_grayscale is None

    assert profile.blur_score is None
    assert profile.blur_level is None

    assert profile.noise_score is None
    assert profile.noise_level is None

    assert profile.brightness is None
    assert profile.contrast is None
    assert profile.histogram is None
    assert profile.color_cast is None


def test_image_profile_can_store_values():
    profile = ImageProfile(
        width=1920,
        height=1080,
        channels=3,
        is_grayscale=False,
    )

    assert profile.width == 1920
    assert profile.height == 1080
    assert profile.channels == 3
    assert profile.is_grayscale is False
