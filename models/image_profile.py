from dataclasses import dataclass
from typing import Optional


@dataclass
class ImageProfile:
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    is_grayscale: Optional[bool] = None

    blur_score: Optional[float] = None
    blur_level: Optional[str] = None

    noise_score: Optional[float] = None
    noise_level: Optional[str] = None

    brightness: Optional[float] = None
    contrast: Optional[float] = None

    histogram: Optional[object] = None

    color_cast: Optional[str] = None
