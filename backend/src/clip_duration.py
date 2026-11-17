"""Task-specific clip length presets shared by selection and rendering."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClipDuration:
    minimum: int
    maximum: int
    ideal_minimum: int
    ideal_maximum: int

    @property
    def target_maximum(self) -> int:
        return self.maximum - 10


CLIP_DURATIONS = {
    "auto": ClipDuration(15, 70, 25, 50),
    # Ten seconds beyond the target are reserved for completing a thought.
    "short": ClipDuration(15, 40, 18, 25),
    "medium": ClipDuration(30, 70, 35, 55),
    "long": ClipDuration(60, 130, 75, 110),
}


def validate_clip_length(value: Any) -> str:
    if not isinstance(value, str) or value not in CLIP_DURATIONS:
        raise ValueError("clip_length must be auto, short, medium, or long")
    return value


def normalize_clip_length(value: Any) -> str:
    """Stored task settings fall back to Auto instead of failing the job."""
    return value if isinstance(value, str) and value in CLIP_DURATIONS else "auto"


def clip_duration(value: str = "auto") -> ClipDuration:
    return CLIP_DURATIONS[validate_clip_length(value)]
