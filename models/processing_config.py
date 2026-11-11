"""Structured Quick Restore configurations and generated restoration plans."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrengthConfiguration:
    """Parameters and activation thresholds for one restoration strength."""

    name: str
    noise_medium_parameters: dict[str, float | int]
    noise_high_parameters: dict[str, float | int]
    clahe_clip_limit: float
    contrast_threshold: float
    dark_gamma: float
    bright_gamma: float
    high_blur_unsharp_amount: float
    medium_blur_unsharp_amount: float
    final_sharpen_amount: float
    vintage_intensity: float


@dataclass(frozen=True)
class RestorationStep:
    """A single GUI-independent operation selected by Quick Restore."""

    operation: str
    label: str
    parameters: dict[str, object]
    reason: str


@dataclass(frozen=True)
class RestorationPlan:
    """A deterministic Quick Restore pipeline and its explainable report."""

    strength: str
    style: str
    steps: tuple[RestorationStep, ...]
    skipped: tuple[str, ...]

    @property
    def report(self) -> str:
        lines = [
            "RESTORATION REPORT",
            f"Strength: {self.strength}",
            f"Style: {self.style}",
            "",
            "Applied:",
        ]
        lines.extend(f"✓ {step.label}\n  {step.reason}" for step in self.steps)
        if self.skipped:
            lines.extend(["", "Skipped:"])
            lines.extend(f"○ {reason}" for reason in self.skipped)
        return "\n".join(lines)
