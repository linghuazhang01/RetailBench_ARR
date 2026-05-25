from __future__ import annotations

from dataclasses import dataclass
import os
import numpy as np


@dataclass(frozen=True)
class ReturnRateBand:
    min_rate: float
    max_rate: float


class ReturnRateModel:
    """
    Fixed smooth mapping from quality score (1-5) to return rate probability.

    Star ranges:
    1 -> 15%-30%
    2 -> 10%-15%
    3 ->  5%-10%
    4 ->  1%-5%
    5 ->  0%-2%
    """

    STAR_POINTS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    MAX_POINTS = np.array([0.30, 0.15, 0.10, 0.05, 0.02])
    MAX_POLY_COEFFS = np.polyfit(STAR_POINTS, MAX_POINTS, deg=4)

    DEFAULT_RATE = 0.08

    @classmethod
    def max_return_rate(cls, quality_score: float | np.ndarray) -> np.ndarray:
        """Return smooth max return rate curve (polynomial fit, clipped to bounds)."""
        r = np.asarray(quality_score, dtype=float)
        r = np.clip(r, cls.STAR_POINTS[0], cls.STAR_POINTS[-1])
        base = np.polyval(cls.MAX_POLY_COEFFS, r)
        return np.clip(base, cls.MAX_POINTS[-1], cls.MAX_POINTS[0])

    @classmethod
    def from_quality(cls, quality_score: float | np.ndarray) -> float | np.ndarray:
        """
        Deterministic mapping used in simulation.
        Returns the smooth max return rate.
        """
        if quality_score is None:
            return cls.DEFAULT_RATE
        rates = cls.max_return_rate(quality_score)
        if np.isscalar(quality_score):
            return float(rates)
        return rates


if __name__ == "__main__":
    # Quick sanity check
    for q in [1.0, 2.0, 3.0, 4.0, 5.0, 4.3, 2.7]:
        print(q, ReturnRateModel.from_quality(q))

    # Plot smooth max mapping curve
    try:
        import matplotlib.pyplot as plt

        xs = np.linspace(1.0, 5.0, 200)
        max_curve = ReturnRateModel.max_return_rate(xs)

        plt.figure(figsize=(6, 4))
        plt.plot(xs, max_curve, label="Max Return Rate")
        plt.scatter(ReturnRateModel.STAR_POINTS, ReturnRateModel.MAX_POINTS, color="black", s=20, label="Star Points")
        plt.xlabel("Quality Score (1-5)")
        plt.ylabel("Return Rate")
        plt.title("Return Rate vs Quality Score (Smooth Max Mapping)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        out_path = os.path.join(os.path.dirname(__file__), "return_rate_curve.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"Plot saved to: {out_path}")
    except Exception as exc:
        print(f"[WARN] Failed to plot return rate curve: {exc}")
