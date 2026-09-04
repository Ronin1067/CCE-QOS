from __future__ import annotations

from typing import Dict, Mapping


class PenaltyTuner:
    """
    Lightweight APR-style penalty updater:

        lambda_k^{t+1} = clip(
            lambda_k^t + eta1 * violation_rate_k + eta2 * avg_magnitude_k,
            [lam_min, lam_max],
        )
    """

    def __init__(self, eta1: float, eta2: float, lam_min: float, lam_max: float) -> None:
        if lam_min > lam_max:
            raise ValueError("lam_min must be <= lam_max.")
        self.eta1 = float(eta1)
        self.eta2 = float(eta2)
        self.lam_min = float(lam_min)
        self.lam_max = float(lam_max)

    def update(
        self,
        penalties: Mapping[str, float],
        violation_rates: Mapping[str, float],
        avg_magnitudes: Mapping[str, float],
    ) -> Dict[str, float]:
        """
        Return updated penalties without mutating inputs.

        Missing violation or magnitude entries default to 0.
        Keys present in either violation_rates or avg_magnitudes are also updated,
        initialized from existing penalties when available, otherwise lam_min.
        """
        keys = set(penalties) | set(violation_rates) | set(avg_magnitudes)
        updated: Dict[str, float] = {}

        for key in keys:
            current = float(penalties.get(key, self.lam_min))
            violation = float(violation_rates.get(key, 0.0))
            magnitude = float(avg_magnitudes.get(key, 0.0))

            proposal = current + self.eta1 * violation + self.eta2 * magnitude
            updated[key] = min(self.lam_max, max(self.lam_min, proposal))

        return updated
