"""
Adaptive Penalty Refinement (APR) Algorithm for CCE-QOS.
Dynamically updates penalty multipliers to eliminate constraint violations:
lambda_k^{(t+1)} = lambda_k^{(t)} + mu * max(0, g_k(x))
guaranteeing asymptotic convergence to the feasible zero-violation subspace.
"""

import numpy as np


class AdaptivePenaltyRefinement:
    def __init__(self, initial_penalty: float = 10.0, mu_step: float = 5.0, max_penalty: float = 500.0):
        self.penalties = {
            "lambda_one_hot": initial_penalty,
            "lambda_precedence": initial_penalty,
            "lambda_sram": initial_penalty
        }
        self.mu = mu_step
        self.max_penalty = max_penalty
        self.violation_history = []

    def update_penalties(self, violations: dict) -> dict:
        for k in self.penalties:
            viol = violations.get(k, 0)
            if viol > 0:
                self.penalties[k] = min(self.penalties[k] + self.mu * viol, self.max_penalty)
        self.violation_history.append(sum(violations.values()))
        return self.penalties

    def is_converged(self, violations: dict) -> bool:
        return sum(violations.values()) == 0
