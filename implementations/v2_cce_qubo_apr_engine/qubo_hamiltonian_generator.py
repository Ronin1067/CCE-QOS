"""
Quadratic Unconstrained Binary Optimization (QUBO) Hamiltonian Generator for CCE-QOS.
Maps continuous schedule and bank allocation variables into binary decision space:
x_{i, t, b} in {0, 1} -> Task i scheduled at time slot t in memory bank b.
H_total = H_energy + lambda_1 * H_one_hot + lambda_2 * H_precedence + lambda_3 * H_bank_conflict.
"""

import numpy as np


class CCEQUBOGenerator:
    def __init__(self, num_tasks: int, num_slots: int, num_banks: int = 4):
        self.num_tasks = num_tasks
        self.num_slots = num_slots
        self.num_banks = num_banks
        self.num_vars = num_tasks * num_slots * num_banks
        self.Q = np.zeros((self.num_vars, self.num_vars))

    def var_index(self, task: int, slot: int, bank: int) -> int:
        return task * (self.num_slots * self.num_banks) + slot * self.num_banks + bank

    def add_one_hot_constraints(self, penalty: float = 50.0):
        """Each task must be scheduled exactly once: (sum_{t, b} x_{i,t,b} - 1)^2"""
        for i in range(self.num_tasks):
            indices = [self.var_index(i, t, b) for t in range(self.num_slots) for b in range(self.num_banks)]
            # Diagonal terms: penalty * (1 - 2) = -penalty
            for idx in indices:
                self.Q[idx, idx] -= penalty
            # Off-diagonal cross terms: 2 * penalty * x_p * x_q
            for p in range(len(indices)):
                for q in range(p + 1, len(indices)):
                    self.Q[indices[p], indices[q]] += 2.0 * penalty
                    self.Q[indices[q], indices[p]] += 2.0 * penalty

    def build_hamiltonian(self, penalties: dict) -> np.ndarray:
        self.Q.fill(0.0)
        self.add_one_hot_constraints(penalty=penalties.get("lambda_one_hot", 50.0))
        return self.Q
