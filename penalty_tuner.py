from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict


@dataclass
class APRState:
    penalties: Dict[str, float]
    round_index: int
    signal_snapshot: Dict[str, Dict[str, float]]


class AdaptivePenaltyRefinement:
    def __init__(self, initial_penalties: Dict[str, float] | None = None):
        self.penalties = dict(initial_penalties or {})
        if not self.penalties:
            self.penalties = {
                "sram_capacity": 1.0,
                "bandwidth_capacity": 1.0,
                "dependency_conflict": 1.0,
                "dram_pressure": 1.0,
                "memory_bank_conflict": 1.0,
                "bandwidth_imbalance": 1.0,
            }
        self.history = []
        self.frequency_counter = defaultdict(float)
        self.signal_ema = defaultdict(float)
        self.rounds = 0
        self.clip_low = 0.2
        self.clip_high = 14.0
        self.ema_decay = 0.72
        self.momentum = 0.18
        self.max_growth = 2.4

    def get(self) -> Dict[str, float]:
        return dict(self.penalties)

    def update(self, violation_rate: Dict[str, float], cost_impact: Dict[str, float]) -> APRState:
        self.rounds += 1
        updated = {}
        snapshots: Dict[str, Dict[str, float]] = {}

        for constraint, old_value in self.penalties.items():
            v_rate_raw = max(0.0, float(violation_rate.get(constraint, 0.0)))
            c_impact_raw = max(0.0, float(cost_impact.get(constraint, 0.0)))

            v_key = f"{constraint}::v"
            c_key = f"{constraint}::c"
            self.signal_ema[v_key] = self.ema_decay * self.signal_ema[v_key] + (1.0 - self.ema_decay) * v_rate_raw
            self.signal_ema[c_key] = self.ema_decay * self.signal_ema[c_key] + (1.0 - self.ema_decay) * c_impact_raw
            v_rate = self.signal_ema[v_key]
            c_impact = self.signal_ema[c_key]

            touched = 1.0 if v_rate_raw > 0.0 else 0.0
            self.frequency_counter[constraint] += touched
            freq = self.frequency_counter[constraint] / max(1.0, self.rounds)

            growth = math.exp(v_rate + c_impact + freq)
            growth = min(self.max_growth, max(1.0 / self.max_growth, growth))

            proposal = old_value * growth
            proposal = (1.0 - self.momentum) * proposal + self.momentum * old_value
            new_value = max(self.clip_low, min(self.clip_high, proposal))
            updated[constraint] = new_value

            snapshots[constraint] = {
                "violation_rate": v_rate,
                "cost_impact": c_impact,
                "constraint_frequency": freq,
                "growth": growth,
            }

        self.penalties = updated
        state = APRState(penalties=self.get(), round_index=self.rounds, signal_snapshot=snapshots)
        self.history.append(state)
        return state
