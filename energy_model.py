from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence

from cost_model import ScheduleCostModel
from fusion_logic import FusionLogic
from graph_builder import OperationNode, WorkloadGraph


@dataclass
class EnergyBreakdown:
    unary_cost: float
    data_reuse_reward: float
    fusion_reward: float
    bandwidth_spike_penalty: float
    memory_conflict_penalty: float
    pairwise_total: float
    total_energy: float


class EnergyScheduleModel:
    def __init__(
        self,
        cost_model: ScheduleCostModel,
        fusion_logic: FusionLogic,
        weights: Dict | None = None,
        hardware_config: Dict | None = None,
    ):
        self.cost_model = cost_model
        self.fusion_logic = fusion_logic
        cfg = dict(weights or {})
        self.weights = {
            "unary_cost": float(cfg.get("unary_cost", 1.0)),
            "data_reuse_reward": float(cfg.get("data_reuse_reward", 0.34)),
            "fusion_reward": float(cfg.get("fusion_reward", 0.92)),
            "bandwidth_spike_penalty": float(cfg.get("bandwidth_spike_penalty", 0.23)),
            "memory_conflict_penalty": float(cfg.get("memory_conflict_penalty", 0.19)),
        }
        hw = dict(hardware_config or {})
        self.bank_count_hint = int(cfg.get("bank_count_hint", hw.get("sram_banks", 8)))
        self.bandwidth_tolerance = float(cfg.get("bandwidth_tolerance", 46.0))

    def _bank_of(self, node_id: str) -> int:
        banks = max(1, self.bank_count_hint)
        return sum(ord(ch) for ch in node_id) % banks

    def _pair_data_reuse(self, left: OperationNode, right: OperationNode) -> float:
        direct = 0.0
        if left.node_id in right.dependencies:
            direct = min(left.output_size, right.input_size)

        shared_deps = set(left.dependencies).intersection(right.dependencies)
        shared = 0.18 * len(shared_deps) * min(left.input_size, right.input_size)

        cycle_gap = abs(left.compute_cycles - right.compute_cycles) / max(
            1.0, max(left.compute_cycles, right.compute_cycles)
        )
        temporal_affinity = 1.0 / (1.0 + cycle_gap)
        return (direct + shared) * temporal_affinity

    def _pair_bandwidth_spike(self, left: OperationNode, right: OperationNode) -> float:
        traffic_left = left.input_size + left.output_size
        traffic_right = right.input_size + right.output_size
        jump = max(0.0, abs(traffic_right - traffic_left) - self.bandwidth_tolerance)

        left_rate = traffic_left / max(1.0, left.compute_cycles)
        right_rate = traffic_right / max(1.0, right.compute_cycles)
        skew = abs(left_rate - right_rate)
        return jump * (1.0 + 0.25 * skew)

    def _pair_memory_conflict(self, left: OperationNode, right: OperationNode) -> float:
        left_hint = float(left.attrs.get("sram_hint", left.output_size * 0.8))
        right_hint = float(right.attrs.get("sram_hint", right.output_size * 0.8))
        over_hint = max(0.0, left.output_size - left_hint) + max(0.0, right.output_size - right_hint)

        same_bank = 1.0 if self._bank_of(left.node_id) == self._bank_of(right.node_id) else 0.0
        shared_deps = set(left.dependencies).intersection(right.dependencies)
        shared_reads = 0.08 * len(shared_deps) * min(left.input_size, right.input_size)

        return same_bank * (0.22 * (left.output_size + right.output_size) + 0.34 * over_hint) + shared_reads

    def _pairwise_terms(self, graph: WorkloadGraph, order: Sequence[str]) -> Dict[str, float]:
        reuse_reward = 0.0
        bandwidth_penalty = 0.0
        memory_penalty = 0.0

        for idx in range(len(order) - 1):
            left = graph.nodes[order[idx]]
            right = graph.nodes[order[idx + 1]]
            reuse_reward += self._pair_data_reuse(left, right)
            bandwidth_penalty += self._pair_bandwidth_spike(left, right)
            memory_penalty += self._pair_memory_conflict(left, right)

        fusion_report = self.fusion_logic.estimate(graph, order)
        fusion_reward = fusion_report.fusion_gain + 12.0 * len(fusion_report.fused_triplets)

        return {
            "data_reuse_reward": reuse_reward,
            "fusion_reward": fusion_reward,
            "bandwidth_spike_penalty": bandwidth_penalty,
            "memory_conflict_penalty": memory_penalty,
        }

    def evaluate(
        self,
        graph: WorkloadGraph,
        order: Sequence[str],
        penalties: Dict[str, float] | None = None,
    ) -> Dict:
        base_eval = self.cost_model.evaluate(graph, order, penalties=penalties)
        unary = self.weights["unary_cost"] * float(base_eval["breakdown"]["total_cost"])

        pair = self._pairwise_terms(graph, order)
        pair_count = max(1.0, len(order) - 1.0)
        fusion_norm = max(1.0, len(order) / 2.0)

        reuse_reward = self.weights["data_reuse_reward"] * (pair["data_reuse_reward"] / pair_count)
        fusion_reward = self.weights["fusion_reward"] * (pair["fusion_reward"] / fusion_norm)
        bandwidth_penalty = self.weights["bandwidth_spike_penalty"] * (pair["bandwidth_spike_penalty"] / pair_count)
        memory_penalty = self.weights["memory_conflict_penalty"] * (pair["memory_conflict_penalty"] / pair_count)

        pairwise_total = bandwidth_penalty + memory_penalty - reuse_reward - fusion_reward
        total_energy = unary + pairwise_total

        report = dict(base_eval)
        report["energy_breakdown"] = asdict(
            EnergyBreakdown(
                unary_cost=unary,
                data_reuse_reward=reuse_reward,
                fusion_reward=fusion_reward,
                bandwidth_spike_penalty=bandwidth_penalty,
                memory_conflict_penalty=memory_penalty,
                pairwise_total=pairwise_total,
                total_energy=total_energy,
            )
        )
        report["objective"] = {
            "formulation": "energy_based",
            "objective_score": total_energy,
            "unary_source": "cost_model.total_cost",
        }
        return report
