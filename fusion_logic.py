from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from core_types import OperatorGraph


@dataclass(slots=True)
class FusionReport:
    fused_edges: List[Tuple[int, int]]
    fusion_gain: float
    fused_triplets: List[Tuple[int, int, int]]


class FusionLogic:
    """
    Fusion heuristic used for both baseline cost modeling and analysis outputs.
    """

    def __init__(self, config: Dict) -> None:
        pair_rules = config.get("fusible_pairs", [])
        triplet_rules = config.get("fusible_triplets", [])
        self.fusible_pairs = {tuple(item) for item in pair_rules}
        self.fusible_triplets = {tuple(item) for item in triplet_rules}
        self.max_tensor_for_fusion = float(config.get("max_tensor_for_fusion", 512.0))
        self.base_gain_factor = float(config.get("base_gain_factor", 0.08))
        self.compute_overlap_factor = float(config.get("compute_overlap_factor", 0.18))
        self.locality_bonus = float(config.get("locality_bonus", 0.05))

    def _is_fusible_pair(self, left_type: str, right_type: str) -> bool:
        return (left_type, right_type) in self.fusible_pairs

    def _is_fusible_triplet(self, a_type: str, b_type: str, c_type: str) -> bool:
        return (a_type, b_type, c_type) in self.fusible_triplets

    def estimate(self, graph: OperatorGraph, order: Sequence[int]) -> FusionReport:
        levels = graph.compute_levels()
        total_gain = 0.0
        fused_edges: List[Tuple[int, int]] = []
        fused_triplets: List[Tuple[int, int, int]] = []
        consumed_edges = set()

        for idx in range(len(order) - 1):
            a_id = int(order[idx])
            b_id = int(order[idx + 1])
            a = graph.node_by_id[a_id]
            b = graph.node_by_id[b_id]

            if a_id not in b.dependencies:
                continue
            if not self._is_fusible_pair(a.op_type, b.op_type):
                continue
            if max(a.output_bytes, b.input_bytes) > self.max_tensor_for_fusion:
                continue

            edge = (a_id, b_id)
            if edge in consumed_edges:
                continue

            memory_saving = min(a.output_bytes, b.input_bytes) * self.base_gain_factor
            compute_overlap = min(a.compute_cycles, b.compute_cycles) * self.compute_overlap_factor
            level_proximity = 1.0 / (1.0 + abs(levels[a_id] - levels[b_id]))
            gain = memory_saving + compute_overlap + self.locality_bonus * level_proximity * (
                a.compute_cycles + b.compute_cycles
            )
            total_gain += gain
            fused_edges.append(edge)
            consumed_edges.add(edge)

        for idx in range(len(order) - 2):
            a_id = int(order[idx])
            b_id = int(order[idx + 1])
            c_id = int(order[idx + 2])
            a = graph.node_by_id[a_id]
            b = graph.node_by_id[b_id]
            c = graph.node_by_id[c_id]

            if a_id not in b.dependencies or b_id not in c.dependencies:
                continue
            if not self._is_fusible_triplet(a.op_type, b.op_type, c.op_type):
                continue
            if max(a.output_bytes, b.output_bytes, c.input_bytes) > self.max_tensor_for_fusion:
                continue

            triplet = (a_id, b_id, c_id)
            fused_triplets.append(triplet)
            chain_overlap = min(a.compute_cycles, b.compute_cycles, c.compute_cycles)
            triplet_gain = 0.85 * self.compute_overlap_factor * chain_overlap + 0.06 * (
                a.output_bytes + b.output_bytes + c.output_bytes
            )
            total_gain += triplet_gain

        return FusionReport(
            fused_edges=fused_edges,
            fusion_gain=total_gain,
            fused_triplets=fused_triplets,
        )
