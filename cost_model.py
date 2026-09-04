from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence

from bandwidth_estimator import BandwidthEstimator
from fusion_logic import FusionLogic
from graph_builder import WorkloadGraph
from memory_hierarchy import MemoryHierarchy


@dataclass
class CostBreakdown:
    dram_access: float
    sram_reuse_loss: float
    bandwidth_congestion: float
    pipeline_stalls: float
    fusion_gain: float
    parallelism_loss: float
    penalty_cost: float
    total_cost: float


class ScheduleCostModel:
    def __init__(
        self,
        memory_hierarchy: MemoryHierarchy,
        bandwidth_estimator: BandwidthEstimator,
        fusion_logic: FusionLogic,
        weights: Dict,
    ):
        self.memory_hierarchy = memory_hierarchy
        self.bandwidth_estimator = bandwidth_estimator
        self.fusion_logic = fusion_logic
        self.weights = {
            "dram_access": float(weights.get("dram_access", 1.0)),
            "sram_reuse_loss": float(weights.get("sram_reuse_loss", 1.0)),
            "bandwidth_congestion": float(weights.get("bandwidth_congestion", 1.0)),
            "pipeline_stalls": float(weights.get("pipeline_stalls", 1.0)),
            "fusion_gain": float(weights.get("fusion_gain", 1.0)),
            "parallelism_loss": float(weights.get("parallelism_loss", 1.0)),
        }

    def _parallelism_loss(self, graph: WorkloadGraph, order: Sequence[str], critical_path: Dict[str, float]) -> float:
        cp_scale = max(1.0, max(critical_path.values()))

        scheduled = set()
        loss = 0.0
        for node_id in order:
            ready = graph.ready_nodes(scheduled)
            node = graph.nodes[node_id]
            local_frontier = len(ready)
            node_cp = critical_path[node_id]

            if ready:
                best_ready_cp = max(critical_path[r] for r in ready)
                slack = max(0.0, best_ready_cp - node_cp) / cp_scale
            else:
                slack = 0.0

            if local_frontier > 1:
                serialization = ((local_frontier - 1) / local_frontier) * node.compute_cycles
                loss += serialization + 0.25 * slack * node.compute_cycles
            else:
                loss += 0.1 * slack * node.compute_cycles

            scheduled.add(node_id)
        return loss

    def evaluate(
        self,
        graph: WorkloadGraph,
        order: Sequence[str],
        penalties: Dict[str, float] | None = None,
    ) -> Dict:
        if not graph.is_valid_order(order):
            invalid_penalty = 1e9
            return {
                "breakdown": asdict(
                    CostBreakdown(
                        dram_access=0.0,
                        sram_reuse_loss=0.0,
                        bandwidth_congestion=0.0,
                        pipeline_stalls=0.0,
                        fusion_gain=0.0,
                        parallelism_loss=0.0,
                        penalty_cost=invalid_penalty,
                        total_cost=invalid_penalty,
                    )
                ),
                "memory": {},
                "bandwidth": {},
                "fusion": {},
                "violation_rate": {
                    "dependency_conflict": 1.0,
                    "sram_capacity": 1.0,
                    "bandwidth_capacity": 1.0,
                    "dram_pressure": 1.0,
                },
                "cost_impact": {"dependency_conflict": 1.0},
                "feasibility": 0.0,
                "latency_cycles": float("inf"),
            }

        penalties = penalties or {}

        memory_report = self.memory_hierarchy.simulate(graph, order)
        bandwidth_report = self.bandwidth_estimator.simulate(graph, order)
        fusion_report = self.fusion_logic.estimate(graph, order)
        critical_path = graph.critical_path_cycles()
        parallelism_loss = self._parallelism_loss(graph, order, critical_path)

        dram_term = self.weights["dram_access"] * (
            memory_report.dram_access + 0.08 * memory_report.spill_count * max(1.0, memory_report.avg_sram_utilization)
        )
        reuse_term = self.weights["sram_reuse_loss"] * (
            memory_report.sram_reuse_loss + 0.45 * memory_report.bank_conflict_cycles
        )
        bw_term = self.weights["bandwidth_congestion"] * (
            bandwidth_report.bandwidth_congestion + 0.35 * bandwidth_report.backlog_pressure
        )
        stall_term = self.weights["pipeline_stalls"] * (
            bandwidth_report.pipeline_stalls + 0.22 * memory_report.bank_conflict_cycles
        )
        fusion_term = self.weights["fusion_gain"] * fusion_report.fusion_gain
        parallel_term = self.weights["parallelism_loss"] * parallelism_loss

        output_mass = sum(graph.nodes[n].output_size for n in order)
        violation_rate = {
            "sram_capacity": memory_report.violations.get("sram_capacity", 0.0) / max(1.0, len(order)),
            "bandwidth_capacity": bandwidth_report.violations.get("bandwidth_capacity", 0.0) / max(1.0, len(order)),
            "dependency_conflict": 0.0,
            "dram_pressure": memory_report.dram_access / max(1.0, output_mass),
            "memory_bank_conflict": memory_report.violations.get("memory_bank_conflict", 0.0),
            "bandwidth_imbalance": bandwidth_report.violations.get("bandwidth_imbalance", 0.0),
        }

        raw_total = dram_term + reuse_term + bw_term + stall_term - fusion_term + parallel_term

        penalty_cost = (
            penalties.get("sram_capacity", 1.0) * violation_rate["sram_capacity"] * 140.0
            + penalties.get("bandwidth_capacity", 1.0) * violation_rate["bandwidth_capacity"] * 100.0
            + penalties.get("dependency_conflict", 1.0) * violation_rate["dependency_conflict"] * 900.0
            + penalties.get("dram_pressure", 1.0) * max(0.0, violation_rate["dram_pressure"] - 1.0) * 70.0
            + penalties.get("memory_bank_conflict", 1.0) * violation_rate["memory_bank_conflict"] * 90.0
            + penalties.get("bandwidth_imbalance", 1.0) * violation_rate["bandwidth_imbalance"] * 65.0
        )

        total_cost = raw_total + penalty_cost

        positive_total = max(1.0, dram_term + reuse_term + bw_term + stall_term + parallel_term)
        cost_impact = {
            "sram_capacity": (reuse_term + 0.4 * memory_report.spill_count) / positive_total,
            "bandwidth_capacity": (bw_term + stall_term) / positive_total,
            "dependency_conflict": 0.0,
            "dram_pressure": dram_term / positive_total,
            "memory_bank_conflict": (memory_report.bank_conflict_cycles + 1.0) / positive_total,
            "bandwidth_imbalance": (bandwidth_report.backlog_pressure + 1.0) / positive_total,
        }

        violation_mass = (
            memory_report.violations.get("sram_capacity", 0.0)
            + bandwidth_report.violations.get("bandwidth_capacity", 0.0)
            + 0.5 * violation_rate["memory_bank_conflict"]
        )
        feasibility = max(0.0, 1.0 - violation_mass / max(1.0, len(order)))
        latency = (
            sum(graph.nodes[node_id].compute_cycles for node_id in order)
            + bandwidth_report.pipeline_stalls
            + memory_report.idle_cycles
            + 0.15 * bandwidth_report.backlog_pressure
        )

        breakdown = CostBreakdown(
            dram_access=dram_term,
            sram_reuse_loss=reuse_term,
            bandwidth_congestion=bw_term,
            pipeline_stalls=stall_term,
            fusion_gain=fusion_term,
            parallelism_loss=parallel_term,
            penalty_cost=penalty_cost,
            total_cost=total_cost,
        )

        return {
            "breakdown": asdict(breakdown),
            "memory": asdict(memory_report),
            "bandwidth": asdict(bandwidth_report),
            "fusion": asdict(fusion_report),
            "violation_rate": violation_rate,
            "cost_impact": cost_impact,
            "feasibility": feasibility,
            "latency_cycles": latency,
            "derived": {
                "critical_path_cycles": max(critical_path.values()),
                "frontier_mean": sum(graph.frontier_profile(order)) / max(1.0, len(order)),
            },
        }
