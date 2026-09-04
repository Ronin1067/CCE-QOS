from __future__ import annotations

from typing import Dict, List


class ScheduleExplainer:
    def _top_contributors(self, breakdown: Dict) -> List[str]:
        candidates = {
            "DRAM pressure": breakdown.get("dram_access", 0.0),
            "SRAM reuse loss": breakdown.get("sram_reuse_loss", 0.0),
            "Bandwidth congestion": breakdown.get("bandwidth_congestion", 0.0),
            "Pipeline stalls": breakdown.get("pipeline_stalls", 0.0),
            "Parallelism loss": breakdown.get("parallelism_loss", 0.0),
        }
        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        return [name for name, _ in ranked[:2]]

    def explain(self, strategy: str, result: Dict, summary: Dict) -> str:
        evaluation = result["evaluation"]
        breakdown = evaluation["breakdown"]
        memory = evaluation.get("memory", {})
        bandwidth = evaluation.get("bandwidth", {})
        fusion = evaluation.get("fusion", {})
        derived = evaluation.get("derived", {})
        drivers = self._top_contributors(breakdown)

        lines = []
        lines.append(f"[{strategy}]")
        lines.append(
            f"Total cost {summary['total_cost']:.2f}, latency {summary['latency']:.2f} cycles, feasibility {summary['feasibility_percent']:.2f}%."
        )
        if "energy_breakdown" in evaluation:
            energy = evaluation["energy_breakdown"]
            lines.append(
                "Energy objective: "
                f"total {energy.get('total_energy', 0.0):.2f} = unary {energy.get('unary_cost', 0.0):.2f} "
                f"+ pairwise {energy.get('pairwise_total', 0.0):.2f}."
            )
        lines.append(
            f"Main contributors were {drivers[0]} and {drivers[1]}. Fusion captured {len(fusion.get('fused_edges', []))} pair merges and {len(fusion.get('fused_triplets', []))} triplet merges."
        )
        lines.append(
            f"Memory profile: peak SRAM {memory.get('peak_sram_usage', 0.0):.2f}, spills {memory.get('spill_count', 0.0):.2f}, average utilization {memory.get('avg_sram_utilization', 0.0):.2f}."
        )
        lines.append(
            f"Bandwidth profile: read util {bandwidth.get('read_utilization', 0.0)*100.0:.2f}%, write util {bandwidth.get('write_utilization', 0.0)*100.0:.2f}%, backlog pressure {bandwidth.get('backlog_pressure', 0.0):.2f}."
        )
        lines.append(f"Frontier mean during execution was {derived.get('frontier_mean', 0.0):.2f}.")

        if "APR" in strategy:
            trace = result.get("metadata", {}).get("round_trace", [])
            if trace:
                first = trace[0]["penalties"]
                last = trace[-1]["penalties"]
                lines.append(
                    "APR trajectory: "
                    f"sram_capacity {first.get('sram_capacity', 0.0):.2f}->{last.get('sram_capacity', 0.0):.2f}, "
                    f"bandwidth_capacity {first.get('bandwidth_capacity', 0.0):.2f}->{last.get('bandwidth_capacity', 0.0):.2f}, "
                    f"bank_conflict {first.get('memory_bank_conflict', 0.0):.2f}->{last.get('memory_bank_conflict', 0.0):.2f}."
                )

        if strategy.startswith("Quantum"):
            backend = result.get("metadata", {}).get("backend", "qaoa_style_multiwalker_simulator")
            walkers = result.get("metadata", {}).get("walkers", 0)
            lines.append(f"Quantum refinement used backend '{backend}' with {walkers} walkers.")

        return "\n".join(lines)
