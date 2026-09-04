from __future__ import annotations

from typing import Dict


class ScheduleAnalysis:
    def summarize(self, strategy: str, evaluation: Dict) -> Dict:
        breakdown = evaluation["breakdown"]
        energy = evaluation.get("energy_breakdown", {})
        memory = evaluation["memory"]
        bandwidth = evaluation["bandwidth"]
        derived = evaluation.get("derived", {})

        return {
            "strategy": strategy,
            "total_cost": round(float(breakdown["total_cost"]), 3),
            "total_energy": round(float(energy.get("total_energy", breakdown["total_cost"])), 3),
            "pairwise_energy": round(float(energy.get("pairwise_total", 0.0)), 3),
            "latency": round(float(evaluation["latency_cycles"]), 3),
            "DRAM_usage": round(float(memory.get("dram_access", 0.0)), 3),
            "bandwidth_utilization": round(float(bandwidth.get("avg_utilization", 0.0)) * 100.0, 2),
            "idle_cycles": round(float(memory.get("idle_cycles", 0.0) + bandwidth.get("pipeline_stalls", 0.0)), 3),
            "feasibility_percent": round(float(evaluation["feasibility"]) * 100.0, 2),
            "peak_sram": round(float(memory.get("peak_sram_usage", 0.0)), 3),
            "spill_count": round(float(memory.get("spill_count", 0.0)), 3),
            "frontier_mean": round(float(derived.get("frontier_mean", 0.0)), 3),
        }

    def comparison_table(self, summaries: Dict[str, Dict]) -> str:
        header = (
            f"{'Strategy':<22}"
            f"{'Cost':>12}"
            f"{'Energy':>12}"
            f"{'Pairwise':>12}"
            f"{'Latency':>12}"
            f"{'DRAM':>12}"
            f"{'BW%':>10}"
            f"{'Idle':>12}"
            f"{'Feasible%':>12}"
            f"{'PeakSRAM':>12}"
            f"{'Spills':>10}"
        )
        line = "-" * len(header)
        rows = [header, line]

        ranked = sorted(summaries.values(), key=lambda item: item["total_cost"])
        for item in ranked:
            rows.append(
                f"{item['strategy']:<22}"
                f"{item['total_cost']:>12.2f}"
                f"{item['total_energy']:>12.2f}"
                f"{item['pairwise_energy']:>12.2f}"
                f"{item['latency']:>12.2f}"
                f"{item['DRAM_usage']:>12.2f}"
                f"{item['bandwidth_utilization']:>10.2f}"
                f"{item['idle_cycles']:>12.2f}"
                f"{item['feasibility_percent']:>12.2f}"
                f"{item['peak_sram']:>12.2f}"
                f"{item['spill_count']:>10.2f}"
            )
        return "\n".join(rows)
