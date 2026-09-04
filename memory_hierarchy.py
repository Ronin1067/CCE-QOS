from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Sequence

from core_types import OperatorGraph


@dataclass(slots=True)
class MemoryReport:
    dram_access: float
    sram_reuse_loss: float
    peak_sram_usage: float
    idle_cycles: float
    bank_conflict_cycles: float
    spill_count: float
    avg_sram_utilization: float
    prefetch_bytes_saved: float
    violations: Dict[str, float]


class MemoryHierarchy:
    """
    SRAM/DRAM simulation used by the classical cost model and diagnostics.
    """

    def __init__(self, config: Dict) -> None:
        self.sram_capacity = float(config.get("sram_capacity", 1024.0))
        self.bank_count = int(config.get("sram_banks", config.get("bank_count", 4)))
        self.eviction_idle_factor = float(config.get("eviction_idle_factor", 0.03))
        self.write_back_factor = float(config.get("write_back_factor", 1.0))
        self.prefetch_ratio = float(config.get("prefetch_ratio", 0.25))
        self.prefetch_slots = int(config.get("prefetch_slots", 2))
        self.bank_conflict_factor = float(config.get("bank_conflict_factor", 0.12))

    def _bank_of(self, tensor_name: str) -> int:
        return sum(ord(ch) for ch in tensor_name) % max(1, self.bank_count)

    def simulate(self, graph: OperatorGraph, order: Sequence[int]) -> MemoryReport:
        out_degree = {node_id: len(graph.children_by_id[node_id]) for node_id in graph.node_by_id}
        remaining_consumers = dict(out_degree)

        sram_tensors: OrderedDict[int, Dict[str, float]] = OrderedDict()
        current_usage = 0.0
        usage_integral = 0.0

        dram_access = 0.0
        reuse_loss = 0.0
        idle_cycles = 0.0
        bank_conflict_cycles = 0.0
        spill_count = 0.0
        sram_violations = 0.0
        prefetch_bytes_saved = 0.0
        peak_usage = 0.0
        prefetch_credit = self.prefetch_slots

        def victim_for_eviction(step: int) -> int | None:
            if not sram_tensors:
                return None

            scored = []
            for tensor_id, entry in sram_tensors.items():
                rem = remaining_consumers.get(tensor_id, 0.0)
                age = max(0.0, step - entry["last_access"])
                size = entry["size"]
                score = 2.1 * rem + 0.03 * size - 0.04 * age
                scored.append((score, tensor_id))
            scored.sort(key=lambda item: item[0])
            return scored[0][1]

        def evict_until(target_size: float, step: int) -> None:
            nonlocal current_usage, reuse_loss, idle_cycles, sram_violations
            while current_usage + target_size > self.sram_capacity and sram_tensors:
                victim_id = victim_for_eviction(step)
                if victim_id is None:
                    break
                entry = sram_tensors.pop(victim_id)
                current_usage -= entry["size"]
                rem = remaining_consumers.get(victim_id, 0.0)
                if rem > 0:
                    reuse_loss += entry["size"] * (0.14 + 0.04 * rem)
                idle_cycles += entry["size"] * self.eviction_idle_factor
            if current_usage + target_size > self.sram_capacity:
                sram_violations += 1.0

        for step, node_id in enumerate(order):
            node = graph.node_by_id[node_id]
            bank_reads: Dict[int, int] = {}

            if not node.dependencies:
                root_fetch = node.input_bytes * (1.0 - self.prefetch_ratio * 0.6)
                dram_access += root_fetch
                prefetch_bytes_saved += node.input_bytes - root_fetch

            for dep_id in node.dependencies:
                dep = graph.node_by_id[dep_id]
                bank = self._bank_of(dep.name)
                bank_reads[bank] = bank_reads.get(bank, 0) + 1

                if dep_id in sram_tensors:
                    sram_tensors[dep_id]["last_access"] = float(step)
                    sram_tensors.move_to_end(dep_id)
                else:
                    fetch_bytes = dep.output_bytes
                    if prefetch_credit > 0:
                        saved = fetch_bytes * self.prefetch_ratio
                        fetch_bytes -= saved
                        prefetch_bytes_saved += saved
                        prefetch_credit -= 1
                    dram_access += fetch_bytes
                    reuse_loss += dep.output_bytes * 0.11

                remaining_consumers[dep_id] = max(0.0, remaining_consumers.get(dep_id, 0.0) - 1.0)
                if remaining_consumers[dep_id] == 0 and dep_id in sram_tensors:
                    current_usage -= sram_tensors.pop(dep_id)["size"]

            for reads in bank_reads.values():
                if reads > 1:
                    bank_conflict_cycles += (reads - 1) * self.bank_conflict_factor * max(1.0, node.compute_cycles)

            prefetch_credit = min(self.prefetch_slots, prefetch_credit + 1)

            if node.output_bytes <= self.sram_capacity:
                evict_until(node.output_bytes, step)
                if current_usage + node.output_bytes <= self.sram_capacity:
                    sram_tensors[node_id] = {
                        "size": float(node.output_bytes),
                        "bank": float(self._bank_of(node.name)),
                        "last_access": float(step),
                    }
                    current_usage += node.output_bytes
                    peak_usage = max(peak_usage, current_usage)
                else:
                    dram_access += node.output_bytes * self.write_back_factor
                    spill_count += 1.0
            else:
                dram_access += node.output_bytes * self.write_back_factor
                spill_count += 1.0
                sram_violations += 1.0

            usage_integral += current_usage

        avg_util = usage_integral / max(1.0, len(order) * max(1.0, self.sram_capacity))
        idle_cycles += 0.25 * bank_conflict_cycles

        return MemoryReport(
            dram_access=dram_access,
            sram_reuse_loss=reuse_loss,
            peak_sram_usage=peak_usage,
            idle_cycles=idle_cycles,
            bank_conflict_cycles=bank_conflict_cycles,
            spill_count=spill_count,
            avg_sram_utilization=avg_util,
            prefetch_bytes_saved=prefetch_bytes_saved,
            violations={
                "sram_capacity": sram_violations,
                "memory_bank_conflict": bank_conflict_cycles / max(1.0, len(order) * 100.0),
            },
        )
