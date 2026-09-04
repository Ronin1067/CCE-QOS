from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Sequence

from graph_builder import WorkloadGraph


@dataclass
class BandwidthReport:
    bandwidth_congestion: float
    pipeline_stalls: float
    avg_utilization: float
    read_utilization: float
    write_utilization: float
    backlog_pressure: float
    violations: Dict[str, float]


class BandwidthEstimator:
    def __init__(self, config: Dict):
        fallback = float(config.get("max_bytes_per_cycle", 8.0))
        self.read_capacity = float(config.get("read_bytes_per_cycle", fallback))
        self.write_capacity = float(config.get("write_bytes_per_cycle", fallback))
        self.stall_factor = float(config.get("stall_factor", 1.3))
        self.burst_sensitivity = float(config.get("burst_sensitivity", 0.5))
        self.window_size = int(config.get("bandwidth_window", 3))
        self.backlog_decay = float(config.get("backlog_decay", 0.68))

    def simulate(self, graph: WorkloadGraph, order: Sequence[str]) -> BandwidthReport:
        congestion = 0.0
        stalls = 0.0
        util_sum = 0.0
        read_util_sum = 0.0
        write_util_sum = 0.0
        violations = 0.0
        backlog_pressure = 0.0

        read_window = deque(maxlen=max(1, self.window_size))
        write_window = deque(maxlen=max(1, self.window_size))
        read_backlog = 0.0
        write_backlog = 0.0

        for node_id in order:
            node = graph.nodes[node_id]
            cycles = max(1.0, node.compute_cycles)

            read_demand = node.input_size / cycles
            write_demand = node.output_size / cycles

            read_backlog *= self.backlog_decay
            write_backlog *= self.backlog_decay
            read_demand += read_backlog
            write_demand += write_backlog

            read_window.append(read_demand)
            write_window.append(write_demand)

            burst_read = sum(read_window) / len(read_window)
            burst_write = sum(write_window) / len(write_window)

            read_util = burst_read / max(1e-6, self.read_capacity)
            write_util = burst_write / max(1e-6, self.write_capacity)
            util = max(read_util, write_util)

            util_sum += min(util, 1.0)
            read_util_sum += min(read_util, 1.0)
            write_util_sum += min(write_util, 1.0)

            overflow_read = max(0.0, read_util - 1.0)
            overflow_write = max(0.0, write_util - 1.0)
            overflow = max(overflow_read, overflow_write)

            if overflow > 0:
                traffic = node.input_size + node.output_size
                burst_scale = 1.0 + 0.15 * max(0, len(read_window) - 1)
                asymmetry = abs(read_util - write_util)
                congestion += overflow * traffic * self.burst_sensitivity * burst_scale * (1.0 + 0.2 * asymmetry)
                stalls += overflow * cycles * self.stall_factor * (1.0 + 0.15 * asymmetry)
                violations += 1.0

                read_backlog += overflow_read * self.read_capacity * 0.4
                write_backlog += overflow_write * self.write_capacity * 0.4

            backlog_pressure += read_backlog + write_backlog

        avg_util = util_sum / max(1, len(order))
        avg_read_util = read_util_sum / max(1, len(order))
        avg_write_util = write_util_sum / max(1, len(order))

        return BandwidthReport(
            bandwidth_congestion=congestion,
            pipeline_stalls=stalls,
            avg_utilization=avg_util,
            read_utilization=avg_read_util,
            write_utilization=avg_write_util,
            backlog_pressure=backlog_pressure / max(1.0, len(order)),
            violations={
                "bandwidth_capacity": violations,
                "bandwidth_imbalance": abs(avg_read_util - avg_write_util),
            },
        )
