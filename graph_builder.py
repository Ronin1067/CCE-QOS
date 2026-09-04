from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from core_types import DVFSState, HardwareConfig, MemoryLevel, OperatorGraph, OperatorNode

# Backward-compatible aliases for older imports.
OperationNode = OperatorNode
WorkloadGraph = OperatorGraph


def _validate_workload(raw: Mapping[str, object]) -> None:
    nodes = raw.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Workload must contain a top-level 'nodes' list.")

    ids: set[str] = set()
    for idx, entry in enumerate(nodes):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Node at index {idx} must be an object.")
        node_id = entry.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise ValueError(f"Node at index {idx} has an invalid 'id'.")
        if node_id in ids:
            raise ValueError(f"Duplicate node id: {node_id}")
        ids.add(node_id)

    for entry in nodes:
        deps = entry.get("dependencies", [])
        if not isinstance(deps, list):
            raise ValueError(f"Node '{entry.get('id')}' has non-list dependencies.")
        for dep in deps:
            if dep not in ids:
                raise ValueError(f"Node '{entry.get('id')}' depends on unknown node '{dep}'.")


def _name_to_idx(names: Sequence[str]) -> Dict[str, int]:
    return {name: idx for idx, name in enumerate(names)}


def _derive_reuse_groups(
    names: Sequence[str],
    deps_by_name: Mapping[str, Sequence[str]],
) -> Dict[str, List[str]]:
    """
    Reuse candidates are approximated by:
    - direct producer/consumer relations
    - siblings sharing a dependency
    """
    reuse: Dict[str, set[str]] = {name: set() for name in names}
    for name in names:
        for dep in deps_by_name.get(name, []):
            reuse[name].add(dep)
            reuse[dep].add(name)

    children: Dict[str, List[str]] = defaultdict(list)
    for name in names:
        for dep in deps_by_name.get(name, []):
            children[dep].append(name)

    for siblings in children.values():
        for i in range(len(siblings)):
            for j in range(i + 1, len(siblings)):
                left, right = siblings[i], siblings[j]
                reuse[left].add(right)
                reuse[right].add(left)

    return {name: sorted(group) for name, group in reuse.items()}


def _derive_fusible_with(
    names: Sequence[str],
    deps_by_name: Mapping[str, Sequence[str]],
    types_by_name: Mapping[str, str],
) -> Dict[str, List[str]]:
    """
    Fusibility prior:
    - parent/child relations
    - same-type ops with direct edges
    """
    fusible: Dict[str, set[str]] = {name: set() for name in names}
    for dst in names:
        for src in deps_by_name.get(dst, []):
            fusible[dst].add(src)
            fusible[src].add(dst)
            if types_by_name.get(src) == types_by_name.get(dst):
                fusible[dst].add(src)
                fusible[src].add(dst)
    return {name: sorted(group) for name, group in fusible.items()}


def load_operator_graph(path: str | Path, flops_per_cycle: float = 1e6) -> OperatorGraph:
    """
    Load a workload JSON into canonical OperatorGraph format.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_workload(raw)

    raw_nodes = list(raw["nodes"])  # validated shape above
    names = [str(entry["id"]) for entry in raw_nodes]
    name_to_id = _name_to_idx(names)

    deps_by_name = {
        str(entry["id"]): [str(dep) for dep in entry.get("dependencies", [])]
        for entry in raw_nodes
    }
    types_by_name = {str(entry["id"]): str(entry.get("type", "op")) for entry in raw_nodes}

    reuse_by_name = _derive_reuse_groups(names, deps_by_name)
    fusible_by_name = _derive_fusible_with(names, deps_by_name, types_by_name)

    nodes: List[OperatorNode] = []
    edges: List[Tuple[int, int]] = []

    for entry in raw_nodes:
        name = str(entry["id"])
        node_id = name_to_id[name]
        deps = [name_to_id[dep] for dep in deps_by_name[name]]
        for dep in deps:
            edges.append((dep, node_id))

        compute_cycles = int(entry.get("compute_cycles", 1))
        flops = float(entry.get("flops", max(1, compute_cycles) * flops_per_cycle))
        input_bytes = float(entry.get("input_bytes", entry.get("input_size", 0.0)))
        output_bytes = float(entry.get("output_bytes", entry.get("output_size", 0.0)))

        node = OperatorNode(
            id=node_id,
            name=name,
            op_type=str(entry.get("type", "op")),
            flops=flops,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            compute_cycles=compute_cycles,
            dependencies=deps,
            reuse_groups=[name_to_id[n] for n in reuse_by_name[name] if n in name_to_id],
            fusible_with=[name_to_id[n] for n in fusible_by_name[name] if n in name_to_id],
            attrs=dict(entry.get("attrs", {})),
        )
        nodes.append(node)

    return OperatorGraph(nodes=nodes, edges=edges)


def load_hardware_config(config: Mapping[str, object], num_nodes: int) -> HardwareConfig:
    """
    Convert experiment config sections into the canonical HardwareConfig.
    """
    hw = dict(config)

    resources_raw = hw.get("resources", ["npu0"])
    if isinstance(resources_raw, list):
        resources = [str(x) for x in resources_raw] or ["npu0"]
    else:
        resources = ["npu0"]

    memory_levels_cfg = hw.get("memory_levels")
    if isinstance(memory_levels_cfg, list) and memory_levels_cfg:
        memory_levels = []
        for item in memory_levels_cfg:
            if not isinstance(item, Mapping):
                continue
            memory_levels.append(
                MemoryLevel(
                    name=str(item.get("name", "L2")),
                    capacity_bytes=float(item.get("capacity_bytes", 0.0)),
                    bandwidth_gbps=float(item.get("bandwidth_gbps", 1.0)),
                )
            )
    else:
        memory_levels = [
            MemoryLevel(name="L1", capacity_bytes=float(hw.get("sram_capacity", 0.0)), bandwidth_gbps=250.0),
            MemoryLevel(name="L2", capacity_bytes=float(hw.get("sram_capacity", 0.0)) * 4.0, bandwidth_gbps=120.0),
            MemoryLevel(name="DRAM", capacity_bytes=float(hw.get("dram_capacity", 1e12)), bandwidth_gbps=40.0),
        ]

    dvfs_cfg = hw.get("dvfs_states")
    if isinstance(dvfs_cfg, list) and dvfs_cfg:
        dvfs_states = []
        for item in dvfs_cfg:
            if not isinstance(item, Mapping):
                continue
            dvfs_states.append(
                DVFSState(
                    name=str(item.get("name", "nominal")),
                    freq_ghz=float(item.get("freq_ghz", 1.0)),
                    voltage_v=float(item.get("voltage_v", 1.0)),
                    energy_per_cycle=float(item.get("energy_per_cycle", 1.0)),
                )
            )
    else:
        dvfs_states = [
            DVFSState(name="eco", freq_ghz=0.8, voltage_v=0.78, energy_per_cycle=0.9),
            DVFSState(name="nominal", freq_ghz=1.0, voltage_v=0.9, energy_per_cycle=1.0),
            DVFSState(name="turbo", freq_ghz=1.2, voltage_v=1.0, energy_per_cycle=1.25),
        ]

    max_time_slots = int(hw.get("max_time_slots", max(1, num_nodes)))
    bank_count = int(hw.get("sram_banks", hw.get("bank_count", max(1, len(resources)))))

    return HardwareConfig(
        resources=resources,
        memory_levels=memory_levels,
        dvfs_states=dvfs_states,
        max_time_slots=max_time_slots,
        bank_count=bank_count,
    )


# Backward-compatible function name used by older code.
def load_workload(path: str | Path) -> OperatorGraph:
    return load_operator_graph(path)
