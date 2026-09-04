from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Set


@dataclass
class OperationNode:
    node_id: str
    op_type: str
    compute_cycles: int
    input_size: float
    output_size: float
    dependencies: List[str] = field(default_factory=list)
    attrs: Dict[str, float] = field(default_factory=dict)


@dataclass
class WorkloadGraph:
    nodes: Dict[str, OperationNode]
    children: Dict[str, List[str]]
    _topo_cache: List[str] | None = field(default=None, init=False, repr=False)
    _levels_cache: Dict[str, int] | None = field(default=None, init=False, repr=False)
    _critical_path_cache: Dict[str, float] | None = field(default=None, init=False, repr=False)
    _desc_cache: Dict[str, int] | None = field(default=None, init=False, repr=False)

    @property
    def indegree(self) -> Dict[str, int]:
        deg = {node_id: 0 for node_id in self.nodes}
        for node in self.nodes.values():
            for dep in node.dependencies:
                deg[node.node_id] += 1
        return deg

    def topological_order(self) -> List[str]:
        if self._topo_cache is not None:
            return list(self._topo_cache)

        deg = self.indegree
        ready = deque(sorted([node_id for node_id, v in deg.items() if v == 0]))
        order: List[str] = []

        while ready:
            current = ready.popleft()
            order.append(current)
            for child in self.children.get(current, []):
                deg[child] -= 1
                if deg[child] == 0:
                    ready.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected in workload graph.")
        self._topo_cache = list(order)
        return order

    def ready_nodes(self, scheduled: Set[str]) -> List[str]:
        ready: List[str] = []
        for node_id, node in self.nodes.items():
            if node_id in scheduled:
                continue
            if all(dep in scheduled for dep in node.dependencies):
                ready.append(node_id)
        return ready

    def is_valid_order(self, order: Iterable[str]) -> bool:
        order_list = list(order)
        if len(order_list) != len(self.nodes):
            return False
        if len(set(order_list)) != len(order_list):
            return False
        if any(node_id not in self.nodes for node_id in order_list):
            return False

        pos = {node_id: idx for idx, node_id in enumerate(order_list)}
        for node_id, node in self.nodes.items():
            for dep in node.dependencies:
                if pos[dep] >= pos[node_id]:
                    return False
        return True

    def compute_levels(self) -> Dict[str, int]:
        if self._levels_cache is not None:
            return dict(self._levels_cache)

        order = self.topological_order()
        levels: Dict[str, int] = {}
        for node_id in order:
            deps = self.nodes[node_id].dependencies
            if not deps:
                levels[node_id] = 0
            else:
                levels[node_id] = 1 + max(levels[dep] for dep in deps)
        self._levels_cache = dict(levels)
        return levels

    def critical_path_cycles(self) -> Dict[str, float]:
        if self._critical_path_cache is not None:
            return dict(self._critical_path_cache)

        reverse_order = list(reversed(self.topological_order()))
        cp: Dict[str, float] = {}
        for node_id in reverse_order:
            node = self.nodes[node_id]
            if not self.children[node_id]:
                cp[node_id] = float(node.compute_cycles)
            else:
                cp[node_id] = float(node.compute_cycles) + max(cp[ch] for ch in self.children[node_id])
        self._critical_path_cache = dict(cp)
        return cp

    def descendant_count(self) -> Dict[str, int]:
        if self._desc_cache is not None:
            return dict(self._desc_cache)

        reverse_order = list(reversed(self.topological_order()))
        descendants: Dict[str, Set[str]] = {node_id: set() for node_id in self.nodes}
        for node_id in reverse_order:
            for ch in self.children[node_id]:
                descendants[node_id].add(ch)
                descendants[node_id].update(descendants[ch])
        collapsed = {node_id: len(desc) for node_id, desc in descendants.items()}
        self._desc_cache = dict(collapsed)
        return collapsed

    def frontier_profile(self, order: Iterable[str]) -> List[int]:
        scheduled = set()
        profile: List[int] = []
        for node_id in order:
            profile.append(len(self.ready_nodes(scheduled)))
            scheduled.add(node_id)
        return profile


def _validate_workload(raw: Dict) -> None:
    if "nodes" not in raw or not isinstance(raw["nodes"], list):
        raise ValueError("Workload file must have a top-level 'nodes' list.")
    node_ids = set()
    for entry in raw["nodes"]:
        if "id" not in entry:
            raise ValueError("Each node needs an 'id'.")
        if entry["id"] in node_ids:
            raise ValueError(f"Duplicate node id found: {entry['id']}")
        node_ids.add(entry["id"])

    for entry in raw["nodes"]:
        deps = entry.get("dependencies", [])
        for dep in deps:
            if dep not in node_ids:
                raise ValueError(f"Node {entry['id']} depends on unknown node {dep}.")


def load_workload(path: str | Path) -> WorkloadGraph:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    _validate_workload(raw)

    nodes: Dict[str, OperationNode] = {}
    children: Dict[str, List[str]] = defaultdict(list)
    for entry in raw["nodes"]:
        node = OperationNode(
            node_id=entry["id"],
            op_type=entry.get("type", "op"),
            compute_cycles=int(entry.get("compute_cycles", 1)),
            input_size=float(entry.get("input_size", 0.0)),
            output_size=float(entry.get("output_size", 0.0)),
            dependencies=list(entry.get("dependencies", [])),
            attrs=dict(entry.get("attrs", {})),
        )
        nodes[node.node_id] = node
        for dep in node.dependencies:
            children[dep].append(node.node_id)

    for node_id in nodes:
        children.setdefault(node_id, [])

    return WorkloadGraph(nodes=nodes, children=dict(children))
