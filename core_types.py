from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


@dataclass(slots=True)
class OperatorNode:
    """
    Canonical operator representation used across classical and quantum paths.
    """

    id: int
    name: str
    op_type: str
    flops: float
    input_bytes: float
    output_bytes: float
    compute_cycles: int
    dependencies: List[int] = field(default_factory=list)
    reuse_groups: List[int] = field(default_factory=list)
    fusible_with: List[int] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OperatorGraph:
    """
    DAG of operators with convenience methods for schedule search.
    """

    nodes: List[OperatorNode]
    edges: List[Tuple[int, int]]
    _node_map: Dict[int, OperatorNode] = field(default_factory=dict, init=False, repr=False)
    _children_map: Dict[int, List[int]] = field(default_factory=dict, init=False, repr=False)
    _topo_cache: List[int] | None = field(default=None, init=False, repr=False)
    _levels_cache: Dict[int, int] | None = field(default=None, init=False, repr=False)
    _critical_path_cache: Dict[int, float] | None = field(default=None, init=False, repr=False)
    _desc_cache: Dict[int, int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._node_map = {int(node.id): node for node in self.nodes}
        self._children_map = {int(node.id): [] for node in self.nodes}
        for src, dst in self.edges:
            src_id, dst_id = int(src), int(dst)
            if src_id not in self._node_map or dst_id not in self._node_map:
                raise ValueError(f"Edge references unknown node id: ({src_id}, {dst_id})")
            self._children_map[src_id].append(dst_id)
        for node in self.nodes:
            for dep in node.dependencies:
                if int(dep) not in self._node_map:
                    raise ValueError(f"Node {node.id} depends on unknown node {dep}")

    @property
    def node_by_id(self) -> Dict[int, OperatorNode]:
        return self._node_map

    @property
    def children_by_id(self) -> Dict[int, List[int]]:
        return self._children_map

    @property
    def indegree(self) -> Dict[int, int]:
        deg = {node_id: 0 for node_id in self._node_map}
        for node in self.nodes:
            deg[int(node.id)] = len(node.dependencies)
        return deg

    def topological_order(self) -> List[int]:
        if self._topo_cache is not None:
            return list(self._topo_cache)

        indeg = self.indegree
        ready = deque(sorted(node_id for node_id, val in indeg.items() if val == 0))
        order: List[int] = []
        while ready:
            current = ready.popleft()
            order.append(current)
            for child in self._children_map.get(current, []):
                indeg[child] -= 1
                if indeg[child] == 0:
                    ready.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected in OperatorGraph.")
        self._topo_cache = list(order)
        return order

    def ready_nodes(self, scheduled: Set[int]) -> List[int]:
        ready: List[int] = []
        for node in self.nodes:
            node_id = int(node.id)
            if node_id in scheduled:
                continue
            if all(int(dep) in scheduled for dep in node.dependencies):
                ready.append(node_id)
        return sorted(ready)

    def is_valid_order(self, order: Iterable[int]) -> bool:
        order_list = [int(node_id) for node_id in order]
        if len(order_list) != len(self.nodes):
            return False
        if len(set(order_list)) != len(order_list):
            return False
        if any(node_id not in self._node_map for node_id in order_list):
            return False

        pos = {node_id: idx for idx, node_id in enumerate(order_list)}
        for node in self.nodes:
            node_id = int(node.id)
            for dep in node.dependencies:
                dep_id = int(dep)
                if pos[dep_id] >= pos[node_id]:
                    return False
        return True

    def compute_levels(self) -> Dict[int, int]:
        if self._levels_cache is not None:
            return dict(self._levels_cache)

        levels: Dict[int, int] = {}
        for node_id in self.topological_order():
            deps = self._node_map[node_id].dependencies
            if not deps:
                levels[node_id] = 0
            else:
                levels[node_id] = 1 + max(levels[int(dep)] for dep in deps)
        self._levels_cache = dict(levels)
        return levels

    def critical_path_cycles(self) -> Dict[int, float]:
        if self._critical_path_cache is not None:
            return dict(self._critical_path_cache)

        cp: Dict[int, float] = {}
        for node_id in reversed(self.topological_order()):
            node = self._node_map[node_id]
            if not self._children_map[node_id]:
                cp[node_id] = float(node.compute_cycles)
            else:
                cp[node_id] = float(node.compute_cycles) + max(cp[ch] for ch in self._children_map[node_id])
        self._critical_path_cache = dict(cp)
        return cp

    def descendant_count(self) -> Dict[int, int]:
        if self._desc_cache is not None:
            return dict(self._desc_cache)

        descendants: Dict[int, Set[int]] = {int(node.id): set() for node in self.nodes}
        for node_id in reversed(self.topological_order()):
            for child in self._children_map[node_id]:
                descendants[node_id].add(child)
                descendants[node_id].update(descendants[child])
        collapsed = {node_id: len(desc) for node_id, desc in descendants.items()}
        self._desc_cache = dict(collapsed)
        return collapsed

    def frontier_profile(self, order: Sequence[int]) -> List[int]:
        scheduled: Set[int] = set()
        profile: List[int] = []
        for node_id in order:
            profile.append(len(self.ready_nodes(scheduled)))
            scheduled.add(int(node_id))
        return profile


@dataclass(slots=True)
class MemoryLevel:
    name: str
    capacity_bytes: float
    bandwidth_gbps: float


@dataclass(slots=True)
class DVFSState:
    name: str
    freq_ghz: float
    voltage_v: float
    energy_per_cycle: float


@dataclass(slots=True)
class HardwareConfig:
    resources: List[str]
    memory_levels: List[MemoryLevel]
    dvfs_states: List[DVFSState]
    max_time_slots: int
    bank_count: int = 0


@dataclass(slots=True)
class Schedule:
    order: List[int]
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ScheduleMetrics:
    objective: float
    total_cost: float
    latency_cycles: float
    dram_bytes: float
    bandwidth_utilization: float
    idle_cycles: float
    feasibility: float
    runtime_seconds: float
