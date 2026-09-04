from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

from qubo_types import QUBOData, QuboKey, VarIndex


class EnergyModel:
    """
    Build the Constraint-Coupled Energy QUBO (CCE-QUBO) for NPU scheduling.

    The generated objective has the form:
        E(z) = sum_i a_i z_i + sum_{i<=j} b_ij z_i z_j + const
    where z is a binary vector containing core decision variables (x, m, f)
    and auxiliary variables introduced to keep higher-order effects quadratic.
    """

    def __init__(
        self,
        graph: "OperatorGraph",
        hw: "HardwareConfig",
        alpha: Dict[str, float],
        beta: Dict[str, float],
        gamma: Dict[str, float],
        initial_penalties: Dict[str, float],
    ) -> None:
        """
        Parameters map directly to the CCE-QUBO formulation:
        - alpha: unary-term weights
        - beta: pairwise-term weights
        - gamma: higher-order / auxiliary-term weights
        - initial_penalties: constraint penalties lambda_k
        """
        self.graph = graph
        self.hw = hw

        self.alpha: Dict[str, float] = {"comp": 1.0, "energy": 1.0, "lat": 1.0, "dvfs": 1.0}
        self.alpha.update({k: float(v) for k, v in alpha.items()})

        self.beta: Dict[str, float] = {"reuse": 0.0, "fuse": 0.0, "bw": 0.0}
        self.beta.update({k: float(v) for k, v in beta.items()})

        self.gamma: Dict[str, float] = {"bank": 0.0, "burst": 0.0, "stall": 0.0, "parallelism": 0.0}
        self.gamma.update({k: float(v) for k, v in gamma.items()})

        self.penalties: Dict[str, float] = {k: float(v) for k, v in initial_penalties.items()}

        self.var_index: Dict[Tuple[str, ...], VarIndex] = {}
        self.var_metadata: Dict[VarIndex, Dict[str, Any]] = {}
        self._next_var: int = 0

        self._x_vars: Dict[Tuple[int, int, str], VarIndex] = {}
        self._m_vars: Dict[Tuple[int, str], VarIndex] = {}
        self._f_vars: Dict[Tuple[int, str], VarIndex] = {}
        self._aux_vars: Dict[Tuple[str, ...], VarIndex] = {}

        self._node_by_id: Dict[int, Any] = {int(node.id): node for node in self.graph.nodes}
        self._edges = {(int(src), int(dst)) for src, dst in self.graph.edges}
        self._resource_to_idx = {resource: idx for idx, resource in enumerate(self.hw.resources)}

    def build_qubo(self) -> QUBOData:
        """
        Construct the complete CCE-QUBO instance:
        1) Create core variables (x, m, f)
        2) Add unary terms
        3) Add pairwise terms
        4) Add higher-order effects via auxiliaries
        5) Add APR-tunable constraint penalties
        """
        linear: Dict[VarIndex, float] = {}
        quadratic: Dict[QuboKey, float] = {}
        constant = 0.0

        self._reset_variable_state()
        self._init_core_variables()

        constant = self._add_unary_terms(linear, quadratic, constant)
        constant = self._add_pairwise_terms(linear, quadratic, constant)
        constant = self._add_higher_order_terms(linear, quadratic, constant)
        constant = self._add_constraint_penalties(linear, quadratic, constant)

        return QUBOData(
            num_variables=self._next_var,
            linear=linear,
            quadratic=quadratic,
            constant=constant,
            var_metadata=dict(self.var_metadata),
            penalty_weights=dict(self.penalties),
        )

    def _reset_variable_state(self) -> None:
        """Clear cached index mappings so each build_qubo call is deterministic."""
        self.var_index.clear()
        self.var_metadata.clear()
        self._next_var = 0
        self._x_vars.clear()
        self._m_vars.clear()
        self._f_vars.clear()
        self._aux_vars.clear()

    def _make_var_key(self, kind: str, attrs: Mapping[str, Any]) -> Tuple[str, ...]:
        return (kind, *(f"{name}={attrs[name]}" for name in sorted(attrs)))

    def _get_or_create_var(self, kind: str, **attrs: Any) -> VarIndex:
        """
        Map conceptual variables to contiguous integer indices.

        Examples:
            kind="x", attrs={"op": 3, "t": 4, "r": "npu0"}
            kind="m", attrs={"op": 3, "level": "L2"}
            kind="f", attrs={"t": 4, "state": "turbo"}
            kind="aux", attrs={"role": "bank_congestion", "bank": 1, "t": 4}
        """
        key = self._make_var_key(kind, attrs)
        if key in self.var_index:
            return self.var_index[key]

        idx = self._next_var
        self._next_var += 1

        self.var_index[key] = idx
        metadata = {"kind": kind, **attrs}
        self.var_metadata[idx] = metadata
        if kind == "aux":
            self._aux_vars[key] = idx
        return idx

    @staticmethod
    def _add_to_linear(linear: MutableMapping[VarIndex, float], idx: VarIndex, value: float) -> None:
        """Accumulate a linear term a_i z_i."""
        if value == 0.0:
            return
        linear[idx] = linear.get(idx, 0.0) + float(value)
        if abs(linear[idx]) < 1e-12:
            del linear[idx]

    @staticmethod
    def _add_to_quadratic(
        quadratic: MutableMapping[QuboKey, float],
        i: VarIndex,
        j: VarIndex,
        value: float,
    ) -> None:
        """Accumulate a quadratic term b_ij z_i z_j with canonical i <= j ordering."""
        if value == 0.0:
            return
        a, b = (i, j) if i <= j else (j, i)
        key = (a, b)
        quadratic[key] = quadratic.get(key, 0.0) + float(value)
        if abs(quadratic[key]) < 1e-12:
            del quadratic[key]

    def _add_squared_penalty(
        self,
        linear: MutableMapping[VarIndex, float],
        quadratic: MutableMapping[QuboKey, float],
        constant: float,
        terms: Mapping[VarIndex, float],
        rhs: float,
        weight: float,
    ) -> float:
        """
        Expand weight * (sum_i c_i z_i - rhs)^2 into linear + quadratic + constant.

        Binary simplification z_i^2 = z_i is applied, so diagonal terms become linear.
        """
        if weight == 0.0 or not terms:
            return constant

        for idx, coeff in terms.items():
            self._add_to_linear(linear, idx, weight * (coeff * coeff - 2.0 * rhs * coeff))

        items = list(terms.items())
        for pos, (idx_i, coeff_i) in enumerate(items):
            for idx_j, coeff_j in items[pos + 1 :]:
                self._add_to_quadratic(quadratic, idx_i, idx_j, weight * (2.0 * coeff_i * coeff_j))

        return constant + weight * (rhs * rhs)

    def _add_and_penalty(
        self,
        linear: MutableMapping[VarIndex, float],
        quadratic: MutableMapping[QuboKey, float],
        left_idx: VarIndex,
        right_idx: VarIndex,
        aux_idx: VarIndex,
        weight: float,
    ) -> None:
        """
        Enforce aux_idx ~= left_idx AND right_idx using:
            weight * (x*y - 2*x*u - 2*y*u + 3*u)
        """
        if weight == 0.0:
            return

        self._add_to_quadratic(quadratic, left_idx, right_idx, weight)
        self._add_to_quadratic(quadratic, left_idx, aux_idx, -2.0 * weight)
        self._add_to_quadratic(quadratic, right_idx, aux_idx, -2.0 * weight)
        self._add_to_linear(linear, aux_idx, 3.0 * weight)

    def _init_core_variables(self) -> None:
        """Create all x_{i,t,r}, m_{i,l}, and f_{t,s} variables."""
        for node in self.graph.nodes:
            op = int(node.id)
            for t in range(self.hw.max_time_slots):
                for resource in self.hw.resources:
                    idx = self._get_or_create_var("x", op=op, t=t, r=resource)
                    self._x_vars[(op, t, resource)] = idx

            for level in self.hw.memory_levels:
                idx = self._get_or_create_var("m", op=op, level=level.name)
                self._m_vars[(op, level.name)] = idx

        for t in range(self.hw.max_time_slots):
            for state in self.hw.dvfs_states:
                idx = self._get_or_create_var("f", t=t, state=state.name)
                self._f_vars[(t, state.name)] = idx

    def _add_unary_terms(
        self,
        linear: MutableMapping[VarIndex, float],
        quadratic: MutableMapping[QuboKey, float],
        constant: float,
    ) -> float:
        """
        Add unary terms corresponding to:
            alpha_comp * comp_cost * x
            alpha_lat  * latency_cost * x
            alpha_energy * memory_energy * m
            alpha_dvfs * dvfs_cost * f
        """
        del quadratic  # kept for a uniform method signature

        alpha_comp = self.alpha.get("comp", 0.0)
        alpha_energy = self.alpha.get("energy", 0.0)
        alpha_lat = self.alpha.get("lat", 0.0)
        alpha_dvfs = self.alpha.get("dvfs", 0.0)

        for (op, t, resource), idx in self._x_vars.items():
            node = self._node_by_id[op]
            comp_cost = self._compute_comp_cost(node, resource)
            lat_cost = self._compute_latency_cost(node, t)
            self._add_to_linear(linear, idx, alpha_comp * comp_cost + alpha_lat * lat_cost)

        level_by_name = {level.name: level for level in self.hw.memory_levels}
        for (op, level_name), idx in self._m_vars.items():
            node = self._node_by_id[op]
            level = level_by_name[level_name]
            energy_cost = self._compute_memory_energy(node, level)
            self._add_to_linear(linear, idx, alpha_energy * energy_cost)

        state_by_name = {state.name: state for state in self.hw.dvfs_states}
        for (t, state_name), idx in self._f_vars.items():
            del t
            state = state_by_name[state_name]
            dvfs_cost = self._compute_dvfs_cost(state)
            self._add_to_linear(linear, idx, alpha_dvfs * dvfs_cost)

        return constant

    def _add_pairwise_terms(
        self,
        linear: MutableMapping[VarIndex, float],
        quadratic: MutableMapping[QuboKey, float],
        constant: float,
    ) -> float:
        """
        Add pairwise CCE-QUBO interactions:
        - data reuse (encouragement => negative coefficients)
        - operator fusion adjacency (encouragement => negative coefficients)
        - bandwidth overlap penalties (positive coefficients)
        """
        del linear

        self._add_reuse_terms(quadratic)
        self._add_fusion_terms(quadratic)
        self._add_bandwidth_terms(quadratic)
        return constant

    def _add_higher_order_terms(
        self,
        linear: MutableMapping[VarIndex, float],
        quadratic: MutableMapping[QuboKey, float],
        constant: float,
    ) -> float:
        """
        Approximate higher-order effects with auxiliaries:
        - bank congestion auxiliaries u_{b,t}
        - optional burst-congestion auxiliaries u_{burst,t}
        """
        constant = self._add_bank_aux_terms(linear, quadratic, constant)
        constant = self._add_burst_aux_terms(linear, quadratic, constant)
        return constant

    def _add_constraint_penalties(
        self,
        linear: MutableMapping[VarIndex, float],
        quadratic: MutableMapping[QuboKey, float],
        constant: float,
    ) -> float:
        """
        Add APR-tunable constraints:
        - unique execution per operator
        - dependency ordering
        - DVFS one-hot per time slot
        - memory capacity (pairwise overflow approximation)
        """
        constant = self._add_unique_exec_penalty(linear, quadratic, constant)
        constant = self._add_dependency_penalty(linear, quadratic, constant)
        constant = self._add_dvfs_one_hot_penalty(linear, quadratic, constant)
        constant = self._add_memory_one_hot_penalty(linear, quadratic, constant)
        constant = self._add_memory_capacity_penalty(linear, quadratic, constant)
        return constant

    def _compute_comp_cost(self, node: Any, resource: str) -> float:
        """
        Placeholder compute-cost model:
        higher FLOPs and slower resources produce larger cost.
        """
        flops = max(0.0, float(node.flops))
        throughput = self._resource_speed_factor(resource) * 1e8
        return flops / max(throughput, 1.0)

    def _compute_latency_cost(self, node: Any, t: int) -> float:
        """
        Placeholder latency model:
        later start slots and longer operator durations cost more.
        """
        duration_slots = self._estimate_duration_slots(node)
        return float(t + duration_slots)

    def _compute_memory_energy(self, node: Any, level: Any) -> float:
        """
        Placeholder memory-energy model from bytes moved and level energy factor.
        """
        bytes_moved = self._node_total_bytes(node)
        level_name = str(level.name).upper()
        if level_name == "L1":
            factor = 0.7
        elif level_name == "L2":
            factor = 1.0
        elif level_name == "DRAM":
            factor = 2.6
        else:
            factor = 1.4
        return factor * bytes_moved / 1e6

    def _compute_dvfs_cost(self, state: Any) -> float:
        """
        Placeholder DVFS energy proxy (dynamic power-style scaling):
            energy_per_cycle * f * V^2
        """
        energy_per_cycle = max(0.0, float(state.energy_per_cycle))
        freq = max(0.0, float(state.freq_ghz))
        voltage = max(0.0, float(state.voltage_v))
        return energy_per_cycle * freq * (voltage**2)

    def _resource_speed_factor(self, resource: str) -> float:
        label = resource.lower()
        if "npu" in label or "tensor" in label:
            return 2.0
        if "gpu" in label:
            return 1.6
        if "dsp" in label:
            return 1.25
        if "cpu" in label:
            return 1.0

        checksum = sum(ord(ch) for ch in label) % 5
        return 0.9 + 0.1 * checksum

    def _estimate_duration_slots(self, node: Any) -> int:
        """
        Coarse duration estimate used by dependency and latency terms.
        """
        peak_freq = max((float(state.freq_ghz) for state in self.hw.dvfs_states), default=1.0)
        normalized_work = max(0.0, float(node.flops)) / max(1.0, peak_freq * 1e8)
        slots = int(math.ceil(normalized_work))
        return max(1, min(self.hw.max_time_slots, slots))

    def _node_total_bytes(self, node: Any) -> float:
        return max(0.0, float(node.input_bytes) + float(node.output_bytes))

    def _node_dram_demand_bytes(self, node: Any) -> float:
        """
        Proxy for DRAM pressure. This can later be replaced by a richer traffic model.
        """
        return self._node_total_bytes(node)

    def _dram_capacity_bytes_per_slot(self) -> float:
        """
        Convert DRAM bandwidth into a coarse per-slot capacity estimate.
        """
        dram_level = None
        for level in self.hw.memory_levels:
            if str(level.name).upper() == "DRAM":
                dram_level = level
                break

        if dram_level is None and self.hw.memory_levels:
            dram_level = max(self.hw.memory_levels, key=lambda lv: float(lv.bandwidth_gbps))

        if dram_level is None:
            return 1e6

        slot_seconds = 1e-6
        bytes_per_second = max(0.0, float(dram_level.bandwidth_gbps)) * 1e9 / 8.0
        return max(1.0, bytes_per_second * slot_seconds)

    def _collect_relation_pairs(self, attr_name: str) -> Sequence[Tuple[int, int]]:
        """
        Collect unique unordered (i, j) pairs from relation lists on each node.
        """
        pairs = set()
        for node in self.graph.nodes:
            src = int(node.id)
            for target_raw in getattr(node, attr_name, []) or []:
                dst = int(target_raw)
                if dst == src or dst not in self._node_by_id:
                    continue
                pair = (src, dst) if src < dst else (dst, src)
                pairs.add(pair)
        return sorted(pairs)

    def _add_reuse_terms(self, quadratic: MutableMapping[QuboKey, float]) -> None:
        beta_reuse = self.beta.get("reuse", 0.0)
        if beta_reuse == 0.0:
            return

        time_window = 2
        for op_i, op_j in self._collect_relation_pairs("reuse_groups"):
            node_i = self._node_by_id[op_i]
            node_j = self._node_by_id[op_j]
            affinity = self._reuse_affinity(node_i, node_j)
            if affinity <= 0.0:
                continue

            base_coeff = -beta_reuse * affinity
            for resource in self.hw.resources:
                for t_i in range(self.hw.max_time_slots):
                    t_min = max(0, t_i - time_window)
                    t_max = min(self.hw.max_time_slots, t_i + time_window + 1)
                    idx_i = self._x_vars[(op_i, t_i, resource)]
                    for t_j in range(t_min, t_max):
                        idx_j = self._x_vars[(op_j, t_j, resource)]
                        proximity = 1.0 / (1.0 + abs(t_i - t_j))
                        self._add_to_quadratic(quadratic, idx_i, idx_j, base_coeff * proximity)

    def _reuse_affinity(self, node_i: Any, node_j: Any) -> float:
        shared_bytes = min(float(node_i.output_bytes), float(node_j.input_bytes)) + min(
            float(node_j.output_bytes), float(node_i.input_bytes)
        )
        total_bytes = max(1.0, self._node_total_bytes(node_i) + self._node_total_bytes(node_j))
        return shared_bytes / total_bytes

    def _add_fusion_terms(self, quadratic: MutableMapping[QuboKey, float]) -> None:
        beta_fuse = self.beta.get("fuse", 0.0)
        if beta_fuse == 0.0 or self.hw.max_time_slots < 2:
            return

        for op_i, op_j in self._collect_relation_pairs("fusible_with"):
            node_i = self._node_by_id[op_i]
            node_j = self._node_by_id[op_j]
            affinity = 1.2 if str(node_i.op_type) == str(node_j.op_type) else 1.0
            coeff = -beta_fuse * affinity

            directions: Sequence[Tuple[int, int]]
            has_ij = (op_i, op_j) in self._edges
            has_ji = (op_j, op_i) in self._edges
            if has_ij and not has_ji:
                directions = [(op_i, op_j)]
            elif has_ji and not has_ij:
                directions = [(op_j, op_i)]
            else:
                directions = [(op_i, op_j), (op_j, op_i)]

            for src, dst in directions:
                for resource in self.hw.resources:
                    for t in range(self.hw.max_time_slots - 1):
                        src_idx = self._x_vars[(src, t, resource)]
                        dst_idx = self._x_vars[(dst, t + 1, resource)]
                        self._add_to_quadratic(quadratic, src_idx, dst_idx, coeff)

    def _add_bandwidth_terms(self, quadratic: MutableMapping[QuboKey, float]) -> None:
        beta_bw = self.beta.get("bw", 0.0)
        if beta_bw == 0.0:
            return

        node_ids = sorted(self._node_by_id)
        capacity = self._dram_capacity_bytes_per_slot()

        for op_i, op_j in combinations(node_ids, 2):
            node_i = self._node_by_id[op_i]
            node_j = self._node_by_id[op_j]
            combined_demand = self._node_dram_demand_bytes(node_i) + self._node_dram_demand_bytes(node_j)
            pressure = max(0.0, (combined_demand - capacity) / max(capacity, 1.0))
            if pressure <= 0.0 and combined_demand < 0.5 * capacity:
                continue

            coeff = beta_bw * max(0.05, pressure)
            for t in range(self.hw.max_time_slots):
                for resource_i in self.hw.resources:
                    idx_i = self._x_vars[(op_i, t, resource_i)]
                    for resource_j in self.hw.resources:
                        idx_j = self._x_vars[(op_j, t, resource_j)]
                        self._add_to_quadratic(quadratic, idx_i, idx_j, coeff)

    def _add_bank_aux_terms(
        self,
        linear: MutableMapping[VarIndex, float],
        quadratic: MutableMapping[QuboKey, float],
        constant: float,
    ) -> float:
        gamma_bank = self.gamma.get("bank", 0.0)
        tie_weight = self.gamma.get("bank_tie", abs(gamma_bank))
        if gamma_bank == 0.0 and tie_weight == 0.0:
            return constant

        bank_count = self._bank_count()
        if bank_count <= 0:
            return constant

        node_ids = sorted(self._node_by_id)
        for t in range(self.hw.max_time_slots):
            for bank in range(bank_count):
                accessed_vars = []
                for op in node_ids:
                    for resource in self.hw.resources:
                        if self._bank_for(op, resource, bank_count) == bank:
                            accessed_vars.append(self._x_vars[(op, t, resource)])

                if not accessed_vars:
                    continue

                aux = self._get_or_create_var("aux", role="bank_congestion", bank=bank, t=t)
                self._add_to_linear(linear, aux, gamma_bank)

                threshold = float(max(1, math.ceil(0.4 * len(accessed_vars))))
                terms: Dict[VarIndex, float] = {aux: -threshold}
                for idx in accessed_vars:
                    terms[idx] = terms.get(idx, 0.0) + 1.0
                constant = self._add_squared_penalty(
                    linear=linear,
                    quadratic=quadratic,
                    constant=constant,
                    terms=terms,
                    rhs=0.0,
                    weight=tie_weight,
                )

        return constant

    def _add_burst_aux_terms(
        self,
        linear: MutableMapping[VarIndex, float],
        quadratic: MutableMapping[QuboKey, float],
        constant: float,
    ) -> float:
        gamma_burst = self.gamma.get("burst", 0.0)
        tie_weight = self.gamma.get("burst_tie", abs(gamma_burst))
        if gamma_burst == 0.0 and tie_weight == 0.0:
            return constant

        capacity = self._dram_capacity_bytes_per_slot()
        node_ids = sorted(self._node_by_id)
        for t in range(self.hw.max_time_slots):
            terms: Dict[VarIndex, float] = {}
            for op in node_ids:
                node = self._node_by_id[op]
                normalized_demand = self._node_dram_demand_bytes(node) / max(1.0, capacity)
                if normalized_demand <= 0.0:
                    continue
                for resource in self.hw.resources:
                    idx = self._x_vars[(op, t, resource)]
                    terms[idx] = terms.get(idx, 0.0) + normalized_demand

            if not terms:
                continue

            aux = self._get_or_create_var("aux", role="burst_congestion", t=t)
            self._add_to_linear(linear, aux, gamma_burst)
            terms[aux] = terms.get(aux, 0.0) - 1.0

            constant = self._add_squared_penalty(
                linear=linear,
                quadratic=quadratic,
                constant=constant,
                terms=terms,
                rhs=0.0,
                weight=tie_weight,
            )

        return constant

    def _bank_count(self) -> int:
        explicit = getattr(self.hw, "bank_count", None)
        if explicit is None:
            explicit = getattr(self.hw, "num_banks", None)
        if explicit is None:
            explicit = getattr(self.hw, "sram_banks", None)
        if explicit is not None:
            return max(1, int(explicit))
        return max(1, len(self.hw.resources))

    def _bank_for(self, op: int, resource: str, bank_count: int) -> int:
        resource_offset = self._resource_to_idx.get(resource, 0)
        return (int(op) + resource_offset) % max(1, bank_count)

    def _add_unique_exec_penalty(
        self,
        linear: MutableMapping[VarIndex, float],
        quadratic: MutableMapping[QuboKey, float],
        constant: float,
    ) -> float:
        lam = self.penalties.get("unique_exec", 0.0)
        if lam == 0.0:
            return constant

        for node in self.graph.nodes:
            op = int(node.id)
            terms = {
                self._x_vars[(op, t, resource)]: 1.0
                for t in range(self.hw.max_time_slots)
                for resource in self.hw.resources
            }
            constant = self._add_squared_penalty(linear, quadratic, constant, terms, rhs=1.0, weight=lam)

        return constant

    def _add_dependency_penalty(
        self,
        linear: MutableMapping[VarIndex, float],
        quadratic: MutableMapping[QuboKey, float],
        constant: float,
    ) -> float:
        """
        Penalize invalid (parent, child) schedule assignments where:
            t_child < t_parent + duration(parent)
        """
        del linear

        lam = self.penalties.get("dep", 0.0)
        if lam == 0.0:
            return constant

        for src, dst in sorted(self._edges):
            src_node = self._node_by_id[src]
            duration = self._estimate_duration_slots(src_node)

            for t_src in range(self.hw.max_time_slots):
                invalid_until = min(self.hw.max_time_slots, t_src + duration)
                for t_dst in range(invalid_until):
                    gap = float((t_src + duration) - t_dst)
                    coeff = lam * gap
                    for resource_src in self.hw.resources:
                        src_idx = self._x_vars[(src, t_src, resource_src)]
                        for resource_dst in self.hw.resources:
                            dst_idx = self._x_vars[(dst, t_dst, resource_dst)]
                            self._add_to_quadratic(quadratic, src_idx, dst_idx, coeff)

        return constant

    def _add_dvfs_one_hot_penalty(
        self,
        linear: MutableMapping[VarIndex, float],
        quadratic: MutableMapping[QuboKey, float],
        constant: float,
    ) -> float:
        lam = self.penalties.get("dvfs_one_hot", 0.0)
        if lam == 0.0:
            return constant

        for t in range(self.hw.max_time_slots):
            terms = {self._f_vars[(t, state.name)]: 1.0 for state in self.hw.dvfs_states}
            constant = self._add_squared_penalty(linear, quadratic, constant, terms, rhs=1.0, weight=lam)

        return constant

    def _add_memory_one_hot_penalty(
        self,
        linear: MutableMapping[VarIndex, float],
        quadratic: MutableMapping[QuboKey, float],
        constant: float,
    ) -> float:
        """
        Optional memory placement one-hot penalty per operator.

        Enable by passing penalty key:
            - "mem_one_hot" or
            - "mem_assign"
        """
        lam = self.penalties.get("mem_one_hot", self.penalties.get("mem_assign", 0.0))
        if lam == 0.0:
            return constant

        for node in self.graph.nodes:
            op = int(node.id)
            terms = {self._m_vars[(op, level.name)]: 1.0 for level in self.hw.memory_levels}
            constant = self._add_squared_penalty(linear, quadratic, constant, terms, rhs=1.0, weight=lam)

        return constant

    def _add_memory_capacity_penalty(
        self,
        linear: MutableMapping[VarIndex, float],
        quadratic: MutableMapping[QuboKey, float],
        constant: float,
    ) -> float:
        """
        Approximate per-(time, memory level) capacity with auxiliary binding vars:

        - y_{i,t,l,r} ~= x_{i,t,r} AND m_{i,l}
        - penalize combinations of active y variables whose total bytes exceed level capacity

        This keeps the model quadratic while giving a practical capacity pressure signal.
        """
        lam = self.penalties.get("mem_cap", 0.0)
        if lam == 0.0:
            return constant

        and_weight = self.penalties.get("mem_bind", lam)

        for t in range(self.hw.max_time_slots):
            for level in self.hw.memory_levels:
                capacity = max(1.0, float(level.capacity_bytes))
                active_terms: list[Tuple[VarIndex, float]] = []

                for node in self.graph.nodes:
                    op = int(node.id)
                    demand = self._node_total_bytes(node)
                    m_idx = self._m_vars[(op, level.name)]

                    for resource in self.hw.resources:
                        x_idx = self._x_vars[(op, t, resource)]
                        y_idx = self._get_or_create_var(
                            "aux",
                            role="mem_bind",
                            op=op,
                            t=t,
                            level=level.name,
                            r=resource,
                        )
                        self._add_and_penalty(linear, quadratic, x_idx, m_idx, y_idx, and_weight)
                        active_terms.append((y_idx, demand))

                for y_idx, demand in active_terms:
                    single_overflow = max(0.0, (demand - capacity) / capacity)
                    if single_overflow > 0.0:
                        self._add_to_linear(linear, y_idx, lam * single_overflow)

                for (y_i, demand_i), (y_j, demand_j) in combinations(active_terms, 2):
                    pair_overflow = max(0.0, (demand_i + demand_j - capacity) / capacity)
                    if pair_overflow > 0.0:
                        self._add_to_quadratic(quadratic, y_i, y_j, lam * pair_overflow)

        return constant
