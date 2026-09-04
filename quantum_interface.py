from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from core_types import HardwareConfig, OperatorGraph
from energy_model import EnergyModel
from qubo_types import QUBOData


@dataclass(slots=True)
class ProblemSpec:
    """
    Quantum-backend input contract.
    """

    graph: OperatorGraph
    hardware: HardwareConfig
    alpha: Dict[str, float]
    beta: Dict[str, float]
    gamma: Dict[str, float]
    penalties: Dict[str, float]


def build_qubo(problem_spec: ProblemSpec) -> QUBOData:
    """
    Build the canonical CCE-QUBO consumed by classical/quantum solvers.
    """
    model = EnergyModel(
        graph=problem_spec.graph,
        hw=problem_spec.hardware,
        alpha=problem_spec.alpha,
        beta=problem_spec.beta,
        gamma=problem_spec.gamma,
        initial_penalties=problem_spec.penalties,
    )
    return model.build_qubo()


def qubo_energy(qubo_data: QUBOData, bits: Sequence[int]) -> float:
    """
    Evaluate E(z) = const + sum_i a_i z_i + sum_{i<=j} b_ij z_i z_j
    for a binary assignment `bits`.
    """
    if len(bits) != qubo_data.num_variables:
        raise ValueError("Bitstring length does not match QUBO variable count.")

    energy = float(qubo_data.constant)
    for idx, coeff in qubo_data.linear.items():
        energy += float(coeff) * int(bits[idx])
    for (i, j), coeff in qubo_data.quadratic.items():
        energy += float(coeff) * int(bits[i]) * int(bits[j])
    return energy


def _decode_schedule_projection(qubo_data: QUBOData, bits: Sequence[int]) -> List[int]:
    """
    Recover a coarse schedule order from active x_{i,t,r} variables:
    sort by time, then op id; keep first occurrence per op.
    """
    active_x: List[tuple[int, int, str]] = []
    for idx, meta in qubo_data.var_metadata.items():
        if int(bits[idx]) != 1:
            continue
        if meta.get("kind") != "x":
            continue
        op = int(meta["op"])
        t = int(meta["t"])
        r = str(meta["r"])
        active_x.append((t, op, r))

    active_x.sort(key=lambda item: (item[0], item[1], item[2]))
    seen = set()
    order: List[int] = []
    for _, op, _ in active_x:
        if op in seen:
            continue
        seen.add(op)
        order.append(op)
    return order


def run_qaoa_stub(
    qubo_data: QUBOData,
    num_samples: int = 64,
    num_steps: int = 220,
    seed: int = 101,
) -> List[Dict[str, Any]]:
    """
    Simple optimizer stub that mimics a QAOA candidate-generation surface.

    It performs multi-start stochastic bit-flip search directly on QUBO energy
    and returns best candidate assignments in a quantum-backend-friendly shape.
    """
    rng = random.Random(seed)
    n = qubo_data.num_variables
    if n <= 0:
        return []

    starts = max(4, min(num_samples, 16))
    pool: List[List[int]] = [[rng.randint(0, 1) for _ in range(n)] for _ in range(starts)]
    pool_energy = [qubo_energy(qubo_data, bits) for bits in pool]

    best_seen: Dict[str, float] = {}
    for walk_idx in range(starts):
        bits = list(pool[walk_idx])
        energy = pool_energy[walk_idx]

        for step in range(max(1, num_steps)):
            temp = max(0.01, 1.5 * (1.0 - step / max(1, num_steps)))
            k = rng.randint(1, 3)
            flip_indices = rng.sample(range(n), k=k)

            proposal = list(bits)
            for fidx in flip_indices:
                proposal[fidx] = 1 - proposal[fidx]

            p_energy = qubo_energy(qubo_data, proposal)
            delta = p_energy - energy
            accept = delta <= 0.0 or rng.random() < pow(2.718281828, -delta / temp)
            if accept:
                bits = proposal
                energy = p_energy

        bitstring = "".join("1" if b else "0" for b in bits)
        prev = best_seen.get(bitstring)
        if prev is None or energy < prev:
            best_seen[bitstring] = energy

    ranked = sorted(best_seen.items(), key=lambda item: item[1])[: max(1, num_samples)]
    results: List[Dict[str, Any]] = []
    for rank, (bitstring, energy) in enumerate(ranked, start=1):
        bits = [1 if ch == "1" else 0 for ch in bitstring]
        active = [idx for idx, b in enumerate(bits) if b == 1]
        results.append(
            {
                "rank": rank,
                "energy": float(energy),
                "bitstring": bitstring,
                "active_variables": active,
                "schedule_projection": _decode_schedule_projection(qubo_data, bits),
                "backend": "qaoa_stub_local_search",
            }
        )
    return results


class QuantumInterface:
    """
    Thin OO wrapper around the required functional API.
    """

    def __init__(self, seed: int = 101) -> None:
        self.seed = int(seed)

    def build_qubo(self, problem_spec: ProblemSpec) -> QUBOData:
        return build_qubo(problem_spec)

    def run_qaoa_stub(
        self,
        qubo_data: QUBOData,
        num_samples: int = 64,
        num_steps: int = 220,
    ) -> List[Dict[str, Any]]:
        return run_qaoa_stub(
            qubo_data=qubo_data,
            num_samples=num_samples,
            num_steps=num_steps,
            seed=self.seed,
        )
