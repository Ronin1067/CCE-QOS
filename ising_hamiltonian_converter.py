from __future__ import annotations
from typing import Dict, Tuple
from dataclasses import dataclass, field
from qubo_types import QUBOData, VarIndex

@dataclass
class IsingHamiltonian:
    r"""
    Standard Ising Hamiltonian representation:
    H = \sum_i h_i s_i + \sum_{i < j} J_{ij} s_i s_j + offset
    where s_i \in {-1, +1}.
    """
    h: Dict[VarIndex, float] = field(default_factory=dict)
    J: Dict[Tuple[VarIndex, VarIndex], float] = field(default_factory=dict)
    offset: float = 0.0

def convert_qubo_to_ising(qubo_data: QUBOData, result_obj=None) -> IsingHamiltonian:
    r"""
    Converts a standard QUBO problem:
        min_{x \in {0,1}} x^T Q x + c^T x + offset
    into an Ising Spin Hamiltonian:
        min_{s \in {-1,+1}} \sum_i h_i s_i + \sum_{i < j} J_{ij} s_i s_j + offset'
    using the standard substitution: x_i = (1 - s_i) / 2.
    
    Expansion:
        c_i x_i = c_i (1 - s_i) / 2 = c_i/2 - (c_i/2) s_i
        Q_{ij} x_i x_j = Q_{ij} (1 - s_i)(1 - s_j) / 4
                       = Q_{ij}/4 - (Q_{ij}/4) s_i - (Q_{ij}/4) s_j + (Q_{ij}/4) s_i s_j
    """
    h: Dict[VarIndex, float] = {}
    J: Dict[Tuple[VarIndex, VarIndex], float] = {}
    offset: float = getattr(qubo_data, "constant", getattr(qubo_data, "offset", 0.0))

    # Linear terms
    for var, coeff in qubo_data.linear.items():
        h[var] = h.get(var, 0.0) - coeff / 2.0
        offset += coeff / 2.0

    # Quadratic terms
    for (u, v), coeff in qubo_data.quadratic.items():
        if u == v:
            # Diagonal terms x_i^2 = x_i
            h[u] = h.get(u, 0.0) - coeff / 2.0
            offset += coeff / 2.0
        else:
            var_i, var_j = (u, v) if u < v else (v, u)
            J[(var_i, var_j)] = J.get((var_i, var_j), 0.0) + coeff / 4.0
            h[var_i] = h.get(var_i, 0.0) - coeff / 4.0
            h[var_j] = h.get(var_j, 0.0) - coeff / 4.0
            offset += coeff / 4.0

    if result_obj is not None:
        if hasattr(result_obj, "ising_h_terms"):
            result_obj.ising_h_terms = h
        if hasattr(result_obj, "ising_J_terms"):
            result_obj.ising_J_terms = J
        if hasattr(result_obj, "ising_offset"):
            result_obj.ising_offset = offset

    return IsingHamiltonian(h=h, J=J, offset=offset)
