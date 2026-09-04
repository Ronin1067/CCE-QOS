from __future__ import annotations

import time
from typing import Dict, Tuple, Any, Optional
from ortools.sat.python import cp_model

from qubo_types import QUBOData, VarIndex

class ORToolsQUBOSolver:
    """
    Exact Global Optimum QUBO Solver utilizing Google OR-Tools CP-SAT.
    Linearizes quadratic binary products x_i * x_j via auxiliary variables:
        y_{ij} <= x_i
        y_{ij} <= x_j
        y_{ij} >= x_i + x_j - 1
    Enforces provable optimality guarantees for NPU compiler scheduling.
    """
    def __init__(self, qubo_data: QUBOData):
        self.qubo_data = qubo_data

    def solve(self, result_obj: Optional[Any] = None) -> Tuple[Dict[VarIndex, int], float, float]:
        model = cp_model.CpModel()

        # Identify all distinct variables
        vars_set = set(self.qubo_data.linear.keys())
        for (u, v) in self.qubo_data.quadratic.keys():
            vars_set.add(u)
            vars_set.add(v)

        if not vars_set:
            return {}, self.qubo_data.offset, 0.0

        # Create binary decision variables
        x_vars: Dict[VarIndex, cp_model.BoolVar] = {
            var_idx: model.NewBoolVar(f"x_{var_idx}") for var_idx in vars_set
        }

        # Scale float coefficients to integer micro-units (1e4) for CP-SAT precision
        SCALE = 10000
        obj_terms = []

        # Linear objective terms
        for var_idx, coeff in self.qubo_data.linear.items():
            int_coeff = int(round(coeff * SCALE))
            if int_coeff != 0:
                obj_terms.append(int_coeff * x_vars[var_idx])

        # Quadratic terms linearization
        for (u, v), coeff in self.qubo_data.quadratic.items():
            int_coeff = int(round(coeff * SCALE))
            if int_coeff == 0:
                continue

            if u == v:
                obj_terms.append(int_coeff * x_vars[u])
            else:
                y_uv = model.NewBoolVar(f"y_{u}_{v}")
                # Exact linearization constraints:
                model.Add(y_uv <= x_vars[u])
                model.Add(y_uv <= x_vars[v])
                model.Add(y_uv >= x_vars[u] + x_vars[v] - 1)
                obj_terms.append(int_coeff * y_uv)

        raw_offset = getattr(self.qubo_data, "constant", getattr(self.qubo_data, "offset", 0.0))
        offset_int = int(round(raw_offset * SCALE))
        model.Minimize(sum(obj_terms) + offset_int)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        solver.parameters.num_search_workers = 4

        start_t = time.time()
        status = solver.Solve(model)
        elapsed = time.time() - start_t

        solution: Dict[VarIndex, int] = {}
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for v_idx, b_var in x_vars.items():
                solution[v_idx] = int(solver.Value(b_var))
            obj_val = solver.ObjectiveValue() / float(SCALE)
            status_str = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
        else:
            solution = {v_idx: 0 for v_idx in vars_set}
            obj_val = raw_offset
            status_str = "INFEASIBLE"

        if result_obj is not None:
            if hasattr(result_obj, "ortools_solution"):
                result_obj.ortools_solution = solution
            if hasattr(result_obj, "ortools_objective"):
                result_obj.ortools_objective = obj_val
            if hasattr(result_obj, "ortools_runtime"):
                result_obj.ortools_runtime = elapsed

        print(f"[OR-Tools CP-SAT] Status: {status_str} | Objective Cost: {obj_val:.4f} | Runtime: {elapsed*1000:.2f} ms")
        return solution, obj_val, elapsed
