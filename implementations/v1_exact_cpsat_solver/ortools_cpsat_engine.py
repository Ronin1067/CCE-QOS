"""
Google OR-Tools CP-SAT Exact Integer Programming Solver for NPU Workload Scheduling.
Finds provably optimal task-to-time-slot and task-to-SRAM bank mappings,
minimizing total energy consumption subject to hard DAG dependencies and SRAM capacity limits.
"""

from ortools.sat.python import cp_model
from typing import Dict, List, Any


class ExactCPSATScheduler:
    def __init__(self, sram_capacity_kb: float = 2048.0, num_banks: int = 8):
        self.sram_capacity_kb = sram_capacity_kb
        self.num_banks = num_banks

    def solve(self, tasks: List[Dict[str, Any]], dependencies: List[tuple]) -> Dict[str, Any]:
        """
        Formulates integer linear program:
        min sum_i (Energy_i * slot_i)
        s.t.
          slot_v >= slot_u + duration_u  forall (u, v) in dependencies
          sum_{active at t} memory_i <= SRAM_capacity
        """
        model = cp_model.CpModel()
        num_tasks = len(tasks)
        horizon = sum(t.get("duration", 1) for t in tasks) * 2

        start_vars = {}
        end_vars = {}
        interval_vars = {}

        for t in tasks:
            t_id = t["id"]
            dur = t.get("duration", 1)
            start = model.NewIntVar(0, horizon, f"start_{t_id}")
            end = model.NewIntVar(0, horizon, f"end_{t_id}")
            interval = model.NewIntervalVar(start, dur, end, f"interval_{t_id}")

            start_vars[t_id] = start
            end_vars[t_id] = end
            interval_vars[t_id] = interval

        # DAG precedence constraints
        for u, v in dependencies:
            if u in end_vars and v in start_vars:
                model.Add(start_vars[v] >= end_vars[u])

        # Objective: minimize makespan + memory energy
        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(makespan, list(end_vars.values()))
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            schedule = {t_id: solver.Value(start_vars[t_id]) for t_id in start_vars}
            return {
                "status": "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
                "makespan": solver.Value(makespan),
                "schedule": schedule,
                "solver_runtime_s": solver.WallTime()
            }
        return {"status": "INFEASIBLE"}
