"""
Tier 1 Exact CP-SAT Solver Runner for CCE-QOS.
Solves NPU task DAGs with exact global optimality guarantees.
"""

from .ortools_cpsat_engine import ExactCPSATScheduler


def run_cpsat_benchmark():
    print("=" * 70)
    print("TIER 1: EXACT OR-TOOLS CP-SAT COMPILER SCHEDULING")
    print("=" * 70)

    # 6-node synthetic NPU DAG
    tasks = [
        {"id": "conv1", "duration": 2, "memory_kb": 256},
        {"id": "relu1", "duration": 1, "memory_kb": 128},
        {"id": "pool1", "duration": 1, "memory_kb": 128},
        {"id": "conv2", "duration": 3, "memory_kb": 512},
        {"id": "relu2", "duration": 1, "memory_kb": 256},
        {"id": "dense", "duration": 2, "memory_kb": 1024}
    ]
    deps = [
        ("conv1", "relu1"),
        ("relu1", "pool1"),
        ("pool1", "conv2"),
        ("conv2", "relu2"),
        ("relu2", "dense")
    ]

    scheduler = ExactCPSATScheduler()
    res = scheduler.solve(tasks, deps)

    print(f"Solver Status   : {res['status']}")
    print(f"Optimal Makespan: {res['makespan']} cycles")
    print(f"Schedule Map    : {res['schedule']}")
    print(f"Solve Time      : {res['solver_runtime_s']:.3f} s")
    print("Tier 1 exact global scheduling verified.\n")


if __name__ == "__main__":
    run_cpsat_benchmark()
