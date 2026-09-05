"""
Tier 2 Benchmark: CCE-QUBO and Adaptive Penalty Refinement (APR)
Alias / wrapper module for cce_qubo_benchmark.py
"""

try:
    from cce_qubo_benchmark import run_cce_qubo_benchmark
except ImportError:
    try:
        from .cce_qubo_benchmark import run_cce_qubo_benchmark
    except ImportError:
        from implementations.v2_cce_qubo_apr_engine.cce_qubo_benchmark import run_cce_qubo_benchmark

if __name__ == "__main__":
    run_cce_qubo_benchmark()
