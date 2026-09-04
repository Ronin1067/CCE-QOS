from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Assuming these are defined elsewhere or can be mocked for this dataclass
# from qubo_types import QUBOData, QuboKey, VarIndex
# from scheduling_engine import ScheduleResult

@dataclass
class ExperimentResult:
    """
    A unified data structure to store results from various stages of the experiment pipeline.
    """
    # Raw Input Data
    workload_data: Dict[str, Any] = field(default_factory=dict)
    npu_slot_ids: List[int] = field(default_factory=list)
    task_ids: List[str] = field(default_factory=list)

    # OperatorGraph (Classical Scheduling Input)
    operator_graph_nodes: Optional[List[Dict[str, Any]]] = None
    operator_graph_edges: Optional[List[Dict[str, Any]]] = None

    # QUBO Generation Results
    qubo_linear_terms: Dict[int, float] = field(default_factory=dict)
    qubo_quadratic_terms: Dict[Tuple[int, int], float] = field(default_factory=dict)
    qubo_offset: float = 0.0
    num_qubo_vars: int = 0

    # Ising Conversion Results
    ising_h_terms: Dict[int, float] = field(default_factory=dict)
    ising_J_terms: Dict[Tuple[int, int], float] = field(default_factory=dict)
    ising_offset: float = 0.0

    # QAOA Execution Results
    qaoa_optimized_parameters: Optional[Dict[str, Any]] = None
    qaoa_energy: Optional[float] = None
    qaoa_raw_results: Optional[Dict[str, Any]] = None

    # Classical Solver Results (SchedulingEngine)
    classical_schedule_strategy: Optional[str] = None
    classical_schedule_order: Optional[List[int]] = None
    classical_schedule_score: Optional[float] = None
    classical_schedule_metadata: Dict[str, Any] = field(default_factory=dict)

    # APR (Adaptive Parameterization Refinement) Results
    apr_initial_penalties: Dict[str, float] = field(default_factory=dict)
    apr_final_penalties: Dict[str, float] = field(default_factory=dict)
    apr_optimized_score: Optional[float] = None
    apr_metrics_history: List[Dict[str, Any]] = field(default_factory=list)

    # General Experiment Metrics
    final_solution: Optional[Dict[str, Any]] = None # The best solution found (e.g., assignment of tasks to slots)
    final_metrics: Optional[Dict[str, Any]] = None # Full evaluation of the final solution
