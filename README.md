# CCE-QOS: Constraint-Coupled Energy – Quantum Operator Scheduling

CCE-QOS (Constraint-Coupled Energy – Quantum Operator Scheduling) is a research-grade scheduling framework for neural processing units (NPUs) that treats operator scheduling as an energy minimization problem and exposes it as a QUBO suitable for both classical and quantum optimization.

Instead of using ad-hoc heuristics or purely additive cost models, CCE-QOS builds a **constraint-coupled energy function** that captures how compute, memory, bandwidth, and feasibility constraints interact across the entire operator graph. This same energy is then optimized by classical algorithms (greedy, beam, annealing, etc.) and, in future work, by a QAOA-based quantum backend.

## Core idea

Given a neural network operator DAG and an NPU hardware configuration (resources, multi-level memory, DVFS states, bandwidth limits), CCE-QOS encodes scheduling decisions into binary variables and defines a total energy

\[
E(\mathbf{z}) = E_{\text{unary}}(\mathbf{z}) + E_{\text{pair}}(\mathbf{z}) + E_{\text{high}}(\mathbf{z}) + E_{\text{constr}}(\mathbf{z}; \boldsymbol{\lambda}),
\]

where:

- **Unary terms** capture per-operator compute, energy, and latency costs.
- **Pairwise terms** capture data reuse, fusion compatibility, and bandwidth interactions between operators.
- **Higher-order terms** model effects that involve many operators at once, such as memory bank conflicts, DRAM burst congestion, and pipeline stall cascades (implemented via auxiliary variables and penalties).
- **Constraint terms** enforce feasibility (unique execution, dependency ordering, DVFS one-hot, memory capacity) via adaptive penalty weights \(\boldsymbol{\lambda}\).

This formulation yields a **Constraint-Coupled Energy QUBO** that can be passed to classical or quantum optimizers without changing the underlying problem definition.

## What CCE-QOS provides

CCE-QOS is structured as a full research pipeline rather than just a single heuristic:

- **Operator and hardware modeling**
  - `graph_builder.py` constructs the operator DAG with metadata.
  - `memory_hierarchy.py` and `bandwidth_estimator.py` model the memory system and DRAM usage.
  - `core_types.py` defines data structures for graphs, hardware, schedules, and metrics.

- **Energy formulation and QUBO construction**
  - `energy_model.py` implements the CCE-QUBO energy function and builds the QUBOData structure.
  - `cost_model.py` provides a baseline additive cost formulation for comparison.
  - `qubo_types.py` defines the QUBO representation shared by classical and quantum solvers.

- **Scheduling and optimization**
  - `scheduling_engine.py` runs classical schedulers (e.g., greedy, lookahead/beam, annealing) on either the baseline cost model or the CCE-QUBO energy.
  - `penalty_tuner.py` implements Adaptive Penalty Refinement (APR), which updates constraint penalties based on observed violations and stabilizes the search.
  - `quantum_interface.py` defines the API that a future QAOA backend will use to consume the same QUBO.

- **Experiments and analysis**
  - `run_experiment.py` orchestrates experiments across multiple workloads and methods, logging metrics (cost, latency, DRAM usage, feasibility, runtime).
  - `schedule_analysis.py` and `schedule_explainer.py` provide aggregate metrics and human-readable explanations of schedules.
  - `plot_results.py` generates publication-ready figures from logged results.

- **Reports and specifications**
  - `classical_report.tex` is a paper-style report for the classical and CCE-QUBO side, with experiments and ablations.
  - `quantum_backend_spec.tex` is an implementation spec for the QAOA-based quantum backend.
  - `project_guide.tex` gives a high-level overview and project roadmap.

## Why "Constraint-Coupled Energy – Quantum Operator Scheduling"?

- **Constraint-Coupled Energy**: The energy is not just a sum of independent costs. It explicitly encodes how constraints (dependencies, memory, bandwidth, DVFS) are coupled across operators and time, making the scheduling landscape highly structured and non-separable.

- **Quantum Operator Scheduling**: By expressing the full scheduling problem as a QUBO, CCE-QOS makes quantum optimization (e.g., QAOA) a first-class citizen. The quantum backend does not see a separate toy problem; it sees the same CCE-QUBO that the classical solvers optimize.

Together, this makes CCE-QOS a **bridge between compiler-style NPU scheduling and quantum optimization**, rather than a classical system with a superficial quantum add-on.

## Project status

The repository currently includes:

- A working classical pipeline (CCE-QUBO + APR + schedulers + experiments).
- Scripts and LaTeX documents for generating and presenting results.
- Stubs and specifications for a QAOA-based quantum backend.

Future work focuses on:

- Implementing and integrating a real quantum backend using QAOA or related variational algorithms.
- Scaling experiments to larger models and more realistic NPU configurations.
- Exploring robustness of CCE-QUBO and APR under noisy or approximate hardware models.
