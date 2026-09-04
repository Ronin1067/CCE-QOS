# CCE-QOS: Constraint-Coupled Energy – Quantum Operator Scheduling

CCE-QOS is a scheduling framework for NPUs where the scheduling problem is formulated as a constraint-coupled energy minimization problem and mapped into a QUBO representation. This allows the same formulation to be optimized using both classical schedulers and (in future work) quantum methods like QAOA.

Instead of using simple additive cost models, this project builds a structured energy function that captures interactions between compute, memory hierarchy, bandwidth, and scheduling constraints across the entire operator graph.

---

## Core Formulation

Given an operator DAG \( G = (V, E) \) and a hardware configuration (resources, memory levels, DVFS states, bandwidth limits), scheduling decisions are encoded using binary variables such as:

- \( x_{i,t,r} \): operator \( i \) executes at time \( t \) on resource \( r \)  
- \( m_{i,\ell} \): operator \( i \) is placed in memory level \( \ell \)  
- \( f_{t,s} \): DVFS state \( s \) at time \( t \)  
- auxiliary variables for higher-order effects  

The total energy is defined as:

\[
E(z) = E_{\text{unary}} + E_{\text{pair}} + E_{\text{high}} + E_{\text{constr}}(\lambda)
\]

### Unary Terms
Capture per-operator effects:
- compute latency and energy  
- DRAM access cost  
- SRAM reuse loss  
- DVFS-dependent scaling  

---

### Pairwise Terms
Capture interactions between operators:
- data reuse benefits  
- fusion compatibility  
- bandwidth contention  

---

### Higher-Order Terms (Key Part of the Project)
Model effects involving multiple operators:
- memory bank conflicts  
- burst DRAM congestion  
- pipeline stall cascades  
- parallelism collapse  

These are approximated using auxiliary variables and penalty terms in the QUBO.

---

### Constraint Terms
Ensure feasibility using penalty weights \( \lambda \):
- unique scheduling of operators  
- dependency ordering  
- memory capacity limits  
- DVFS one-hot constraints  

---

## Adaptive Penalty Refinement (APR)

Instead of fixed penalties, CCE-QOS uses an adaptive scheme:

\[
\lambda_k^{t+1} = \text{clip}\left(
\lambda_k^t + \eta_1 \cdot \text{violation\_rate}_k + \eta_2 \cdot \text{cost\_impact}_k,
[\lambda_{\min}, \lambda_{\max}]
\right)
\]

APR helps:
- reduce constraint violations over time  
- stabilize optimization  
- avoid overly aggressive penalties  

---

## Project Structure

### Modeling Layer
- `graph_builder.py` → builds operator DAG  
- `memory_hierarchy.py` → models multi-level memory  
- `bandwidth_estimator.py` → estimates DRAM/bandwidth behavior  
- `core_types.py` → defines graph, hardware, and schedule structures  

---

### Formulation Layer
- `energy_model.py` → implements full CCE-QUBO formulation  
- `cost_model.py` → baseline additive cost model  
- `qubo_types.py` → QUBO representation (shared across solvers)  

---

### Optimization Layer
- `scheduling_engine.py`:
  - greedy scheduling  
  - lookahead / beam search  
  - simulated annealing  

- `penalty_tuner.py`:
  - APR implementation  
  - penalty updates and tracking  

- `quantum_interface.py`:
  - interface for QAOA backend  
  - consumes same QUBO formulation  

---

### Experiments and Analysis
- `run_experiment.py`:
  - runs multiple workloads and seeds  
  - logs metrics (cost, latency, DRAM, feasibility, runtime)

- `schedule_analysis.py`:
  - aggregates metrics  
  - computes comparisons  

- `schedule_explainer.py`:
  - generates human-readable explanations  

- `plot_results.py`:
  - creates figures (cost comparison, APR behavior, ablations)  

---

## Key Idea

The main idea is that scheduling should not be treated as independent decisions.  
In real hardware, compute, memory, and bandwidth constraints are tightly coupled.

CCE-QOS models this coupling explicitly using an energy-based formulation, which makes the optimization landscape more realistic but also more complex.

---

## Why QUBO + Quantum?

By expressing the entire scheduling problem as a QUBO:

- classical solvers can still be used  
- quantum algorithms (like QAOA) can operate directly on the same formulation  

So the quantum backend is not solving a simplified problem—it works on the exact same energy function.

---
