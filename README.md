# CCE-QOS: Constraint-Coupled Energy – Quantum Operator Scheduling

CCE-QOS is a scheduling framework for NPUs where the scheduling problem is modeled as an energy minimization problem and then converted into a QUBO form. This makes it possible to use both classical optimization methods and (in future work) quantum algorithms like QAOA on the same problem.

Instead of using simple heuristics or basic cost functions, this project builds a structured energy model that captures how compute, memory, bandwidth, and constraints interact across the entire operator graph.

---

## Core Idea

Given:
- an operator DAG G = (V, E)
- an NPU hardware configuration (resources, memory hierarchy, DVFS states, bandwidth limits)

we encode scheduling decisions using binary variables:

- x(i,t,r): operator i executes at time t on resource r  
- m(i,l): operator i is placed in memory level l  
- f(t,s): DVFS state s at time t  
- auxiliary variables for higher-order effects  

The total energy is defined as:

E(z) = E_unary + E_pair + E_high + E_constr(λ)

---

## Energy Components

### Unary Terms
Capture per-operator effects:
- compute latency and energy  
- DRAM access cost  
- SRAM reuse loss  
- DVFS-dependent behavior  

---

### Pairwise Terms
Capture interactions between operators:
- data reuse benefits  
- fusion opportunities  
- bandwidth contention  

---

### Higher-Order Terms (Key Contribution)
Model effects involving multiple operators:
- memory bank conflicts  
- burst DRAM congestion  
- pipeline stall cascades  
- parallelism collapse  

These are approximated using auxiliary variables and penalty terms.

---

### Constraint Terms
Ensure feasibility using penalty weights λ:
- unique execution of each operator  
- dependency ordering  
- memory capacity limits  
- DVFS one-hot constraints  

---

## Adaptive Penalty Refinement (APR)

Instead of fixed penalties, CCE-QOS uses an adaptive update rule:

λ_k(t+1) = clip(
    λ_k(t) + η1 * violation_rate_k + η2 * cost_impact_k,
    [λ_min, λ_max]
)

APR helps to:
- reduce constraint violations over time  
- stabilize optimization  
- avoid overly aggressive penalties  

---

## Project Structure

### Modeling Layer
- `graph_builder.py` → builds operator DAG  
- `memory_hierarchy.py` → models memory system  
- `bandwidth_estimator.py` → models DRAM and bandwidth  
- `core_types.py` → defines core data structures  

---

### Formulation Layer
- `energy_model.py` → implements full CCE-QUBO formulation  
- `cost_model.py` → baseline additive model  
- `qubo_types.py` → shared QUBO representation  

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

---

### Experiments and Analysis
- `run_experiment.py`:
  - runs multiple workloads and seeds  
  - logs metrics (cost, latency, DRAM usage, feasibility, runtime)

- `schedule_analysis.py`:
  - aggregates results  
  - computes comparisons  

- `schedule_explainer.py`:
  - generates human-readable explanations  

- `plot_results.py`:
  - generates graphs (cost comparison, APR behavior, ablations)

---

## Why This Approach?

### Constraint-Coupled Energy
In real hardware, scheduling decisions are not independent. Memory, bandwidth, and execution order are tightly linked. This approach models those interactions directly instead of treating everything separately.

---

### QUBO + Quantum Compatibility
By expressing the scheduling problem as a QUBO:
- classical optimization methods can still be used  
- quantum algorithms like QAOA can operate on the same formulation  

So the quantum backend is not solving a simplified version — it uses the exact same problem definition.

---

## Summary

This project tries to move beyond simple scheduling heuristics by:
- modeling scheduling as a structured energy problem  
- capturing interactions between constraints  
- enabling both classical and quantum optimization  

The goal is to build a system where quantum optimization naturally fits into the formulation.
