from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from bandwidth_estimator import BandwidthEstimator
from core_types import HardwareConfig, OperatorGraph
from cost_model import ScheduleCostModel
from fusion_logic import FusionLogic
from graph_builder import load_hardware_config, load_operator_graph
from memory_hierarchy import MemoryHierarchy
from penalty_tuner import PenaltyTuner
from quantum_interface import ProblemSpec, build_qubo, qubo_energy, run_qaoa_stub
from qubo_types import QUBOData
from schedule_analysis import ScheduleAnalysis
from schedule_explainer import ScheduleExplainer
from scheduling_engine import ScheduleResult, SchedulingEngine


def load_config(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Configuration root must be an object.")
        return parsed
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install PyYAML or keep config JSON-compatible.") from exc
        parsed = yaml.safe_load(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Configuration root must be an object.")
        return parsed


def _derive_weights(config: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    energy = dict(config.get("energy_weights", {}))
    alpha = dict(config.get("alpha", {}))
    beta = dict(config.get("beta", {}))
    gamma = dict(config.get("gamma", {}))
    return (
        {
            "comp": float(alpha.get("comp", energy.get("unary_cost", 1.0))),
            "energy": float(alpha.get("energy", 1.0)),
            "lat": float(alpha.get("lat", 0.25)),
            "dvfs": float(alpha.get("dvfs", 0.2)),
        },
        {
            "reuse": float(beta.get("reuse", energy.get("data_reuse_reward", 1.0))),
            "fuse": float(beta.get("fuse", energy.get("fusion_reward", 1.0))),
            "bw": float(beta.get("bw", energy.get("bandwidth_spike_penalty", 1.0))),
        },
        {
            "bank": float(gamma.get("bank", energy.get("memory_conflict_penalty", 0.8))),
            "burst": float(gamma.get("burst", 0.6)),
            "stall": float(gamma.get("stall", 0.4)),
            "parallelism": float(gamma.get("parallelism", 0.3)),
        },
    )


def _map_penalties_to_qubo(p: Mapping[str, float]) -> Dict[str, float]:
    return {
        "unique_exec": float(p.get("unique_exec", p.get("dependency_conflict", 1.0) * 1.6)),
        "dep": float(p.get("dep", p.get("dependency_conflict", 1.0))),
        "dvfs_one_hot": float(p.get("dvfs_one_hot", p.get("bandwidth_capacity", 1.0) * 0.6)),
        "mem_cap": float(p.get("mem_cap", p.get("sram_capacity", 1.0))),
        "mem_bind": float(p.get("mem_bind", p.get("sram_capacity", 1.0))),
    }


def _complete_order(order: Sequence[int], graph: OperatorGraph) -> list[int]:
    seen = set()
    candidate = []
    for x in order:
        node_id = int(x)
        if node_id in seen or node_id not in graph.node_by_id:
            continue
        seen.add(node_id)
        candidate.append(node_id)

    final: list[int] = []
    scheduled = set()
    pending = list(candidate)
    while pending:
        changed = False
        rest = []
        for node_id in pending:
            deps = graph.node_by_id[node_id].dependencies
            if all(dep in scheduled for dep in deps):
                final.append(node_id)
                scheduled.add(node_id)
                changed = True
            else:
                rest.append(node_id)
        pending = rest
        if not changed:
            break

    while len(final) < len(graph.nodes):
        ready = [n for n in graph.ready_nodes(scheduled) if n not in scheduled]
        if not ready:
            break
        chosen = ready[0]
        final.append(chosen)
        scheduled.add(chosen)
    return final


class CCEEvaluator:
    def __init__(
        self,
        graph: OperatorGraph,
        hardware: HardwareConfig,
        cost_model: ScheduleCostModel,
        alpha: Dict[str, float],
        beta: Dict[str, float],
        gamma: Dict[str, float],
        base_penalties: Dict[str, float],
    ) -> None:
        self.graph = graph
        self.hardware = hardware
        self.cost_model = cost_model
        self.alpha = dict(alpha)
        self.beta = dict(beta)
        self.gamma = dict(gamma)
        self.base_penalties = dict(base_penalties)
        self._cache: Dict[Tuple[Any, ...], QUBOData] = {}

    def _qubo(self, penalties: Dict[str, float], ablation: str | None) -> QUBOData:
        beta = dict(self.beta)
        gamma = dict(self.gamma)
        if ablation == "remove_pairwise":
            beta = {k: 0.0 for k in beta}
        if ablation == "remove_higher_order":
            gamma = {k: 0.0 for k in gamma}
        key = (
            tuple(sorted(_map_penalties_to_qubo(penalties).items())),
            tuple(sorted(beta.items())),
            tuple(sorted(gamma.items())),
        )
        if key in self._cache:
            return self._cache[key]
        spec = ProblemSpec(
            graph=self.graph,
            hardware=self.hardware,
            alpha=self.alpha,
            beta=beta,
            gamma=gamma,
            penalties=_map_penalties_to_qubo(penalties),
        )
        qubo = build_qubo(spec)
        self._cache[key] = qubo
        return qubo

    def _encode(self, order: Sequence[int], qubo: QUBOData) -> list[int]:
        bits = [0] * qubo.num_variables
        x_lookup: Dict[Tuple[int, int, str], int] = {}
        x_fallback: Dict[int, int] = {}
        m_lookup: Dict[Tuple[int, str], int] = {}
        m_fallback: Dict[int, int] = {}
        f_lookup: Dict[Tuple[int, str], int] = {}
        f_fallback: Dict[int, int] = {}

        for idx, meta in qubo.var_metadata.items():
            kind = meta.get("kind")
            if kind == "x":
                op = int(meta["op"])
                x_lookup[(op, int(meta["t"]), str(meta["r"]))] = idx
                x_fallback.setdefault(op, idx)
            elif kind == "m":
                op = int(meta["op"])
                m_lookup[(op, str(meta["level"]))] = idx
                m_fallback.setdefault(op, idx)
            elif kind == "f":
                t = int(meta["t"])
                f_lookup[(t, str(meta["state"]))] = idx
                f_fallback.setdefault(t, idx)

        primary_r = self.hardware.resources[0] if self.hardware.resources else ""
        levels = [lv.name for lv in self.hardware.memory_levels]
        level = "L2" if "L2" in levels else (levels[0] if levels else "")
        states = [s.name for s in self.hardware.dvfs_states]
        state = "nominal" if "nominal" in states else (states[0] if states else "")

        for pos, op in enumerate(order):
            t = min(pos, self.hardware.max_time_slots - 1)
            x = x_lookup.get((int(op), t, primary_r), x_fallback.get(int(op)))
            m = m_lookup.get((int(op), level), m_fallback.get(int(op)))
            if x is not None:
                bits[x] = 1
            if m is not None:
                bits[m] = 1
        for t in range(self.hardware.max_time_slots):
            f = f_lookup.get((t, state), f_fallback.get(t))
            if f is not None:
                bits[f] = 1
        return bits

    def evaluate(self, order: Sequence[int], penalties: Dict[str, float], ablation: str | None = None) -> Dict[str, Any]:
        valid_order = _complete_order(order, self.graph)
        qubo = self._qubo(penalties, ablation)
        bits = self._encode(valid_order, qubo)
        linear = sum(float(c) * bits[i] for i, c in qubo.linear.items())
        quadratic = sum(float(c) * bits[i] * bits[j] for (i, j), c in qubo.quadratic.items())
        total = float(qubo.constant) + linear + quadratic
        base = self.cost_model.evaluate(self.graph, valid_order, penalties=penalties)
        out = dict(base)
        out["energy_breakdown"] = {
            "unary_cost": linear,
            "pairwise_total": quadratic,
            "constant": float(qubo.constant),
            "total_energy": total,
        }
        out["qubo_snapshot"] = {
            "num_variables": qubo.num_variables,
            "linear_terms": len(qubo.linear),
            "quadratic_terms": len(qubo.quadratic),
            "active_bits": int(sum(bits)),
            "ablation": ablation or "none",
        }
        return out

    def build_qubo_data(self, penalties: Dict[str, float], ablation: str | None = None) -> QUBOData:
        return self._qubo(penalties, ablation)


def _rank(evaluation: Dict[str, Any], objective: float) -> float:
    return objective - 120.0 * float(evaluation.get("feasibility", 0.0))


def _run_trials(
    graph: OperatorGraph,
    trials: int,
    seed: int,
    builder,
    evaluate,
    objective_fn,
) -> Tuple[ScheduleResult, Dict[str, Any]]:
    best_result = None
    best_eval = None
    best_rank = float("inf")
    for idx in range(trials):
        engine = SchedulingEngine(graph, random_seed=seed + idx * 37)
        result = builder(engine)
        order = _complete_order(result.order, graph)
        evaluation = evaluate(order)
        objective = float(objective_fn(evaluation))
        rank = _rank(evaluation, objective)
        if rank < best_rank:
            best_rank = rank
            result.order = order
            result.score = objective
            best_result = result
            best_eval = evaluation
    if best_result is None or best_eval is None:
        raise RuntimeError("No valid schedule found.")
    return best_result, best_eval


def _run_suite(graph, config, penalties, evaluate, objective_fn, seed_offset):
    exp = dict(config.get("experiment", {}))
    search = dict(config.get("search", {}))
    trials = int(exp.get("search_trials", 5))
    seed = int(exp.get("seed", 17)) + seed_offset
    methods = {
        "Greedy": lambda e: e.greedy(penalties=penalties),
        "Lookahead": lambda e: e.lookahead(
            penalties=penalties,
            lookahead_depth=int(search.get("lookahead_depth", 3)),
            evaluator=lambda o: float(objective_fn(evaluate(o))),
        ),
        "Beam Search": lambda e: e.beam_search(
            penalties=penalties,
            beam_width=int(search.get("beam_width", 5)),
            evaluator=lambda o: float(objective_fn(evaluate(o))),
        ),
        "Simulated Annealing": lambda e: e.simulated_annealing(
            penalties=penalties,
            evaluator=lambda o: float(objective_fn(evaluate(o))),
            iterations=int(search.get("annealing_iterations", 240)),
            start_temp=float(search.get("annealing_start_temp", 3.2)),
            end_temp=float(search.get("annealing_end_temp", 0.05)),
        ),
    }
    out = {}
    for i, (name, builder) in enumerate(methods.items()):
        best, ev = _run_trials(graph, trials, seed + i * 97, builder, evaluate, objective_fn)
        out[name] = {"order": best.order, "metadata": best.metadata, "evaluation": ev}
    return out


def _x_metric(base_eval, cand_eval, weights):
    def improve(old, new):
        return 0.0 if abs(old) < 1e-9 else (old - new) / abs(old)

    base_idle = float(base_eval["memory"].get("idle_cycles", 0.0)) + float(base_eval["bandwidth"].get("pipeline_stalls", 0.0))
    cand_idle = float(cand_eval["memory"].get("idle_cycles", 0.0)) + float(cand_eval["bandwidth"].get("pipeline_stalls", 0.0))
    return (
        float(weights.get("cost", 0.3)) * improve(base_eval["breakdown"]["total_cost"], cand_eval["breakdown"]["total_cost"])
        + float(weights.get("latency", 0.25)) * improve(base_eval["latency_cycles"], cand_eval["latency_cycles"])
        + float(weights.get("dram", 0.25)) * improve(base_eval["memory"].get("dram_access", 0.0), cand_eval["memory"].get("dram_access", 0.0))
        + float(weights.get("stalls", 0.2)) * improve(base_idle, cand_idle)
    )


def main() -> None:
    start = time.perf_counter()
    root = Path(__file__).resolve().parent
    config = load_config(root / "config.yaml")
    graph = load_operator_graph(root / str(config["input"]["workload"]))
    hardware = load_hardware_config(config.get("hardware", {}), num_nodes=len(graph.nodes))
    penalties = dict(config.get("apr", {}).get("initial_penalties", {}))

    cost_model = ScheduleCostModel(
        memory_hierarchy=MemoryHierarchy(dict(config.get("hardware", {}))),
        bandwidth_estimator=BandwidthEstimator(dict(config.get("hardware", {}))),
        fusion_logic=FusionLogic(dict(config.get("fusion", {}))),
        weights=dict(config.get("cost_weights", {})),
    )
    alpha, beta, gamma = _derive_weights(config)
    cce = CCEEvaluator(graph, hardware, cost_model, alpha, beta, gamma, penalties)

    eval_cost = lambda order: cost_model.evaluate(graph, order, penalties=penalties)
    eval_cce = lambda order: cce.evaluate(order, penalties=penalties)
    obj_cost = lambda ev: float(ev["breakdown"]["total_cost"])
    obj_energy = lambda ev: float(ev["energy_breakdown"]["total_energy"])

    baseline = _run_suite(graph, config, penalties, eval_cost, obj_cost, seed_offset=0)
    cce_suite = _run_suite(graph, config, penalties, eval_cce, obj_energy, seed_offset=400)

    apr_cfg = dict(config.get("apr", {}))
    tuner = PenaltyTuner(
        eta1=float(apr_cfg.get("eta1", 0.9)),
        eta2=float(apr_cfg.get("eta2", 0.6)),
        lam_min=float(apr_cfg.get("lam_min", 0.1)),
        lam_max=float(apr_cfg.get("lam_max", 20.0)),
    )
    cur_pen = dict(penalties)
    best_apr = None
    best_apr_obj = float("inf")
    apr_trace = []
    for rnd in range(int(apr_cfg.get("rounds", 5))):
        engine = SchedulingEngine(graph, random_seed=900 + rnd * 17)
        res = engine.simulated_annealing(
            penalties=cur_pen,
            evaluator=lambda o: float(obj_energy(cce.evaluate(o, penalties=cur_pen))),
            iterations=int(apr_cfg.get("iterations_per_round", config.get("search", {}).get("annealing_iterations", 180))),
            start_temp=float(config.get("search", {}).get("annealing_start_temp", 3.0)),
            end_temp=float(config.get("search", {}).get("annealing_end_temp", 0.05)),
        )
        ev = cce.evaluate(res.order, penalties=cur_pen)
        obj = obj_energy(ev)
        if obj < best_apr_obj:
            best_apr_obj = obj
            best_apr = {"order": _complete_order(res.order, graph), "evaluation": ev}
        cur_pen = tuner.update(cur_pen, ev.get("violation_rate", {}), ev.get("cost_impact", {}))
        apr_trace.append({"round": rnd + 1, "objective": obj, "penalties": dict(cur_pen)})
    cce_apr = {"order": best_apr["order"], "evaluation": best_apr["evaluation"], "metadata": {"round_trace": apr_trace, "final_penalties": cur_pen}}

    qubo_data = cce.build_qubo_data(penalties=penalties)
    candidates = run_qaoa_stub(qubo_data, num_samples=int(config.get("quantum", {}).get("samples", 48)), num_steps=int(config.get("quantum", {}).get("iterations", 220)), seed=int(config.get("experiment", {}).get("seed", 17)) + 1234)
    best_q = None
    best_q_obj = float("inf")
    for c in candidates:
        order = _complete_order(c.get("schedule_projection", []), graph)
        ev = cce.evaluate(order, penalties=penalties)
        obj = obj_energy(ev)
        if obj < best_q_obj:
            best_q_obj = obj
            best_q = {"order": order, "evaluation": ev, "metadata": {"candidate": c, "backend": "qaoa_stub_local_search"}}
    quantum_stub = best_q

    ablations = {
        "full_cce": cce_suite["Simulated Annealing"],
        "no_pairwise": {"order": quantum_stub["order"], "evaluation": cce.evaluate(quantum_stub["order"], penalties=penalties, ablation="remove_pairwise"), "metadata": {}},
        "no_higher_order": {"order": quantum_stub["order"], "evaluation": cce.evaluate(quantum_stub["order"], penalties=penalties, ablation="remove_higher_order"), "metadata": {}},
    }

    analysis = ScheduleAnalysis()
    explainer = ScheduleExplainer()
    baseline_summary = {k: analysis.summarize(k, v["evaluation"]) for k, v in baseline.items()}
    cce_summary = {k: analysis.summarize(k, v["evaluation"]) for k, v in cce_suite.items()}
    extra = {
        "CCE + APR": analysis.summarize("CCE + APR", cce_apr["evaluation"]),
        "Quantum (Stub)": analysis.summarize("Quantum (Stub)", quantum_stub["evaluation"]),
    }

    best_base = min(baseline.items(), key=lambda kv: obj_cost(kv[1]["evaluation"]))
    best_cce = min(cce_suite.items(), key=lambda kv: obj_energy(kv[1]["evaluation"]))
    x_weights = dict(config.get("x_metric_weights", {"cost": 0.3, "latency": 0.25, "dram": 0.25, "stalls": 0.2}))
    x_cce = _x_metric(best_base[1]["evaluation"], best_cce[1]["evaluation"], x_weights)
    x_quantum = _x_metric(best_base[1]["evaluation"], quantum_stub["evaluation"], x_weights)

    output_dir = root / str(config.get("output", {}).get("directory", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "workload": {"num_nodes": len(graph.nodes), "num_edges": len(graph.edges)},
        "results": {
            "baseline_cost": baseline,
            "cce_qubo": cce_suite,
            "cce_qubo_apr": cce_apr,
            "quantum_stub": quantum_stub,
        },
        "ablations": ablations,
        "x_metric": {"weights": x_weights, "cce_vs_baseline": x_cce, "quantum_vs_baseline": x_quantum},
        "runtime_seconds": time.perf_counter() - start,
    }
    (output_dir / "schedules.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "metrics.txt").write_text(
        "[Baseline Cost Objective]\n"
        + analysis.comparison_table(baseline_summary)
        + "\n\n[CCE-QUBO Objective]\n"
        + analysis.comparison_table(cce_summary)
        + "\n\n[APR / Quantum Stub]\n"
        + analysis.comparison_table(extra)
        + "\n\n[X Metric]\n"
        + f"CCE vs baseline: {x_cce:.4f}\nQuantum stub vs baseline: {x_quantum:.4f}\n",
        encoding="utf-8",
    )
    explanations = [explainer.explain(name, payload, baseline_summary[name]) for name, payload in baseline.items()]
    explanations.extend(explainer.explain(name, payload, cce_summary[name]) for name, payload in cce_suite.items())
    explanations.append(explainer.explain("CCE + APR", cce_apr, extra["CCE + APR"]))
    explanations.append(explainer.explain("Quantum (Stub)", quantum_stub, extra["Quantum (Stub)"]))
    (output_dir / "explanations.txt").write_text("\n\n".join(explanations) + "\n", encoding="utf-8")

    print("Experiment completed.")
    print(f"Best baseline: {best_base[0]}")
    print(f"Best CCE-QUBO: {best_cce[0]}")
    print(f"X (CCE vs baseline): {x_cce:.4f}")
    print(f"X (Quantum stub vs baseline): {x_quantum:.4f}")
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
