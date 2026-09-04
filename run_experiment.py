from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

from bandwidth_estimator import BandwidthEstimator
from cost_model import ScheduleCostModel
from energy_model import EnergyScheduleModel
from fusion_logic import FusionLogic
from graph_builder import WorkloadGraph, load_workload
from memory_hierarchy import MemoryHierarchy
from penalty_tuner import AdaptivePenaltyRefinement
from quantum_interface import QuantumInterface
from schedule_analysis import ScheduleAnalysis
from schedule_explainer import ScheduleExplainer
from scheduling_engine import ScheduleResult, SchedulingEngine

ObjectiveFromEvaluation = Callable[[Dict], float]
OrderEvaluator = Callable[[Sequence[str], Dict[str, float] | None], Dict]


def load_config(path: Path) -> Dict:
    raw_text = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "config.yaml uses non-JSON YAML syntax but PyYAML is not installed. "
                "Either install 'pyyaml' or keep config.yaml JSON-compatible."
            ) from exc
        loaded = yaml.safe_load(raw_text)
        if not isinstance(loaded, dict):
            raise ValueError("Configuration root must be a mapping/object.")
        return loaded


def objective_cost(evaluation: Dict) -> float:
    return float(evaluation["breakdown"]["total_cost"])


def objective_energy(evaluation: Dict) -> float:
    return float(evaluation.get("energy_breakdown", {}).get("total_energy", evaluation["breakdown"]["total_cost"]))


def rank_score(evaluation: Dict, objective_from_evaluation: ObjectiveFromEvaluation) -> float:
    objective = objective_from_evaluation(evaluation)
    return objective - 120.0 * float(evaluation.get("feasibility", 0.0))


def select_best_result(
    trials: int,
    base_seed: int,
    builder: Callable[[int], ScheduleResult],
    evaluate_with_penalties: Callable[[Sequence[str]], Dict],
    objective_from_evaluation: ObjectiveFromEvaluation,
) -> Tuple[ScheduleResult, Dict]:
    best_result: ScheduleResult | None = None
    best_eval: Dict | None = None
    best_rank = float("inf")
    trial_log = []

    for idx in range(trials):
        seed = base_seed + idx * 37
        result = builder(seed)
        evaluation = evaluate_with_penalties(result.order)
        objective = objective_from_evaluation(evaluation)
        score = rank_score(evaluation, objective_from_evaluation)
        trial_log.append(
            {
                "trial": idx + 1,
                "seed": seed,
                "rank_score": score,
                "objective": objective,
                "cost": evaluation["breakdown"]["total_cost"],
                "feasibility": evaluation["feasibility"],
            }
        )
        if score < best_rank:
            best_rank = score
            best_result = result
            best_eval = evaluation

    if best_result is None or best_eval is None:
        raise RuntimeError("No trial produced a valid schedule result.")

    best_result.metadata = dict(best_result.metadata)
    best_result.metadata["trials"] = trial_log
    best_result.metadata["selected_rank_score"] = best_rank
    best_result.metadata["selected_objective"] = objective_from_evaluation(best_eval)
    return best_result, best_eval


def run_strategy_suite(
    graph: WorkloadGraph,
    config: Dict,
    evaluate: OrderEvaluator,
    objective_from_evaluation: ObjectiveFromEvaluation,
    objective_name: str,
) -> Dict[str, Dict]:
    base_seed = int(config["experiment"]["seed"])
    search_trials = int(config["experiment"].get("search_trials", 2))
    quantum_trials = int(config["experiment"].get("quantum_trials", 2))

    initial_penalties = dict(config["apr"]["initial_penalties"])
    baseline_penalties = dict(initial_penalties)

    def greedy_builder(seed: int) -> ScheduleResult:
        scheduler = SchedulingEngine(graph=graph, random_seed=seed)
        return scheduler.greedy(penalties=baseline_penalties)

    greedy, greedy_eval = select_best_result(
        trials=search_trials,
        base_seed=base_seed + 11,
        builder=greedy_builder,
        evaluate_with_penalties=lambda order: evaluate(order, baseline_penalties),
        objective_from_evaluation=objective_from_evaluation,
    )

    def lookahead_builder(seed: int) -> ScheduleResult:
        scheduler = SchedulingEngine(graph=graph, random_seed=seed)
        return scheduler.lookahead(
            penalties=baseline_penalties,
            lookahead_depth=int(config["search"]["lookahead_depth"]),
            evaluator=lambda order: objective_from_evaluation(evaluate(order, baseline_penalties)),
        )

    lookahead, lookahead_eval = select_best_result(
        trials=search_trials,
        base_seed=base_seed + 67,
        builder=lookahead_builder,
        evaluate_with_penalties=lambda order: evaluate(order, baseline_penalties),
        objective_from_evaluation=objective_from_evaluation,
    )

    def beam_builder(seed: int) -> ScheduleResult:
        scheduler = SchedulingEngine(graph=graph, random_seed=seed)
        return scheduler.beam_search(
            penalties=baseline_penalties,
            beam_width=int(config["search"]["beam_width"]),
            evaluator=lambda order: objective_from_evaluation(evaluate(order, baseline_penalties)),
        )

    beam, beam_eval = select_best_result(
        trials=search_trials,
        base_seed=base_seed + 101,
        builder=beam_builder,
        evaluate_with_penalties=lambda order: evaluate(order, baseline_penalties),
        objective_from_evaluation=objective_from_evaluation,
    )

    def anneal_builder(seed: int) -> ScheduleResult:
        scheduler = SchedulingEngine(graph=graph, random_seed=seed)
        return scheduler.simulated_annealing(
            penalties=baseline_penalties,
            evaluator=lambda order: objective_from_evaluation(evaluate(order, baseline_penalties)),
            iterations=int(config["search"]["annealing_iterations"]),
            start_temp=float(config["search"]["annealing_start_temp"]),
            end_temp=float(config["search"]["annealing_end_temp"]),
        )

    anneal, anneal_eval = select_best_result(
        trials=search_trials,
        base_seed=base_seed + 149,
        builder=anneal_builder,
        evaluate_with_penalties=lambda order: evaluate(order, baseline_penalties),
        objective_from_evaluation=objective_from_evaluation,
    )

    def quantum_builder(seed: int) -> ScheduleResult:
        q = QuantumInterface(graph=graph, random_seed=seed)
        q_result = q.qaoa_refine(
            seed_order=anneal.order,
            evaluator=evaluate,
            objective_from_evaluation=objective_from_evaluation,
            penalties=baseline_penalties,
            layers=int(config["quantum"]["layers"]),
            iterations=int(config["quantum"]["iterations"]),
            walkers=int(config["quantum"].get("walkers", 6)),
        )
        return ScheduleResult(
            strategy=q_result.strategy,
            order=q_result.order,
            score=q_result.score,
            metadata=q_result.metadata,
        )

    quantum_baseline, quantum_eval = select_best_result(
        trials=quantum_trials,
        base_seed=base_seed + 191,
        builder=quantum_builder,
        evaluate_with_penalties=lambda order: evaluate(order, baseline_penalties),
        objective_from_evaluation=objective_from_evaluation,
    )

    apr_warm = AdaptivePenaltyRefinement(initial_penalties=initial_penalties)
    warm_scheduler = SchedulingEngine(graph=graph, random_seed=base_seed + 509)
    apr_warm_sa = warm_scheduler.simulated_annealing(
        penalties=apr_warm.get(),
        evaluator=lambda order: objective_from_evaluation(evaluate(order, apr_warm.get())),
        iterations=int(config["apr"]["classical_warmup_iterations"]),
        start_temp=float(config["search"]["annealing_start_temp"]),
        end_temp=float(config["search"]["annealing_end_temp"]),
    )

    warm_candidates = [lookahead.order, beam.order, anneal.order, apr_warm_sa.order]
    warm_logs = []
    for order in warm_candidates:
        e = evaluate(order, apr_warm.get())
        apr_warm.update(e["violation_rate"], e["cost_impact"])
        warm_logs.append(
            {
                "objective": objective_from_evaluation(e),
                "cost": e["breakdown"]["total_cost"],
                "feasibility": e["feasibility"],
                "penalties_after_update": apr_warm.get(),
            }
        )

    ranked_warm = []
    for order in warm_candidates:
        probe = evaluate(order, apr_warm.get())
        ranked_warm.append((-probe["feasibility"], objective_from_evaluation(probe), order))
    ranked_warm.sort(key=lambda item: (item[0], item[1]))
    apr_seed_order = ranked_warm[0][2]

    def quantum_apr_builder(seed: int) -> ScheduleResult:
        q = QuantumInterface(graph=graph, random_seed=seed)
        apr_instance = AdaptivePenaltyRefinement(initial_penalties=apr_warm.get())
        q_result = q.qaoa_with_apr(
            seed_order=apr_seed_order,
            evaluator=evaluate,
            apr=apr_instance,
            objective_from_evaluation=objective_from_evaluation,
            rounds=int(config["apr"]["rounds"]),
            layers=int(config["quantum"]["layers"]),
            iterations_per_round=int(config["apr"]["iterations_per_round"]),
            walkers=int(config["quantum"].get("walkers", 6)),
        )
        metadata = dict(q_result.metadata)
        metadata["warmup_trace"] = warm_logs
        return ScheduleResult(
            strategy=q_result.strategy,
            order=q_result.order,
            score=q_result.score,
            metadata=metadata,
        )

    quantum_apr: ScheduleResult | None = None
    quantum_apr_eval: Dict | None = None
    quantum_apr_rank = float("inf")
    quantum_apr_trials = []

    for idx in range(quantum_trials):
        seed = base_seed + 239 + idx * 37
        candidate = quantum_apr_builder(seed)
        candidate_penalties = candidate.metadata.get("final_penalties", apr_warm.get())
        candidate_eval = evaluate(candidate.order, baseline_penalties)
        candidate_objective = objective_from_evaluation(candidate_eval)
        candidate_rank = rank_score(candidate_eval, objective_from_evaluation)
        quantum_apr_trials.append(
            {
                "trial": idx + 1,
                "seed": seed,
                "rank_score": candidate_rank,
                "objective": candidate_objective,
                "cost": candidate_eval["breakdown"]["total_cost"],
                "feasibility": candidate_eval["feasibility"],
                "selection_penalties": baseline_penalties,
                "apr_final_penalties": candidate_penalties,
            }
        )
        if candidate_rank < quantum_apr_rank:
            quantum_apr_rank = candidate_rank
            quantum_apr = candidate
            quantum_apr_eval = candidate_eval

    if quantum_apr is None or quantum_apr_eval is None:
        raise RuntimeError("No Quantum + APR trial produced a valid schedule result.")
    quantum_apr.metadata = dict(quantum_apr.metadata)
    quantum_apr.metadata["trials"] = quantum_apr_trials
    quantum_apr.metadata["selected_rank_score"] = quantum_apr_rank
    quantum_apr.metadata["selected_objective"] = objective_from_evaluation(quantum_apr_eval)

    results = {
        "Greedy": {"order": greedy.order, "metadata": greedy.metadata, "evaluation": greedy_eval},
        "Lookahead": {"order": lookahead.order, "metadata": lookahead.metadata, "evaluation": lookahead_eval},
        "Beam Search": {"order": beam.order, "metadata": beam.metadata, "evaluation": beam_eval},
        "Simulated Annealing": {"order": anneal.order, "metadata": anneal.metadata, "evaluation": anneal_eval},
        "Quantum (QAOA)": {
            "order": quantum_baseline.order,
            "metadata": quantum_baseline.metadata,
            "evaluation": quantum_eval,
        },
        "Quantum + APR": {"order": quantum_apr.order, "metadata": quantum_apr.metadata, "evaluation": quantum_apr_eval},
    }

    for payload in results.values():
        payload["metadata"] = dict(payload["metadata"])
        payload["metadata"]["objective_name"] = objective_name
    return results


def build_formulation_comparison(cost_results: Dict[str, Dict], energy_results: Dict[str, Dict]) -> Dict[str, Dict]:
    comparison = {}
    for strategy in cost_results:
        old_cost = float(cost_results[strategy]["evaluation"]["breakdown"]["total_cost"])
        new_cost = float(energy_results[strategy]["evaluation"]["breakdown"]["total_cost"])
        old_latency = float(cost_results[strategy]["evaluation"]["latency_cycles"])
        new_latency = float(energy_results[strategy]["evaluation"]["latency_cycles"])
        old_feasible = float(cost_results[strategy]["evaluation"]["feasibility"])
        new_feasible = float(energy_results[strategy]["evaluation"]["feasibility"])
        delta = old_cost - new_cost
        pct = 0.0 if abs(old_cost) < 1e-9 else delta / old_cost
        comparison[strategy] = {
            "old_cost": old_cost,
            "new_cost": new_cost,
            "delta_cost": delta,
            "improvement_percent": pct,
            "latency_delta": old_latency - new_latency,
            "feasibility_delta": (new_feasible - old_feasible) * 100.0,
        }
    return comparison


def formulation_comparison_table(comparison: Dict[str, Dict]) -> str:
    header = (
        f"{'Strategy':<22}"
        f"{'Old Cost':>12}"
        f"{'New Cost':>12}"
        f"{'Delta':>12}"
        f"{'Improve%':>12}"
        f"{'LatDelta':>10}"
        f"{'FeasDelta%':>12}"
    )
    line = "-" * len(header)
    rows = [header, line]
    ranked = sorted(comparison.items(), key=lambda item: item[1]["improvement_percent"], reverse=True)
    for strategy, item in ranked:
        rows.append(
            f"{strategy:<22}"
            f"{item['old_cost']:>12.2f}"
            f"{item['new_cost']:>12.2f}"
            f"{item['delta_cost']:>12.2f}"
            f"{item['improvement_percent'] * 100.0:>12.2f}"
            f"{item['latency_delta']:>10.2f}"
            f"{item['feasibility_delta']:>12.2f}"
        )
    return "\n".join(rows)


def x_factor(baseline_cost: float, optimized_cost: float) -> float:
    if abs(baseline_cost) < 1e-9:
        return 0.0
    return (baseline_cost - optimized_cost) / baseline_cost


def _best_classical_cost(results: Dict[str, Dict]) -> float:
    keys = ["Greedy", "Lookahead", "Beam Search", "Simulated Annealing"]
    return min(float(results[name]["evaluation"]["breakdown"]["total_cost"]) for name in keys)


def build_x_factors(cost_results: Dict[str, Dict], energy_results: Dict[str, Dict]) -> Dict[str, Dict]:
    baseline = float(cost_results["Greedy"]["evaluation"]["breakdown"]["total_cost"])
    classical_optimized = _best_classical_cost(cost_results)
    quantum_optimized = float(cost_results["Quantum (QAOA)"]["evaluation"]["breakdown"]["total_cost"])
    quantum_new_optimized = min(
        float(energy_results["Quantum (QAOA)"]["evaluation"]["breakdown"]["total_cost"]),
        float(energy_results["Quantum + APR"]["evaluation"]["breakdown"]["total_cost"]),
    )

    return {
        "classical_methods": {
            "baseline_cost": baseline,
            "optimized_cost": classical_optimized,
            "x_factor": x_factor(baseline, classical_optimized),
        },
        "quantum": {
            "baseline_cost": baseline,
            "optimized_cost": quantum_optimized,
            "x_factor": x_factor(baseline, quantum_optimized),
        },
        "quantum_plus_new_formulation": {
            "baseline_cost": baseline,
            "optimized_cost": quantum_new_optimized,
            "x_factor": x_factor(baseline, quantum_new_optimized),
        },
    }


def xfactor_table(x_factors: Dict[str, Dict]) -> str:
    header = f"{'Track':<30}{'Baseline':>12}{'Optimized':>12}{'X':>10}"
    line = "-" * len(header)
    rows = [header, line]
    for track, payload in x_factors.items():
        rows.append(
            f"{track:<30}"
            f"{payload['baseline_cost']:>12.2f}"
            f"{payload['optimized_cost']:>12.2f}"
            f"{payload['x_factor']:>10.4f}"
        )
    return "\n".join(rows)


def build_explanations(results: Dict[str, Dict], summaries: Dict[str, Dict], explainer: ScheduleExplainer) -> str:
    blocks = []
    for name, payload in results.items():
        blocks.append(explainer.explain(name, payload, summaries[name]))
        blocks.append("")
    return "\n".join(blocks).strip()


def main() -> None:
    root = Path(__file__).resolve().parent
    config = load_config(root / "config.yaml")
    graph = load_workload(root / config["input"]["workload"])

    memory_hierarchy = MemoryHierarchy(config["hardware"])
    bandwidth_estimator = BandwidthEstimator(config["hardware"])
    fusion_logic = FusionLogic(config["fusion"])

    cost_model = ScheduleCostModel(
        memory_hierarchy=memory_hierarchy,
        bandwidth_estimator=bandwidth_estimator,
        fusion_logic=fusion_logic,
        weights=config["cost_weights"],
    )
    energy_model = EnergyScheduleModel(
        cost_model=cost_model,
        fusion_logic=fusion_logic,
        weights=config.get("energy_weights", {}),
        hardware_config=config.get("hardware", {}),
    )

    analysis = ScheduleAnalysis()
    explainer = ScheduleExplainer()

    def evaluate_cost(order: Sequence[str], penalties: Dict[str, float] | None = None) -> Dict:
        return cost_model.evaluate(graph, order, penalties=penalties)

    def evaluate_energy(order: Sequence[str], penalties: Dict[str, float] | None = None) -> Dict:
        return energy_model.evaluate(graph, order, penalties=penalties)

    cost_results = run_strategy_suite(
        graph=graph,
        config=config,
        evaluate=evaluate_cost,
        objective_from_evaluation=objective_cost,
        objective_name="cost",
    )
    energy_results = run_strategy_suite(
        graph=graph,
        config=config,
        evaluate=evaluate_energy,
        objective_from_evaluation=objective_energy,
        objective_name="energy",
    )

    cost_summaries = {name: analysis.summarize(name, payload["evaluation"]) for name, payload in cost_results.items()}
    energy_summaries = {name: analysis.summarize(name, payload["evaluation"]) for name, payload in energy_results.items()}

    cost_table = analysis.comparison_table(cost_summaries)
    energy_table = analysis.comparison_table(energy_summaries)

    comparison = build_formulation_comparison(cost_results, energy_results)
    comparison_table = formulation_comparison_table(comparison)

    x_factors = build_x_factors(cost_results, energy_results)
    x_table = xfactor_table(x_factors)

    cost_explanations = build_explanations(cost_results, cost_summaries, explainer)
    energy_explanations = build_explanations(energy_results, energy_summaries, explainer)

    output_dir = root / config["output"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "schedules.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config_snapshot": config,
                "formulations": {
                    "cost_based": {
                        "results": cost_results,
                        "summaries": cost_summaries,
                    },
                    "energy_based": {
                        "results": energy_results,
                        "summaries": energy_summaries,
                    },
                },
                "comparison": comparison,
                "x_factors": x_factors,
            },
            f,
            indent=2,
        )

    with (output_dir / "metrics.txt").open("w", encoding="utf-8") as f:
        f.write("[Old Formulation: Cost-Based]\n")
        f.write(cost_table)
        f.write("\n\n[New Formulation: Energy-Based]\n")
        f.write(energy_table)
        f.write("\n\n[Old vs New Comparison]\n")
        f.write(comparison_table)
        f.write("\n\n[X Factor]\n")
        f.write(x_table)
        f.write("\n")

    with (output_dir / "explanations.txt").open("w", encoding="utf-8") as f:
        f.write("[Old Formulation]\n")
        f.write(cost_explanations)
        f.write("\n\n[Energy-Based Formulation]\n")
        f.write(energy_explanations)
        f.write("\n")

    print("Experiment completed.")
    print("[Old Formulation: Cost-Based]")
    print(cost_table)
    print()
    print("[New Formulation: Energy-Based]")
    print(energy_table)
    print()
    print("[Old vs New Comparison]")
    print(comparison_table)
    print()
    print("[X Factor]")
    print(x_table)
    print(f"Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
