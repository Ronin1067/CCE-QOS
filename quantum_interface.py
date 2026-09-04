from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

from graph_builder import WorkloadGraph
from penalty_tuner import AdaptivePenaltyRefinement


@dataclass
class QuantumResult:
    strategy: str
    order: List[str]
    score: float
    metadata: Dict


class QuantumInterface:
    def __init__(self, graph: WorkloadGraph, random_seed: int = 101):
        self.graph = graph
        self.rng = random.Random(random_seed)

    def _valid(self, order: Sequence[str]) -> bool:
        return self.graph.is_valid_order(order)

    def _sample_neighbor(self, order: Sequence[str], exploration: float) -> List[str]:
        base = list(order)
        n = len(base)
        tries = max(14, int(65 * exploration))
        for _ in range(tries):
            move = self.rng.choice(["swap", "insert", "block"])
            proposal = list(base)

            if move == "swap":
                i = self.rng.randrange(0, n)
                j = self.rng.randrange(0, n)
                if i == j:
                    continue
                proposal[i], proposal[j] = proposal[j], proposal[i]
            elif move == "insert":
                i = self.rng.randrange(0, n)
                j = self.rng.randrange(0, n)
                if i == j:
                    continue
                value = proposal.pop(i)
                proposal.insert(j, value)
            else:
                if n < 4:
                    continue
                left = self.rng.randrange(0, n - 2)
                right = self.rng.randrange(left + 1, min(n, left + 5))
                proposal[left:right] = reversed(proposal[left:right])

            if self._valid(proposal):
                return proposal
        return base

    def _objective(
        self,
        order: Sequence[str],
        evaluator: Callable[[Sequence[str], Dict[str, float] | None], Dict],
        objective_from_evaluation: Callable[[Dict], float],
        penalties: Dict[str, float] | None,
    ) -> Tuple[float, Dict]:
        if not self._valid(order):
            return float("inf"), {
                "breakdown": {"total_cost": float("inf")},
                "feasibility": 0.0,
                "latency_cycles": float("inf"),
            }
        eval_report = evaluator(order, penalties)
        objective = float(objective_from_evaluation(eval_report))
        return objective, eval_report

    def qaoa_refine(
        self,
        seed_order: Sequence[str],
        evaluator: Callable[[Sequence[str], Dict[str, float] | None], Dict],
        objective_from_evaluation: Callable[[Dict], float] | None = None,
        penalties: Dict[str, float] | None = None,
        layers: int = 2,
        iterations: int = 80,
        walkers: int = 6,
    ) -> QuantumResult:
        objective_from_evaluation = objective_from_evaluation or (lambda report: report["breakdown"]["total_cost"])
        layers = max(1, layers)
        walkers = max(2, walkers)

        gamma = [self.rng.uniform(0.3, 1.2) for _ in range(layers)]
        beta = [self.rng.uniform(0.2, 1.0) for _ in range(layers)]

        chains = [list(seed_order)]
        for _ in range(walkers - 1):
            chains.append(self._sample_neighbor(seed_order, exploration=1.1))

        chain_energy = []
        chain_eval = []
        for order in chains:
            e, rep = self._objective(order, evaluator, objective_from_evaluation, penalties)
            chain_energy.append(e)
            chain_eval.append(rep)

        best_idx = min(range(len(chains)), key=lambda i: chain_energy[i])
        best_order = list(chains[best_idx])
        best_energy = chain_energy[best_idx]
        best_eval = chain_eval[best_idx]
        accepted = 0
        moves = 0

        for step in range(iterations):
            layer = step % layers
            local_accept = 0

            for w in range(walkers):
                moves += 1
                base_order = chains[w]
                exploration = max(0.2, min(2.4, 1.2 + beta[layer] - 0.55 * gamma[layer]))
                proposal = self._sample_neighbor(base_order, exploration)
                prop_energy, prop_eval = self._objective(proposal, evaluator, objective_from_evaluation, penalties)

                delta = prop_energy - chain_energy[w]
                temp = max(0.04, gamma[layer])
                accept_prob = 1.0 if delta <= 0 else math.exp(-delta / temp)
                if self.rng.random() < accept_prob:
                    chains[w] = proposal
                    chain_energy[w] = prop_energy
                    chain_eval[w] = prop_eval
                    accepted += 1
                    local_accept += 1

                if chain_energy[w] < best_energy:
                    best_energy = chain_energy[w]
                    best_order = list(chains[w])
                    best_eval = chain_eval[w]

            accept_ratio = local_accept / max(1, walkers)
            gamma[layer] = max(0.05, min(2.0, gamma[layer] * (0.97 + 0.06 * (1.0 - accept_ratio))))
            beta[layer] = max(0.05, min(2.0, beta[layer] * (0.95 + 0.08 * accept_ratio)))

        return QuantumResult(
            strategy="Quantum (QAOA)",
            order=best_order,
            score=best_energy,
            metadata={
                "layers": layers,
                "iterations": iterations,
                "walkers": walkers,
                "acceptance_ratio": accepted / max(1, moves),
                "backend": "qaoa_style_multiwalker_simulator",
                "final_gamma": gamma,
                "final_beta": beta,
                "best_objective": best_energy,
                "best_cost": best_eval["breakdown"]["total_cost"],
                "final_breakdown": best_eval["breakdown"],
            },
        )

    def qaoa_with_apr(
        self,
        seed_order: Sequence[str],
        evaluator: Callable[[Sequence[str], Dict[str, float] | None], Dict],
        apr: AdaptivePenaltyRefinement,
        objective_from_evaluation: Callable[[Dict], float] | None = None,
        rounds: int = 4,
        layers: int = 2,
        iterations_per_round: int = 55,
        walkers: int = 6,
    ) -> QuantumResult:
        objective_from_evaluation = objective_from_evaluation or (lambda report: report["breakdown"]["total_cost"])
        order = list(seed_order)
        best_order = list(seed_order)
        best_objective = float("inf")
        round_trace = []

        for idx in range(rounds):
            penalties = apr.get()
            adaptive_iters = int(iterations_per_round * (1.0 + 0.15 * idx))
            round_result = self.qaoa_refine(
                seed_order=order,
                evaluator=evaluator,
                objective_from_evaluation=objective_from_evaluation,
                penalties=penalties,
                layers=layers,
                iterations=adaptive_iters,
                walkers=walkers,
            )
            round_eval = evaluator(round_result.order, penalties)
            objective_score = float(objective_from_evaluation(round_eval))
            cost_score = float(round_eval["breakdown"]["total_cost"])
            order = list(round_result.order)

            apr_state = apr.update(round_eval["violation_rate"], round_eval["cost_impact"])
            round_trace.append(
                {
                    "round": idx + 1,
                    "objective_score": objective_score,
                    "cost_score": cost_score,
                    "feasibility": round_eval.get("feasibility", 0.0),
                    "penalties": apr_state.penalties,
                    "apr_signal": apr_state.signal_snapshot,
                    "violation_rate": round_eval["violation_rate"],
                }
            )
            if objective_score < best_objective:
                best_objective = objective_score
                best_order = list(round_result.order)

        return QuantumResult(
            strategy="Quantum + APR",
            order=best_order,
            score=best_objective,
            metadata={
                "backend": "qaoa_style_multiwalker_simulator",
                "rounds": rounds,
                "layers": layers,
                "walkers": walkers,
                "round_trace": round_trace,
                "final_penalties": apr.get(),
                "best_objective": best_objective,
            },
        )
