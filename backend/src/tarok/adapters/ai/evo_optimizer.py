"""Evolutionary hyperparameter optimization for PPO training using DEAP.

Uses a (μ+λ) evolutionary strategy with Gaussian mutation and uniform
crossover to search the hyperparameter space.  Each individual encodes a
full set of PPO + agent hyperparameters.  Fitness is evaluated by running
a short training session and measuring win rate + reward trend.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from deap import base, creator, tools, algorithms

from tarok.adapters.ai.agent import RLAgent
from tarok.adapters.ai.trainer import PPOTrainer, TrainingMetrics


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

@dataclass
class IndividualResult:
    """Fitness evaluation result for one individual."""
    index: int
    genes: list[float]
    hparams: dict[str, Any]
    fitness: float
    win_rate: float
    reward_trend: float


@dataclass
class EvoProgress:
    """Live progress state for the evolution dashboard."""
    generation: int = 0
    total_generations: int = 0
    evaluating_index: int = 0        # Which individual is being evaluated
    evaluating_total: int = 0        # Total individuals to evaluate this gen
    phase: str = "idle"              # idle | evaluating | selecting | done
    elapsed_seconds: float = 0.0
    # Per-generation stats: list of {gen, avg, std, min, max}
    gen_stats: list[dict[str, Any]] = field(default_factory=list)
    # Current population: list of IndividualResult
    population: list[IndividualResult] = field(default_factory=list)
    # Hall of fame (top 5)
    hall_of_fame: list[IndividualResult] = field(default_factory=list)
    # Best fitness seen so far
    best_fitness: float = 0.0
    best_hparams: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "total_generations": self.total_generations,
            "evaluating_index": self.evaluating_index,
            "evaluating_total": self.evaluating_total,
            "phase": self.phase,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "gen_stats": self.gen_stats,
            "population": [
                {"index": r.index, "hparams": r.hparams, "fitness": round(r.fitness, 4),
                 "win_rate": round(r.win_rate, 4), "reward_trend": round(r.reward_trend, 4)}
                for r in self.population
            ],
            "hall_of_fame": [
                {"index": r.index, "hparams": r.hparams, "fitness": round(r.fitness, 4),
                 "win_rate": round(r.win_rate, 4), "reward_trend": round(r.reward_trend, 4)}
                for r in self.hall_of_fame
            ],
            "best_fitness": round(self.best_fitness, 4),
            "best_hparams": self.best_hparams,
        }

@dataclass
class HyperparamBounds:
    """Defines a single hyperparameter with its search bounds and type."""
    name: str
    low: float
    high: float
    log_scale: bool = False   # Sample/mutate in log-space (for lr, etc.)
    integer: bool = False     # Round to int after mutation

    def clip(self, value: float) -> float:
        v = max(self.low, min(self.high, value))
        return int(round(v)) if self.integer else v


# Default search space — covers the most impactful PPO hyperparameters
DEFAULT_SEARCH_SPACE: list[HyperparamBounds] = [
    HyperparamBounds("lr",               1e-5, 1e-2, log_scale=True),
    HyperparamBounds("gamma",            0.9,  0.999),
    HyperparamBounds("gae_lambda",       0.8,  0.99),
    HyperparamBounds("clip_epsilon",     0.05, 0.4),
    HyperparamBounds("value_coef",       0.1,  1.0),
    HyperparamBounds("entropy_coef",     0.001, 0.1, log_scale=True),
    HyperparamBounds("epochs_per_update", 1,   10,   integer=True),
    HyperparamBounds("batch_size",       16,   256,  integer=True),
    HyperparamBounds("hidden_size",      64,   512,  integer=True),
    HyperparamBounds("explore_rate",     0.01, 0.3),
    HyperparamBounds("fsp_ratio",        0.0,  0.5),
]


def _param_index(space: list[HyperparamBounds]) -> dict[str, int]:
    return {p.name: i for i, p in enumerate(space)}


# ---------------------------------------------------------------------------
# Individual ↔ hyperparameter dict conversion
# ---------------------------------------------------------------------------

def individual_to_hparams(ind: list[float], space: list[HyperparamBounds]) -> dict[str, Any]:
    """Convert a DEAP individual (flat float list) to a hyperparameter dict."""
    hparams: dict[str, Any] = {}
    for gene, bound in zip(ind, space):
        val = bound.clip(gene)
        if bound.log_scale:
            val = bound.clip(10 ** gene)
        hparams[bound.name] = val
    return hparams


def hparams_to_individual(hparams: dict[str, Any], space: list[HyperparamBounds]) -> list[float]:
    """Convert a hyperparameter dict back to a flat gene list."""
    genes: list[float] = []
    for bound in space:
        val = hparams.get(bound.name, (bound.low + bound.high) / 2)
        if bound.log_scale:
            val = math.log10(max(val, 1e-10))
        genes.append(float(val))
    return genes


def _random_individual(space: list[HyperparamBounds], rng: random.Random) -> list[float]:
    """Generate a random individual in gene-space."""
    genes: list[float] = []
    for b in space:
        if b.log_scale:
            lo = math.log10(b.low)
            hi = math.log10(b.high)
            genes.append(rng.uniform(lo, hi))
        else:
            genes.append(rng.uniform(b.low, b.high))
    return genes


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

async def evaluate_individual(
    individual: list[float],
    space: list[HyperparamBounds],
    eval_sessions: int,
    games_per_session: int,
    oracle: bool,
    device: str,
) -> tuple[float, float, float]:
    """Train with the given hyperparameters and return (fitness, win_rate, trend_score).

    Fitness = 0.6 * final_win_rate + 0.4 * normalized_reward_trend.
    This balances raw performance with learning progress.
    """
    hparams = individual_to_hparams(individual, space)

    hidden_size = int(hparams.pop("hidden_size", 256))
    explore_rate = float(hparams.pop("explore_rate", 0.1))

    agents = [
        RLAgent(
            name=f"Evo-{i}",
            hidden_size=hidden_size,
            device=device,
            explore_rate=explore_rate,
            oracle_critic=oracle,
        )
        for i in range(4)
    ]

    trainer = PPOTrainer(
        agents,
        lr=hparams.get("lr", 3e-4),
        gamma=hparams.get("gamma", 0.99),
        gae_lambda=hparams.get("gae_lambda", 0.95),
        clip_epsilon=hparams.get("clip_epsilon", 0.2),
        value_coef=hparams.get("value_coef", 0.5),
        entropy_coef=hparams.get("entropy_coef", 0.01),
        epochs_per_update=int(hparams.get("epochs_per_update", 4)),
        batch_size=int(hparams.get("batch_size", 64)),
        games_per_session=games_per_session,
        device=device,
        fsp_ratio=hparams.get("fsp_ratio", 0.3),
        save_dir="checkpoints/evo_tmp",
    )

    result = await trainer.train(eval_sessions)

    # --- Compute fitness ---
    win_rate = result.win_rate

    # Reward trend: compare last 30% of sessions vs first 30%
    scores = result.session_avg_score_history
    if len(scores) >= 4:
        split = max(1, len(scores) // 3)
        early = sum(scores[:split]) / split
        late = sum(scores[-split:]) / split
        # Normalize improvement to [0, 1] range (cap at ±200 points)
        trend = max(-1.0, min(1.0, (late - early) / 200.0))
        trend_score = (trend + 1.0) / 2.0  # shift to [0, 1]
    else:
        trend_score = 0.5

    fitness = 0.6 * win_rate + 0.4 * trend_score
    return (fitness, win_rate, trend_score)


# ---------------------------------------------------------------------------
# DEAP setup & evolutionary loop
# ---------------------------------------------------------------------------

@dataclass
class EvoConfig:
    """Configuration for the evolutionary optimizer."""
    population_size: int = 12
    num_generations: int = 10
    eval_sessions: int = 20     # Training sessions per fitness evaluation
    games_per_session: int = 10  # Games per training session (shorter for speed)
    cxpb: float = 0.5           # Crossover probability
    mutpb: float = 0.3          # Mutation probability
    mu_plus_lambda: bool = True  # Use (μ+λ) selection
    tournament_size: int = 3
    oracle: bool = False
    device: str = "cpu"
    seed: int = 42
    search_space: list[HyperparamBounds] = field(default_factory=lambda: list(DEFAULT_SEARCH_SPACE))
    output_dir: str = "checkpoints/evo_results"
    progress_callback: Any = None  # async callable(EvoProgress) or None


def _setup_deap(config: EvoConfig) -> tuple[base.Toolbox, tools.HallOfFame, tools.Statistics]:
    """Register DEAP types, operators, and statistics."""
    # Avoid re-registering if already done (e.g. in tests)
    if not hasattr(creator, "EvoFitness"):
        creator.create("EvoFitness", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.EvoFitness)

    rng = random.Random(config.seed)
    toolbox = base.Toolbox()

    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        lambda: _random_individual(config.search_space, rng),
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("mate", tools.cxUniform, indpb=0.5)

    # Gaussian mutation scaled per-gene
    sigmas = []
    for b in config.search_space:
        if b.log_scale:
            s = (math.log10(b.high) - math.log10(b.low)) * 0.15
        else:
            s = (b.high - b.low) * 0.15
        sigmas.append(s)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=sigmas, indpb=0.3)

    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)

    # Hall of fame keeps the best individual across all generations
    hof = tools.HallOfFame(5)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    return toolbox, hof, stats


def _clip_individual(ind: list[float], space: list[HyperparamBounds]) -> list[float]:
    """Clip all genes to their valid bounds (in gene-space)."""
    for i, b in enumerate(space):
        if b.log_scale:
            lo = math.log10(b.low)
            hi = math.log10(b.high)
            ind[i] = max(lo, min(hi, ind[i]))
        else:
            ind[i] = max(b.low, min(b.high, ind[i]))
        if b.integer and not b.log_scale:
            ind[i] = round(ind[i])
    return ind


async def run_evolution(config: EvoConfig | None = None) -> dict[str, Any]:
    """Run the full evolutionary hyperparameter search.

    Returns a dict with the best hyperparameters, fitness history,
    and the full DEAP logbook.
    """
    if config is None:
        config = EvoConfig()

    toolbox, hof, stats = _setup_deap(config)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = EvoProgress(
        total_generations=config.num_generations,
        phase="evaluating",
    )

    async def _notify():
        progress.elapsed_seconds = time.time() - start_time
        if config.progress_callback:
            await config.progress_callback(progress)

    def _make_ind_result(idx: int, ind, fit_tuple) -> IndividualResult:
        hp = individual_to_hparams(ind, config.search_space)
        return IndividualResult(
            index=idx, genes=list(ind), hparams=hp,
            fitness=fit_tuple[0], win_rate=fit_tuple[1], reward_trend=fit_tuple[2],
        )

    def _update_population_and_hof(population, hof):
        progress.population = [
            _make_ind_result(i, ind, (ind.fitness.values[0], 0, 0))
            for i, ind in enumerate(population) if ind.fitness.valid
        ]
        progress.hall_of_fame = [
            _make_ind_result(i, ind, (ind.fitness.values[0], 0, 0))
            for i, ind in enumerate(hof)
        ]
        if hof:
            best = hof[0]
            progress.best_fitness = best.fitness.values[0]
            progress.best_hparams = individual_to_hparams(best, config.search_space)

    print(f"🧬 Evolutionary Hyperparameter Optimization")
    print(f"   Population: {config.population_size} | Generations: {config.num_generations}")
    print(f"   Eval budget per individual: {config.eval_sessions} sessions × {config.games_per_session} games")
    print(f"   Search space: {len(config.search_space)} hyperparameters")
    print(f"   Device: {config.device} | Oracle: {config.oracle}")
    print()

    population = toolbox.population(n=config.population_size)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    start_time = time.time()

    # --- Evaluate initial population ---
    progress.generation = 0
    progress.evaluating_total = len(population)
    print(f"  Gen 0: evaluating {len(population)} individuals...")
    fitnesses = []
    for i, ind in enumerate(population):
        _clip_individual(ind, config.search_space)
        progress.evaluating_index = i + 1
        await _notify()
        fit = await evaluate_individual(
            ind, config.search_space,
            config.eval_sessions, config.games_per_session,
            config.oracle, config.device,
        )
        fitnesses.append(fit)
        hparams = individual_to_hparams(ind, config.search_space)
        print(f"    [{i+1}/{len(population)}] fitness={fit[0]:.4f} | lr={hparams['lr']:.2e} hidden={hparams['hidden_size']} ε={hparams['clip_epsilon']:.3f}")

    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = (fit[0],)

    hof.update(population)
    record = stats.compile(population)
    logbook.record(gen=0, nevals=len(population), **record)
    progress.gen_stats.append({"gen": 0, "nevals": len(population), **{k: float(v) for k, v in record.items()}})
    _update_population_and_hof(population, hof)
    # Backfill win_rate/trend for gen 0
    for i, (ind, fit) in enumerate(zip(population, fitnesses)):
        if i < len(progress.population):
            progress.population[i].win_rate = fit[1]
            progress.population[i].reward_trend = fit[2]
    await _notify()
    print(f"  Gen 0 stats: {record}")
    print()

    # --- Generational loop ---
    for gen in range(1, config.num_generations + 1):
        progress.generation = gen
        progress.phase = "selecting"
        await _notify()

        if config.mu_plus_lambda:
            offspring = algorithms.varOr(
                population, toolbox,
                lambda_=config.population_size,
                cxpb=config.cxpb,
                mutpb=config.mutpb,
            )
        else:
            offspring = algorithms.varAnd(population, toolbox, config.cxpb, config.mutpb)

        for ind in offspring:
            _clip_individual(ind, config.search_space)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        progress.phase = "evaluating"
        progress.evaluating_total = len(invalid_ind)
        print(f"  Gen {gen}: evaluating {len(invalid_ind)} individuals...")

        gen_fit_results: list[tuple] = []
        for i, ind in enumerate(invalid_ind):
            progress.evaluating_index = i + 1
            await _notify()
            fit = await evaluate_individual(
                ind, config.search_space,
                config.eval_sessions, config.games_per_session,
                config.oracle, config.device,
            )
            gen_fit_results.append(fit)
            ind.fitness.values = (fit[0],)
            hparams = individual_to_hparams(ind, config.search_space)
            print(f"    [{i+1}/{len(invalid_ind)}] fitness={fit[0]:.4f} | lr={hparams['lr']:.2e} hidden={hparams['hidden_size']} ε={hparams['clip_epsilon']:.3f}")

        if config.mu_plus_lambda:
            population = toolbox.select(population + offspring, config.population_size)
        else:
            population = toolbox.select(offspring, config.population_size)

        hof.update(population)
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        progress.gen_stats.append({"gen": gen, "nevals": len(invalid_ind), **{k: float(v) for k, v in record.items()}})
        _update_population_and_hof(population, hof)
        await _notify()
        print(f"  Gen {gen} stats: {record}")

        _save_evo_checkpoint(gen, population, hof, logbook, config)
        print()

    elapsed = time.time() - start_time

    # --- Results ---
    best_ind = hof[0]
    best_hparams = individual_to_hparams(best_ind, config.search_space)
    best_fitness = best_ind.fitness.values[0]

    result = {
        "best_hparams": best_hparams,
        "best_fitness": best_fitness,
        "elapsed_seconds": round(elapsed, 1),
        "logbook": [dict(r) for r in logbook],
        "top_5": [
            {"hparams": individual_to_hparams(ind, config.search_space), "fitness": ind.fitness.values[0]}
            for ind in hof
        ],
    }

    results_path = output_dir / "evo_best_hparams.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    progress.phase = "done"
    await _notify()

    print()
    print(f"🏆 Best hyperparameters (fitness={best_fitness:.4f}):")
    for k, v in best_hparams.items():
        print(f"   {k}: {v}")
    print()
    print(f"   Total time: {elapsed/60:.1f} min")
    print(f"   Results saved to {results_path}")

    return result


def _save_evo_checkpoint(
    gen: int,
    population: list,
    hof: tools.HallOfFame,
    logbook: tools.Logbook,
    config: EvoConfig,
) -> None:
    """Save evolutionary search state after each generation."""
    output_dir = Path(config.output_dir)
    checkpoint = {
        "generation": gen,
        "population": [
            {"genes": list(ind), "fitness": ind.fitness.values[0] if ind.fitness.valid else None}
            for ind in population
        ],
        "hall_of_fame": [
            {"genes": list(ind), "fitness": ind.fitness.values[0]}
            for ind in hof
        ],
        "logbook": [dict(r) for r in logbook],
    }
    path = output_dir / f"evo_gen{gen}.json"
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Convenience: train with best evolved hyperparameters
# ---------------------------------------------------------------------------

async def train_with_best(
    results_path: str | Path,
    num_sessions: int = 100,
    games_per_session: int = 20,
    oracle: bool = False,
    device: str = "cpu",
) -> TrainingMetrics:
    """Load best hyperparameters from an evo run and do a full training."""
    with open(results_path) as f:
        result = json.load(f)

    hparams = result["best_hparams"]
    print(f"🧬 Training with evolved hyperparameters (fitness={result['best_fitness']:.4f})")
    for k, v in hparams.items():
        print(f"   {k}: {v}")
    print()

    hidden_size = int(hparams.pop("hidden_size", 256))
    explore_rate = float(hparams.pop("explore_rate", 0.1))

    agents = [
        RLAgent(
            name=f"Evo-Agent-{i}",
            hidden_size=hidden_size,
            device=device,
            explore_rate=explore_rate,
            oracle_critic=oracle,
        )
        for i in range(4)
    ]

    trainer = PPOTrainer(
        agents,
        lr=hparams.get("lr", 3e-4),
        gamma=hparams.get("gamma", 0.99),
        gae_lambda=hparams.get("gae_lambda", 0.95),
        clip_epsilon=hparams.get("clip_epsilon", 0.2),
        value_coef=hparams.get("value_coef", 0.5),
        entropy_coef=hparams.get("entropy_coef", 0.01),
        epochs_per_update=int(hparams.get("epochs_per_update", 4)),
        batch_size=int(hparams.get("batch_size", 64)),
        games_per_session=games_per_session,
        device=device,
        fsp_ratio=hparams.get("fsp_ratio", 0.3),
    )

    return await trainer.train(num_sessions)
