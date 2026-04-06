"""Multi-phase behavioral breeding pipeline.

Evolves agent *personality* (behavioral traits that bias the policy) rather
than hyperparameters.  The pipeline cycles through:

  Phase 1 — Warmup:       Train a base model via standard self-play
  Phase 2 — Breed:        Fork the base into a population of behavioral variants
  Phase 3 — Evaluate:     Each variant plays evaluation games; measure fitness
  Phase 4 — Refine:       Top 2 variants do further self-play training together
  (repeat from Phase 2 for the configured number of breeding cycles)

The base model weights are shared — only the behavioral profile (logit biases,
temperature, exploration schedule) differs between individuals.
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from deap import base, creator, tools, algorithms

from tarok.adapters.ai.agent import RLAgent
from tarok.adapters.ai.behavioral_profile import (
    BehavioralProfile,
    GENE_BOUNDS,
    GENE_SIGMAS,
    NUM_GENES,
)
from tarok.adapters.ai.trainer import PPOTrainer, TrainingMetrics
from tarok.use_cases.game_loop import GameLoop


# ---------------------------------------------------------------------------
# Breeding progress (for dashboard)
# ---------------------------------------------------------------------------

@dataclass
class BreedingIndividual:
    index: int
    profile: dict[str, float]
    fitness: float
    win_rate: float
    avg_reward: float
    bid_rate: float
    solo_rate: float

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "profile": self.profile,
            "fitness": round(self.fitness, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_reward": round(self.avg_reward, 4),
            "bid_rate": round(self.bid_rate, 4),
            "solo_rate": round(self.solo_rate, 4),
        }


@dataclass
class BreedingProgress:
    """Live progress state for the breeding dashboard."""
    phase: str = "idle"                     # idle | warmup | breeding | evaluating | refining | done
    cycle: int = 0
    total_cycles: int = 0
    generation: int = 0
    total_generations: int = 0
    evaluating_index: int = 0
    evaluating_total: int = 0
    warmup_session: int = 0
    warmup_total_sessions: int = 0
    refine_session: int = 0
    refine_total_sessions: int = 0
    elapsed_seconds: float = 0.0
    model_name: str = ""
    # Current population
    population: list[BreedingIndividual] = field(default_factory=list)
    # Hall of fame
    hall_of_fame: list[BreedingIndividual] = field(default_factory=list)
    best_fitness: float = 0.0
    best_profile: dict[str, float] = field(default_factory=dict)
    # Per-generation fitness stats
    gen_stats: list[dict[str, Any]] = field(default_factory=list)
    # Per-cycle summary
    cycle_summaries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "cycle": self.cycle,
            "total_cycles": self.total_cycles,
            "generation": self.generation,
            "total_generations": self.total_generations,
            "evaluating_index": self.evaluating_index,
            "evaluating_total": self.evaluating_total,
            "warmup_session": self.warmup_session,
            "warmup_total_sessions": self.warmup_total_sessions,
            "refine_session": self.refine_session,
            "refine_total_sessions": self.refine_total_sessions,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "model_name": self.model_name,
            "population": [ind.to_dict() for ind in self.population],
            "hall_of_fame": [ind.to_dict() for ind in self.hall_of_fame],
            "best_fitness": round(self.best_fitness, 4),
            "best_profile": self.best_profile,
            "gen_stats": self.gen_stats,
            "cycle_summaries": self.cycle_summaries,
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BreedingConfig:
    # --- Warmup phase ---
    warmup_sessions: int = 50        # Sessions of self-play for the base model
    warmup_games_per_session: int = 20

    # --- Breeding evolution ---
    population_size: int = 12
    num_generations: int = 5          # DEAP generations per breeding cycle
    num_cycles: int = 3               # Full breed→refine cycles
    eval_games: int = 100             # Games per individual for fitness evaluation

    # --- Refinement phase ---
    refine_sessions: int = 30         # Self-play sessions with top 2 variants
    refine_games_per_session: int = 20

    # --- DEAP parameters ---
    cxpb: float = 0.5
    mutpb: float = 0.4
    tournament_size: int = 3

    # --- General ---
    oracle: bool = False
    device: str = "cpu"
    seed: int = 42
    output_dir: str = "checkpoints/breeding_results"
    progress_callback: Any = None     # async callable(BreedingProgress)
    resume: bool = False
    resume_from: str | None = None
    model_name: str | None = None
    _running: bool = True
    # --- StockŠkis opponents ---
    stockskis_eval: bool = False      # Evaluate fitness against StockŠkis bots
    stockskis_strength: float = 1.0   # StockŠkis bot strength (0..1)


# ---------------------------------------------------------------------------
# Fitness evaluation — play a behavioral variant against the base
# ---------------------------------------------------------------------------

async def _evaluate_variant(
    profile: BehavioralProfile,
    base_weights: dict,
    hidden_size: int,
    eval_games: int,
    oracle: bool,
    device: str,
    use_stockskis: bool = False,
    stockskis_strength: float = 1.0,
) -> dict[str, float]:
    """Play eval_games with a profiled agent vs 3 opponents, return metrics.

    When use_stockskis is True, opponents are StockŠkis heuristic bots
    instead of base-model clones, providing a fixed external benchmark.
    """
    from tarok.adapters.ai.stockskis_player import StockSkisPlayer

    # The profiled agent (player 0)
    agent0 = RLAgent(
        name="Variant",
        hidden_size=hidden_size,
        device=device,
        explore_rate=max(profile.explore_floor, 0.05),
        oracle_critic=oracle,
        profile=profile,
    )
    agent0.network.load_state_dict(base_weights)
    agent0.set_training(False)

    if use_stockskis:
        opponents: list = [
            StockSkisPlayer(name=f"StockŠkis-{i+1}", strength=stockskis_strength)
            for i in range(3)
        ]
    else:
        # Opponents: base model without behavioral bias
        opponents = []
        for i in range(3):
            opp = RLAgent(
                name=f"Base-{i+1}",
                hidden_size=hidden_size,
                device=device,
                oracle_critic=oracle,
            )
            opp.network.load_state_dict(base_weights)
            opp.set_training(False)
            opponents.append(opp)

    agents = [agent0] + opponents
    wins = 0
    total_reward = 0.0
    bids = 0
    solos = 0

    for g in range(eval_games):
        for a in agents:
            a.clear_experiences()
        game = GameLoop(agents)
        state, scores = await game.run(dealer=g % 4)

        score0 = scores.get(0, 0)
        total_reward += score0 / 100.0
        if score0 > 0:
            wins += 1
        agent0_bids = [b for b in state.bids if b.player == 0 and b.contract is not None]
        if agent0_bids:
            bids += 1
        if state.contract and state.contract.is_solo and state.declarer == 0:
            solos += 1

        # Yield to event loop every 5 games so FastAPI can serve requests
        if (g + 1) % 5 == 0:
            await asyncio.sleep(0)

    n = max(eval_games, 1)
    return {
        "win_rate": wins / n,
        "avg_reward": total_reward / n,
        "bid_rate": bids / n,
        "solo_rate": solos / n,
    }


def _fitness_from_metrics(m: dict[str, float]) -> float:
    """Compute scalar fitness from evaluation metrics.

    Weights: 50% win rate, 30% avg reward (normalized), 20% bid engagement.
    """
    wr = m["win_rate"]
    # Normalize reward: typical range is -1 to +1, map to 0..1
    reward_norm = max(0.0, min(1.0, (m["avg_reward"] + 1.0) / 2.0))
    # Bid engagement: reward profiles that actually bid (not always passing)
    bid_bonus = min(1.0, m["bid_rate"])
    return 0.5 * wr + 0.3 * reward_norm + 0.2 * bid_bonus


# ---------------------------------------------------------------------------
# DEAP setup for behavioral breeding
# ---------------------------------------------------------------------------

def _setup_breeding_deap(config: BreedingConfig):
    if not hasattr(creator, "BreedFitness"):
        creator.create("BreedFitness", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "BreedIndividual"):
        creator.create("BreedIndividual", list, fitness=creator.BreedFitness)

    rng = random.Random(config.seed)
    toolbox = base.Toolbox()

    def _random_genes():
        profile = BehavioralProfile.random(rng)
        return profile.to_genes()

    toolbox.register("individual", tools.initIterate, creator.BreedIndividual, _random_genes)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=GENE_SIGMAS, indpb=0.4)
    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)

    hof = tools.HallOfFame(5)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    return toolbox, hof, stats


def _clip_genes(ind: list[float]) -> list[float]:
    for i, (lo, hi) in enumerate(GENE_BOUNDS):
        ind[i] = max(lo, min(hi, ind[i]))
    return ind


# ---------------------------------------------------------------------------
# Main breeding pipeline
# ---------------------------------------------------------------------------

async def run_breeding(config: BreedingConfig | None = None) -> dict[str, Any]:
    if config is None:
        config = BreedingConfig()

    progress = BreedingProgress(
        total_cycles=config.num_cycles,
        total_generations=config.num_generations,
        warmup_total_sessions=config.warmup_sessions,
        refine_total_sessions=config.refine_sessions,
    )
    start_time = time.time()

    async def _notify():
        progress.elapsed_seconds = time.time() - start_time
        if config.progress_callback:
            await config.progress_callback(progress)
        # Yield to the event loop so FastAPI can serve HTTP requests
        await asyncio.sleep(0)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = config.device
    oracle = config.oracle
    hidden_size = 256

    from shortuuid import ShortUUID
    base_model_name = config.model_name or f"BredModel-{ShortUUID().random(length=6)}"
    progress.model_name = base_model_name
    
    print("🧬 Multi-Phase Behavioral Breeding Pipeline")
    print(f"   Model Name: {base_model_name}")
    print(f"   Warmup: {config.warmup_sessions} sessions × {config.warmup_games_per_session} games")
    print(f"   Breeding: {config.num_cycles} cycles × {config.num_generations} gens × {config.population_size} pop")
    print(f"   Eval: {config.eval_games} games per individual")
    print(f"   Refinement: {config.refine_sessions} sessions × {config.refine_games_per_session} games")
    print()

    # ===================================================================
    # PHASE 1: WARMUP — train base model via standard self-play
    # ===================================================================
    progress.phase = "warmup"
    await _notify()
    
    base_agents = [
        RLAgent(name=f"Base-{i}", hidden_size=hidden_size, device=device, oracle_critic=oracle)
        for i in range(4)
    ]
    warmup_trainer = PPOTrainer(
        base_agents,
        device=device,
        games_per_session=config.warmup_games_per_session,
        save_dir=str(output_dir / "warmup"),
    )

    if config.resume and config.resume_from:
        ckpt_path = Path("checkpoints") / config.resume_from
        if ckpt_path.exists():
            print(f"━━━ Resuming Base Model from {ckpt_path} ━━━")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            warmup_trainer.shared_network.load_state_dict(ckpt.get("model_state_dict", ckpt))
        else:
            print(f"━━━ Phase 1: Warmup (standard self-play) ━━━")
    else:
        print("━━━ Phase 1: Warmup (standard self-play) ━━━")

    async def _warmup_metrics(metrics):
        progress.warmup_session = metrics.session
        await _notify()

    warmup_trainer.add_metrics_callback(_warmup_metrics)
    warmup_result = await warmup_trainer.train(config.warmup_sessions)

    base_weights = copy.deepcopy(warmup_trainer.shared_network.state_dict())
    print(f"\n  Warmup done: win_rate={warmup_result.win_rate:.1%} avg_reward={warmup_result.avg_reward:+.2f}")
    print()

    # ===================================================================
    # BREEDING CYCLES
    # ===================================================================
    toolbox, hof, stats = _setup_breeding_deap(config)

    for cycle in range(config.num_cycles):
        if not config._running:
            break
        progress.cycle = cycle + 1
        print(f"━━━ Breeding Cycle {cycle + 1}/{config.num_cycles} ━━━")

        # ---------------------------------------------------------------
        # PHASE 2: BREED — create population of behavioral variants
        # ---------------------------------------------------------------
        progress.phase = "breeding"
        population = toolbox.population(n=config.population_size)
        for ind in population:
            _clip_genes(ind)
        await _notify()

        # ---------------------------------------------------------------
        # PHASE 3: EVALUATE — fitness via games against base model
        # ---------------------------------------------------------------
        for gen in range(config.num_generations + 1):
            progress.generation = gen
            progress.phase = "evaluating"

            if gen == 0:
                to_eval = population
            else:
                # Breed offspring
                offspring = algorithms.varOr(
                    population, toolbox,
                    lambda_=config.population_size,
                    cxpb=config.cxpb,
                    mutpb=config.mutpb,
                )
                for ind in offspring:
                    _clip_genes(ind)
                to_eval = [ind for ind in offspring if not ind.fitness.valid]

            progress.evaluating_total = len(to_eval)
            print(f"  Gen {gen}: evaluating {len(to_eval)} individuals...")

            eval_results: list[tuple[list, dict]] = []
            for i, ind in enumerate(to_eval):
                if not config._running:
                    break
                progress.evaluating_index = i + 1
                await _notify()

                profile = BehavioralProfile.from_genes(ind)
                metrics = await _evaluate_variant(
                    profile, base_weights, hidden_size,
                    config.eval_games, oracle, device,
                    use_stockskis=config.stockskis_eval,
                    stockskis_strength=config.stockskis_strength,
                )
                fitness = _fitness_from_metrics(metrics)
                ind.fitness.values = (fitness,)
                eval_results.append((ind, metrics))

                print(f"    [{i+1}/{len(to_eval)}] fit={fitness:.4f} wr={metrics['win_rate']:.1%} "
                      f"bid={metrics['bid_rate']:.0%} "
                      f"agg={profile.bid_aggression:+.2f} temp={profile.temperature:.2f}")

            if not config._running:
                break

            if gen > 0:
                population = toolbox.select(population + offspring, config.population_size)
            hof.update(population)

            record = stats.compile(population)
            progress.gen_stats.append({
                "cycle": cycle + 1, "gen": gen,
                **{k: float(v) for k, v in record.items()},
            })

            # Update progress population
            progress.population = []
            for idx, ind in enumerate(population):
                if ind.fitness.valid:
                    p = BehavioralProfile.from_genes(ind)
                    progress.population.append(BreedingIndividual(
                        index=idx, profile=p.to_dict(), fitness=ind.fitness.values[0],
                        win_rate=0, avg_reward=0, bid_rate=0, solo_rate=0,
                    ))

            # Update HoF in progress
            progress.hall_of_fame = []
            for idx, ind in enumerate(hof):
                p = BehavioralProfile.from_genes(ind)
                progress.hall_of_fame.append(BreedingIndividual(
                    index=idx, profile=p.to_dict(), fitness=ind.fitness.values[0],
                    win_rate=0, avg_reward=0, bid_rate=0, solo_rate=0,
                ))

            if hof:
                best = hof[0]
                progress.best_fitness = best.fitness.values[0]
                progress.best_profile = BehavioralProfile.from_genes(best).to_dict()

            await _notify()
            print(f"  Gen {gen} stats: avg={record['avg']:.4f} max={record['max']:.4f}")
            print()

        if not config._running:
            break

        # For the last eval, store proper metrics on HoF individuals
        for idx, ind in enumerate(hof):
            profile = BehavioralProfile.from_genes(ind)
            metrics = await _evaluate_variant(
                profile, base_weights, hidden_size,
                config.eval_games, oracle, device,
                use_stockskis=config.stockskis_eval,
                stockskis_strength=config.stockskis_strength,
            )
            if idx < len(progress.hall_of_fame):
                progress.hall_of_fame[idx].win_rate = metrics["win_rate"]
                progress.hall_of_fame[idx].avg_reward = metrics["avg_reward"]
                progress.hall_of_fame[idx].bid_rate = metrics["bid_rate"]
                progress.hall_of_fame[idx].solo_rate = metrics["solo_rate"]

        if not config._running:
            break

        # ---------------------------------------------------------------
        # PHASE 4: REFINE — self-play between top 2 variants
        # ---------------------------------------------------------------
        progress.phase = "refining"
        await _notify()
        print(f"  Refining: self-play between top 2 variants...")

        top2_profiles = [BehavioralProfile.from_genes(hof[i]) for i in range(min(2, len(hof)))]

        refine_agents = []
        for i in range(4):
            profile = top2_profiles[i % len(top2_profiles)]
            agent = RLAgent(
                name=f"Refine-{i}",
                hidden_size=hidden_size,
                device=device,
                oracle_critic=oracle,
                profile=profile,
            )
            refine_agents.append(agent)

        refine_trainer = PPOTrainer(
            refine_agents,
            device=device,
            games_per_session=config.refine_games_per_session,
            save_dir=str(output_dir / f"refine_c{cycle + 1}"),
        )
        # Load base weights into the shared network
        refine_trainer.shared_network.load_state_dict(base_weights)
        for agent in refine_agents:
            agent.network = refine_trainer.shared_network

        async def _refine_metrics(metrics, _c=cycle):
            progress.refine_session = metrics.session
            await _notify()

        refine_trainer.add_metrics_callback(_refine_metrics)
        refine_result = await refine_trainer.train(config.refine_sessions)

        # Decay exploration for refinement agents
        for agent in refine_agents:
            agent.decay_exploration()

        # The refined weights become the new base
        base_weights = copy.deepcopy(refine_trainer.shared_network.state_dict())

        cycle_summary = {
            "cycle": cycle + 1,
            "best_fitness": progress.best_fitness,
            "best_profile": progress.best_profile,
            "refine_win_rate": refine_result.win_rate,
            "refine_avg_reward": refine_result.avg_reward,
        }
        progress.cycle_summaries.append(cycle_summary)

        print(f"  Cycle {cycle + 1} refined: wr={refine_result.win_rate:.1%} reward={refine_result.avg_reward:+.2f}")
        print()

        # Save per-cycle checkpoint
        _save_breeding_checkpoint(cycle + 1, base_weights, hof, progress, config)

    # ===================================================================
    # Final results
    # ===================================================================
    progress.phase = "done"
    await _notify()

    best_profile = BehavioralProfile.from_genes(hof[0]) if hof else BehavioralProfile()

    result = {
        "model_name": base_model_name,
        "best_profile": best_profile.to_dict(),
        "best_fitness": progress.best_fitness,
        "elapsed_seconds": round(time.time() - start_time, 1),
        "cycle_summaries": progress.cycle_summaries,
        "gen_stats": progress.gen_stats,
        "top_5": [
            {"profile": BehavioralProfile.from_genes(ind).to_dict(), "fitness": ind.fitness.values[0]}
            for ind in hof if ind.fitness.valid
        ],
    }

    results_path = output_dir / f"breeding_best_{base_model_name}.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Save final model with best profile embedded
    final_ckpt = {
        "model_name": base_model_name,
        "model_state_dict": base_weights,
        "behavioral_profile": best_profile.to_dict(),
        "breeding_result": result,
    }
    torch.save(final_ckpt, output_dir / f"bred_model_{base_model_name}.pt")

    print()
    print(f"🏆 Breeding complete for model: {base_model_name}! Best fitness: {progress.best_fitness:.4f}")
    print(f"   Best profile: {best_profile.to_dict()}")
    print(f"   Results: {results_path}")
    print(f"   Model: {output_dir / f'bred_model_{base_model_name}.pt'}")

    return result


def _save_breeding_checkpoint(
    cycle: int,
    weights: dict,
    hof: tools.HallOfFame,
    progress: BreedingProgress,
    config: BreedingConfig,
) -> None:
    output_dir = Path(config.output_dir)
    checkpoint = {
        "cycle": cycle,
        "model_state_dict": weights,
        "hall_of_fame": [
            {"genes": list(ind), "fitness": ind.fitness.values[0]}
            for ind in hof
        ],
        "progress": progress.to_dict(),
    }
    torch.save(checkpoint, output_dir / f"breeding_cycle{cycle}.pt")


# ---------------------------------------------------------------------------
# Convenience: train with a bred model + profile
# ---------------------------------------------------------------------------

async def train_with_bred_model(
    checkpoint_path: str | Path,
    num_sessions: int = 100,
    games_per_session: int = 20,
    oracle: bool = False,
    device: str = "cpu",
) -> TrainingMetrics:
    """Load a bred model checkpoint and continue training with its profile."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    profile_dict = ckpt.get("behavioral_profile", {})
    profile = BehavioralProfile.from_dict(profile_dict) if profile_dict else None
    weights = ckpt["model_state_dict"]

    print(f"🧬 Training with bred model")
    if profile:
        print(f"   Profile: {profile.to_dict()}")
    print()

    agents = [
        RLAgent(
            name=f"Bred-{i}",
            device=device,
            oracle_critic=oracle,
            profile=profile,
        )
        for i in range(4)
    ]

    trainer = PPOTrainer(
        agents, device=device, games_per_session=games_per_session,
    )
    trainer.shared_network.load_state_dict(weights)
    for agent in agents:
        agent.network = trainer.shared_network

    return await trainer.train(num_sessions)
