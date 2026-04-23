"""Use case: collect self-play experience data for one training iteration."""

from __future__ import annotations

import time

from training.entities.experience_bundle import ExperienceBundle
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import LEARNER_SEAT_LABELS, TrainingConfig
from training.ports.ppo_port import PPOPort
from training.ports.presenter_port import PresenterPort
from training.ports.selfplay_port import SelfPlayPort


class CollectExperiences:
    """Run self-play, optionally blend in human data, and compute run stats.

    Single responsibility: gather the experience payload that the PPO update
    will consume.  All timing, seat-label parsing, and oracle-state decisions
    live here so that the orchestrator stays declarative.
    """

    def __init__(
        self,
        selfplay: SelfPlayPort,
        ppo: PPOPort,
        presenter: PresenterPort,
    ) -> None:
        self._selfplay = selfplay
        self._ppo = ppo
        self._presenter = presenter

    def execute(
        self,
        config: TrainingConfig,
        identity: ModelIdentity,
        ts_path: str,
        seats_override: str | None = None,
        iter_imitation_coef: float | None = None,
        iter_behavioral_clone_coef: float | None = None,
        iter_explore_rate: float | None = None,
    ) -> ExperienceBundle:
        effective_seats = seats_override if seats_override is not None else config.seats
        effective_explore_rate = (
            iter_explore_rate if iter_explore_rate is not None else config.explore_rate
        )
        self._presenter.on_selfplay_start(
            config,
            effective_seats=effective_seats,
            iter_explore_rate=effective_explore_rate,
        )

        t0 = time.time()
        include_oracle_states = bool(
            identity.oracle_critic
            and (iter_imitation_coef if iter_imitation_coef is not None else config.imitation_coef) > 0.0
        )
        raw = self._selfplay.run(
            ts_path,
            config.games,
            effective_seats,
            effective_explore_rate,
            config.concurrency,
            include_oracle_states=include_oracle_states,
            lapajne_mc_worlds=config.lapajne_mc_worlds,
            lapajne_mc_sims=config.lapajne_mc_sims,
            centaur_handoff_trick=config.centaur_handoff_trick,
            centaur_pimc_worlds=config.centaur_pimc_worlds,
            centaur_endgame_solver=config.centaur_endgame_solver,
            centaur_alpha_mu_depth=config.centaur_alpha_mu_depth,
            centaur_deterministic_seed=config.centaur_deterministic_seed,
        )
        seat_labels = [s.strip() for s in effective_seats.split(",")]
        nn_seats = [i for i, s in enumerate(seat_labels) if s in LEARNER_SEAT_LABELS]
        n_total = len(raw["players"])
        sp_time = time.time() - t0

        n_learner, mean_scores, seat_outcomes = self._selfplay.compute_run_stats(
            raw,
            seat_labels,
            session_size=config.outplace_session_size,
        )
        self._presenter.on_selfplay_done(n_total, n_learner, sp_time)

        if config.human_data_dir:
            human_raw = self._ppo.load_human_data(config.human_data_dir)
            if human_raw is not None:
                raw = self._ppo.merge_experiences(raw, human_raw)

        effective_bc_coef = (
            iter_behavioral_clone_coef
            if iter_behavioral_clone_coef is not None
            else config.behavioral_clone_coef
        )

        if (
            effective_bc_coef > 0.0
            and config.behavioral_clone_games_per_iteration > 0
            and config.behavioral_clone_teacher is not None
        ):
            expert_raw = self._ppo.load_expert_data(
                teacher=config.behavioral_clone_teacher,
                num_games=config.behavioral_clone_games_per_iteration,
            )
            if expert_raw is not None:
                raw = self._ppo.merge_experiences(raw, expert_raw)

        return ExperienceBundle(
            raw=raw,
            nn_seats=nn_seats,
            seat_labels=seat_labels,
            effective_seats=effective_seats,
            n_total=n_total,
            n_learner=n_learner,
            mean_scores=mean_scores,
            seat_outcomes=seat_outcomes,
            sp_time=sp_time,
        )
