"""Use case: collect duplicate-pod experiences for one training iteration.

Parallel to :class:`training.use_cases.collect_experiences.CollectExperiences`
but driven by the duplicate pairing / seeded self-play / shadow-diff reward
stack (see ``docs/double_rl.md``).

Framework-independent: imports only entities and ports per the clean
architecture contract. The orchestrator is responsible for choosing between
this use case and the legacy ``CollectExperiences`` based on
``config.duplicate.enabled``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from training.entities.duplicate_config import DuplicateConfig
from training.entities.experience_bundle import ExperienceBundle
from training.ports.duplicate_iteration_stats_port import DuplicateIterationStatsPort
from training.ports.duplicate_pairing_port import DuplicatePairingPort
from training.ports.duplicate_reward_port import DuplicateRewardPort
from training.ports.presenter_port import PresenterPort
from training.ports.selfplay_port import SelfPlayPort
from training.use_cases.learner_contract_stats import compute_learner_contract_stats

if TYPE_CHECKING:
    from training.entities.league import LeagueConfig, LeaguePool


class CollectDuplicateExperiences:
    """Build pods, run seeded self-play twice (active + shadow), compute rewards.

    Returns an ``ExperienceBundle`` whose ``raw`` dict carries the
    ``precomputed_rewards`` key expected by
    :mod:`training.adapters.ppo.ppo_batch_preparation` (see §4.1 of the
    design doc). The legacy PPO reward path is thereby bypassed exactly for
    this iteration's rollout.
    """

    def __init__(
        self,
        selfplay: SelfPlayPort,
        pairing: DuplicatePairingPort,
        reward: DuplicateRewardPort,
        presenter: PresenterPort,
        iteration_stats: DuplicateIterationStatsPort | None = None,
    ) -> None:
        self._selfplay = selfplay
        self._pairing = pairing
        self._reward = reward
        self._presenter = presenter
        self._iteration_stats = iteration_stats

    def execute(
        self,
        duplicate_config: DuplicateConfig,
        concurrency: int,
        explore_rate: float,
        learner_path: str,
        shadow_path: str,
        pool: "LeaguePool | LeagueConfig | None",
        outplace_session_size: int,
        include_oracle_states: bool = False,
        iter_explore_rate: float | None = None,
        seats_label_for_stats: str | None = None,
        centaur_handoff_trick: int | None = None,
        centaur_pimc_worlds: int | None = None,
        centaur_endgame_solver: str | None = None,
        centaur_alpha_mu_depth: int | None = None,
        centaur_deterministic_seed: int | None = None,
        shadow_seat_token: str | None = None,
        variant: int = 0,
    ) -> ExperienceBundle:
        if not duplicate_config.enabled:
            raise ValueError(
                "CollectDuplicateExperiences invoked with duplicate.enabled=False; "
                "the orchestrator must route to CollectExperiences instead."
            )
        if duplicate_config.pods_per_iteration <= 0:
            raise ValueError(
                "duplicate.pods_per_iteration must be > 0 when duplicate is enabled"
            )

        effective_explore_rate = (
            iter_explore_rate if iter_explore_rate is not None else explore_rate
        )

        # 1. Build pods.
        learner_token = duplicate_config.learner_seat_token
        # When the shadow is a heuristic bot (e.g. ``bot_v3``), its seat
        # token differs from the learner token so the Rust engine
        # instantiates the heuristic player at the shadow seat instead of
        # loading ``shadow_path`` as an NN. Otherwise the shadow is NN too
        # and shares the learner seat token (engine loads the frozen
        # snapshot via ``model_path=shadow_path``).
        resolved_shadow_token = (
            shadow_seat_token if shadow_seat_token is not None else learner_token
        )
        pods = self._pairing.build_pods(
            pool=pool,
            learner_seat_token=learner_token,
            shadow_seat_token=resolved_shadow_token,
            n_pods=duplicate_config.pods_per_iteration,
            rng_seed=int(duplicate_config.rng_seed),
        )

        # Announce the self-play phase. In duplicate mode the learner
        # rotates across all four seats, so we render the opponent-token
        # set the league sampled into this iteration's pods (dedup'd,
        # sorted) — the single best "who's at the table this iter?"
        # diagnostic.
        unique_opponents = tuple(sorted({tok for pod in pods for tok in pod.opponents}))
        n_games_per_pod = pods[0].n_games_per_group if pods else 0
        self._presenter.on_duplicate_selfplay_start(
            n_pods=len(pods),
            n_games_per_pod=n_games_per_pod,
            unique_opponents=unique_opponents,
            explore_rate=effective_explore_rate,
        )

        # 2. Seeded self-play.
        t0 = time.time()
        result = self._selfplay.run_seeded_pods(
            learner_path=learner_path,
            shadow_path=shadow_path,
            pods=pods,
            explore_rate=effective_explore_rate,
            concurrency=concurrency,
            include_oracle_states=include_oracle_states,
            centaur_handoff_trick=centaur_handoff_trick,
            centaur_pimc_worlds=centaur_pimc_worlds,
            centaur_endgame_solver=centaur_endgame_solver,
            centaur_alpha_mu_depth=centaur_alpha_mu_depth,
            centaur_deterministic_seed=centaur_deterministic_seed,
            variant=variant,
        )
        sp_time = time.time() - t0

        # 3. Compute shaped rewards and attach to the active experience dict
        #    under ``precomputed_rewards`` so ppo_batch_preparation §4.1 picks
        #    it up instead of the default scores/100 + bonuses path.
        raw = result.active
        # Adapter-visible payload needs a few extra keys for the reward
        # computation (see ShadowScoreRewardAdapter.compute_rewards).
        n_games_per_group = pods[0].n_games_per_group if pods else 0
        game_idx_within_pod = raw["game_ids"].astype("int64") % max(n_games_per_group, 1)
        raw_for_reward = dict(raw)
        raw_for_reward["pod_ids"] = result.pod_ids
        raw_for_reward["learner_positions_flat"] = _flatten_learner_positions(
            result.learner_positions, raw["game_ids"], n_games_per_group
        )
        raw_for_reward["game_idx_within_pod"] = game_idx_within_pod

        precomputed = self._reward.compute_rewards(
            active_raw=raw_for_reward,
            shadow_scores=result.shadow_scores,
            pods=pods,
            shadow_contracts=result.shadow_contracts,
        )
        raw["precomputed_rewards"] = precomputed
        # Phase 3: actor-only mode drops the critic and switches PPO batch
        # prep to the terminal-advantage broadcast path (§4.2). Propagate
        # the flag on the raw dict so adapters/ppo_batch_preparation picks
        # it up without needing a config reference.
        if duplicate_config.actor_only:
            raw["actor_only"] = True

        # 4. Build ExperienceBundle in the shape the PPO update expects.
        #    Duplicate training treats every active game as a learner game; all
        #    rows in ``raw`` come from learner seats by construction.
        n_total = int(raw["players"].shape[0])
        n_learner = n_total

        # seat_labels is only used downstream for presenter/outplace stats. In
        # duplicate mode the learner rotates across all four seats, so every
        # seat qualifies as a learner seat from the PPO guard's perspective —
        # regardless of what the stats label looks like.
        n_seats = pods[0].n_seats if pods else 4
        effective_seats = seats_label_for_stats or ",".join([learner_token] * n_seats)
        seat_labels = [s.strip() for s in effective_seats.split(",")]
        nn_seats = list(range(n_seats))

        # Per-seat mean scores from the active runs. Outplace outcomes are
        # not meaningful in duplicate mode (the learner rotates across all
        # four seats within each pod, so "learner vs opponent at seat i" is
        # ill-defined), so we surface an empty dict. Downstream code treats
        # empty seat_outcomes as "no data" and skips the per-opponent
        # summary, which is the correct behavior for duplicate iterations.
        scores = raw["scores"]
        total_games = scores.shape[0]
        if total_games > 0:
            col_means = scores.mean(axis=0)
            mean_seat_scores = [float(col_means[s]) for s in range(n_seats)]
            # Pad to 4-tuple for the legacy ExperienceBundle signature; the
            # extra slots are zero for ThreePlayer.
            while len(mean_seat_scores) < 4:
                mean_seat_scores.append(0.0)
            mean_scores = (
                mean_seat_scores[0],
                mean_seat_scores[1],
                mean_seat_scores[2],
                mean_seat_scores[3],
            )
        else:
            mean_scores = (0.0, 0.0, 0.0, 0.0)
        seat_outcomes: dict = {}
        _ = outplace_session_size  # reserved for future duplicate-aware stats

        # Compute per-iteration duplicate stats (per-opponent outplaces +
        # duplicate advantage) when the stats port is injected. This is
        # what powers presenter visibility and the league Elo update for
        # opponents the learner actually faced in this iteration's pods.
        duplicate_stats = None
        if self._iteration_stats is not None:
            duplicate_stats = self._iteration_stats.compute(
                active_raw=raw,
                shadow_scores=result.shadow_scores,
                pods=pods,
                pod_ids=result.pod_ids,
                learner_positions=result.learner_positions,
                active_game_ids=result.active_game_ids,
            )

        self._presenter.on_selfplay_done(n_total, n_learner, sp_time)
        # In duplicate mode the learner rotates across all four seats within
        # each pod, so a single constant ``learner_seats`` set would count
        # opponent bots' bids as if they were the learner's. Pass a
        # per-game learner-position array instead so bid stats reflect
        # only the real learner for each game.
        learner_seat_per_game = _derive_learner_seat_per_game(
            raw.get("game_ids"),
            result.learner_positions,
            n_games_per_group,
            n_games=int(raw["scores"].shape[0]),
        )
        learner_contract_stats = compute_learner_contract_stats(
            raw,
            nn_seats,
            learner_seat_per_game=learner_seat_per_game,
        )
        if learner_contract_stats:
            self._presenter.on_learner_contract_stats(learner_contract_stats)
        if duplicate_stats is not None:
            self._presenter.on_duplicate_iteration_stats(duplicate_stats)

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
            learner_contract_stats=learner_contract_stats,
            duplicate_stats=duplicate_stats,
        )


def _flatten_learner_positions(
    learner_positions,  # shape (n_pods, n_games_per_group)
    game_ids,
    n_games_per_group: int,
):
    """Expand ``learner_positions[pod, variant]`` to per-step alignment.

    The reward adapter reads one seat index per active experience row; we
    derive it from the row's ``game_id`` using the globally-unique id encoding
    (pod * n_games_per_group + variant).
    """
    # All imports that are numerical live in adapters/entities; here we fall
    # back to use the port's own abstractions. But because ExperienceBundle's
    # raw dict is a plain dict carrying numpy arrays, we treat those arrays as
    # duck-typed sequences. This keeps use_cases free of numpy while still
    # producing the shape the reward adapter expects.
    if n_games_per_group <= 0:
        # No pods; return an empty sequence matching game_ids.
        return game_ids * 0
    pod_idx = game_ids // n_games_per_group
    variant_idx = game_ids % n_games_per_group
    # ``learner_positions`` is an ndarray-like; indexing with two int arrays
    # produces a 1D array aligned with game_ids.
    return learner_positions[pod_idx, variant_idx]


def _derive_learner_seat_per_game(
    game_ids,  # per-experience global game ids (unused; kept for API symmetry)
    learner_positions,  # shape (n_pods, n_games_per_group)
    n_games_per_group: int,
    n_games: int,
):
    """Build a per-game learner seat index aligned with ``scores`` / ``contracts``.

    The merged ``raw`` dict indexes per-game metadata by ``gid = pod *
    n_games_per_group + variant``. The returned object is a flat sequence
    of length ``n_games`` where element ``gid`` is the learner's seat for
    that game — exactly what ``compute_learner_contract_stats`` needs to
    attribute bids to the real learner (the seat rotates per variant in
    duplicate mode).

    Follows the same ``_flatten_learner_positions`` convention of treating
    the numpy-backed ``learner_positions`` as a duck-typed sequence so this
    use case stays free of a direct numpy import.
    """
    _ = game_ids  # not needed: we enumerate all games in [0, n_games)
    if n_games_per_group <= 0 or n_games <= 0:
        return []
    # Build index arrays the same way _flatten_learner_positions does.
    # ``range(n_games)`` is a plain Python sequence; numpy fancy indexing
    # accepts any integer sequence, so we avoid importing numpy here.
    pods = [gid // n_games_per_group for gid in range(n_games)]
    variants = [gid % n_games_per_group for gid in range(n_games)]
    return learner_positions[pods, variants]
