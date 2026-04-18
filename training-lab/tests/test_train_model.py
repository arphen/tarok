"""Tests for the TrainModel use case orchestrator."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.adapters.persistence import JsonLeagueStatePersistence
from training.entities.iteration_result import IterationResult
from training.entities.league import LeagueConfig, LeagueOpponent
from training.entities.model_identity import ModelIdentity
from training.entities.training_config import TrainingConfig, scheduled_lr
from training.use_cases.train_model import TrainModel
from training.use_cases.train_model.policies import EloDecayEntropyPolicy, EloGaussianILPolicy, elo_based_lr


@pytest.fixture
def mock_iteration_runner() -> MagicMock:
    runner = MagicMock()
    runner.run_iteration.return_value = IterationResult(
        iteration=1,
        placement=2.5,
        loss=0.42,
        policy_loss=0.31,
        value_loss=0.27,
        entropy=0.9,
        n_experiences=128,
        selfplay_time=1.0,
        ppo_time=2.0,
        bench_time=0.5,
        seat_config_used="nn,bot_v5,bot_v5,bot_v5",
        seat_outcomes={1: (10, 0, 0)},
    )
    return runner


@pytest.fixture
def mock_benchmark() -> MagicMock:
    bench = MagicMock()
    bench.measure_placement.return_value = 3.1
    return bench


@pytest.fixture
def mock_model_port() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_presenter() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_league_persistence() -> MagicMock:
    return MagicMock()


@pytest.fixture
def base_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        model_arch="v4",
        save_dir=str(tmp_path / "test_run"),
        iterations=2,
        bench_games=10,
        benchmark_checkpoints=(),
        league=LeagueConfig(enabled=True),
    )


@pytest.fixture
def identity() -> ModelIdentity:
    return ModelIdentity(
        name="TestModel",
        hidden_size=64,
        oracle_critic=False,
        model_arch="v4",
        is_new=True,
    )


def test_train_model_basic_execution_flow(
    mock_iteration_runner: MagicMock,
    mock_benchmark: MagicMock,
    mock_model_port: MagicMock,
    mock_presenter: MagicMock,
    base_config: TrainingConfig,
    identity: ModelIdentity,
    mock_league_persistence: MagicMock,
) -> None:
    """Verifies setup -> benchmark -> iterations -> teardown flow."""
    use_case = TrainModel(
        iteration_runner=mock_iteration_runner,
        benchmark=mock_benchmark,
        model=mock_model_port,
        presenter=mock_presenter,
        league_persistence=mock_league_persistence,
    )

    run_result = use_case.execute(
        config=base_config,
        identity=identity,
        weights={"dummy": "weights"},
        device="cpu",
    )

    # Setup / initial benchmark
    mock_model_port.export_for_inference.assert_called_once()
    mock_iteration_runner.setup.assert_called_once()
    mock_benchmark.measure_placement.assert_called_once()

    # 2 iterations requested
    assert mock_iteration_runner.run_iteration.call_count == 2
    assert len(run_result.results) == 2

    # Always tears down and finalizes run
    mock_iteration_runner.teardown.assert_called_once()
    mock_model_port.copy_best.assert_called_once()
    mock_presenter.on_training_complete.assert_called_once_with(run_result)


def test_train_model_requires_enabled_league(
    mock_iteration_runner: MagicMock,
    mock_benchmark: MagicMock,
    mock_model_port: MagicMock,
    mock_presenter: MagicMock,
    base_config: TrainingConfig,
    identity: ModelIdentity,
    mock_league_persistence: MagicMock,
) -> None:
    use_case = TrainModel(
        iteration_runner=mock_iteration_runner,
        benchmark=mock_benchmark,
        model=mock_model_port,
        presenter=mock_presenter,
        league_persistence=mock_league_persistence,
    )

    cfg = replace(base_config, league=None)
    with pytest.raises(ValueError, match="requires league.enabled=true"):
        use_case.execute(config=cfg, identity=identity, weights={}, device="cpu")

    mock_iteration_runner.teardown.assert_not_called()


def test_train_model_uses_injected_lr_policy(
    mock_iteration_runner: MagicMock,
    mock_benchmark: MagicMock,
    mock_model_port: MagicMock,
    mock_presenter: MagicMock,
    base_config: TrainingConfig,
    identity: ModelIdentity,
    mock_league_persistence: MagicMock,
) -> None:
    lr_policy = MagicMock()
    lr_policy.compute.return_value = 1.23e-4

    use_case = TrainModel(
        iteration_runner=mock_iteration_runner,
        benchmark=mock_benchmark,
        model=mock_model_port,
        presenter=mock_presenter,
        lr_policy=lr_policy,
        league_persistence=mock_league_persistence,
    )

    use_case.execute(config=base_config, identity=identity, weights={}, device="cpu")

    assert lr_policy.compute.call_count == 2
    first_call = lr_policy.compute.call_args_list[0]
    assert first_call.kwargs["config"] == base_config
    assert first_call.kwargs["iteration"] == 1
    assert first_call.kwargs["learner_elo"] == pytest.approx(800.0)

    first_iter_call = mock_iteration_runner.run_iteration.call_args_list[0]
    assert first_iter_call.kwargs["iter_lr"] == pytest.approx(1.23e-4)


def test_train_model_uses_injected_imitation_policy(
    mock_iteration_runner: MagicMock,
    mock_benchmark: MagicMock,
    mock_model_port: MagicMock,
    mock_presenter: MagicMock,
    base_config: TrainingConfig,
    identity: ModelIdentity,
    mock_league_persistence: MagicMock,
) -> None:
    imitation_policy = MagicMock()
    imitation_policy.compute.return_value = 0.77

    use_case = TrainModel(
        iteration_runner=mock_iteration_runner,
        benchmark=mock_benchmark,
        model=mock_model_port,
        presenter=mock_presenter,
        imitation_policy=imitation_policy,
        league_persistence=mock_league_persistence,
    )

    use_case.execute(config=base_config, identity=identity, weights={}, device="cpu")

    assert imitation_policy.compute.call_count == 2
    first_call = imitation_policy.compute.call_args_list[0]
    assert first_call.kwargs["config"] == base_config
    assert first_call.kwargs["iteration"] == 1

    first_iter_call = mock_iteration_runner.run_iteration.call_args_list[0]
    assert first_iter_call.kwargs["iter_imitation_coef"] == pytest.approx(0.77)


@patch("training.use_cases.train_model.orchestrator.UpdateLeagueElo")
@patch("training.use_cases.train_model.orchestrator.SampleLeagueSeats")
@patch("training.use_cases.train_model.orchestrator.shutil.copy2")
def test_train_model_with_league_and_snapshots(
    mock_copy2: MagicMock,
    MockSampleSeats: MagicMock,
    MockUpdateElo: MagicMock,
    mock_iteration_runner: MagicMock,
    mock_benchmark: MagicMock,
    mock_model_port: MagicMock,
    mock_presenter: MagicMock,
    base_config: TrainingConfig,
    identity: ModelIdentity,
    mock_league_persistence: MagicMock,
) -> None:
    """Verifies league sampling, Elo updates, and snapshot cadence behavior."""
    cfg = replace(base_config, league=LeagueConfig(enabled=True, snapshot_interval=2))

    mock_sampler = MockSampleSeats.return_value
    mock_sampler.execute.return_value = "nn,bot_m6,bot_v5,nn"

    # Two distinct results for two iterations.
    mock_iteration_runner.run_iteration.side_effect = [
        IterationResult(
            iteration=1,
            placement=2.8,
            loss=0.8,
            policy_loss=0.5,
            value_loss=0.3,
            entropy=1.0,
            n_experiences=256,
            selfplay_time=1.0,
            ppo_time=1.0,
            bench_time=0.0,
            seat_config_used="nn,bot_m6,bot_v5,nn",
            seat_outcomes={1: (1, 0, 0), 2: (0, 1, 0)},
        ),
        IterationResult(
            iteration=2,
            placement=2.6,
            loss=0.6,
            policy_loss=0.4,
            value_loss=0.2,
            entropy=0.9,
            n_experiences=256,
            selfplay_time=1.0,
            ppo_time=1.0,
            bench_time=0.0,
            seat_config_used="nn,bot_m6,bot_v5,nn",
            seat_outcomes={1: (1, 0, 0), 2: (0, 1, 0)},
        ),
    ]

    use_case = TrainModel(
        iteration_runner=mock_iteration_runner,
        benchmark=mock_benchmark,
        model=mock_model_port,
        presenter=mock_presenter,
        league_persistence=mock_league_persistence,
    )

    use_case.execute(config=cfg, identity=identity, weights={}, device="cpu")

    # League sampling and ELO update happen every iteration.
    assert mock_sampler.execute.call_count == 2
    mock_updater = MockUpdateElo.return_value
    assert mock_updater.execute.call_count == 2

    # Snapshot interval=2 => exactly one snapshot on iteration 2.
    save_dir = Path(cfg.save_dir)
    expected_snap = save_dir / "league_pool" / "iter_002.pt"

    mock_copy2.assert_called_once_with(str(save_dir / "_current.pt"), str(expected_snap))
    mock_presenter.on_league_snapshot_added.assert_called_once_with(2, str(expected_snap))

    # Teardown still called in successful path.
    mock_iteration_runner.teardown.assert_called_once()


def test_train_model_ensures_teardown_on_error(
    mock_iteration_runner: MagicMock,
    mock_benchmark: MagicMock,
    mock_model_port: MagicMock,
    mock_presenter: MagicMock,
    base_config: TrainingConfig,
    identity: ModelIdentity,
    mock_league_persistence: MagicMock,
) -> None:
    """Verifies teardown is always called if an iteration raises."""
    use_case = TrainModel(
        iteration_runner=mock_iteration_runner,
        benchmark=mock_benchmark,
        model=mock_model_port,
        presenter=mock_presenter,
        league_persistence=mock_league_persistence,
    )

    mock_iteration_runner.run_iteration.side_effect = RuntimeError("CUDA Out of Memory")

    with pytest.raises(RuntimeError, match="CUDA Out of Memory"):
        use_case.execute(config=base_config, identity=identity, weights={}, device="cpu")

    mock_iteration_runner.teardown.assert_called_once()

    # No successful completion path after crash.
    mock_model_port.copy_best.assert_not_called()
    mock_presenter.on_training_complete.assert_not_called()


@patch("training.use_cases.train_model.orchestrator.UpdateLeagueElo")
@patch("training.use_cases.train_model.orchestrator.SampleLeagueSeats")
@patch("training.use_cases.train_model.orchestrator.shutil.copy2")
def test_train_model_caps_active_nn_snapshots(
    mock_copy2: MagicMock,
    MockSampleSeats: MagicMock,
    MockUpdateElo: MagicMock,
    mock_iteration_runner: MagicMock,
    mock_benchmark: MagicMock,
    mock_model_port: MagicMock,
    mock_presenter: MagicMock,
    base_config: TrainingConfig,
    identity: ModelIdentity,
    mock_league_persistence: MagicMock,
) -> None:
    cfg = replace(
        base_config,
        iterations=5,
        league=LeagueConfig(
            enabled=True,
            snapshot_interval=1,
            max_active_snapshots=2,
            opponents=(LeagueOpponent(name="Anchor", type="bot_v1"),),
        ),
    )

    sampled_checkpoint_counts: list[int] = []
    sampled_total_counts: list[int] = []

    mock_sampler = MockSampleSeats.return_value
    mock_updater = MockUpdateElo.return_value

    def _bump_elo(pool, _seat_config_used, _seat_outcomes) -> None:
        pool.learner_elo += 60.0

    mock_updater.execute.side_effect = _bump_elo

    def _capture_pool(pool) -> str:
        sampled_total_counts.append(len(pool.entries))
        sampled_checkpoint_counts.append(
            sum(1 for entry in pool.entries if entry.opponent.type == "nn_checkpoint")
        )
        return "nn,bot_v1,nn,nn"

    mock_sampler.execute.side_effect = _capture_pool

    use_case = TrainModel(
        iteration_runner=mock_iteration_runner,
        benchmark=mock_benchmark,
        model=mock_model_port,
        presenter=mock_presenter,
        league_persistence=mock_league_persistence,
    )

    use_case.execute(config=cfg, identity=identity, weights={}, device="cpu")

    # Pool seen before each iteration should cap checkpoints at configured max,
    # the heuristic anchor remains present.
    assert sampled_checkpoint_counts == [0, 1, 2, 2, 2]
    assert sampled_total_counts == [1, 2, 3, 3, 3]

    assert mock_updater.execute.call_count == 5
    assert mock_copy2.call_count == 5
    assert mock_presenter.on_league_snapshot_added.call_count == 5


@patch("training.use_cases.train_model.orchestrator.UpdateLeagueElo")
@patch("training.use_cases.train_model.orchestrator.SampleLeagueSeats")
@patch("training.use_cases.train_model.orchestrator.shutil.copy2")
def test_train_model_snapshot_admission_requires_elo_milestone(
    mock_copy2: MagicMock,
    MockSampleSeats: MagicMock,
    MockUpdateElo: MagicMock,
    mock_iteration_runner: MagicMock,
    mock_benchmark: MagicMock,
    mock_model_port: MagicMock,
    mock_presenter: MagicMock,
    base_config: TrainingConfig,
    identity: ModelIdentity,
    mock_league_persistence: MagicMock,
) -> None:
    cfg = replace(
        base_config,
        iterations=4,
        league=LeagueConfig(
            enabled=True,
            snapshot_interval=1,
            opponents=(LeagueOpponent(name="Anchor", type="bot_v1"),),
        ),
    )

    mock_sampler = MockSampleSeats.return_value
    mock_sampler.execute.return_value = "nn,bot_v1,nn,nn"

    # Simulate plateau: learner Elo does not improve after first admission.
    mock_updater = MockUpdateElo.return_value
    mock_updater.execute.side_effect = lambda *_args, **_kwargs: None

    use_case = TrainModel(
        iteration_runner=mock_iteration_runner,
        benchmark=mock_benchmark,
        model=mock_model_port,
        presenter=mock_presenter,
        league_persistence=mock_league_persistence,
    )

    use_case.execute(config=cfg, identity=identity, weights={}, device="cpu")

    # First snapshot is admitted (no baseline yet), then blocked by +50 Elo gate.
    assert mock_copy2.call_count == 1
    assert mock_presenter.on_league_snapshot_added.call_count == 1


@patch("training.use_cases.train_model.orchestrator.UpdateLeagueElo")
@patch("training.use_cases.train_model.orchestrator.SampleLeagueSeats")
@patch("training.use_cases.train_model.orchestrator.shutil.copy2")
def test_train_model_snapshot_admission_uses_configured_elo_delta(
    mock_copy2: MagicMock,
    MockSampleSeats: MagicMock,
    MockUpdateElo: MagicMock,
    mock_iteration_runner: MagicMock,
    mock_benchmark: MagicMock,
    mock_model_port: MagicMock,
    mock_presenter: MagicMock,
    base_config: TrainingConfig,
    identity: ModelIdentity,
    mock_league_persistence: MagicMock,
) -> None:
    cfg = replace(
        base_config,
        iterations=3,
        league=LeagueConfig(
            enabled=True,
            snapshot_interval=1,
            snapshot_elo_delta=10.0,
            opponents=(LeagueOpponent(name="Anchor", type="bot_v1"),),
        ),
    )

    mock_sampler = MockSampleSeats.return_value
    mock_sampler.execute.return_value = "nn,bot_v1,nn,nn"

    mock_updater = MockUpdateElo.return_value

    def _bump_elo(pool, _seat_config_used, _seat_outcomes) -> None:
        pool.learner_elo += 20.0

    mock_updater.execute.side_effect = _bump_elo

    use_case = TrainModel(
        iteration_runner=mock_iteration_runner,
        benchmark=mock_benchmark,
        model=mock_model_port,
        presenter=mock_presenter,
        league_persistence=mock_league_persistence,
    )

    use_case.execute(config=cfg, identity=identity, weights={}, device="cpu")

    # +20 Elo gains clear configured +10 gate each iteration.
    assert mock_copy2.call_count == 3
    assert mock_presenter.on_league_snapshot_added.call_count == 3


@patch("training.use_cases.train_model.orchestrator.UpdateLeagueElo")
@patch("training.use_cases.train_model.orchestrator.SampleLeagueSeats")
def test_train_model_restores_persisted_league_state(
    MockSampleSeats: MagicMock,
    MockUpdateElo: MagicMock,
    mock_iteration_runner: MagicMock,
    mock_benchmark: MagicMock,
    mock_model_port: MagicMock,
    mock_presenter: MagicMock,
    base_config: TrainingConfig,
    identity: ModelIdentity,
) -> None:
    cfg = replace(
        base_config,
        iterations=1,
        league=LeagueConfig(
            enabled=True,
            opponents=(LeagueOpponent(name="Anchor", type="bot_v1", initial_elo=900.0),),
        ),
    )

    save_dir = Path(cfg.save_dir)
    league_pool_dir = save_dir / "league_pool"
    snapshot_path = league_pool_dir / "iter_005.pt"
    league_pool_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(b"checkpoint")
    (league_pool_dir / "state.json").write_text(
        json.dumps(
            {
                "learner_elo": 1444.0,
                "entries": [
                    {
                        "opponent": {
                            "name": "Anchor",
                            "type": "bot_v1",
                            "path": None,
                            "initial_elo": 900.0,
                        },
                        "elo": 900.0,
                        "games_played": 10,
                        "learner_outplaces": 6,
                    },
                    {
                        "opponent": {
                            "name": "snapshot_iter_005",
                            "type": "nn_checkpoint",
                            "path": str(snapshot_path),
                            "initial_elo": 1410.0,
                        },
                        "elo": 1410.0,
                        "games_played": 4,
                        "learner_outplaces": 2,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    mock_sampler = MockSampleSeats.return_value

    def _capture_pool(pool) -> str:
        assert pool.learner_elo == pytest.approx(1444.0)
        assert [entry.opponent.name for entry in pool.entries] == ["Anchor", "snapshot_iter_005"]
        assert pool.entries[1].elo == pytest.approx(1410.0)
        return "nn,bot_v1,nn,nn"

    mock_sampler.execute.side_effect = _capture_pool

    use_case = TrainModel(
        iteration_runner=mock_iteration_runner,
        benchmark=mock_benchmark,
        model=mock_model_port,
        presenter=mock_presenter,
        league_persistence=JsonLeagueStatePersistence(),
    )

    use_case.execute(config=cfg, identity=identity, weights={}, device="cpu")

    first_lr_call = mock_iteration_runner.run_iteration.call_args_list[0]
    assert first_lr_call.kwargs["iter_lr"] < cfg.lr

    persisted = json.loads((league_pool_dir / "state.json").read_text(encoding="utf-8"))
    assert persisted["learner_elo"] == pytest.approx(1444.0)
    assert [entry["opponent"]["name"] for entry in persisted["entries"]] == [
        "Anchor",
        "snapshot_iter_005",
    ]


@patch("training.use_cases.train_model.orchestrator.UpdateLeagueElo")
@patch("training.use_cases.train_model.orchestrator.SampleLeagueSeats")
def test_train_model_elo_based_lr_decays_smoothly(
    MockSampleSeats: MagicMock,
    MockUpdateElo: MagicMock,
    mock_iteration_runner: MagicMock,
    mock_benchmark: MagicMock,
    mock_model_port: MagicMock,
    mock_presenter: MagicMock,
    base_config: TrainingConfig,
    identity: ModelIdentity,
    mock_league_persistence: MagicMock,
) -> None:
    cfg = replace(
        base_config,
        iterations=3,
        league=LeagueConfig(
            enabled=True,
            snapshot_interval=999,
            opponents=(LeagueOpponent(name="Anchor", type="bot_v1"),),
        ),
    )

    mock_sampler = MockSampleSeats.return_value
    mock_sampler.execute.return_value = "nn,bot_v1,nn,nn"

    mock_updater = MockUpdateElo.return_value

    def _bump_elo(pool, _seat_config_used, _seat_outcomes) -> None:
        pool.learner_elo += 600.0

    mock_updater.execute.side_effect = _bump_elo

    use_case = TrainModel(
        iteration_runner=mock_iteration_runner,
        benchmark=mock_benchmark,
        model=mock_model_port,
        presenter=mock_presenter,
        league_persistence=mock_league_persistence,
    )

    use_case.execute(config=cfg, identity=identity, weights={}, device="cpu")

    lrs = [call.kwargs["iter_lr"] for call in mock_iteration_runner.run_iteration.call_args_list]
    assert len(lrs) == 3
    assert lrs[0] == pytest.approx(cfg.lr, rel=1e-9)
    assert lrs[0] > lrs[1] > lrs[2]
    assert lrs[2] == pytest.approx(cfg.effective_lr_min, rel=1e-9)


def test_elo_based_lr_hits_expected_floor_and_ceiling_values() -> None:
    assert elo_based_lr(current_elo=800.0, base_lr=0.0003, min_lr=0.00001) == pytest.approx(0.0003)
    assert elo_based_lr(current_elo=2000.0, base_lr=0.0003, min_lr=0.00001) == pytest.approx(0.00001)


def test_scheduled_lr_elo_is_passthrough() -> None:
    # Guard against accidental double-decay: elo schedule must not apply iteration decay.
    lr0 = scheduled_lr(iteration=0, total_iterations=100, lr_max=0.0003, lr_min=0.00001, schedule="elo")
    lr50 = scheduled_lr(iteration=50, total_iterations=100, lr_max=0.0003, lr_min=0.00001, schedule="elo")
    lr99 = scheduled_lr(iteration=99, total_iterations=100, lr_max=0.0003, lr_min=0.00001, schedule="elo")

    assert lr0 == pytest.approx(0.0003, rel=1e-12)
    assert lr50 == pytest.approx(0.0003, rel=1e-12)
    assert lr99 == pytest.approx(0.0003, rel=1e-12)


# ---------------------------------------------------------------------------
# EloGaussianILPolicy
# ---------------------------------------------------------------------------


def test_elo_gaussian_il_peaks_at_centre() -> None:
    cfg = TrainingConfig(imitation_coef=0.05, imitation_center_elo=1500.0, imitation_width_elo=250.0)
    policy = EloGaussianILPolicy(floor=0.0)
    coef = policy.compute(config=cfg, iteration=1, learner_elo=1500.0)
    assert coef == pytest.approx(0.05, rel=1e-6)


def test_elo_gaussian_il_near_zero_at_floor_elo() -> None:
    # At 800 Elo (700 below centre, width=250) the Gaussian value is negligible.
    cfg = TrainingConfig(imitation_coef=0.05, imitation_center_elo=1500.0, imitation_width_elo=250.0)
    policy = EloGaussianILPolicy(floor=0.001)
    coef = policy.compute(config=cfg, iteration=1, learner_elo=800.0)
    assert coef == 0.0  # below floor → hard zero


def test_elo_gaussian_il_is_symmetric() -> None:
    cfg = TrainingConfig(imitation_coef=0.05, imitation_center_elo=1500.0, imitation_width_elo=250.0)
    policy = EloGaussianILPolicy(floor=0.0)
    below = policy.compute(config=cfg, iteration=1, learner_elo=1250.0)
    policy2 = EloGaussianILPolicy(floor=0.0)
    above = policy2.compute(config=cfg, iteration=1, learner_elo=1750.0)
    assert below == pytest.approx(above, rel=1e-6)


def test_elo_gaussian_il_ema_smooths_spike() -> None:
    # A sudden Elo spike should be dampened by EMA (alpha=0.05).
    cfg = TrainingConfig(imitation_coef=0.05, imitation_center_elo=1500.0, imitation_width_elo=250.0)
    policy = EloGaussianILPolicy(alpha=0.05, floor=0.0)
    # Prime EMA at 1500 (peak)
    first = policy.compute(config=cfg, iteration=1, learner_elo=1500.0)
    # Next call with Elo jumping to 2500 — smoothed_elo should still be near 1500
    second = policy.compute(config=cfg, iteration=2, learner_elo=2500.0)
    assert second > 0.0           # not floored — EMA held near centre
    assert second < first         # moved away from peak but not collapsed


# ---------------------------------------------------------------------------
# EloDecayEntropyPolicy
# ---------------------------------------------------------------------------


def test_elo_decay_entropy_max_at_floor_elo() -> None:
    policy = EloDecayEntropyPolicy()
    cfg = TrainingConfig(entropy_coef=0.05, entropy_coef_min=0.0005)
    coef = policy.compute(config=cfg, iteration=1, learner_elo=800.0)
    assert coef == pytest.approx(0.05, rel=1e-6)


def test_elo_decay_entropy_min_at_ceiling_elo() -> None:
    policy = EloDecayEntropyPolicy()
    cfg = TrainingConfig(entropy_coef=0.05, entropy_coef_min=0.0005)
    coef = policy.compute(config=cfg, iteration=1, learner_elo=2000.0)
    assert coef == pytest.approx(0.0005, rel=1e-6)


def test_elo_decay_entropy_monotone_decreasing() -> None:
    cfg = TrainingConfig(entropy_coef=0.05, entropy_coef_min=0.0005)
    elos = [800.0, 1000.0, 1300.0, 1600.0, 2000.0]
    coefs = []
    for elo in elos:
        p = EloDecayEntropyPolicy()  # fresh instance per point (no EMA carry-over)
        coefs.append(p.compute(config=cfg, iteration=1, learner_elo=elo))
    assert coefs == sorted(coefs, reverse=True)  # strictly decreasing


def test_elo_decay_entropy_ema_smooths_spike() -> None:
    policy = EloDecayEntropyPolicy(alpha=0.05)
    cfg = TrainingConfig(entropy_coef=0.05, entropy_coef_min=0.0005)
    # Prime at 800 (max entropy)
    first = policy.compute(config=cfg, iteration=1, learner_elo=800.0)
    # Sudden jump to 2000 — EMA should hold smoothed_elo near 800
    second = policy.compute(config=cfg, iteration=2, learner_elo=2000.0)
    assert second > cfg.entropy_coef_min   # EMA kept it high
    assert second < first                  # but nudged down
