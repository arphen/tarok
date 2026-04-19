"""Focused unit tests for iteration-level use cases.

Covers CollectExperiences, UpdatePolicy, and SaveCheckpoint.
All ports are MagicMocks — no Rust/torch imports required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from training.entities.experience_bundle import ExperienceBundle
from training.entities.model_identity import ModelIdentity
from training.entities.policy_update_result import PolicyUpdateResult
from training.entities.training_config import TrainingConfig
from training.use_cases.collect_experiences import CollectExperiences
from training.use_cases.save_checkpoint import SaveCheckpoint
from training.use_cases.update_policy import UpdatePolicy


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _raw_data(n: int = 8) -> dict:
    return {"players": list(range(n)), "states": [], "actions": []}


def _make_selfplay(raw: dict | None = None, *, n_learner: int = 2) -> MagicMock:
    sp = MagicMock()
    sp.run.return_value = raw or _raw_data()
    sp.compute_run_stats.return_value = (
        n_learner,
        (1.0, -0.5, -0.5, 0.0),
        {1: (3, 2, 1), 2: (2, 3, 1), 3: (2, 2, 2)},
    )
    return sp


def _make_ppo(
    new_weights: dict | None = None,
    human_raw: dict | None = None,
    expert_raw: dict | None = None,
) -> MagicMock:
    ppo = MagicMock()
    metrics = {
        "total_loss": 0.42,
        "policy_loss": 0.20,
        "value_loss": 0.15,
        "entropy": 0.07,
    }
    ppo.update.return_value = (metrics, new_weights or {"w": 1.0})
    ppo.load_human_data.return_value = human_raw
    ppo.load_expert_data.return_value = expert_raw
    ppo.merge_experiences.side_effect = lambda a, b: {**a, **b}
    return ppo


def _base_config(tmp_path: Path, **kwargs) -> TrainingConfig:
    return TrainingConfig(
        model_arch="v4",
        save_dir=str(tmp_path / "run"),
        iterations=3,
        bench_games=10,
        **kwargs,
    )


def _identity(*, oracle_critic: bool = False) -> ModelIdentity:
    return ModelIdentity(
        name="M",
        hidden_size=64,
        oracle_critic=oracle_critic,
        model_arch="v4",
        is_new=True,
    )


# ---------------------------------------------------------------------------
# CollectExperiences
# ---------------------------------------------------------------------------


class TestCollectExperiences:
    def test_oracle_states_included_when_oracle_critic_and_positive_imitation(
        self, tmp_path: Path
    ) -> None:
        sp = _make_selfplay()
        ppo = _make_ppo()
        presenter = MagicMock()
        config = _base_config(tmp_path, imitation_coef=0.5)
        identity = _identity(oracle_critic=True)

        uc = CollectExperiences(sp, ppo, presenter)
        uc.execute(config, identity, "model.pt", iter_imitation_coef=0.5)

        _, call_kwargs = sp.run.call_args
        assert call_kwargs["include_oracle_states"] is True

    def test_oracle_states_excluded_when_imitation_coef_zero(
        self, tmp_path: Path
    ) -> None:
        sp = _make_selfplay()
        ppo = _make_ppo()
        config = _base_config(tmp_path, imitation_coef=0.0)
        identity = _identity(oracle_critic=True)

        uc = CollectExperiences(sp, ppo, MagicMock())
        uc.execute(config, identity, "model.pt", iter_imitation_coef=0.0)

        _, call_kwargs = sp.run.call_args
        assert call_kwargs["include_oracle_states"] is False

    def test_oracle_states_excluded_when_not_oracle_critic(
        self, tmp_path: Path
    ) -> None:
        sp = _make_selfplay()
        config = _base_config(tmp_path, imitation_coef=0.9)
        identity = _identity(oracle_critic=False)

        uc = CollectExperiences(sp, _make_ppo(), MagicMock())
        uc.execute(config, identity, "model.pt", iter_imitation_coef=0.9)

        _, call_kwargs = sp.run.call_args
        assert call_kwargs["include_oracle_states"] is False

    def test_seats_override_takes_precedence(self, tmp_path: Path) -> None:
        sp = _make_selfplay()
        config = _base_config(tmp_path, seats="nn,bot,bot,bot")

        uc = CollectExperiences(sp, _make_ppo(), MagicMock())
        uc.execute(config, _identity(), "m.pt", seats_override="nn,nn,bot,bot")

        args, _ = sp.run.call_args
        # second positional arg is seats string
        assert args[2] == "nn,nn,bot,bot"

    def test_nn_seats_derived_from_effective_seats(self, tmp_path: Path) -> None:
        sp = _make_selfplay()
        config = _base_config(tmp_path, seats="nn,bot,nn,bot")

        uc = CollectExperiences(sp, _make_ppo(), MagicMock())
        bundle = uc.execute(config, _identity(), "m.pt")

        assert bundle.nn_seats == [0, 2]

    def test_human_data_merged_when_dir_configured(self, tmp_path: Path) -> None:
        human_raw = _raw_data(4)
        sp = _make_selfplay()
        ppo = _make_ppo(human_raw=human_raw)
        config = _base_config(tmp_path, human_data_dir=str(tmp_path / "human"))

        uc = CollectExperiences(sp, ppo, MagicMock())
        uc.execute(config, _identity(), "m.pt")

        ppo.load_human_data.assert_called_once_with(str(tmp_path / "human"))
        ppo.merge_experiences.assert_called_once()

    def test_human_data_skipped_when_dir_not_configured(self, tmp_path: Path) -> None:
        sp = _make_selfplay()
        ppo = _make_ppo()
        config = _base_config(tmp_path)  # human_data_dir defaults to None/empty

        uc = CollectExperiences(sp, ppo, MagicMock())
        uc.execute(config, _identity(), "m.pt")

        ppo.load_human_data.assert_not_called()
        ppo.merge_experiences.assert_not_called()

    def test_expert_data_merged_when_behavior_clone_enabled(self, tmp_path: Path) -> None:
        expert_raw = _raw_data(4)
        sp = _make_selfplay()
        ppo = _make_ppo(expert_raw=expert_raw)
        config = _base_config(
            tmp_path,
            behavioral_clone_coef=1.0,
            behavioral_clone_teacher="bot_v5",
            behavioral_clone_games_per_iteration=200,
        )

        uc = CollectExperiences(sp, ppo, MagicMock())
        uc.execute(config, _identity(), "m.pt")

        ppo.load_expert_data.assert_called_once_with(teacher="bot_v5", num_games=200)
        ppo.merge_experiences.assert_called_once()

    def test_expert_data_skipped_when_behavior_clone_disabled(self, tmp_path: Path) -> None:
        sp = _make_selfplay()
        ppo = _make_ppo(expert_raw=_raw_data(4))
        config = _base_config(tmp_path, behavioral_clone_coef=0.0)

        uc = CollectExperiences(sp, ppo, MagicMock())
        uc.execute(config, _identity(), "m.pt")

        ppo.load_expert_data.assert_not_called()

    def test_bundle_fields_populated_correctly(self, tmp_path: Path) -> None:
        raw = _raw_data(12)
        sp = _make_selfplay(raw, n_learner=3)
        config = _base_config(tmp_path, seats="nn,bot,bot,bot", games=100)

        uc = CollectExperiences(sp, _make_ppo(), MagicMock())
        bundle = uc.execute(config, _identity(), "m.pt")

        assert bundle.n_total == 12
        assert bundle.n_learner == 3
        assert bundle.seat_labels == ["nn", "bot", "bot", "bot"]
        assert bundle.nn_seats == [0]
        assert bundle.sp_time >= 0.0

    def test_presenter_notified_on_selfplay_done(self, tmp_path: Path) -> None:
        sp = _make_selfplay(_raw_data(8), n_learner=2)
        presenter = MagicMock()
        config = _base_config(tmp_path)

        uc = CollectExperiences(sp, _make_ppo(), presenter)
        uc.execute(config, _identity(), "m.pt")

        presenter.on_selfplay_done.assert_called_once()

    def test_outplace_session_size_forwarded_to_selfplay(self, tmp_path: Path) -> None:
        sp = _make_selfplay(_raw_data(8), n_learner=2)
        config = _base_config(tmp_path, outplace_session_size=17)

        uc = CollectExperiences(sp, _make_ppo(), MagicMock())
        uc.execute(config, _identity(), "m.pt")

        sp.compute_run_stats.assert_called_once_with(
            sp.run.return_value,
            ["nn", "bot_v5", "bot_v5", "bot_v5"],
            session_size=17,
        )


# ---------------------------------------------------------------------------
# UpdatePolicy
# ---------------------------------------------------------------------------


class TestUpdatePolicy:
    def test_lr_override_calls_set_lr(self) -> None:
        ppo = _make_ppo()
        config = TrainingConfig(model_arch="v4", save_dir="/tmp", iterations=1, bench_games=5)

        uc = UpdatePolicy(ppo, MagicMock())
        uc.execute(raw={}, nn_seats=[0], config=config, iter_lr=1e-4)

        ppo.set_lr.assert_called_once_with(1e-4)

    def test_lr_not_set_when_none(self) -> None:
        ppo = _make_ppo()
        config = TrainingConfig(model_arch="v4", save_dir="/tmp", iterations=1, bench_games=5)

        uc = UpdatePolicy(ppo, MagicMock())
        uc.execute(raw={}, nn_seats=[0], config=config, iter_lr=None)

        ppo.set_lr.assert_not_called()

    def test_imitation_coef_override(self) -> None:
        ppo = _make_ppo()
        config = TrainingConfig(model_arch="v4", save_dir="/tmp", iterations=1, bench_games=5)

        uc = UpdatePolicy(ppo, MagicMock())
        uc.execute(raw={}, nn_seats=[0], config=config, iter_imitation_coef=0.3)

        ppo.set_imitation_coef.assert_called_once_with(0.3)

    def test_imitation_coef_not_set_when_none(self) -> None:
        ppo = _make_ppo()
        config = TrainingConfig(model_arch="v4", save_dir="/tmp", iterations=1, bench_games=5)

        uc = UpdatePolicy(ppo, MagicMock())
        uc.execute(raw={}, nn_seats=[0], config=config, iter_imitation_coef=None)

        ppo.set_imitation_coef.assert_not_called()

    def test_entropy_coef_override(self) -> None:
        ppo = _make_ppo()
        config = TrainingConfig(model_arch="v4", save_dir="/tmp", iterations=1, bench_games=5)

        uc = UpdatePolicy(ppo, MagicMock())
        uc.execute(raw={}, nn_seats=[0], config=config, iter_entropy_coef=0.01)

        ppo.set_entropy_coef.assert_called_once_with(0.01)

    def test_behavioral_clone_coef_override(self) -> None:
        ppo = _make_ppo()
        config = TrainingConfig(model_arch="v4", save_dir="/tmp", iterations=1, bench_games=5)

        uc = UpdatePolicy(ppo, MagicMock())
        uc.execute(raw={}, nn_seats=[0], config=config, iter_behavioral_clone_coef=0.25)

        ppo.set_behavioral_clone_coef.assert_called_once_with(0.25)

    def test_behavioral_clone_coef_not_set_when_none(self) -> None:
        ppo = _make_ppo()
        config = TrainingConfig(model_arch="v4", save_dir="/tmp", iterations=1, bench_games=5)

        uc = UpdatePolicy(ppo, MagicMock())
        uc.execute(raw={}, nn_seats=[0], config=config, iter_behavioral_clone_coef=None)

        ppo.set_behavioral_clone_coef.assert_not_called()

    def test_result_carries_new_weights_and_metrics(self) -> None:
        weights = {"layer.weight": [1, 2, 3]}
        ppo = _make_ppo(new_weights=weights)
        config = TrainingConfig(model_arch="v4", save_dir="/tmp", iterations=1, bench_games=5)

        uc = UpdatePolicy(ppo, MagicMock())
        result = uc.execute(raw={}, nn_seats=[0], config=config)

        assert result.new_weights is weights
        assert result.metrics["total_loss"] == pytest.approx(0.42)
        assert result.ppo_time >= 0.0

    def test_presenter_notified_on_start_and_done(self) -> None:
        ppo = _make_ppo()
        presenter = MagicMock()
        config = TrainingConfig(model_arch="v4", save_dir="/tmp", iterations=1, bench_games=5)

        uc = UpdatePolicy(ppo, presenter)
        uc.execute(raw={}, nn_seats=[0], config=config)

        presenter.on_ppo_start.assert_called_once()
        presenter.on_ppo_done.assert_called_once()


# ---------------------------------------------------------------------------
# SaveCheckpoint
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    def _bundle(self) -> ExperienceBundle:
        return ExperienceBundle(
            raw=_raw_data(),
            nn_seats=[0],
            seat_labels=["nn", "bot", "bot", "bot"],
            effective_seats="nn,bot,bot,bot",
            n_total=8,
            n_learner=2,
            mean_scores=(1.0, -0.5, -0.5, 0.0),
            seat_outcomes={1: (3, 2, 1)},
            sp_time=1.5,
        )

    def _update(self, *, total_loss: float = 0.42) -> PolicyUpdateResult:
        return PolicyUpdateResult(
            new_weights={"w": 1.0},
            metrics={
                "total_loss": total_loss,
                "policy_loss": 0.20,
                "value_loss": 0.15,
                "entropy": 0.07,
            },
            ppo_time=0.8,
        )

    def test_checkpoint_path_uses_three_digit_iteration(self, tmp_path: Path) -> None:
        model = MagicMock()
        identity = _identity()

        uc = SaveCheckpoint(model)
        uc.execute(
            iteration=7,
            bundle=self._bundle(),
            update=self._update(),
            identity=identity,
            save_dir=tmp_path,
            placement=2.5,
            bench_time=1.0,
        )

        call_args = model.save_checkpoint.call_args[0]
        assert call_args[-1] == str(tmp_path / "iter_007.pt")

    def test_model_save_called_with_identity_fields(self, tmp_path: Path) -> None:
        model = MagicMock()
        identity = _identity(oracle_critic=True)

        uc = SaveCheckpoint(model)
        uc.execute(
            iteration=1,
            bundle=self._bundle(),
            update=self._update(),
            identity=identity,
            save_dir=tmp_path,
            placement=3.0,
            bench_time=0.5,
        )

        args = model.save_checkpoint.call_args[0]
        hidden_size_arg = args[1]
        oracle_arg = args[2]
        assert hidden_size_arg == 64
        assert oracle_arg is True

    def test_iteration_result_fields_match_inputs(self, tmp_path: Path) -> None:
        model = MagicMock()
        bundle = self._bundle()
        update = self._update(total_loss=0.55)

        uc = SaveCheckpoint(model)
        result = uc.execute(
            iteration=5,
            bundle=bundle,
            update=update,
            identity=_identity(),
            save_dir=tmp_path,
            placement=2.0,
            bench_time=3.2,
        )

        assert result.iteration == 5
        assert result.placement == pytest.approx(2.0)
        assert result.loss == pytest.approx(0.55)
        assert result.policy_loss == pytest.approx(0.20)
        assert result.value_loss == pytest.approx(0.15)
        assert result.entropy == pytest.approx(0.07)
        assert result.n_experiences == 8
        assert result.selfplay_time == pytest.approx(1.5)
        assert result.ppo_time == pytest.approx(0.8)
        assert result.bench_time == pytest.approx(3.2)
        assert result.seat_config_used == "nn,bot,bot,bot"
        assert result.mean_scores == (1.0, -0.5, -0.5, 0.0)
        assert result.seat_outcomes == {1: (3, 2, 1)}

    def test_iteration_result_is_immutable(self, tmp_path: Path) -> None:
        model = MagicMock()

        uc = SaveCheckpoint(model)
        result = uc.execute(
            iteration=1,
            bundle=self._bundle(),
            update=self._update(),
            identity=_identity(),
            save_dir=tmp_path,
            placement=2.0,
            bench_time=0.0,
        )

        with pytest.raises((AttributeError, TypeError)):
            result.iteration = 99  # type: ignore[misc]
