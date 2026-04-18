"""Infrastructure adapter package.

Structure is organized by delivery concern to keep modules discoverable:
- configuration/: config file loaders
- evaluation/: benchmark runners
- modeling/: model persistence/export
- persistence/: state stores
- policies/: schedule and Elo policies
- presentation/: terminal/UI presenters
- self_play/: game data producers
"""

from training.adapters.configuration import YAMLConfigLoader
from training.adapters.evaluation import SessionBenchmark
from training.adapters.modeling import TorchModelAdapter
from training.adapters.persistence import JsonLeagueStatePersistence
from training.adapters.policies import LeagueEloLearningRatePolicy, ScheduledImitationCoefPolicy
from training.adapters.presentation import TerminalPresenter
from training.adapters.self_play import RustSelfPlay

__all__ = [
	"YAMLConfigLoader",
	"SessionBenchmark",
	"TorchModelAdapter",
	"JsonLeagueStatePersistence",
	"LeagueEloLearningRatePolicy",
	"ScheduledImitationCoefPolicy",
	"TerminalPresenter",
	"RustSelfPlay",
]
