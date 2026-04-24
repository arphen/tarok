"""League calibration adapters (swappable via ``LeagueCalibrationPort``)."""

from training.adapters.league_calibration.self_play_calibration import (
    SelfPlayLeagueCalibrationAdapter,
)
from training.adapters.league_calibration.duplicate_tournament_calibration import (
    DuplicateTournamentLeagueCalibrationAdapter,
)

__all__ = [
    "SelfPlayLeagueCalibrationAdapter",
    "DuplicateTournamentLeagueCalibrationAdapter",
]
