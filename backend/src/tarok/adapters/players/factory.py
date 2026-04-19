"""Canonical player factory and registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, cast

from tarok.ports.player_port import PlayerPort

from tarok.adapters.players.human_player import HumanPlayer
from tarok.adapters.players.neural_player import NeuralPlayer
from tarok.adapters.players.stockskis_player import StockskisPlayer


@dataclass(frozen=True)
class PlayerTypeInfo:
    id: str
    name: str
    description: str
    category: str
    version: int | None = None
    default_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
        }


PlayerCreator = Callable[..., PlayerPort]


_STOCKSKIS_DESCRIPTIONS: dict[int, str] = {
    5: "Rust-backed engine for maximum speed (delegates to Rust)",
    6: "Improved v5 with better discarding and king protection",
}

_STOCKSKIS_NAMED_VARIANTS: dict[str, tuple[str, str]] = {
    "m6": ("StockSkisPlayerM6", "Refined v5/v6 with cautious pagat, point-securing discards"),
    "pozrl": ("StockSkisPlayerPozrl", "Heuristic policy based on Domen Požrl's 2021 Tarock thesis"),
}


class PlayerFactory:
    """Registry-backed factory for all PlayerPort implementations."""

    def __init__(self) -> None:
        self._players: dict[str, tuple[PlayerTypeInfo, PlayerCreator]] = {}

    def register(self, info: PlayerTypeInfo, creator: PlayerCreator) -> None:
        self._players[info.id] = (info, creator)

    def discover(self) -> None:
        self._register_builtins()
        self._register_stockskis()

    def _register_builtins(self) -> None:
        self.register(
            PlayerTypeInfo(
                id="nn",
                name="Neural Network",
                description="TarokNet player loaded from checkpoint or initialized fresh",
                category="neural",
                default_params={"hidden_size": 256, "device": "cpu"},
            ),
            _make_nn_creator(),
        )
        self.register(
            PlayerTypeInfo(
                id="human",
                name="Human",
                description="Human player via WebSocket",
                category="human",
            ),
            lambda name="Human", **kwargs: cast(PlayerPort, HumanPlayer(name=name)),
        )

    def _register_stockskis(self) -> None:
        version_variants = {"v5": 5}
        for variant, version in version_variants.items():
            bot_id = f"stockskis_{variant}"
            desc = _STOCKSKIS_DESCRIPTIONS.get(
                version, f"StockSkis heuristic bot version {version}"
            )
            self.register(
                PlayerTypeInfo(
                    id=bot_id,
                    name=f"StockSkis v{version}",
                    description=desc,
                    category="heuristic",
                    version=version,
                ),
                lambda name=f"StockSkis-v{version}", v=variant, **kw: StockskisPlayer(
                    variant=v, name=name
                ),
            )

        for variant, (_, desc) in _STOCKSKIS_NAMED_VARIANTS.items():
            bot_id = f"stockskis_{variant}"
            self.register(
                PlayerTypeInfo(
                    id=bot_id,
                    name=f"StockSkis {variant}",
                    description=desc,
                    category="heuristic",
                    version=None,
                ),
                lambda name=f"StockSkis-{variant}", v=variant, **kw: StockskisPlayer(
                    variant=v, name=name
                ),
            )

    def list_players(self) -> list[dict[str, Any]]:
        order = {"heuristic": 1, "neural": 3, "human": 4}
        items = sorted(
            self._players.values(),
            key=lambda x: (order.get(x[0].category, 99), x[0].version or 0, x[0].name),
        )
        return [info.to_dict() for info, _ in items]

    def get(self, player_type: str) -> tuple[PlayerTypeInfo, PlayerCreator] | None:
        return self._players.get(player_type)

    def has(self, player_type: str) -> bool:
        return player_type in self._players

    def create(self, player_type: str, *, name: str | None = None, **kwargs: Any) -> PlayerPort:
        entry = self._players.get(player_type)
        if entry is None:
            raise KeyError(
                f"Unknown player type: {player_type!r}. Available: {list(self._players)}"
            )
        info, creator = entry
        if name is None:
            name = info.name
        return creator(name=name, **kwargs)

    @property
    def stockskis_versions(self) -> list[int]:
        versions: list[int] = []
        for _player_type, (info, _) in self._players.items():
            if info.category == "heuristic" and info.version is not None:
                versions.append(info.version)
        return sorted(versions)

    @property
    def stockskis_types(self) -> list[str]:
        types: list[str] = []
        for player_type, (info, _) in self._players.items():
            if info.category == "heuristic":
                types.append(player_type)
        return sorted(types)


def _make_nn_creator() -> PlayerCreator:
    def creator(
        name: str = "NN",
        checkpoint: str | None = None,
        hidden_size: int = 256,
        device: str = "cpu",
        **kwargs: Any,
    ) -> PlayerPort:
        if checkpoint:
            return NeuralPlayer.from_checkpoint(checkpoint, name=name, device=device)
        return NeuralPlayer(name=name, hidden_size=hidden_size, device=device)

    return creator


_default_factory: PlayerFactory | None = None


def get_player_factory() -> PlayerFactory:
    global _default_factory
    if _default_factory is None:
        _default_factory = PlayerFactory()
        _default_factory.discover()
    return _default_factory
