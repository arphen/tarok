"""Player factory — unified plugin system for creating any player type.

Supports bots (heuristic, baseline, search), neural network agents,
and human players — all through the same interface. Any module under
``tarok.adapters.ai`` whose filename matches ``stockskis*.py`` is
auto-discovered.

Usage::

    from tarok.adapters.ai.bot_registry import get_registry

    factory = get_registry()

    # Bots
    bot = factory.create("stockskis_v5", name="Bot-0")

    # Neural network agent from checkpoint
    ai = factory.create("nn", name="AI", checkpoint="path/to/model.pt")

    # Fresh neural network (for self-play training)
    ai = factory.create("nn", name="Lab-0", hidden_size=256, device="cpu")

    # Human player (WebSocket)
    human = factory.create("human", name="You")

    # List all available player types
    info = factory.list_bots()
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class BotInfo:
    """Metadata for a registered bot plugin."""

    id: str  # e.g. "stockskis_v3", "random", "lookahead"
    name: str  # human-readable, e.g. "StockŠkis v3"
    description: str
    category: str  # "heuristic", "neural", "baseline", "search"
    version: int | None = None  # numeric version for sorting (StockŠkis only)
    default_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
        }


# Type for factory callables: (name: str, **kwargs) -> PlayerPort
BotFactory = Callable[..., Any]

_STOCKSKIS_FILE_RE = re.compile(r"^stockskis(?:_v(\d+))?(?:_\d+)?\.py$")

# Descriptions for known StockSkis versions
_STOCKSKIS_DESCRIPTIONS: dict[int, str] = {
    1: "Basic heuristics ported from the original Dart engine",
    2: "Card counting + positional awareness + partner signaling",
    3: "Bayesian inference + endgame awareness + game phase tracking",
    4: "Tighter béraç gating + clearer declarer/defender roles",
    5: "Rust-backed engine for maximum speed (delegates to Rust)",
}


class BotRegistry:
    """Central registry for bot plugins.

    Supports auto-discovery of built-in StockŠkis versions and manual
    registration of external plugins.
    """

    def __init__(self) -> None:
        self._bots: dict[str, tuple[BotInfo, BotFactory]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, info: BotInfo, factory: BotFactory) -> None:
        """Register a bot plugin.  Overwrites if *id* already exists."""
        self._bots[info.id] = (info, factory)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self) -> None:
        """Auto-discover built-in bots (StockŠkis v1-v*, random, lookahead)."""
        self._register_builtins()
        self._discover_stockskis()

    def _register_builtins(self) -> None:
        """Register non-StockŠkis built-in bots."""
        from tarok.adapters.ai.random_agent import RandomPlayer

        self.register(
            BotInfo(
                id="random",
                name="Random",
                description="Uniformly random legal moves (biased toward passing)",
                category="baseline",
            ),
            lambda name="Random", **kw: RandomPlayer(name=name),
        )

        # Lookahead (Monte Carlo search) — import lazily to avoid hard dep
        try:
            from tarok.adapters.ai.lookahead_agent import LookaheadAgent

            self.register(
                BotInfo(
                    id="lookahead",
                    name="Lookahead (MCTS)",
                    description="Monte Carlo tree search with configurable simulations",
                    category="search",
                    default_params={"n_simulations": 50},
                ),
                lambda name="Lookahead", n_simulations=50, **kw: LookaheadAgent(
                    n_simulations=n_simulations, name=name
                ),
            )
        except ImportError:
            pass

        # Neural network agent
        self.register(
            BotInfo(
                id="nn",
                name="Neural Network",
                description="TarokNet RL agent — load from checkpoint or create fresh",
                category="neural",
                default_params={"hidden_size": 256, "device": "cpu"},
            ),
            _make_nn_factory(),
        )

        # Human player (WebSocket)
        self.register(
            BotInfo(
                id="human",
                name="Human",
                description="Human player via WebSocket connection",
                category="human",
            ),
            _make_human_factory(),
        )

    def _discover_stockskis(self) -> None:
        """Scan the ai/ directory for stockskis*.py files and register them."""
        ai_dir = Path(__file__).resolve().parent
        seen_versions: set[int] = set()

        for fpath in sorted(ai_dir.iterdir()):
            m = _STOCKSKIS_FILE_RE.match(fpath.name)
            if not m:
                continue

            version = int(m.group(1)) if m.group(1) else 1
            if version in seen_versions:
                continue
            seen_versions.add(version)

            # Determine module and class name
            module_name = fpath.stem  # e.g. "stockskis_v5"
            class_name = f"StockSkisPlayerV{version}"
            full_module = f"tarok.adapters.ai.{module_name}"

            bot_id = f"stockskis_v{version}"
            desc = _STOCKSKIS_DESCRIPTIONS.get(version, f"StockŠkis heuristic bot version {version}")

            # Capture variables for the lambda
            self.register(
                BotInfo(
                    id=bot_id,
                    name=f"StockŠkis v{version}",
                    description=desc,
                    category="heuristic",
                    version=version,
                ),
                _make_stockskis_factory(full_module, class_name, version),
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_bots(self) -> list[dict]:
        """Return metadata for all registered bots, sorted by category then name."""
        order = {"baseline": 0, "heuristic": 1, "search": 2, "neural": 3}
        items = sorted(
            self._bots.values(),
            key=lambda x: (order.get(x[0].category, 99), x[0].version or 0, x[0].name),
        )
        return [info.to_dict() for info, _ in items]

    def get(self, bot_id: str) -> tuple[BotInfo, BotFactory] | None:
        return self._bots.get(bot_id)

    def has(self, bot_id: str) -> bool:
        return bot_id in self._bots

    def create(self, bot_id: str, *, name: str | None = None, **kwargs: Any) -> Any:
        """Instantiate a bot by its registered *id*.

        Raises ``KeyError`` if *bot_id* is not registered.
        """
        entry = self._bots.get(bot_id)
        if entry is None:
            raise KeyError(f"Unknown bot: {bot_id!r}. Available: {list(self._bots)}")
        info, factory = entry
        if name is None:
            name = info.name
        return factory(name=name, **kwargs)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def stockskis_versions(self) -> list[int]:
        """Return sorted list of available StockŠkis version numbers."""
        versions = []
        for bot_id, (info, _) in self._bots.items():
            if info.category == "heuristic" and info.version is not None:
                versions.append(info.version)
        return sorted(versions)


def _make_stockskis_factory(module_path: str, class_name: str, version: int) -> BotFactory:
    """Create a factory that lazily imports and instantiates a StockŠkis player."""

    def factory(name: str = f"StockŠkis-v{version}", **kwargs: Any) -> Any:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls(name=name, **kwargs)

    return factory


def _make_nn_factory() -> BotFactory:
    """Create a factory for neural network agents."""

    def factory(
        name: str = "NN",
        checkpoint: str | None = None,
        hidden_size: int = 256,
        device: str = "cpu",
        explore_rate: float = 0.0,
        **kwargs: Any,
    ) -> Any:
        from tarok.adapters.ai.agent import RLAgent

        if checkpoint:
            return RLAgent.from_checkpoint(checkpoint, name=name, device=device)
        return RLAgent(name=name, hidden_size=hidden_size, device=device, explore_rate=explore_rate)

    return factory


def _make_human_factory() -> BotFactory:
    """Create a factory for human WebSocket players."""

    def factory(name: str = "Human", **kwargs: Any) -> Any:
        from tarok.adapters.api.human_player import HumanPlayer

        return HumanPlayer(name=name)

    return factory


# ------------------------------------------------------------------
# Singleton — lazily initialized on first access
# ------------------------------------------------------------------

_default_registry: BotRegistry | None = None


def get_registry() -> BotRegistry:
    """Return the default global bot registry (auto-discovers on first call)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = BotRegistry()
        _default_registry.discover()
    return _default_registry
