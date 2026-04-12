"""ManageHoF — Hall of Fame management use case."""

from __future__ import annotations

import hashlib
import logging

from training_lab.entities.checkpoint import Checkpoint
from training_lab.entities.network import TarokNet
from training_lab.ports.hof import HoFPort

log = logging.getLogger(__name__)


class ManageHoF:
    """Save, pin, unpin, and manage Hall of Fame entries."""

    def __init__(self, hof: HoFPort):
        self.hof = hof

    def save_model(
        self,
        network: TarokNet,
        checkpoint: Checkpoint,
        pinned: bool = False,
    ) -> Checkpoint:
        """Save a model to the Hall of Fame."""
        state_dict = network.state_dict()
        if not checkpoint.model_hash:
            h = hashlib.sha256()
            for key in sorted(state_dict.keys()):
                h.update(state_dict[key].cpu().numpy().tobytes()[:64])
            checkpoint.model_hash = h.hexdigest()[:8]

        result = self.hof.save(state_dict, checkpoint, pinned=pinned)
        log.info("Saved to HoF: %s (pinned=%s)", result.model_hash, pinned)
        return result

    def load_model(self, model_hash: str) -> tuple[TarokNet, Checkpoint] | None:
        """Load a model from the HoF by hash."""
        result = self.hof.load(model_hash)
        if result is None:
            return None
        state_dict, checkpoint = result
        network = TarokNet(hidden_size=checkpoint.hidden_size)
        network.load_state_dict(state_dict)
        return network, checkpoint

    def pin(self, model_hash: str) -> bool:
        return self.hof.pin(model_hash)

    def unpin(self, model_hash: str) -> bool:
        return self.hof.unpin(model_hash)

    def remove(self, model_hash: str) -> bool:
        return self.hof.remove(model_hash)

    def list(self) -> list[Checkpoint]:
        return self.hof.list()
