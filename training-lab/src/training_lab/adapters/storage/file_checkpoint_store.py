"""File-based checkpoint storage."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import torch

from training_lab.entities.checkpoint import Checkpoint
from training_lab.ports.checkpoint_store import CheckpointStorePort


class FileCheckpointStore(CheckpointStorePort):
    """Save/load model checkpoints as .pt files on disk."""

    def __init__(self, checkpoint_dir: str | Path):
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, state_dict: dict, checkpoint: Checkpoint) -> Path:
        if not checkpoint.timestamp:
            checkpoint.timestamp = time.time()

        data = {
            "model_state_dict": state_dict,
            "model_hash": checkpoint.model_hash,
            "train_step": checkpoint.train_step,
            "win_rate": checkpoint.win_rate,
            "persona": checkpoint.persona,
            "phase_label": checkpoint.phase_label,
            "hidden_size": checkpoint.hidden_size,
            "oracle_critic": checkpoint.oracle_critic,
            "saved_at": checkpoint.timestamp,
            "eval_history": checkpoint.eval_history,
            "metrics": checkpoint.metrics,
        }

        filename = f"tarok_agent_{checkpoint.phase_label or 'latest'}.pt"
        path = self._dir / filename
        tmp = path.with_suffix(".tmp")
        torch.save(data, tmp)
        tmp.replace(path)
        return path

    def load(self, identifier: str) -> tuple[dict, Checkpoint] | None:
        # Try as filename first
        path = self._dir / identifier
        if not path.exists():
            path = self._dir / f"{identifier}.pt"
        if not path.exists():
            # Search by hash
            for p in self._dir.glob("*.pt"):
                try:
                    data = torch.load(p, map_location="cpu", weights_only=False)
                    if data.get("model_hash") == identifier:
                        path = p
                        break
                except Exception:
                    continue
            else:
                return None

        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return None

        state_dict = data.get("model_state_dict")
        if state_dict is None:
            return None

        checkpoint = Checkpoint(
            model_hash=data.get("model_hash", ""),
            train_step=data.get("train_step", 0),
            win_rate=data.get("win_rate", 0.0),
            persona=data.get("persona", {}),
            phase_label=data.get("phase_label", ""),
            hidden_size=data.get("hidden_size", 256),
            oracle_critic=data.get("oracle_critic", False),
            timestamp=data.get("saved_at", 0.0),
            eval_history=data.get("eval_history", []),
            metrics=data.get("metrics", {}),
        )
        return state_dict, checkpoint

    def list(self) -> list[Checkpoint]:
        results = []
        for p in sorted(self._dir.glob("*.pt")):
            try:
                data = torch.load(p, map_location="cpu", weights_only=False)
                results.append(Checkpoint(
                    model_hash=data.get("model_hash", ""),
                    train_step=data.get("train_step", 0),
                    win_rate=data.get("win_rate", 0.0),
                    persona=data.get("persona", {}),
                    phase_label=data.get("phase_label", ""),
                    hidden_size=data.get("hidden_size", 256),
                    timestamp=data.get("saved_at", 0.0),
                ))
            except Exception:
                continue
        return results

    def delete(self, identifier: str) -> bool:
        path = self._dir / identifier
        if not path.exists():
            path = self._dir / f"{identifier}.pt"
        if path.exists():
            path.unlink()
            return True
        return False
