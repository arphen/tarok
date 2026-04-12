"""Hall of Fame Manager — pinning, auto-eviction, and size limits.

Models in the HoF are either:
  - **pinned**: manually promoted/protected, never auto-evicted
  - **auto**: saved automatically during training, subject to max-size limit
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import torch

from training_lab.entities.checkpoint import Checkpoint
from training_lab.entities.network import TarokNet
from training_lab.ports.hof import HoFPort

log = logging.getLogger(__name__)


def _model_hash(state_dict: dict) -> str:
    """Short deterministic hash of model weights."""
    h = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        h.update(state_dict[key].cpu().numpy().tobytes()[:64])
    return h.hexdigest()[:8]


class FileHoFManager(HoFPort):
    """Manage the Hall of Fame directory with pinning and auto-eviction."""

    def __init__(self, hof_dir: str | Path, max_auto: int = 10):
        self.hof_dir = Path(hof_dir)
        self.pinned_dir = self.hof_dir / "pinned"
        self.max_auto = max_auto
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.hof_dir.mkdir(parents=True, exist_ok=True)
        self.pinned_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _manifest_path(self) -> Path:
        return self.hof_dir / "manifest.json"

    def _load_manifest(self) -> list[dict]:
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text())
            except Exception:
                return []
        return []

    def _save_manifest(self, manifest: list[dict]) -> None:
        tmp = self._manifest_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest, indent=2))
        tmp.replace(self._manifest_path)

    def list(self) -> list[Checkpoint]:
        manifest = self._load_manifest()
        pinned = [self._entry_to_checkpoint(e) for e in manifest if e.get("pinned", False)]
        auto = [self._entry_to_checkpoint(e) for e in manifest if not e.get("pinned", False)]
        return pinned + auto

    def list_auto(self) -> list[Checkpoint]:
        return [self._entry_to_checkpoint(e) for e in self._load_manifest() if not e.get("pinned", False)]

    def save(
        self,
        state_dict: dict,
        checkpoint: Checkpoint,
        pinned: bool = False,
    ) -> Checkpoint:
        self._ensure_dirs()
        h = checkpoint.model_hash or _model_hash(state_dict)
        checkpoint.model_hash = h
        checkpoint.pinned = pinned
        if not checkpoint.timestamp:
            checkpoint.timestamp = time.time()

        persona = checkpoint.persona
        first = persona.get("first_name", "Agent")
        last = persona.get("last_name", "Unknown")
        age = persona.get("age", 0)
        filename = f"hof_{first}_{last}_age{age}_{h}.pt"

        latest_eval = checkpoint.eval_history[-1] if checkpoint.eval_history else {}

        data = {
            "model_state_dict": state_dict,
            "persona": persona,
            "model_hash": h,
            "phase_label": checkpoint.phase_label,
            "eval_history": checkpoint.eval_history,
            "hidden_size": checkpoint.hidden_size,
            "pinned": pinned,
            "metrics": checkpoint.metrics,
            "saved_at": checkpoint.timestamp,
        }

        target_dir = self.pinned_dir if pinned else self.hof_dir
        torch.save(data, target_dir / filename)

        manifest = self._load_manifest()
        entry = {
            "filename": filename,
            "persona": persona,
            "model_hash": h,
            "phase_label": checkpoint.phase_label,
            "pinned": pinned,
            "metrics": checkpoint.metrics,
            "saved_at": checkpoint.timestamp,
        }
        manifest.append(entry)
        self._save_manifest(manifest)

        if not pinned:
            self._evict_auto(manifest)

        return checkpoint

    def load(self, model_hash: str) -> tuple[dict, Checkpoint] | None:
        manifest = self._load_manifest()
        entry = next((e for e in manifest if e.get("model_hash") == model_hash), None)
        if entry is None:
            return None

        filename = entry["filename"]
        is_pinned = entry.get("pinned", False)

        for d in [self.pinned_dir if is_pinned else self.hof_dir, self.hof_dir, self.pinned_dir]:
            pt_path = d / filename
            if pt_path.exists():
                try:
                    data = torch.load(pt_path, map_location="cpu", weights_only=False)
                except Exception:
                    return None
                state_dict = data.get("model_state_dict")
                if state_dict is None:
                    return None
                return state_dict, self._entry_to_checkpoint(entry)
        return None

    def pin(self, model_hash: str) -> bool:
        manifest = self._load_manifest()
        entry = next((e for e in manifest if e.get("model_hash") == model_hash), None)
        if entry is None:
            return False
        if entry.get("pinned", False):
            return True

        filename = entry["filename"]
        src = self.hof_dir / filename
        dst = self.pinned_dir / filename

        if src.exists():
            try:
                data = torch.load(src, map_location="cpu", weights_only=False)
                data["pinned"] = True
                torch.save(data, dst)
                src.unlink()
            except Exception:
                return False

        entry["pinned"] = True
        self._save_manifest(manifest)
        return True

    def unpin(self, model_hash: str) -> bool:
        manifest = self._load_manifest()
        entry = next((e for e in manifest if e.get("model_hash") == model_hash), None)
        if entry is None:
            return False
        if not entry.get("pinned", False):
            return True

        filename = entry["filename"]
        src = self.pinned_dir / filename
        dst = self.hof_dir / filename

        if src.exists():
            try:
                data = torch.load(src, map_location="cpu", weights_only=False)
                data["pinned"] = False
                torch.save(data, dst)
                src.unlink()
            except Exception:
                return False

        entry["pinned"] = False
        self._save_manifest(manifest)
        self._evict_auto(manifest)
        return True

    def remove(self, model_hash: str) -> bool:
        manifest = self._load_manifest()
        targets = [e for e in manifest if e.get("model_hash") == model_hash]
        if not targets:
            return False
        for entry in targets:
            self._delete_entry(entry)
        return True

    def _evict_auto(self, manifest: list[dict] | None = None) -> None:
        if manifest is None:
            manifest = self._load_manifest()

        auto_entries = [e for e in manifest if not e.get("pinned", False)]
        if len(auto_entries) <= self.max_auto:
            return

        def _score(entry: dict) -> float:
            m = entry.get("metrics", {})
            return sum(m.get(k, 0) * w for k, w in [
                ("vs_v5", 3.0), ("vs_v3", 2.0), ("vs_v3.2", 1.5),
                ("vs_v2", 1.0), ("vs_v1", 0.5),
            ])

        auto_entries.sort(key=_score)
        n_to_remove = len(auto_entries) - self.max_auto
        for entry in auto_entries[:n_to_remove]:
            self._delete_entry(entry)

    def _delete_entry(self, entry: dict) -> None:
        filename = entry["filename"]
        is_pinned = entry.get("pinned", False)

        for d in [self.pinned_dir if is_pinned else self.hof_dir, self.hof_dir, self.pinned_dir]:
            pt_path = d / filename
            if pt_path.exists():
                pt_path.unlink()
                break

        manifest = self._load_manifest()
        model_hash = entry.get("model_hash", "")
        manifest = [e for e in manifest if e.get("model_hash") != model_hash]
        self._save_manifest(manifest)

    @staticmethod
    def _entry_to_checkpoint(entry: dict) -> Checkpoint:
        m = entry.get("metrics", {})
        return Checkpoint(
            model_hash=entry.get("model_hash", ""),
            win_rate=m.get("win_rate", m.get("vs_v1", 0.0)),
            persona=entry.get("persona", {}),
            phase_label=entry.get("phase_label", ""),
            timestamp=entry.get("saved_at", 0.0),
            pinned=entry.get("pinned", False),
            metrics=m,
        )
