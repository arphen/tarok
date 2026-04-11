"""Hall of Fame Manager — pinning, auto-eviction, and size limits.

Models in the HoF are either:
  - **pinned**: manually promoted/protected, never auto-evicted, stored in
    ``hall_of_fame/pinned/`` subdirectory.
  - **auto**: saved automatically during training, subject to a max-size
    limit (default 10). When the limit is exceeded the weakest auto models
    are evicted.

The manifest tracks both kinds with a ``pinned`` boolean flag.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch

from tarok.adapters.ai.network import TarokNet

log = logging.getLogger(__name__)


def _model_hash(network: TarokNet) -> str:
    """Short deterministic hash of model weights."""
    import hashlib
    h = hashlib.sha256()
    for p in network.parameters():
        h.update(p.data.cpu().numpy().tobytes()[:64])
    return h.hexdigest()[:8]


def _display_name(persona: dict, hash_str: str) -> str:
    return f"{persona['first_name']} {persona['last_name']} (age {persona['age']}) #{hash_str}"


class HoFManager:
    """Manage the Hall of Fame directory with pinning and auto-eviction.

    Parameters
    ----------
    hof_dir : Path
        Root HoF directory (e.g. ``checkpoints/hall_of_fame``).
    max_auto : int
        Maximum number of *auto* (non-pinned) models to keep.
    """

    def __init__(self, hof_dir: str | Path, max_auto: int = 10):
        self.hof_dir = Path(hof_dir)
        self.pinned_dir = self.hof_dir / "pinned"
        self.max_auto = max_auto
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.hof_dir.mkdir(parents=True, exist_ok=True)
        self.pinned_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list(self) -> list[dict]:
        """Return all HoF entries (pinned first, then auto by score desc)."""
        manifest = self._load_manifest()
        pinned = [e for e in manifest if e.get("pinned", False)]
        auto = [e for e in manifest if not e.get("pinned", False)]
        return pinned + auto

    def list_auto(self) -> list[dict]:
        return [e for e in self._load_manifest() if not e.get("pinned", False)]

    def list_pinned(self) -> list[dict]:
        return [e for e in self._load_manifest() if e.get("pinned", False)]

    @property
    def auto_count(self) -> int:
        return len(self.list_auto())

    @property
    def pinned_count(self) -> int:
        return len(self.list_pinned())

    def save(
        self,
        network: TarokNet,
        persona: dict,
        eval_history: list[dict],
        phase_label: str = "",
        pinned: bool = False,
    ) -> dict:
        """Save a model to the HoF. Auto-evicts if over limit.

        Returns the entry dict with filename, display_name, model_hash.
        """
        self._ensure_dirs()
        h = _model_hash(network)
        display = _display_name(persona, h)
        filename = f"hof_{persona['first_name']}_{persona['last_name']}_age{persona['age']}_{h}.pt"

        latest_eval = eval_history[-1] if eval_history else {}

        data = {
            "model_state_dict": network.state_dict(),
            "persona": persona,
            "model_hash": h,
            "display_name": display,
            "model_name": display,
            "phase_label": phase_label,
            "eval_history": eval_history,
            "hidden_size": network.shared[0].out_features,
            "pinned": pinned,
            "metrics": {
                "win_rate": latest_eval.get("vs_v1", 0),
                "vs_v1": latest_eval.get("vs_v1", 0),
                "vs_v2": latest_eval.get("vs_v2", 0),
                "vs_v3": latest_eval.get("vs_v3", 0),
                "vs_v3.2": latest_eval.get("vs_v3.2", 0),
                "vs_v5": latest_eval.get("vs_v5", 0),
                "avg_score_v1": latest_eval.get("avg_score_v1", 0),
            },
            "saved_at": time.time(),
        }

        # Save to pinned/ subdir or main HoF dir
        target_dir = self.pinned_dir if pinned else self.hof_dir
        torch.save(data, target_dir / filename)

        # Update manifest
        manifest = self._load_manifest()
        entry = {
            "filename": filename,
            "display_name": display,
            "persona": persona,
            "model_hash": h,
            "phase_label": phase_label,
            "pinned": pinned,
            "vs_v1": latest_eval.get("vs_v1", 0),
            "vs_v2": latest_eval.get("vs_v2", 0),
            "vs_v3": latest_eval.get("vs_v3", 0),
            "vs_v3.2": latest_eval.get("vs_v3.2", 0),
            "vs_v5": latest_eval.get("vs_v5", 0),
            "saved_at": data["saved_at"],
        }
        manifest.append(entry)
        self._save_manifest(manifest)

        # Auto-evict if over limit (only auto entries)
        if not pinned:
            self._evict_auto(manifest)

        return {"filename": filename, "display_name": display, "model_hash": h, "pinned": pinned}

    def _evict_auto(self, manifest: list[dict] | None = None) -> None:
        """Remove weakest auto models if count exceeds max_auto."""
        if manifest is None:
            manifest = self._load_manifest()

        auto_entries = [e for e in manifest if not e.get("pinned", False)]
        if len(auto_entries) <= self.max_auto:
            return

        # Sort by composite score (vs_v5 > vs_v3 > vs_v1), weakest first
        def _score(entry: dict) -> float:
            return (
                entry.get("vs_v5", 0) * 3.0
                + entry.get("vs_v3", 0) * 2.0
                + entry.get("vs_v3.2", 0) * 1.5
                + entry.get("vs_v2", 0) * 1.0
                + entry.get("vs_v1", 0) * 0.5
            )

        auto_entries.sort(key=_score)
        n_to_remove = len(auto_entries) - self.max_auto
        to_remove = auto_entries[:n_to_remove]

        for entry in to_remove:
            self._delete_entry(entry)
            log.info("Auto-evicted HoF model: %s (score=%.2f)", entry.get("display_name", "?"), _score(entry))

    def _delete_entry(self, entry: dict) -> None:
        """Delete a model's .pt file and remove from manifest."""
        filename = entry["filename"]
        is_pinned = entry.get("pinned", False)

        # Try both locations
        for d in [self.pinned_dir if is_pinned else self.hof_dir, self.hof_dir, self.pinned_dir]:
            pt_path = d / filename
            if pt_path.exists():
                pt_path.unlink()
                break

        manifest = self._load_manifest()
        model_hash = entry.get("model_hash", "")
        manifest = [e for e in manifest if e.get("model_hash") != model_hash]
        self._save_manifest(manifest)

    def remove(self, model_hash: str) -> bool:
        """Remove a model by hash. Works for both pinned and auto."""
        manifest = self._load_manifest()
        targets = [e for e in manifest if e.get("model_hash") == model_hash]
        if not targets:
            return False
        for entry in targets:
            self._delete_entry(entry)
        return True

    def pin(self, model_hash: str) -> bool:
        """Pin an existing auto model (move to pinned/ dir, exempt from eviction)."""
        manifest = self._load_manifest()
        entry = next((e for e in manifest if e.get("model_hash") == model_hash), None)
        if entry is None:
            return False
        if entry.get("pinned", False):
            return True  # Already pinned

        filename = entry["filename"]
        src = self.hof_dir / filename
        dst = self.pinned_dir / filename

        if src.exists():
            # Update the pinned flag inside the .pt file
            try:
                data = torch.load(src, map_location="cpu", weights_only=False)
                data["pinned"] = True
                torch.save(data, dst)
                src.unlink()
            except Exception:
                log.warning("Failed to move %s to pinned/", filename)
                return False
        elif dst.exists():
            pass  # Already in pinned dir
        else:
            return False

        entry["pinned"] = True
        self._save_manifest(manifest)
        return True

    def unpin(self, model_hash: str) -> bool:
        """Unpin a model (move back to main HoF dir, subject to eviction)."""
        manifest = self._load_manifest()
        entry = next((e for e in manifest if e.get("model_hash") == model_hash), None)
        if entry is None:
            return False
        if not entry.get("pinned", False):
            return True  # Already unpinned

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
                log.warning("Failed to move %s out of pinned/", filename)
                return False

        entry["pinned"] = False
        self._save_manifest(manifest)

        # May trigger eviction now that we have one more auto
        self._evict_auto(manifest)
        return True

    def load_model(self, model_hash: str) -> tuple[TarokNet, dict] | None:
        """Load a HoF model by hash. Returns (network, metadata) or None."""
        manifest = self._load_manifest()
        entry = next((e for e in manifest if e.get("model_hash") == model_hash), None)
        if entry is None:
            return None

        filename = entry["filename"]
        is_pinned = entry.get("pinned", False)

        # Search both dirs
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
                hidden_size = data.get("hidden_size", 256)
                net = TarokNet(hidden_size=hidden_size)
                net.load_state_dict(state_dict)
                return net, data
        return None

    def all_model_paths(self) -> list[Path]:
        """Return paths to all .pt files (pinned + auto)."""
        paths = []
        if self.hof_dir.exists():
            paths.extend(sorted(self.hof_dir.glob("hof_*.pt")))
        if self.pinned_dir.exists():
            paths.extend(sorted(self.pinned_dir.glob("hof_*.pt")))
        return paths

    def set_max_auto(self, max_auto: int) -> None:
        """Update auto limit and evict if needed."""
        self.max_auto = max(1, max_auto)
        self._evict_auto()

    def migrate_existing(self) -> int:
        """Migrate existing manifest entries that lack a 'pinned' field.

        All existing entries default to auto (pinned=False).
        Returns the number of entries migrated.
        """
        manifest = self._load_manifest()
        migrated = 0
        for entry in manifest:
            if "pinned" not in entry:
                entry["pinned"] = False
                migrated += 1
        if migrated > 0:
            self._save_manifest(manifest)
        return migrated
