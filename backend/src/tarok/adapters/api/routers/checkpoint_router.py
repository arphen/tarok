"""Checkpoint discovery and metadata endpoint."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

router = APIRouter(tags=["checkpoints"])

# Checkpoint metadata cache: {filepath_str: (mtime, metadata_dict)}
_checkpoint_cache: dict[str, tuple[float, dict]] = {}


def _load_checkpoint_meta(fpath: Path, ckpt_dir: Path) -> dict:
    """Load checkpoint metadata, using cache when file hasn't changed."""
    import torch as _torch

    fstr = str(fpath)
    mtime = fpath.stat().st_mtime
    cached = _checkpoint_cache.get(fstr)
    if cached and cached[0] == mtime:
        return cached[1]

    fname = fpath.name

    try:
        meta = _torch.load(fpath, map_location="cpu", weights_only=False)
        entry = {
            "filename": fname,
            "episode": meta.get("episode", 0),
            "session": meta.get("session", 0),
            "win_rate": meta.get("metrics", {}).get("win_rate", 0),
            "avg_reward": meta.get("metrics", {}).get("avg_reward", 0),
            "model_name": meta.get("model_name", None),
        }
    except Exception:
        entry = {"filename": fname, "episode": 0}

    _checkpoint_cache[fstr] = (mtime, entry)
    return entry


@router.get("/api/checkpoints")
async def list_checkpoints():
    """List checkpoints from the top-level checkpoints/ directory.

    - HOF: checkpoints/hall_of_fame/*.pt  (manually curated, committed to git)
    - Persona models: checkpoints/{PersonaName}/_current.pt and optional _test.pt
    """
    import torch as _torch

    root_ckpt_dir = Path("../data/checkpoints")
    legacy_ckpt_dirs = [Path("checkpoints"), Path("../checkpoints")]
    hof_dir = root_ckpt_dir / "hall_of_fame"
    result = []
    seen_filenames: set[str] = set()

    if not root_ckpt_dir.exists():
        # Still return legacy checkpoints even when the new canonical dir is absent.
        for legacy_dir in legacy_ckpt_dirs:
            if not legacy_dir.exists():
                continue
            for f in sorted(legacy_dir.glob("*.pt")):
                if f.name in seen_filenames:
                    continue
                seen_filenames.add(f.name)
                result.append(_load_checkpoint_meta(f, legacy_dir))
        return {"checkpoints": result}

    # 0. Legacy flat checkpoint dirs (backward compatibility for older tests/tools)
    for legacy_dir in legacy_ckpt_dirs:
        if not legacy_dir.exists():
            continue
        for f in sorted(legacy_dir.glob("*.pt")):
            if f.name in seen_filenames:
                continue
            seen_filenames.add(f.name)
            result.append(_load_checkpoint_meta(f, legacy_dir))

    # 1. HOF files (manually placed, committed to git)
    hof_files = sorted(hof_dir.glob("*.pt")) if hof_dir.exists() else []
    for f in hof_files:
        try:
            meta = _torch.load(f, map_location="cpu", weights_only=False)
            model_name = meta.get("model_name") or meta.get("display_name") or f.stem
            result.append(
                {
                    "filename": f"hall_of_fame/{f.name}",
                    "persona": meta.get("persona") or f.stem,
                    "model_name": model_name,
                    "episode": meta.get("episode", 0),
                    "session": meta.get("session", 0),
                    "win_rate": meta.get("metrics", {}).get("win_rate", 0),
                    "avg_reward": meta.get("metrics", {}).get("avg_reward", 0),
                    "is_hof": True,
                }
            )
            seen_filenames.add(f.name)
        except Exception:
            result.append({"filename": f"hall_of_fame/{f.name}", "is_hof": True, "episode": 0})
            seen_filenames.add(f.name)

    # 2. Persona subdirectories — _current.pt and optional _test.pt
    for persona_dir in sorted(root_ckpt_dir.iterdir()):
        if not persona_dir.is_dir() or persona_dir.name == "hall_of_fame":
            continue
        persona_name = persona_dir.name
        for basename in ("_current.pt", "_test.pt"):
            ckpt_file = persona_dir / basename
            if not ckpt_file.exists():
                continue
            rel = f"{persona_name}/{basename}"
            if rel in seen_filenames:
                continue
            seen_filenames.add(rel)
            try:
                meta = _torch.load(ckpt_file, map_location="cpu", weights_only=False)
                model_name = meta.get("model_name") or meta.get("display_name") or persona_name
                if basename == "_test.pt":
                    model_name = f"{model_name} (test)"
                result.append(
                    {
                        "filename": rel,
                        "persona": persona_name,
                        "model_name": model_name,
                        "episode": meta.get("episode", 0),
                        "session": meta.get("session", 0),
                        "win_rate": meta.get("metrics", {}).get("win_rate", 0),
                        "avg_reward": meta.get("metrics", {}).get("avg_reward", 0),
                        "is_hof": False,
                    }
                )
            except Exception:
                result.append(
                    {
                        "filename": rel,
                        "persona": persona_name,
                        "model_name": f"{persona_name} (test)"
                        if basename == "_test.pt"
                        else persona_name,
                        "is_hof": False,
                        "episode": 0,
                    }
                )

    return {"checkpoints": result}
