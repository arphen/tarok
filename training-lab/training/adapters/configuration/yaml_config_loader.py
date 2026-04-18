"""Adapter: YAML config file loader."""

from __future__ import annotations

from typing import Any

import yaml

from training.ports import ConfigPort


class YAMLConfigLoader(ConfigPort):
    def load(self, path: str) -> dict[str, Any]:
        with open(path) as f:
            return yaml.safe_load(f)
