#!/usr/local/bin/python3
"""
claude_model_manager.py

Reusable model validation/selection/cache manager for Anthropic clients.
Can be imported by other scripts or run standalone.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import tomllib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anthropic
from configuration_manager_class import ConfigurationManager
from logger import logger    

DEFAULT_REFRESH_DAYS = 7

class ModelSelectionError(Exception):
    """Raised when a model cannot be validated or selected."""


class ClaudeModelManager:
    """Resolve, validate, and cache Anthropic model selection."""

    def __init__(self, config_manager: ConfigurationManager) -> None:
        self.config_manager = config_manager
        self.config = self.config_manager.get_class_config(self.__class__.__name__)
        self.client = anthropic.Anthropic()
        self.model   = self.config["model"]
        self.max_tokens = self.config["max_tokens"]                         # short structured answer; no need for more
        self.temperature = self.config["temperature"]                        # deterministic — classification task
        self.model_recheck_days = self.config["model_recheck_days"]                   # only re-validate/auto-select model once every N days
        self.model_cache_file = self.config["model_cache_file"]  # stores last validated model + timestamp

    def resolve_model(self) -> str:
        """Validate configured model or auto-select one, then update cache and cfg."""
        configured_model = str(self.model).strip()
        configured_from_cache = False
        refresh_days = int(self.model_recheck_days or DEFAULT_REFRESH_DAYS)
        refresh_days = max(1, refresh_days)

        cache_file = self.model_cache_file
        cache_data = self._load_model_cache()
        cached_model = str(cache_data.get("model", "")).strip()
        checked_at = self._parse_checked_at(str(cache_data.get("checked_at", "")).strip())
        now_utc = datetime.now(timezone.utc)
        cache_is_fresh = bool(
            cached_model
            and checked_at
            and (now_utc - checked_at) < timedelta(days=refresh_days)
        )

        # If user changes configured model, force immediate re-check.
        if configured_model and cached_model and configured_model != cached_model:
            cache_is_fresh = False

        if cache_is_fresh and (configured_model or cached_model):
            selected = configured_model or cached_model
            self.model = selected
            logger.info(
                f"[info] Reusing cached model '{selected}' "
                f"(last checked {checked_at.date()}, recheck every {refresh_days}d)."
            )
            return selected

        if not configured_model and cached_model:
            configured_model = cached_model
            self.model = configured_model
            configured_from_cache = True

        try:
            models_response = self.client.models.list(limit=200)
            available_models = self._extract_model_ids(models_response)
        except Exception as e:
            available_models = []
            logger.info(f"[warn] Could not list models from API: {e}")

        if available_models and configured_model and configured_model   in available_models:
            self._save_model_cache(configured_model)
            return configured_model

        if available_models and configured_model and configured_model not in available_models:
            if configured_from_cache:
                selected = self._auto_select_model(available_models)
                if selected:
                    self.model = selected
                    self._save_model_cache(selected)
                    logger.info(f"[warn] Cached model unavailable. Auto-selected model: {selected}")
                    return selected
            suggestions = self._build_model_suggestions(configured_model, available_models)
            hint = "\n".join(f"  - {s}" for s in suggestions) if suggestions else "  (none available)"
            raise ModelSelectionError(
                f"Configured model not found: {configured_model}\n"
                f"Try one of these available models:\n{hint}"
            )

        if available_models and not configured_model:
            selected = self._auto_select_model(available_models)
            if not selected:
                raise ModelSelectionError("Could not auto-select a model from available model list.")
            self.model = selected
            self._save_model_cache(selected)
            logger.info(f"[info] Auto-selected model: {selected}")
            return selected

        if not configured_model:
            # No configured model and listing unavailable: use sensible fallback and probe.
            fallback_model = "claude-sonnet-4-6"
            self.model = fallback_model
            configured_model = fallback_model
            logger.info(f"[warn] Could not list models; using fallback model: {fallback_model}")

        # If list endpoint is unavailable, do a small probe call.
        try:
            self.client.messages.create(
                model=configured_model,
                max_tokens=1,
                temperature=0,
                system="Model validation check.",
                messages=[{"role": "user", "content": "ping"}],
            )
            self._save_model_cache(configured_model)
            return configured_model
        except anthropic.APIError as e:
            raise ModelSelectionError(
                f"Model validation failed for '{configured_model}': {e}\n"
                "Update [claude].model in config to a valid model available to your API key."
            ) from e

    def _extract_model_ids(self, models_response) -> list[str]:
        """Extract model IDs from Anthropic models.list() response shapes."""
        ids: list[str] = []

        data = getattr(models_response, "data", None)
        if data is not None:
            for model in data:
                model_id = getattr(model, "id", None)
                if model_id:
                    ids.append(str(model_id))
            return ids

        try:
            for model in models_response:
                model_id = getattr(model, "id", None)
                if model_id:
                    ids.append(str(model_id))
        except TypeError:
            pass

        return ids

    def _build_model_suggestions(self, configured_model: str, available_models: list[str]) -> list[str]:
        if not available_models:
            return []

        close = difflib.get_close_matches(configured_model, available_models, n=5, cutoff=0.4)
        if close:
            return close

        preferred = [
            m for m in available_models
            if any(k in m.lower() for k in ["sonnet", "haiku", "opus", "claude-4"])
        ]
        return preferred[:5] if preferred else available_models[:5]

    def _auto_select_model(self, available_models: list[str]) -> str | None:
        """Pick a balanced default model from available IDs (quality vs cost/runtime)."""
        if not available_models:
            return None

        preference_patterns = [
            "claude-sonnet-4-6",
            "claude-sonnet-4",
            "sonnet",
            "haiku",
        ]

        for pattern in preference_patterns:
            for model in available_models:
                if pattern in model.lower():
                    return model

        return available_models[0]


    def _load_model_cache(self) -> dict:
        cache_file = Path(self.model_cache_file)
        if not cache_file.exists():
            return {}
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_model_cache(self, model: str) -> None:
        cache_file = Path(self.model_cache_file)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": model,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _parse_checked_at(self, value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed.astimezone(timezone.utc)
        except ValueError:
            return None


def main(cmd_line: list[str]) -> str:
    config_manager = ConfigurationManager(cmd_line)
    model_manager = ClaudeModelManager(config_manager)
    
    try:
        selected = model_manager.resolve_model()
        print(f"Selected model: {selected}")
        return "---\nDone!"
    except ModelSelectionError as e:
        return f"[error] {e}"
    

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
