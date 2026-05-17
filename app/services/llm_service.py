from __future__ import annotations

from os import getenv
from typing import Any

try:
    from transformers import pipeline as hf_pipeline
except ModuleNotFoundError:  # pragma: no cover - depends on installed packages
    hf_pipeline = None

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_HF_MODEL = "Qwen/Qwen2.5-3B-Instruct"
_LEGACY_NON_HF_DEFAULTS = {"gpt-4o-mini"}


class LLMService:
    """Hugging Face Transformers-backed LLM service."""

    def __init__(self) -> None:
        if hf_pipeline is None:
            raise RuntimeError(
                "Hugging Face backend requires transformers. "
                "Install the project dependencies before starting the API."
            )

        settings = get_settings()
        self._settings = settings
        self._provider = "huggingface"
        self._model_name = self._resolve_model_name(settings.llm_model)
        self._task = getenv("HUGGINGFACE_TASK") or self._infer_task(self._model_name)
        self._generator = hf_pipeline(
            self._task,
            model=self._model_name,
            device=0 if settings.embedding_device == "cuda" else -1,
        )

        logger.info(
            "llm_service_ready",
            provider=self._provider,
            model=self._model_name,
            task=self._task,
            runtime="transformers",
        )

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        effective_max_tokens = max_tokens or self._settings.llm_max_tokens
        effective_temperature = (
            temperature if temperature is not None else self._settings.llm_temperature
        )
        prompt = self._build_prompt(system_prompt=system_prompt, user_prompt=user_prompt)

        do_sample = effective_temperature > 0
        kwargs: dict[str, Any] = {
            "max_new_tokens": effective_max_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            kwargs["temperature"] = effective_temperature

        tokenizer = getattr(self._generator, "tokenizer", None)
        if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
            kwargs["pad_token_id"] = tokenizer.eos_token_id

        logger.debug(
            "llm_request",
            provider=self._provider,
            model=self._model_name,
            task=self._task,
            max_tokens=effective_max_tokens,
        )

        outputs = self._generator(prompt, **kwargs)
        text = self._extract_text(outputs)
        if self._task == "text-generation" and text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()

    @staticmethod
    def _resolve_model_name(configured_model: str) -> str:
        env_model = getenv("HUGGINGFACE_MODEL")
        if env_model:
            return env_model

        if configured_model.startswith(("hf:", "huggingface:")):
            return configured_model.split(":", 1)[1]

        if configured_model in _LEGACY_NON_HF_DEFAULTS:
            return _DEFAULT_HF_MODEL

        return configured_model or _DEFAULT_HF_MODEL

    @staticmethod
    def _infer_task(model_name: str) -> str:
        lowered = model_name.lower()
        if any(token in lowered for token in ("t5", "flan", "bart")):
            return "text2text-generation"
        return "text-generation"

    @staticmethod
    def _build_prompt(system_prompt: str, user_prompt: str) -> str:
        return (
            f"System instructions:\n{system_prompt}\n\n"
            f"User request:\n{user_prompt}\n\n"
            "Answer:"
        )

    @staticmethod
    def _extract_text(outputs: Any) -> str:
        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            if isinstance(first, dict):
                return str(
                    first.get("generated_text")
                    or first.get("summary_text")
                    or first.get("text")
                    or ""
                )
        return str(outputs)
