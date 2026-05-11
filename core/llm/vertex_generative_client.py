"""Optional real Vertex AI ``GenerativeModel`` client (Gemini 1.5 Flash).

Used only when ``RAG_LLM_BACKEND=vertex`` is set AND the ``vertex`` extra is
installed (``pip install '.[vertex]'``). Production code only sees the
:class:`LLMClient` Protocol, so this module is *truly* optional: the import
is lazy so unit tests don't pull in google-cloud-aiplatform.
"""

from __future__ import annotations

import os

from core.utils.logger import logger

from .llm_client import LLMClient, LLMResponse


class VertexGenerativeClient(LLMClient):
    """Thin wrapper around ``vertexai.generative_models.GenerativeModel``."""

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not self._project:
            raise RuntimeError(
                "VertexGenerativeClient requires GOOGLE_CLOUD_PROJECT to be set."
            )
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "google-cloud-aiplatform is not installed. "
                "Reinstall with: pip install '.[vertex]'"
            ) from exc

        vertexai.init(project=self._project, location=self._location)
        self._model = GenerativeModel(self._model_name)
        logger.info(
            "VertexGenerativeClient ready (model={}, project={}, location={}).",
            self._model_name,
            self._project,
            self._location,
        )

    def generate(self, prompt: str) -> LLMResponse:
        response = self._model.generate_content(prompt)
        text = getattr(response, "text", None) or _extract_text(response)
        return LLMResponse(text=text or "", model=self._model_name)

    @property
    def model_name(self) -> str:
        return self._model_name


def _extract_text(response: object) -> str:  # pragma: no cover - real SDK path
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []):
            text = getattr(part, "text", None)
            if text:
                return str(text)
    return ""


__all__ = ["VertexGenerativeClient"]
