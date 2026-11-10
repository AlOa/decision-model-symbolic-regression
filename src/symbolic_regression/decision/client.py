"""HTTP client for Ollama's SystemOne decision API."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import httpx


class OllamaError(RuntimeError):
    """Base error for Ollama interactions."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama server cannot be reached."""


class OllamaTimeoutError(OllamaError):
    """Raised when an Ollama request times out."""


class ModelNotFoundError(OllamaError):
    """Raised when the requested model is not installed."""


class DecisionCapabilityError(OllamaError):
    """Raised when the model does not advertise the decision capability."""


class DecisionResponseError(OllamaError):
    """Raised when a SystemOne response is malformed or incomplete."""


@dataclass(frozen=True, slots=True)
class ModelInfo:
    name: str
    capabilities: tuple[str, ...]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SystemOneChoice:
    selected: str
    probabilities: dict[str, float]
    confidence: float | None
    raw: dict[str, Any]


class OllamaClient:
    """Small synchronous client for ``/api/tags`` and ``/v1/systemone``."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        retries: int = 1,
        transport: httpx.BaseTransport | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if retries < 0:
            raise ValueError("retries must be non-negative")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            transport=transport,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OllamaClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        last_error: Exception | None = None
        for _attempt in range(self.retries + 1):
            try:
                response = self._client.request(method, path, json=payload)
            except httpx.TimeoutException as exc:
                last_error = OllamaTimeoutError(
                    f"Timed out after {self.timeout:g}s calling {self.base_url}{path}"
                )
                continue
            except httpx.ConnectError as exc:
                last_error = OllamaConnectionError(
                    f"Could not connect to Ollama at {self.base_url}. "
                    "Check that the server is running and the base URL is correct."
                )
                continue
            except httpx.HTTPError as exc:
                last_error = OllamaError(f"HTTP error calling {path}: {exc}")
                continue
            if response.status_code >= 400:
                message = _error_message(response)
                if response.status_code == 404:
                    raise ModelNotFoundError(message)
                raise OllamaError(f"HTTP {response.status_code} calling {path}: {message}")
            try:
                data = response.json()
            except ValueError as exc:
                raise DecisionResponseError(
                    f"Response from {path} is not valid JSON: {response.text[:200]!r}"
                ) from exc
            if not isinstance(data, dict):
                raise DecisionResponseError(f"Response from {path} is not a JSON object")
            return data
        if last_error is not None:
            raise last_error
        raise OllamaError(f"Request to {path} failed")

    def list_models(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/api/tags")
        models = payload.get("models")
        if not isinstance(models, list):
            raise DecisionResponseError("Ollama /api/tags response is missing the models list")
        return [model for model in models if isinstance(model, dict)]

    def readiness(self, model: str) -> ModelInfo:
        """Verify that ``model`` exists and advertises the decision capability."""

        models = self.list_models()
        match = next(
            (
                candidate
                for candidate in models
                if candidate.get("name") == model or candidate.get("model") == model
            ),
            None,
        )
        if match is None:
            available = ", ".join(
                str(candidate.get("name") or candidate.get("model")) for candidate in models
            )
            raise ModelNotFoundError(
                f"Model {model!r} is not installed. Available models: {available or 'none'}"
            )
        capabilities = match.get("capabilities")
        if not isinstance(capabilities, list) or not capabilities:
            show = self._request("POST", "/api/show", {"model": model})
            capabilities = show.get("capabilities", [])
        capabilities = tuple(str(capability) for capability in capabilities)
        if "decision" not in capabilities:
            raise DecisionCapabilityError(
                f"Model {model!r} does not advertise the 'decision' capability. "
                "Use a decision model such as d3:4b."
            )
        return ModelInfo(
            name=str(match.get("name") or match.get("model") or model),
            capabilities=capabilities,
            details=dict(match.get("details") or {}),
        )

    def systemone(
        self,
        model: str,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        keep_alive: str | None = None,
    ) -> dict[str, Any]:
        """Call ``POST /v1/systemone`` and return the raw response."""

        if not questions:
            raise ValueError("At least one SystemOne question is required")
        payload: dict[str, Any] = {
            "model": model,
            "state": dict(state),
            "questions": dict(questions),
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        data = self._request("POST", "/v1/systemone", payload)
        if not isinstance(data.get("answers"), dict):
            raise DecisionResponseError("SystemOne response is missing the answers object")
        return data

    def choice(
        self,
        model: str,
        state: Mapping[str, Any],
        instructions: str,
        criteria: Mapping[str, str],
        question_id: str = "operator",
        keep_alive: str | None = None,
    ) -> SystemOneChoice:
        """Ask one ``choice`` question and validate the returned distribution."""

        if len(criteria) < 2:
            raise DecisionResponseError("A choice question needs at least two candidates")
        questions = {
            question_id: {
                "type": "choice",
                "instructions": instructions,
                "criteria": dict(criteria),
            }
        }
        data = self.systemone(model, state, questions, keep_alive=keep_alive)
        answer = data["answers"].get(question_id)
        if not isinstance(answer, dict):
            raise DecisionResponseError(
                f"SystemOne response is missing the answer for question {question_id!r}"
            )
        answer_type = answer.get("type")
        if answer_type not in (None, "choice"):
            raise DecisionResponseError(
                f"Expected a choice answer, got {answer_type!r}"
            )
        selected = answer.get("choice")
        if not isinstance(selected, str) or selected not in criteria:
            raise DecisionResponseError(
                f"Selected candidate {selected!r} is not one of {list(criteria)}"
            )
        raw_probabilities = answer.get("probabilities")
        if not isinstance(raw_probabilities, dict):
            raise DecisionResponseError("Choice answer is missing the probabilities object")
        probabilities: dict[str, float] = {}
        missing = [name for name in criteria if name not in raw_probabilities]
        if missing:
            raise DecisionResponseError(
                f"Choice answer is missing probabilities for: {', '.join(missing)}"
            )
        for name in criteria:
            value = raw_probabilities[name]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise DecisionResponseError(
                    f"Probability for {name!r} is not a finite number: {value!r}"
                )
            probabilities[name] = float(value)
        confidence = answer.get("confidence")
        if confidence is not None:
            if not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
                raise DecisionResponseError(f"Confidence is not a finite number: {confidence!r}")
            confidence = float(confidence)
        return SystemOneChoice(
            selected=selected,
            probabilities=probabilities,
            confidence=confidence,
            raw=dict(answer),
        )


def _error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300] or f"HTTP {response.status_code}"
    if isinstance(payload, dict) and "error" in payload:
        return str(payload["error"])
    return json.dumps(payload)[:300]
