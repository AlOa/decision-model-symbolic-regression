"""Operator-selection policies backed by decision models or deterministic fallbacks."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from typing import Mapping, Protocol, Sequence

from ..context import DecisionContext
from ..expression import OPERATORS
from .client import OllamaClient, OllamaError

DEFAULT_OPERATORS: tuple[str, ...] = tuple(OPERATORS)

DEFAULT_INSTRUCTIONS = (
    "Choose the operator that should be added next to the current expression "
    "to reduce the residual on the target data. Consider the current expression, "
    "the sampled feature and target values, and the residual values. Prefer an "
    "operator that can move the prediction toward the target."
)


@dataclass(frozen=True, slots=True)
class PolicyResult:
    probabilities: dict[str, float]
    selected: str | None
    source: str
    confidence: float | None = None
    model: str | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0

    def to_trace(self) -> dict[str, object]:
        return {
            "source": self.source,
            "model": self.model,
            "selected": self.selected,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "error": self.error,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        }


class OperatorPolicy(Protocol):
    name: str

    def probabilities(self, context: DecisionContext) -> PolicyResult: ...


@dataclass(frozen=True, slots=True)
class OperatorCatalog:
    names: tuple[str, ...] = DEFAULT_OPERATORS

    def __post_init__(self) -> None:
        if len(self.names) < 1:
            raise ValueError("An operator catalogue needs at least one operator")
        unknown = [name for name in self.names if name not in OPERATORS]
        if unknown:
            raise ValueError(f"Unknown operators in catalogue: {', '.join(unknown)}")

    def criteria(self) -> dict[str, str]:
        return {name: OPERATORS[name].description for name in self.names}


def _uniform(names: Sequence[str]) -> dict[str, float]:
    weight = 1.0 / len(names)
    return {name: weight for name in names}


def _normalize(probabilities: Mapping[str, float], names: Sequence[str]) -> dict[str, float]:
    cleaned = {name: max(0.0, float(probabilities.get(name, 0.0))) for name in names}
    total = sum(cleaned.values())
    if total <= 0 or not math.isfinite(total):
        return _uniform(names)
    return {name: value / total for name, value in cleaned.items()}


def select_highest(
    probabilities: Mapping[str, float], names: Sequence[str], tolerance: float = 1e-9
) -> str:
    """Select the highest-probability candidate, resolving ties by catalogue order."""

    if not probabilities:
        raise ValueError("Cannot select from an empty probability distribution")
    best = max(float(probabilities.get(name, 0.0)) for name in names)
    for name in names:
        if best - float(probabilities.get(name, 0.0)) <= tolerance:
            return name
    return next(iter(probabilities))


def _cache_key(model: str, state: Mapping[str, object], criteria: Mapping[str, str]) -> str:
    payload = json.dumps(
        {"model": model, "state": state, "criteria": criteria},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OllamaOperatorPolicy:
    """Ask an Ollama decision model to rank operators for the current context."""

    name = "decision"

    def __init__(
        self,
        client: OllamaClient,
        model: str,
        operators: Sequence[str] | None = None,
        instructions: str = DEFAULT_INSTRUCTIONS,
        allow_fallback: bool = True,
        cache: bool = True,
        question_id: str = "operator",
        keep_alive: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.catalog = OperatorCatalog(tuple(operators) if operators else DEFAULT_OPERATORS)
        if len(self.catalog.names) < 2:
            raise ValueError("The Ollama decision policy needs at least two operators")
        self.instructions = instructions
        self.allow_fallback = allow_fallback
        self.cache = cache
        self.question_id = question_id
        self.keep_alive = keep_alive
        self._cache: dict[str, PolicyResult] = {}

    def probabilities(self, context: DecisionContext) -> PolicyResult:
        criteria = self.catalog.criteria()
        state = context.to_dict()
        key = _cache_key(self.model, state, criteria)
        if self.cache and key in self._cache:
            return replace(self._cache[key], source="cache", elapsed_seconds=0.0)
        started = time.perf_counter()
        try:
            choice = self.client.choice(
                model=self.model,
                state=state,
                instructions=self.instructions,
                criteria=criteria,
                question_id=self.question_id,
                keep_alive=self.keep_alive,
            )
        except OllamaError as exc:
            if not self.allow_fallback:
                raise
            names = self.catalog.names
            return PolicyResult(
                probabilities=_uniform(names),
                selected=names[0],
                source="fallback",
                confidence=None,
                model=self.model,
                error=str(exc),
                elapsed_seconds=time.perf_counter() - started,
            )
        probabilities = _normalize(choice.probabilities, self.catalog.names)
        result = PolicyResult(
            probabilities=probabilities,
            selected=select_highest(probabilities, self.catalog.names),
            source="ollama",
            confidence=choice.confidence,
            model=self.model,
            elapsed_seconds=time.perf_counter() - started,
        )
        if self.cache:
            self._cache[key] = result
        return result


class UniformOperatorPolicy:
    """Assign equal probability to every operator."""

    name = "uniform"

    def __init__(self, operators: Sequence[str] | None = None) -> None:
        self.catalog = OperatorCatalog(tuple(operators) if operators else DEFAULT_OPERATORS)

    def probabilities(self, context: DecisionContext) -> PolicyResult:
        names = self.catalog.names
        return PolicyResult(
            probabilities=_uniform(names),
            selected=names[0],
            source="uniform",
            confidence=None,
        )


class OfflineOperatorPolicy:
    """Deterministic pseudo-random policy that never contacts a model."""

    name = "offline"

    def __init__(self, seed: int = 0, operators: Sequence[str] | None = None) -> None:
        self.seed = seed
        self.catalog = OperatorCatalog(tuple(operators) if operators else DEFAULT_OPERATORS)

    def probabilities(self, context: DecisionContext) -> PolicyResult:
        names = self.catalog.names
        scores: dict[str, float] = {}
        for name in names:
            digest = hashlib.sha256(
                f"{self.seed}:{context.step}:{context.expression}:{name}".encode("utf-8")
            ).digest()
            score = int.from_bytes(digest[:8], "big") / 2**64
            scores[name] = score + 1e-12
        probabilities = _normalize(scores, names)
        return PolicyResult(
            probabilities=probabilities,
            selected=select_highest(probabilities, names),
            source="offline",
            confidence=None,
        )


def make_policy(
    kind: str,
    *,
    client: OllamaClient | None = None,
    model: str = "d3:4b",
    seed: int = 0,
    operators: Sequence[str] | None = None,
) -> OperatorPolicy:
    normalized = kind.lower()
    if normalized == "decision":
        if client is None:
            raise ValueError("The decision policy requires an Ollama client")
        return OllamaOperatorPolicy(client=client, model=model, operators=operators)
    if normalized == "uniform":
        return UniformOperatorPolicy(operators)
    if normalized == "offline":
        return OfflineOperatorPolicy(seed=seed, operators=operators)
    raise ValueError(f"Unknown policy {kind!r}; expected decision, uniform, or offline")
