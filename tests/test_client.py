import json

import httpx
import pytest

from symbolic_regression.decision.client import (
    DecisionCapabilityError,
    DecisionResponseError,
    ModelNotFoundError,
    OllamaClient,
    OllamaConnectionError,
    OllamaTimeoutError,
)

MODEL = "d3:4b"


def make_client(handler, retries: int = 0) -> OllamaClient:
    return OllamaClient(
        base_url="http://testserver",
        retries=retries,
        transport=httpx.MockTransport(handler),
    )


def tags_response(capabilities=None):
    model = {"name": MODEL, "model": MODEL}
    if capabilities is not None:
        model["capabilities"] = capabilities
    return httpx.Response(200, json={"models": [model]})


def choice_response(choice="sin", probabilities=None, confidence=0.9):
    probabilities = probabilities or {"sin": 0.8, "cos": 0.2}
    return httpx.Response(
        200,
        json={
            "model": MODEL,
            "answers": {
                "operator": {
                    "type": "choice",
                    "choice": choice,
                    "probabilities": probabilities,
                    "confidence": confidence,
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 0},
        },
    )


def test_readiness_accepts_decision_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return tags_response(["decision", "vision"])

    with make_client(handler) as client:
        info = client.readiness(MODEL)
    assert info.name == MODEL
    assert "decision" in info.capabilities


def test_readiness_rejects_model_without_decision_capability() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return tags_response(["completion"])

    with make_client(handler) as client:
        with pytest.raises(DecisionCapabilityError, match="decision"):
            client.readiness(MODEL)


def test_readiness_falls_back_to_show() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return tags_response(None)
        assert request.url.path == "/api/show"
        assert json.loads(request.read())["model"] == MODEL
        return httpx.Response(200, json={"capabilities": ["decision"]})

    with make_client(handler) as client:
        info = client.readiness(MODEL)
    assert "decision" in info.capabilities


def test_readiness_reports_missing_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": []})

    with make_client(handler) as client:
        with pytest.raises(ModelNotFoundError, match="not installed"):
            client.readiness(MODEL)


def test_unreachable_server_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with make_client(handler) as client:
        with pytest.raises(OllamaConnectionError, match="Could not connect"):
            client.readiness(MODEL)


def test_timeout_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow", request=request)

    with make_client(handler) as client:
        with pytest.raises(OllamaTimeoutError, match="Timed out"):
            client.readiness(MODEL)


def test_retry_then_success() -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise httpx.TimeoutException("slow", request=request)
        return tags_response(["decision"])

    with make_client(handler, retries=1) as client:
        info = client.readiness(MODEL)
    assert info.name == MODEL
    assert attempts["count"] == 2


def test_choice_request_and_parse() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["payload"] = json.loads(request.read())
        return choice_response("sin", {"sin": 0.7, "cos": 0.3})

    state = {"step": 0, "expression": "x"}
    criteria = {"sin": "sine", "cos": "cosine"}
    with make_client(handler) as client:
        result = client.choice(MODEL, state, "choose", criteria)
    assert seen["path"] == "/v1/systemone"
    assert seen["payload"]["model"] == MODEL
    assert list(seen["payload"]["questions"]["operator"]["criteria"]) == ["sin", "cos"]
    assert result.selected == "sin"
    assert result.probabilities == {"sin": 0.7, "cos": 0.3}
    assert result.confidence == 0.9


def test_choice_rejects_unknown_selected_candidate() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return choice_response("exp", {"sin": 0.5, "cos": 0.5})

    with make_client(handler) as client:
        with pytest.raises(DecisionResponseError, match="not one"):
            client.choice(MODEL, {}, "choose", {"sin": "sine", "cos": "cosine"})


def test_choice_rejects_missing_probability() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return choice_response("sin", {"sin": 1.0})

    with make_client(handler) as client:
        with pytest.raises(DecisionResponseError, match="missing probabilities"):
            client.choice(MODEL, {}, "choose", {"sin": "sine", "cos": "cosine"})


def test_malformed_json_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    with make_client(handler) as client:
        with pytest.raises(DecisionResponseError, match="not valid JSON"):
            client.readiness(MODEL)
