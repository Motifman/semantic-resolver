"""Jev との通信契約と、評価値に応じた採否を保証する。"""

import json

import httpx
import pytest

from semantic_resolver import BackendError, Candidate, Resolved, SemanticResolver, Unresolved
from semantic_resolver.jev import JevBackend, JevPolicy


def answer(request, *, choice="harvest", probability=0.9, confidence=0.8):
    criteria = json.loads(request.content)["questions"]["resolve"]["criteria"]
    sentinel = next(key for key in criteria if key != "harvest")
    return {
        "model": "jev-test-version",
        "answers": {
            "resolve": {
                "type": "choice",
                "choice": choice if choice is not None else sentinel,
                "probabilities": {"harvest": probability, sentinel: 1 - probability},
                "confidence": confidence,
            }
        },
    }


def resolve_with(handler, *, candidates=None, value="gather", context=None, policy=None):
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        backend = JevBackend(api_key="test-key", model="jev-test", client=client)
        resolver = SemanticResolver(
            backend=backend,
            policy=policy or JevPolicy(min_probability=0.8, min_confidence=0.7),
        )
        return resolver.resolve(
            value=value,
            candidates=candidates or [Candidate("harvest", "小麦を収穫する")],
            context=context,
        )


def test_request_carries_input_descriptions_context_and_no_match():
    """元の表現・候補の意味・文脈・対応なしを一回の Choice 要求に載せる。"""
    requests = []

    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        assert body["state"] == {"value": "gather", "context": {"target": "畑"}}
        assert body["questions"]["resolve"]["criteria"]["harvest"] == "小麦を収穫する"
        assert len(body["questions"]["resolve"]["criteria"]) == 2
        assert body["questions"]["resolve"]["type"] == "choice"
        assert body["model"] == "jev-test"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        return httpx.Response(200, json=answer(request))

    result = resolve_with(handler, context={"target": "畑"})
    assert isinstance(result, Resolved)
    assert result.value == "harvest"
    assert result.judgment.model == "jev-test-version"
    assert result.judgment.evidence.probabilities["harvest"] == 0.9
    assert None in result.judgment.evidence.probabilities
    assert result.judgment.latency_ms >= 0
    assert len(requests) == 1


def test_no_match_is_exposed_as_none():
    """Jev が対応なしを選ぶと、内部の選択肢名を返さず no_match を返す。"""
    result = resolve_with(
        lambda r: httpx.Response(200, json=answer(r, choice=None, probability=0.1))
    )
    assert isinstance(result, Unresolved)
    assert result.reason == "no_match"
    assert result.judgment.value is None


def test_internal_no_match_name_does_not_reserve_candidate_values():
    """対応なしと同じ綴りの正式値も、通常の候補として解決できる。"""

    def handler(request):
        criteria = json.loads(request.content)["questions"]["resolve"]["criteria"]
        assert len(criteria) == 2
        probabilities = {key: float(key == "__no_match__") for key in criteria}
        return httpx.Response(
            200,
            json={
                "model": "test",
                "answers": {
                    "resolve": {
                        "type": "choice",
                        "choice": "__no_match__",
                        "probabilities": probabilities,
                        "confidence": 1.0,
                    }
                },
            },
        )

    result = resolve_with(handler, candidates=[Candidate("__no_match__", "通常の操作")])
    assert isinstance(result, Resolved)
    assert result.value == "__no_match__"


@pytest.mark.parametrize(
    "probability,confidence,reason",
    [
        (0.8, 0.7, None),
        (0.79, 0.9, "below_threshold"),
        (0.9, 0.69, "below_threshold"),
        (0.5, 1.0, "ambiguous"),
    ],
)
def test_policy_checks_each_threshold_and_ties(probability, confidence, reason):
    """両閾値の境界は採用し、片方が未達なら棄却、同率なら曖昧として棄却する。"""
    result = resolve_with(
        lambda r: httpx.Response(
            200, json=answer(r, probability=probability, confidence=confidence)
        )
    )
    if reason is None:
        assert isinstance(result, Resolved)
    else:
        assert isinstance(result, Unresolved)
        assert result.reason == reason
    assert result.assessment.thresholds == {"min_probability": 0.8, "min_confidence": 0.7}


@pytest.mark.parametrize(
    "field,value",
    [
        ("confidence", None),
        ("confidence", True),
        ("confidence", "0.9"),
        ("confidence", float("nan")),
        ("confidence", float("inf")),
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("probabilities", {}),
        ("probabilities", {"harvest": 0.9}),
        ("probabilities", {"harvest": 0.9, "__no_match__": 0.9}),
        ("probabilities", {"harvest": True, "__no_match__": 0}),
        ("probabilities", {"harvest": 1.1, "__no_match__": -0.1}),
        ("probabilities", {"harvest": 0.2, "__no_match__": 0.8}),
        ("choice", "invented"),
        ("choice", None),
        ("type", "score"),
    ],
)
def test_invalid_answer_is_not_treated_as_a_decision(field, value):
    """欠落・型違い・不正な確率・候補外の応答を障害として拒否する。"""

    def handler(request):
        payload = answer(request)
        payload["answers"]["resolve"][field] = value
        return httpx.Response(200, content=json.dumps(payload))

    with pytest.raises(BackendError) as exc:
        resolve_with(handler)
    assert exc.value.code == "invalid_response"


@pytest.mark.parametrize("payload", [[], {}, {"answers": {}}, {"model": "test", "answers": []}])
def test_missing_answer_is_backend_error(payload):
    """回答そのものが欠けた応答を、対応なしとして扱わない。"""
    with pytest.raises(BackendError):
        resolve_with(lambda r: httpx.Response(200, json=payload))


@pytest.mark.parametrize("status", [400, 401, 429, 500, 529])
def test_http_errors_do_not_expose_body_or_retry(status):
    """HTTP 障害は状態コードを残し、応答本文を載せず、自動再試行もしない。"""
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="private-state test-key")

    with pytest.raises(BackendError) as exc:
        resolve_with(handler)
    assert exc.value.code == "http_error"
    assert exc.value.status_code == status
    assert "private-state" not in str(exc.value)
    assert "test-key" not in str(exc.value)
    assert len(calls) == 1


def test_timeout_is_transport_error():
    """タイムアウトを棄却と区別し、呼び出し側に再試行判断を戻す。"""

    def handler(request):
        raise httpx.ReadTimeout("private-state", request=request)

    with pytest.raises(BackendError) as exc:
        resolve_with(handler)
    assert exc.value.code == "transport_error"
    assert "private-state" not in str(exc.value)


def test_malformed_json_is_backend_error():
    """JSON でない成功応答でも、既定値へ置き換えず障害として通知する。"""
    with pytest.raises(BackendError):
        resolve_with(lambda r: httpx.Response(200, text="not json"))


@pytest.mark.parametrize("value", [0, -1, 1.1, True, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["min_probability", "min_confidence"])
def test_invalid_thresholds_are_rejected(field, value):
    """閾値は有限な (0, 1] の数値に限り、真偽値も拒否する。"""
    kwargs = {"min_probability": 0.8, "min_confidence": 0.7, field: value}
    with pytest.raises(ValueError):
        JevPolicy(**kwargs)


def test_too_many_candidates_fail_before_http():
    """対応なしを含め255件を超える要求は、通信する前に拒否する。"""

    def handler(request):
        pytest.fail("HTTP must not be called")

    with pytest.raises(ValueError, match="254"):
        resolve_with(handler, candidates=[Candidate(str(i), "説明") for i in range(255)])


def test_injected_client_is_not_closed_by_backend():
    """利用者が渡した HTTP クライアントは、判断器を閉じても利用可能なままにする。"""
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))) as client:
        with JevBackend(api_key="test-key", model="test", client=client):
            pass
        assert not client.is_closed


def test_extreme_numeric_response_is_backend_error():
    """浮動小数に収まらない応答値も、生の数値変換例外にせず障害として通知する。"""

    def handler(request):
        payload = answer(request)
        payload["answers"]["resolve"]["confidence"] = 10**1000
        return httpx.Response(200, json=payload)

    with pytest.raises(BackendError) as exc:
        resolve_with(handler)
    assert exc.value.code == "invalid_response"


@pytest.mark.parametrize("context", [{"value": object()}, {"value": float("nan")}])
def test_invalid_context_fails_before_http(context):
    """JSON にできない値や非有限値を含む文脈は、判断前に入力不備として拒否する。"""

    def handler(request):
        pytest.fail("HTTP must not be called")

    with pytest.raises(ValueError, match="context"):
        resolve_with(handler, context=context)


def test_maximum_candidate_count_leaves_room_for_no_match():
    """254件の候補は対応なしを加えた255件として通信でき、正しい正式値を返す。"""

    def handler(request):
        criteria = json.loads(request.content)["questions"]["resolve"]["criteria"]
        assert len(criteria) == 255
        return httpx.Response(
            200,
            json={
                "model": "test",
                "answers": {
                    "resolve": {
                        "type": "choice",
                        "choice": "candidate_253",
                        "probabilities": {key: float(key == "candidate_253") for key in criteria},
                        "confidence": 1.0,
                    }
                },
            },
        )

    result = resolve_with(
        handler, candidates=[Candidate(f"candidate_{i}", "説明") for i in range(254)]
    )
    assert isinstance(result, Resolved)
    assert result.value == "candidate_253"
