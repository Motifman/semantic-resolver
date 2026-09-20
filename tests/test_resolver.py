"""意味判断を呼ぶ境界と、採用・棄却・障害の区別を保証する。"""

import pytest

from semantic_resolver import (
    Assessment,
    BackendError,
    Candidate,
    Judgment,
    Resolved,
    SemanticResolver,
    Unresolved,
)


class RecordingBackend:
    def __init__(self, value="harvest"):
        self.value = value
        self.requests = []

    def judge(self, request):
        self.requests.append(request)
        return Judgment(
            value=self.value, evidence="evidence", backend="fake", model="test", latency_ms=1
        )


class FixedPolicy:
    def __init__(self, reason="accepted"):
        self.reason = reason

    def assess(self, judgment):
        return Assessment(reason=self.reason, thresholds={"test_threshold": 0.8})


OPTIONS = [Candidate("harvest", "小麦を収穫する")]


def test_exact_input_still_calls_backend():
    """正式値と完全一致しても意味判断を一度行い、その結果と採用基準を返す。"""
    backend = RecordingBackend()
    result = SemanticResolver(backend=backend, policy=FixedPolicy()).resolve(
        value="harvest", candidates=OPTIONS, context={"intent": "食料を集める"}
    )
    assert isinstance(result, Resolved)
    assert result.value == "harvest"
    assert result.original == "harvest"
    assert result.judgment.evidence == "evidence"
    assert result.assessment.thresholds == {"test_threshold": 0.8}
    assert len(backend.requests) == 1
    assert backend.requests[0].context == {"intent": "食料を集める"}


def test_no_match_is_not_a_candidate():
    """判断器の対応なしは正式値へ変換せず、判断情報付きの未解決として返す。"""
    result = SemanticResolver(backend=RecordingBackend(None), policy=FixedPolicy()).resolve(
        value="配線を締める", candidates=OPTIONS
    )
    assert isinstance(result, Unresolved)
    assert result.reason == "no_match"
    assert result.judgment.value is None


def test_empty_candidates_skip_backend():
    """候補が空なら課金する判断を呼ばず、候補なしを返す。"""
    backend = RecordingBackend()
    result = SemanticResolver(backend=backend, policy=FixedPolicy()).resolve(
        value="gather", candidates=[]
    )
    assert isinstance(result, Unresolved)
    assert result.reason == "no_candidates"
    assert result.judgment is None
    assert backend.requests == []


@pytest.mark.parametrize("reason", ["ambiguous", "below_threshold"])
def test_policy_rejection_retains_proposed_value(reason):
    """採用基準で棄却しても、提案された値と採用基準を後から確認できる。"""
    result = SemanticResolver(backend=RecordingBackend(), policy=FixedPolicy(reason)).resolve(
        value="gather", candidates=OPTIONS
    )
    assert isinstance(result, Unresolved)
    assert result.reason == reason
    assert result.judgment.value == "harvest"
    assert result.assessment.reason == reason


def test_value_outside_candidates_is_backend_error():
    """候補外の値を返した判断器は契約違反とし、解決成功にも対応なしにもしない。"""
    with pytest.raises(BackendError, match="candidate"):
        SemanticResolver(backend=RecordingBackend("other"), policy=FixedPolicy()).resolve(
            value="gather", candidates=OPTIONS
        )


def test_backend_failure_is_not_no_match():
    """判断器が失敗した場合は例外を伝え、意味判断による棄却と混同しない。"""

    class Unavailable:
        def judge(self, request):
            raise BackendError("unavailable")

    with pytest.raises(BackendError, match="unavailable"):
        SemanticResolver(backend=Unavailable(), policy=FixedPolicy()).resolve(
            value="gather", candidates=OPTIONS
        )


def test_each_call_uses_its_own_candidates():
    """同じ解決器に別の候補を渡すと、各呼び出しの候補だけが判断器へ届く。"""
    backend = RecordingBackend(None)
    resolver = SemanticResolver(backend=backend, policy=FixedPolicy())
    resolver.resolve(value="gather", candidates=OPTIONS)
    resolver.resolve(value="gather", candidates=[Candidate("collect", "貝を採る")])
    assert backend.requests[0].candidates == tuple(OPTIONS)
    assert backend.requests[1].candidates == (Candidate("collect", "貝を採る"),)


def test_duplicate_values_fail_before_backend():
    """正式値が重複する候補は判断前に拒否し、説明を上書きしない。"""
    backend = RecordingBackend()
    with pytest.raises(ValueError, match="unique"):
        SemanticResolver(backend=backend, policy=FixedPolicy()).resolve(
            value="gather", candidates=OPTIONS * 2
        )
    assert backend.requests == []
