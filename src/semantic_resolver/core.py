"""候補の収集や実行から独立した、意味による解決の契約。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, Literal, Protocol, TypeVar

Evidence = TypeVar("Evidence")
EvidenceCo = TypeVar("EvidenceCo", covariant=True)
EvidenceContra = TypeVar("EvidenceContra", contravariant=True)
AssessmentReason = Literal["accepted", "ambiguous", "below_threshold"]
UnresolvedReason = Literal["no_candidates", "no_match", "ambiguous", "below_threshold"]


class BackendError(Exception):
    """判断を得られない障害。意味による棄却とは区別する。"""

    def __init__(
        self, message: str, *, code: str = "invalid_response", status_code: int | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _nonempty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class Candidate:
    """返してよい正式値と、その意味を表す説明。"""

    value: str
    description: str

    def __post_init__(self) -> None:
        _nonempty(self.value, "candidate value")
        _nonempty(self.description, "candidate description")


@dataclass(frozen=True)
class ResolutionRequest:
    """一回の判断に必要な入力。文脈は入力の意味を絞る補助情報。"""

    value: str
    candidates: tuple[Candidate, ...]
    context: Mapping[str, object]


@dataclass(frozen=True)
class Judgment(Generic[EvidenceCo]):
    """判断器の提案。None は対応なし、評価値の意味は判断器ごとに定義する。"""

    value: str | None
    evidence: EvidenceCo
    backend: str
    model: str
    latency_ms: float


@dataclass(frozen=True)
class Assessment:
    """提案の採否と、それを決めた基準を対で残す。"""

    reason: AssessmentReason
    thresholds: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.reason not in ("accepted", "ambiguous", "below_threshold"):
            raise ValueError("unknown assessment reason")
        object.__setattr__(self, "thresholds", MappingProxyType(dict(self.thresholds)))


class Backend(Protocol[EvidenceCo]):
    """候補または対応なしを提案し、方式固有の評価値を返す。"""

    def judge(self, request: ResolutionRequest) -> Judgment[EvidenceCo]: ...


class Policy(Protocol[EvidenceContra]):
    """判断器の評価値に合う基準で、候補を採用するか決める。"""

    def assess(self, judgment: Judgment[EvidenceContra]) -> Assessment: ...


@dataclass(frozen=True)
class Resolved(Generic[EvidenceCo]):
    """候補の正式値へ解決できた。実行の可否は呼び出し側で検証する。"""

    original: str
    value: str
    judgment: Judgment[EvidenceCo]
    assessment: Assessment


@dataclass(frozen=True)
class Unresolved(Generic[EvidenceCo]):
    """解決しなかった理由と、得られた判断情報。"""

    original: str
    reason: UnresolvedReason
    judgment: Judgment[EvidenceCo] | None = None
    assessment: Assessment | None = None


class SemanticResolver(Generic[Evidence]):
    """呼ばれるたびに意味を判断する。候補なしだけは判断器を呼ばない。"""

    def __init__(self, *, backend: Backend[Evidence], policy: Policy[Evidence]) -> None:
        self._backend = backend
        self._policy = policy

    def resolve(
        self,
        *,
        value: str,
        candidates: Sequence[Candidate],
        context: Mapping[str, object] | None = None,
    ) -> Resolved[Evidence] | Unresolved[Evidence]:
        """入力を候補へ対応づけ、対応なし・曖昧さ・確信不足なら棄却する。"""
        _nonempty(value, "value")
        snapshot = tuple(candidates)
        if not all(isinstance(candidate, Candidate) for candidate in snapshot):
            raise TypeError("candidates must contain Candidate instances")
        values = {candidate.value for candidate in snapshot}
        if len(values) != len(snapshot):
            raise ValueError("candidate values must be unique")
        if not snapshot:
            return Unresolved(original=value, reason="no_candidates")
        request = ResolutionRequest(
            value=value,
            candidates=snapshot,
            context=deepcopy(dict(context)) if context is not None else {},
        )
        judgment = self._backend.judge(request)
        if judgment.value is None:
            return Unresolved(original=value, reason="no_match", judgment=judgment)
        if not isinstance(judgment.value, str) or judgment.value not in values:
            raise BackendError("backend returned a value outside the candidate set")
        assessment = self._policy.assess(judgment)
        if assessment.reason != "accepted":
            return Unresolved(
                original=value, reason=assessment.reason, judgment=judgment, assessment=assessment
            )
        return Resolved(
            original=value, value=judgment.value, judgment=judgment, assessment=assessment
        )
