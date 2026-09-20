"""デモの入力と、キーなしで制御の流れを確認する固定の判断結果。"""

from dataclasses import dataclass

from semantic_resolver import BackendError, Candidate, Judgment, ResolutionRequest
from semantic_resolver.jev import JevEvidence


@dataclass(frozen=True)
class Case:
    name: str
    title: str
    value: str
    candidates: tuple[Candidate, ...]
    context: dict[str, object]
    # 以下は模擬モード専用。実 Jev へは渡さない。
    choice: str | None
    probabilities: tuple[float, ...]  # 候補順、最後が対応なし
    confidence: float


WHEAT = (
    Candidate("reap_wheat", "畑の小麦を収穫する"),
    Candidate("inspect_field", "畑の状態を調べる"),
)

CASES = (
    Case(
        "harvest",
        "畑で gather を解決する",
        "gather",
        WHEAT,
        {"target": "小麦畑", "intent": "食料にするため、小麦を集めたい"},
        "reap_wheat",
        (0.93, 0.04, 0.03),
        0.86,
    ),
    Case(
        "shellfish",
        "同じ gather に別の候補を渡す",
        "gather",
        (
            Candidate("gather_shellfish", "浜辺で食べられる貝を採る"),
            Candidate("search_debris", "浜辺の漂着物から道具を探す"),
        ),
        {"target": "浜辺", "intent": "夕食のために貝を集めたい"},
        "gather_shellfish",
        (0.94, 0.03, 0.03),
        0.9,
    ),
    Case(
        "no-match",
        "話題が近くても、別の操作には変えない",
        "配線を締め直す",
        (
            Candidate("inspect_generator", "発電機を目で見て点検する"),
            Candidate("start_generator", "発電機の電源を入れる"),
        ),
        {"target": "発電機", "intent": "緩んだ配線の結束を締め直したい"},
        None,
        (0.04, 0.04, 0.92),
        0.84,
    ),
    Case(
        "uncertain",
        "曖昧な入力の採否を見る",
        "装置を調整する",
        (
            Candidate("tighten_wiring", "装置の配線の結束を締め直す"),
            Candidate("adjust_voltage", "装置の出力電圧を調整する"),
        ),
        {"target": "装置"},
        "tighten_wiring",
        (0.52, 0.4, 0.08),
        0.12,
    ),
    Case("empty", "候補なしでは判断器を呼ばない", "調べる", (), {}, None, (1.0,), 1.0),
    Case(
        "exact",
        "完全一致でも意味判断する",
        "reap_wheat",
        WHEAT,
        {"target": "小麦畑", "intent": "小麦を収穫したい"},
        "reap_wheat",
        (0.96, 0.02, 0.02),
        0.93,
    ),
    Case("error", "障害を対応なしと区別する", "通信障害の例", WHEAT, {}, None, (1.0,), 1.0),
)


class FixtureBackend:
    """固定の値を返すだけで、意味を推論しない。精度評価には使えない。"""

    def __init__(self) -> None:
        self.calls = 0

    def judge(self, request: ResolutionRequest) -> Judgment[JevEvidence]:
        self.calls += 1
        case = next(
            case
            for case in CASES
            if case.value == request.value and case.candidates == request.candidates
        )
        if case.name == "error":
            raise BackendError("模擬モードで発生させた通信障害", code="transport_error")
        keys = [candidate.value for candidate in request.candidates] + [None]
        return Judgment(
            value=case.choice,
            evidence=JevEvidence(dict(zip(keys, case.probabilities)), case.confidence),
            backend="fixture",
            model="固定のデモ値・推論なし",
            latency_ms=0.0,
        )
