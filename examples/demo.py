"""入力・公開 API の呼び出し・採否と根拠を順に表示する。"""

from __future__ import annotations

import argparse
import os
from contextlib import ExitStack
from pprint import pformat

from _demo_cases import CASES, Case, FixtureBackend

from semantic_resolver import Backend, BackendError, Resolved, SemanticResolver, Unresolved
from semantic_resolver.jev import JevBackend, JevEvidence, JevPolicy


def show_call(case: Case) -> None:
    print(f"\n--- {case.name}: {case.title} ---")
    print("result = resolver.resolve(")
    print(f"    value={case.value!r},")
    print("    candidates=[")
    for candidate in case.candidates:
        print(f"        Candidate({candidate.value!r}, {candidate.description!r}),")
    print("    ],")
    print(f"    context={pformat(case.context, sort_dicts=False)},")
    print(")")


def show_result(result: Resolved[JevEvidence] | Unresolved[JevEvidence]) -> None:
    if isinstance(result, Resolved):
        print(f"→ Resolved: {result.original!r} → {result.value!r}")
    else:
        print(f"→ Unresolved: reason={result.reason!r}")
    if result.judgment is None:
        print("  判断器の呼び出し: なし")
        return
    judgment = result.judgment
    print(f"  判断器: {judgment.backend} / {judgment.model}")
    print(f"  提案値: {judgment.value!r} / confidence={judgment.evidence.confidence:.4f}")
    for value, probability in judgment.evidence.probabilities.items():
        print(f"  {value if value is not None else '〈対応なし〉'}: {probability:.4f}")
    if result.assessment:
        print(f"  採用基準: {dict(result.assessment.thresholds)}")
    print(f"  待ち時間: {judgment.latency_ms:.1f} ms")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=("fixture", "typesafe", "openrouter"), default="fixture"
    )
    parser.add_argument("--case", choices=[case.name for case in CASES])
    parser.add_argument("--api-key-env", help="使用するキーの環境変数名")
    parser.add_argument("--min-probability", type=float, default=0.85)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    args = parser.parse_args()
    if args.provider != "fixture" and args.case == "error":
        parser.error("error は模擬モード専用です。実 API に意図的な障害を起こしません。")
    try:
        policy = JevPolicy(min_probability=args.min_probability, min_confidence=args.min_confidence)
    except ValueError as exc:
        parser.error(str(exc))

    with ExitStack() as stack:
        backend: Backend[JevEvidence]
        if args.provider == "fixture":
            print("模擬モード: 固定の判断結果で API の使い方を確認します。推論・通信はありません。")
            backend = FixtureBackend()
        else:
            key_env = args.api_key_env or (
                "TYPESAFE_API_KEY" if args.provider == "typesafe" else "OPENROUTER_API_KEY"
            )
            key = os.environ.get(key_env, "").strip()
            if not key:
                parser.error(f"環境変数 {key_env} に API キーを設定してください。")
            print(f"実 Jev モード ({args.provider}): 結果は実際の判断で変わります。")
            backend = stack.enter_context(
                JevBackend(
                    api_key=key,
                    model="jev-1.13.0" if args.provider == "typesafe" else "typesafe/jev-1.13",
                    endpoint=(
                        "https://api.typesafe.ai/v1/systemone"
                        if args.provider == "typesafe"
                        else "https://openrouter.ai/api/alpha/decisions"
                    ),
                    timeout_seconds=30.0,
                )
            )

        # 同じ解決器を使い、呼び出しごとに候補と文脈だけを変える。
        resolver = SemanticResolver(backend=backend, policy=policy)
        failures = 0
        for case in CASES:
            if args.case and args.case != case.name:
                continue
            if args.provider != "fixture" and case.name == "error":
                continue
            show_call(case)
            try:
                result = resolver.resolve(
                    value=case.value, candidates=case.candidates, context=case.context
                )
            except BackendError as exc:
                print(f"→ BackendError: code={exc.code!r}, status_code={exc.status_code!r}")
                print("  判断を取得できませんでした。対応なしとは区別して扱います。")
                if args.provider != "fixture":
                    failures += 1
            else:
                show_result(result)
        if isinstance(backend, FixtureBackend):
            print(f"\n模擬判断器の呼び出し回数: {backend.calls}（候補なしは呼ばない）")
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
