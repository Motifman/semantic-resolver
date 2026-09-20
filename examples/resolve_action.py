"""説明付きの操作候補へ解決する。実行すると TypeSafe API を呼ぶ。"""

import os

from semantic_resolver import Candidate, Resolved, SemanticResolver
from semantic_resolver.jev import JevBackend, JevPolicy


def main() -> None:
    with JevBackend(api_key=os.environ["TYPESAFE_API_KEY"], model="jev-1.13.0") as backend:
        resolver = SemanticResolver(
            backend=backend,
            policy=JevPolicy(min_probability=0.85, min_confidence=0.80),
        )
        result = resolver.resolve(
            value="gather",
            candidates=[
                Candidate("reap_wheat", "畑の小麦を収穫する"),
                Candidate("inspect_field", "畑の状態を調べる"),
            ],
            context={"target": "小麦畑", "intent": "食料にするため、小麦を集めたい"},
        )
        if isinstance(result, Resolved):
            print(f"解決: {result.original} → {result.value}")
        else:
            print(f"未解決: {result.reason}")
        print(result)


if __name__ == "__main__":
    main()
