# semantic-resolver

固定スキーマのまま、入力を動的な説明付き候補へ意味で解決する Python ライブラリ。

候補が変わるたびにツールの enum を書き換えると、共通プロンプトのキャッシュを
再利用しにくくなることがあります。`semantic-resolver` は候補をスキーマから分離し、
モデルが書いた引数を、説明と文脈から正式値へ対応づけます。

**対応する候補がなければ棄却します。単に最も近い候補へ置き換える処理ではありません。**

```text
固定されたツールスキーマ + 末尾の現在状態・候補
    → モデルが引数を書く
    → 必要な場合に SemanticResolver を呼ぶ
    → 正式値または未解決
    → 利用者側で実行条件を確認
```

## インストール

Python 3.10 以上が必要です。PyPI へは未公開です。

```bash
git clone https://github.com/Motifman/semantic-resolver.git
cd semantic-resolver
uv sync --extra jev
```

中核の実行時依存はありません。Jev 接続を使うときだけ `httpx` が必要です。
他のプロジェクトから使う場合は `pip install '.[jev]'` でインストールできます。

## 使い方

```python
import os

from semantic_resolver import Candidate, Resolved, SemanticResolver
from semantic_resolver.jev import JevBackend, JevPolicy

with JevBackend(
    api_key=os.environ["TYPESAFE_API_KEY"],
    model="jev-1.13.0",
) as backend:
    resolver = SemanticResolver(
        backend=backend,
        # 使用データに合わせて調整する値。推奨精度を保証する既定値ではない。
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
        print(result.value)  # 採用できた場合の候補の正式値
    else:
        print(result.reason)  # 対応なし・曖昧・確信不足など
```

同じ `resolver` へ毎回異なる候補を渡せます。前回の候補や判断は保持しません。
`context` は省略可能な補助情報で、Jev 接続では JSON にできる辞書を渡します。
入力の意味を絞るために使い、元の入力と矛盾する別の行動を作るためには使いません。

実行できる例は [examples/resolve_action.py](examples/resolve_action.py) にあります。
外部 API 呼び出しを行い、利用料金が発生します。

```bash
# TYPESAFE_API_KEY を環境変数へ設定して実行する
uv run --extra jev python examples/resolve_action.py
```

## 契約

| 場面 | 振る舞い |
|---|---|
| 完全一致した入力 | 判断器を呼ぶ。自動的な省略はしない |
| 候補が空 | 判断器を呼ばず `no_candidates` |
| 対応する候補がない | `no_match` |
| Jev が候補を同率で評価 | `ambiguous`（対応なしを選んだ場合は `no_match`） |
| 提案された候補の評価が閾値未達 | `below_threshold` |
| 意味判断と採用基準を通過 | `Resolved` に候補の正式値を返す |
| 重複した正式値・空の入力など | `ValueError` / `TypeError` |
| 通信障害・不正な応答 | `BackendError`。未解決には読み替えない |

正式値と説明は空でない文字列にします。正式値は重複できません。
入力の空白や大文字・小文字は勝手に変換しません。完全一致・別名・表示崩れの補正を
先に行いたい場合は、利用者側で行ってから未解決値だけを渡してください。

「対応なし」は常に内部で追加します。利用者側で追加する必要はなく、予約された
候補名もありません。Jev はこの一件を含めて最大255選択肢のため、候補は254件までです。

`Resolved` / `Unresolved` は元の入力を `original` に保持します。
判断した場合の `judgment` には提案値、判断器、実際に応答したモデル、所要時間、
方式固有の `evidence` が残ります。棄却した提案値も確認できます。
採否を評価した場合は `assessment` に理由と閾値を残します。
対応なしと候補なしでは閾値評価を行わないため `assessment` は `None` です。
ライブラリ自体は入力・文脈・結果をログやファイルへ保存しません。

## Jev 接続

- 公式の `POST https://api.typesafe.ai/v1/systemone` を使用します。
- 一回の解決につき一回の要求を送ります。自動再試行はしません。
- タイムアウトは既定で10秒です。`timeout_seconds` で変更できます。
- `client=` に `httpx.Client` を渡すと接続を共有できます。この場合、所有者が閉じます。
- 障害の `BackendError.code` は `transport_error` / `http_error` / `invalid_response`。
  HTTP 障害では `status_code` も確認できます。応答本文やキーは例外文へ載せません。
- 全候補の確率、対応なしの確率、確信度を検証します。欠けた値を補いません。
  確率の合計は丸め誤差として絶対誤差0.0001まで許容し、正規化し直しません。
- `JevPolicy` は確率・確信度の両閾値を必須にします。同率判定の許容差は1e-12です。
  `confidence` は分布の集中度であり、他の判断器の類似度や実測正解率とは異なります。

別の対応接続先を使う場合は `endpoint` と、その接続先の `model` を明示します。
接続先ごとの認証・対応状況は利用者側で確認してください。

一次資料：[Choice](https://docs.typesafe.ai/primitives/choice)、
[HTTP API](https://docs.typesafe.ai/api)、[モデルの版固定](https://docs.typesafe.ai/models)。

## 別の判断器を接続する

`Backend[Evidence]` の `judge(request)` と、`Policy[Evidence]` の
`assess(judgment)` を実装します。継承は不要です。
`Evidence` は判断器固有の型なので、確率と類似度を同じ値として扱う必要はありません。

`judge()` は候補の正式値または `None` を含む `Judgment` を返します。
`None` は対応なしです。判断器は候補の意味・文脈・対応なしを扱う契約を守り、
障害時には `BackendError` を送出します。
`assess()` は `accepted` / `ambiguous` / `below_threshold` と採用基準を含む
`Assessment` を返します。対応なしの判断には呼ばれません。

中核は判断器の提案が候補内かを確認します。確率の正しさなど方式固有の応答検証は、
各判断器が担当します。Jev 以外の接続実装は初版には含みません。

## 効果と範囲

候補は実行直前の状態に合わせて利用者が収集します。権限・実行条件・対象の存続や、
状態が変わっていないことの検証も利用者側で行います。複数引数の依存関係は、対象を
先に確定してから候補を渡すなど、利用者側で扱います。

主モデルのスキーマを変更せずに組み込めますが、このライブラリがキャッシュを操作する
わけではありません。命中率・費用・待ち時間の改善は、利用する API と組み込み方に
依存します。実モデルの意味精度や、SLM のツール成功率の改善はまだ測定していません。

初版は同期 API です。JSON 修復、別名辞書、ツール実行、再計画、判断結果のキャッシュは
提供しません。

## 開発

```bash
uv sync --locked --extra jev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv build
```

テストは HTTP の模擬応答を使い、外部 API や秘密情報を必要としません。

MIT ライセンス。利用条件は [LICENSE](LICENSE) を参照してください。
