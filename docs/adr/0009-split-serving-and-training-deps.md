# ADR-0009: 推論の依存と学習の依存を分離する

- 状態: 採用
- 日付: 2026-08-16

## 背景

このプロジェクトが必要とするライブラリは、用途が明確に2つに分かれる。

- **推論に必要**: FastAPI、LightGBM、Polars、joblib、pydantic
- **学習にしか使わない**: MLflow、Optuna、matplotlib

すべてを1つの `dependencies` にまとめると、推論コンテナにも MLflow が入る。
MLflow は Flask・SQLAlchemy・Alembic などを引き込むため、
イメージサイズが数百MB増え、そのぶん脆弱性の対象面も増える。

推論サーバは外部に露出する。そこに、推論では一度も import されないコードを置くのは筋が悪い。

## 決定

`pyproject.toml` で依存を2層に分ける。

```toml
[project]
dependencies = [          # 推論サーバに必要な最小限
    "fastapi", "joblib", "lightgbm", "numpy", "polars",
    "pyarrow", "pydantic", "pydantic-settings", "pyyaml",
    "scikit-learn", "typer", "uvicorn[standard]",
]

[project.optional-dependencies]
train = ["matplotlib", "mlflow", "optuna"]   # 学習・分析にだけ必要

[dependency-groups]
dev = ["httpx", "mypy", "pytest", "pytest-cov", "ruff", "types-pyyaml"]
```

- 開発時: `uv sync --extra train`
- 推論イメージ: `uv sync --frozen --no-dev`（`train` は入らない）

学習用の依存を使うコードは、**モジュールの先頭ではなく関数の中で import する**。
未インストールでも、その機能を呼ばない限り動く。

```python
try:
    import mlflow
except ImportError:
    logger.warning("mlflow が未インストールのため実験記録をスキップします")
    return
```

## 理由

**なぜ `optional-dependencies` と `dependency-groups` を使い分けるか**

この2つは似ているが役割が違う。

| 仕組み | 役割 | 配布されるか |
| --- | --- | --- |
| `optional-dependencies` | パッケージの機能拡張（`pip install pkg[train]`） | される |
| `dependency-groups` | 開発ツール（PEP 735） | されない |

学習機能は「このパッケージの機能の一部」なので `optional-dependencies`。
ruff や mypy は「開発するときだけ必要」なので `dependency-groups`。
この区別により、`uv sync --no-dev` で開発ツールだけを的確に除外できる。

**なぜ遅延 import にするか**

`train` を入れていない環境で `import demand_forecast.models.train` が
`ImportError` で落ちると、CLI 全体が起動しなくなる。
関数内 import にすることで、「学習コマンドを打ったときだけ必要」という
実際の依存関係とコードの構造を一致させている。

CLI 本体（`cli.py`）でも、重いサブコマンドの import は関数の中に置いた。
`dfc version` を打つのに MLflow を読み込む必要はなく、起動が速くなる副次効果もある。

**マルチステージビルドとの組み合わせ**

Dockerfile ではビルドステージで `.venv` を作り、実行ステージにはそれだけをコピーしている。
uv 自体もビルドツールも実行イメージには残らない。

さらに、依存の解決とプロジェクトのインストールを2段階に分けた。

```dockerfile
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --no-editable   # 依存だけ

COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable                        # プロジェクト本体
```

アプリのコードを変えただけでライブラリを再インストールしないための定石である。

## 結果

**得たもの**

- 推論イメージから MLflow / Optuna / matplotlib とその依存群が消えた
- 外部に露出するコンテナの構成要素が減った
- コードの構造（関数内 import）が、実際の依存関係と一致している
- コード変更時の Docker ビルドがキャッシュに乗る

**諦めたもの**

- 「とりあえず全部入れる」より設定が複雑になる。
  ただし `Makefile` の `setup` ターゲットに `uv sync --extra train` を書いてあるので、
  開発者が意識する場面はほとんどない。
- 遅延 import は静的解析からは見えにくい。`TYPE_CHECKING` ブロックで型だけは
  読み込むようにして、型検査は効くようにしてある。

## 補足: 本リポジトリでの検証状況

Dockerfile とマルチステージ構成は記述済みだが、
**この開発環境ではコンテナレジストリへの通信がネットワークポリシーで遮断されており、
イメージのビルド検証はできていない**。
CI（`.github/workflows/ci.yml`）の `docker` ジョブでビルドと疎通確認まで行う構成にしてあるので、
GitHub 上で初めて検証されることになる。
