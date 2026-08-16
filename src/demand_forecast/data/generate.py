"""需要データの合成生成器。

実データ（Kaggle 等）を前提にすると、認証情報やダウンロード手順がないと
誰も再現できないリポジトリになる。そこで「現実の需要データが持つ性質」を
明示的に組み込んだ生成器を用意し、``uv run dfc generate-data`` だけで
誰でも同じデータから同じ結果に到達できるようにしている。

組み込んでいる性質:

- 系列（店舗×商品）ごとに水準が大きく異なる
- 曜日周期（週末が高い）と年周期（季節性）
- 緩やかなトレンド（伸びている商品・落ちている商品が混在）
- 価格弾力性（値下げすると売れる）
- 販促期間のかさ上げ効果
- 祝日効果
- 過分散のカウントノイズ（ガンマ・ポアソン混合）

意図的に組み込んでいない性質（実データとの差分。docs/adr/0002 に記載）:

- 欠品（在庫切れ）による打ち切り
- 新商品の投入・終売による系列の出入り
- 店舗改装などのイベントによる構造変化
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from demand_forecast.config import DataConfig
from demand_forecast.features.calendar import japanese_holiday_flags

# 月曜〜日曜の基本的な曜日プロファイル（小売の典型的な形）
_BASE_WEEKLY_PROFILE = np.array([0.92, 0.86, 0.89, 0.96, 1.18, 1.38, 1.21])

# ガンマ・ポアソン混合の形状パラメータ。小さいほど過分散が強くなる。
# 誤差の理論下限を求めるのに使うため公開している（analysis.estimate_noise_floor）。
OVERDISPERSION_SHAPE = 9.0


def _generate_promo_flags(rng: np.random.Generator, n_days: int) -> np.ndarray:
    """販促フラグを「連続した期間」として生成する。

    販促は1日単位でランダムに立つのではなく、数日〜1週間続く。
    独立なベルヌーイ試行で作ると自己相関のない非現実的な系列になるため、
    開始日を疎に抽選してから期間を伸ばす。
    """
    flags = np.zeros(n_days, dtype=np.int8)
    day = 0
    while day < n_days:
        # 平均するとおよそ 5 週に 1 回、販促期間が始まる
        gap = int(rng.integers(21, 56))
        day += gap
        if day >= n_days:
            break
        duration = int(rng.integers(3, 8))
        flags[day : day + duration] = 1
        day += duration
    return flags


def _generate_series(
    rng: np.random.Generator,
    dates: np.ndarray,
    day_of_week: np.ndarray,
    day_of_year: np.ndarray,
    holiday: np.ndarray,
) -> dict[str, np.ndarray]:
    """1系列（店舗×商品の組）分の需要と価格を生成する。"""
    n_days = len(dates)
    t = np.arange(n_days, dtype=np.float64)

    # --- 水準とトレンド ---
    base_level = float(rng.uniform(8.0, 48.0))
    yearly_growth = float(rng.uniform(-0.18, 0.35))
    trend = (1.0 + yearly_growth) ** (t / 365.25)

    # --- 曜日周期（系列ごとに少しばらつかせる） ---
    weekly_profile = _BASE_WEEKLY_PROFILE * rng.normal(1.0, 0.06, size=7)
    weekly = weekly_profile[day_of_week]

    # --- 年周期 ---
    amplitude = float(rng.uniform(0.08, 0.30))
    phase = float(rng.uniform(0.0, 1.0))
    yearly = 1.0 + amplitude * np.sin(2 * np.pi * (day_of_year / 365.25 + phase))

    # --- 価格と販促 ---
    base_price = float(np.round(rng.uniform(180, 920), -1))
    promo = _generate_promo_flags(rng, n_days)
    discount = np.where(promo == 1, rng.uniform(0.12, 0.35, size=n_days), 0.0)
    # 販促期間外にも小さな価格変動（改定）を入れる
    drift = np.cumsum(rng.normal(0.0, 0.0015, size=n_days))
    price = base_price * (1.0 + drift) * (1.0 - discount)
    price = np.maximum(price, base_price * 0.4)

    # --- 価格弾力性と販促の直接効果 ---
    elasticity = float(rng.uniform(-2.4, -1.1))
    price_effect = (price / base_price) ** elasticity
    # 値下げとは別に、売場露出が増えることによる上乗せ
    promo_effect = np.where(promo == 1, rng.uniform(1.10, 1.45), 1.0)

    # --- 祝日効果 ---
    holiday_effect = np.where(holiday == 1, float(rng.uniform(1.05, 1.30)), 1.0)

    lam = base_level * trend * weekly * yearly * price_effect * promo_effect * holiday_effect
    lam = np.maximum(lam, 0.05)

    # --- 過分散カウントノイズ（ガンマ・ポアソン混合 = 負の二項分布） ---
    gamma_noise = rng.gamma(OVERDISPERSION_SHAPE, 1.0 / OVERDISPERSION_SHAPE, size=n_days)
    units = rng.poisson(lam * gamma_noise)

    return {
        "units_sold": units.astype(np.int32),
        "price": np.round(price, 1),
        "promo_flag": promo,
        # ノイズを乗せる前の期待需要。ガンマ項の期待値は 1 なので、
        # これがその日の「真の平均需要」にあたる。誤差の理論下限を求めるのに使う。
        "expected_demand": lam,
    }


def generate_demand_data(
    cfg: DataConfig, seed: int = 42, *, include_expected: bool = False
) -> pl.DataFrame:
    """店舗×商品×日付の需要データを生成する。

    Args:
        cfg: 生成範囲（期間・店舗数・商品数）。
        seed: 乱数シード。同じシードなら常に同じデータになる。
        include_expected: True なら、ノイズを乗せる前の期待需要
            ``expected_demand`` も返す。実データには存在しない情報なので、
            学習には使わず、誤差の理論下限を求める分析にだけ使う。

    Returns:
        ``date``/``store_id``/``sku_id``/``units_sold``/``price``/``promo_flag``
        を持つ Polars DataFrame。日付・店舗・商品の順にソート済み。
    """
    dates_list: list[dt.date] = []
    current = cfg.start_date
    while current <= cfg.end_date:
        dates_list.append(current)
        current += dt.timedelta(days=1)

    dates = np.array(dates_list, dtype="datetime64[D]")
    n_days = len(dates)

    # 日付から決まる特徴は全系列で共通なので一度だけ計算する
    day_of_week = (dates.astype("datetime64[D]").astype(int) + 3) % 7  # 1970-01-01 は木曜
    day_of_year = np.array([d.timetuple().tm_yday for d in dates_list], dtype=np.int32)
    holiday = japanese_holiday_flags(dates_list)

    stores = [f"S{i + 1:02d}" for i in range(cfg.n_stores)]
    skus = [f"SKU{i + 1:02d}" for i in range(cfg.n_skus)]

    frames: list[pl.DataFrame] = []
    for store_idx, store_id in enumerate(stores):
        for sku_idx, sku_id in enumerate(skus):
            # 系列ごとに独立したシードを与え、店舗数や商品数を増やしても
            # 既存系列の値が変わらないようにする
            series_seed = (seed, store_idx, sku_idx)
            rng = np.random.default_rng(series_seed)
            series = _generate_series(rng, dates, day_of_week, day_of_year, holiday)
            frames.append(
                pl.DataFrame(
                    {
                        "date": dates_list,
                        "store_id": [store_id] * n_days,
                        "sku_id": [sku_id] * n_days,
                        "units_sold": series["units_sold"],
                        "price": series["price"],
                        "promo_flag": series["promo_flag"],
                        "expected_demand": series["expected_demand"],
                    }
                )
            )

    combined = (
        pl.concat(frames)
        .with_columns(
            pl.col("date").cast(pl.Date),
            pl.col("store_id").cast(pl.Utf8),
            pl.col("sku_id").cast(pl.Utf8),
            pl.col("units_sold").cast(pl.Int32),
            pl.col("price").cast(pl.Float64),
            pl.col("promo_flag").cast(pl.Int8),
            pl.col("expected_demand").cast(pl.Float64),
        )
        .sort(["date", "store_id", "sku_id"])
    )
    return combined if include_expected else combined.drop("expected_demand")
