"""課題①: 売上の着地予測（回帰・時系列）。

> **「今日時点で、今月はいくらで着地するか」**

経営層と、そこから数字をコミットされている管理職に向けた予測。
詳細は ``docs/objective.md``。
"""

from sales_analytics.tasks.landing.baselines import BASELINES, add_baselines
from sales_analytics.tasks.landing.dataset import (
    COMPANY_LABEL,
    GROUP_COL,
    STEP_COL,
    TARGET_COL,
    build_landing_frame,
    feature_columns,
)

__all__ = [
    "BASELINES",
    "COMPANY_LABEL",
    "GROUP_COL",
    "STEP_COL",
    "TARGET_COL",
    "add_baselines",
    "build_landing_frame",
    "feature_columns",
]
