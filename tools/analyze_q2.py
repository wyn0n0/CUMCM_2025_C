import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


# -----------------------------
# 数据结构与配置
# -----------------------------

@dataclass
class Q2Config:
    input_path: str
    output_csv: str
    k_groups: int = 5
    alpha_attain: float = 0.9  # 平均达标概率阈值（软约束目标）
    lambda_fail: float = 1.0   # 未达标风险权重
    lambda_time: float = 0.1   # 时间窗口惩罚权重
    c0_early: float = 1.2      # <12周的早期惩罚（提高以避免扎堆早期）
    c1_mid: float = 1.0        # 12-27周区间惩罚
    c2_late: float = 3.0       # >=28周惩罚（远大于 c1）
    t_min: float = 10.0
    t_max: float = 25.0
    t_step: float = 0.1
    random_state: int = 42
    min_group_size: int = 20
    lambda_alpha: float = 0.5  # 未达标软惩罚权重（推动孕周更晚以满足alpha）
    lambda_balance: float = 0.0  # 组均衡惩罚（默认0，可调）
    lambda_trend: float = 0.0  # 时点随BMI递增的趋势惩罚（鼓励高BMI更晚孕周）
    w1_mid: float = 12.0       # 平滑过渡的第一阈值（早->中）
    w2_late: float = 28.0      # 平滑过渡的第二阈值（中->晚）
    smooth_s: float = 0.5      # 平滑强度（越大过渡越缓）


# -----------------------------
# 工具函数
# -----------------------------

def _pick_first_nonnull(series: pd.Series) -> Optional[float]:
    for v in series:
        if pd.notna(v):
            return v
    return None


def _get_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    # 再尝试大小写忽略
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _parse_gestation_weeks(x: object) -> Optional[float]:
    """解析孕周表示，支持如下形式：
    - 数值（周）
    - 字符串 "12+3" 或 "12＋3"（周+天）
    - 字符串含中文，如 "12周+3天"、"12周3天"
    """
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        # 视为周
        if np.isfinite(x):
            return float(x)
        return None
    try:
        s = str(x)
        # 提取数字
        # 优先匹配 A+B 形式
        if "+" in s or "＋" in s:
            s = s.replace("＋", "+")
            parts = s.split("+")
            if len(parts) == 2:
                w = _safe_float(parts[0])
                d = _safe_float(parts[1])
                if w is not None and d is not None:
                    return float(w) + float(d) / 7.0
        # 匹配中文“周/天”
        import re
        m = re.findall(r"(\d+)[^\d]+(\d+)", s)
        if m:
            w, d = m[0]
            return float(w) + float(d) / 7.0
        # 只有周
        w = _safe_float(s)
        if w is not None:
            return float(w)
    except Exception:
        return None
    return None


def _safe_float(x: object) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _parse_datetime(col: pd.Series) -> pd.Series:
    return pd.to_datetime(col, errors="coerce")


def _phi_time_penalty(t: np.ndarray, p0: float, p1: float, p2: float, w1: float, w2: float, s: float) -> np.ndarray:
    """平滑单调时间惩罚：用两个Logistic阶跃叠加实现 <w1 -> [p0->p1]，<w2 -> [p1->p2] 的平滑过渡。"""
    t = np.asarray(t, dtype=float)
    s = max(1e-6, float(s))
    r1 = 1.0 / (1.0 + np.exp(-(t - w1) / s))
    r2 = 1.0 / (1.0 + np.exp(-(t - w2) / s))
    penalties = p0 + (p1 - p0) * r1 + (p2 - p1) * r2
    return penalties


# -----------------------------
# 数据读取与构造
# -----------------------------

def load_and_prepare_longitudinal(input_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取附件.xlsx，构造：
    - 测次级别数据 df_long: 每次检测一行，包含 [subject_id, bmi, age, t_weeks, y_conc, hit]
    - 个体级别数据 df_subj: 每位受试者一行，包含 [subject_id, bmi, age]
    """
    df = pd.read_excel(input_path)

    # 可能的列名候选（不做字母列名强制回退）
    col_subject = _get_col(df, [
        "孕妇代码", "subject", "subject_id", "ID", "样本序号", "样本ID", "受试者编号", "B", "A"
    ])
    col_bmi = _get_col(df, [
        "孕妇BMI指标", "孕妇BMI", "BMI指标", "BMI（kg/m²）", "BMI（kg/m^2）", "BMI (kg/m2)", "BMI(kg/m2)",
        "BMI", "bmi", "体质指数", "身高体重指数", "K"
    ])
    col_weeks = _get_col(df, [
        "检测孕周",
        "孕妇本次检测时的孕周（周数+天数）", "孕周（周数+天数）", "孕周(周数+天数)", "孕周(周+天)", "孕周（周+天）",
        "孕周", "gestational_weeks", "J"
    ])
    col_time = _get_col(df, ["检测日期", "检测时间", "采样时间", "sample_time", "H"])  # 可空
    col_lmp = _get_col(df, ["末次月经", "末次月经时间", "LMP", "F"])  # 可空
    col_age = _get_col(df, ["孕妇年龄", "年龄", "age", "C"])  # 可空
    col_yz = _get_col(df, ["U", "Y染色体的Z值", "Y_Z"])  # 可空
    col_yconc = _get_col(df, ["Y染色体浓度", "Y染色体浓度（比例）", "Y_conc", "y_conc", "V"])  # 关键
    col_height = _get_col(df, ["孕妇身高", "身高", "height", "D"])  # 可选（用于补算BMI）
    col_weight = _get_col(df, ["孕妇体重", "体重", "weight", "E"])  # 可选（用于补算BMI）

    # 仅保留必要列
    candidate_cols = [col_subject, col_bmi, col_weeks, col_time, col_lmp, col_age, col_yz, col_yconc, col_height, col_weight]
    keep_cols = [c for c in candidate_cols if c and c in df.columns]
    if not keep_cols:
        raise ValueError("未识别到任何所需列名，请检查附件.xlsx 的表头。")
    df = df[keep_cols].copy()

    # 仅保留男胎：Y浓度或Y-Z 有数据
    mask_male = False
    if col_yconc and col_yconc in df.columns:
        mask_male = df[col_yconc].notna()
    elif col_yz and col_yz in df.columns:
        mask_male = df[col_yz].notna()
    else:
        # 若两者都没有，无法判断，保留全部（后续会失败提示）
        mask_male = pd.Series([True] * len(df), index=df.index)
    df = df[mask_male].copy()

    # 解析孕周
    weeks = df[col_weeks].apply(_parse_gestation_weeks) if (col_weeks and col_weeks in df.columns) else pd.Series([np.nan] * len(df))
    if weeks.isna().any() and col_time and col_lmp:
        # 用日期推算
        t_dt = _parse_datetime(df[col_time]) if col_time in df.columns else pd.Series([pd.NaT] * len(df))
        lmp_dt = _parse_datetime(df[col_lmp]) if col_lmp in df.columns else pd.Series([pd.NaT] * len(df))
        delta_days = (t_dt - lmp_dt).dt.days
        weeks2 = delta_days / 7.0
        weeks = weeks.fillna(weeks2)
    df["t_weeks"] = weeks

    # 命名对齐
    rename_map = {}
    if col_subject and col_subject in df.columns:
        rename_map[col_subject] = "subject_id"
    if col_bmi and col_bmi in df.columns:
        rename_map[col_bmi] = "bmi"
    if col_age and col_age in df.columns:
        rename_map[col_age] = "age"
    if col_yconc and col_yconc in df.columns:
        rename_map[col_yconc] = "y_conc"
    if col_height and col_height in df.columns:
        rename_map[col_height] = "height"
    if col_weight and col_weight in df.columns:
        rename_map[col_weight] = "weight"
    df.rename(columns=rename_map, inplace=True)

    if "age" not in df.columns:
        df.rename(columns={col_age: "age"}, inplace=True)
    if "age" not in df.columns:
        df["age"] = np.nan
    if "y_conc" not in df.columns:
        df["y_conc"] = np.nan

    # 若缺少 BMI，尝试用身高体重补算：BMI = weight / (height_m^2)
    if "bmi" not in df.columns and ("height" in df.columns and "weight" in df.columns):
        h = pd.to_numeric(df["height"], errors="coerce")
        w = pd.to_numeric(df["weight"], errors="coerce")
        # 将 cm 转 m：若 h>3 视为 cm
        h_m = np.where(h > 3.0, h / 100.0, h)
        bmi_calc = w / (h_m ** 2)
        df["bmi"] = bmi_calc

    # 清洗：去除缺少关键字段的记录
    essential_cols = ["subject_id", "bmi", "t_weeks"]
    missing_ess = [c for c in essential_cols if c not in df.columns]
    if missing_ess:
        available = list(df.columns)
        raise ValueError(f"缺少关键列: {missing_ess}，请检查附件.xlsx 的表头与内容。当前可用列: {available}")
    df = df[(df["subject_id"].notna()) & (df["bmi"].notna()) & (df["t_weeks"].notna())]

    # 命中标签
    df["hit"] = (df["y_conc"] >= 0.04).astype(float)

    # 归并到个体级
    aggs = df.groupby("subject_id").agg({
        "bmi": _pick_first_nonnull,
        "age": _pick_first_nonnull,
    }).reset_index()
    aggs.rename(columns={"bmi": "bmi", "age": "age"}, inplace=True)
    df_subj = aggs

    # 最终长表
    df_long = df[["subject_id", "bmi", "age", "t_weeks", "y_conc", "hit"]].dropna(subset=["t_weeks"]).copy()

    if len(df_long) == 0 or len(df_subj) == 0:
        raise ValueError("数据不足：无法构建长表或个体表，请检查附件.xlsx 列名与内容。")

    return df_long, df_subj


# -----------------------------
# 模型：纵向Logit 达标概率 π(t, x)
# -----------------------------

class AttainmentProbModel:
    def __init__(self, random_state: int = 42):
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)

    def _features(self, t: np.ndarray, bmi: np.ndarray) -> np.ndarray:
        X = np.vstack([t, bmi]).T  # shape (n, 2)
        X_poly = self.poly.fit_transform(X) if not hasattr(self.poly, "n_output_features_") else self.poly.transform(X)
        Xs = self.scaler.fit_transform(X_poly) if not hasattr(self.scaler, "n_features_in_") else self.scaler.transform(X_poly)
        return Xs

    def fit(self, df_long: pd.DataFrame) -> None:
        # 特征：t, bmi；标签：hit
        t = df_long["t_weeks"].to_numpy(dtype=float)
        bmi = df_long["bmi"].to_numpy(dtype=float)
        y = df_long["hit"].to_numpy(dtype=int)

        X = np.vstack([t, bmi]).T
        X_poly = self.poly.fit_transform(X)
        Xs = self.scaler.fit_transform(X_poly)
        self.clf.fit(Xs, y)

    def predict_pi(self, t: np.ndarray, bmi: np.ndarray) -> np.ndarray:
        Xs = self._features(t, bmi)
        proba = self.clf.predict_proba(Xs)[:, 1]
        # 限幅避免数值问题
        proba = np.clip(proba, 1e-6, 1 - 1e-6)
        return proba


# -----------------------------
# DP优化：按BMI一维连续分组 + 组内时点选择
# -----------------------------

@dataclass
class IntervalCost:
    total_cost: float
    best_t: float
    avg_attain: float
    avg_risk_report: float


def _precompute_subject_risk_grid(
    df_subj: pd.DataFrame,
    model: AttainmentProbModel,
    t_grid: np.ndarray,
    cfg: Q2Config,
) -> Tuple[np.ndarray, np.ndarray]:
    """返回：
    - probs: shape (n_subjects, G), 在各个 t 上的达标概率 π
    - risks: shape (n_subjects, G), 在各个 t 上的个体风险 L
    """
    n = len(df_subj)
    G = len(t_grid)
    bmi_arr = df_subj["bmi"].to_numpy(dtype=float)
    # 扩展到网格
    T_mat = np.tile(t_grid, (n, 1))
    BMI_mat = np.tile(bmi_arr.reshape(-1, 1), (1, G))

    probs = model.predict_pi(T_mat.ravel(), BMI_mat.ravel()).reshape(n, G)
    S = 1.0 - probs
    phi = _phi_time_penalty(t_grid, cfg.c0_early, cfg.c1_mid, cfg.c2_late, cfg.w1_mid, cfg.w2_late, cfg.smooth_s)
    # 风险 L = λ_fail * S + λ_time * φ(t)
    risks = cfg.lambda_fail * S + cfg.lambda_time * phi.reshape(1, -1)
    return probs, risks


def _compute_interval_cost(
    probs: np.ndarray,
    risks: np.ndarray,
    bmi_sorted: np.ndarray,
    l: int,
    r: int,
    t_grid: np.ndarray,
    alpha: float,
    lambda_alpha: float,
    target_size: float,
    lambda_balance: float,
    lambda_trend: float,
    global_bmi_mean: float,
    global_bmi_std: float,
) -> IntervalCost:
    """对连续区间 [l, r]（含）计算区间代价：最小平均风险及对应 t。
    代价 = 总风险 + 未达标软惩罚 + 组均衡惩罚。
    """
    sub_probs = probs[l : r + 1]  # shape (m, G)
    sub_risks = risks[l : r + 1]
    avg_probs = sub_probs.mean(axis=0)
    avg_risks = sub_risks.mean(axis=0)
    m = (r - l + 1)

    # 总风险 = 平均风险 * 组大小
    total_risks = avg_risks * float(m)
    # 未达标软惩罚：m * lambda_alpha * max(0, alpha - avg_prob)^2
    alpha_short = np.maximum(0.0, alpha - avg_probs)
    alpha_penalty = float(m) * float(lambda_alpha) * (alpha_short ** 2)
    # 组均衡惩罚：lambda_balance * ((m - target)/target)^2 * m
    if target_size > 0 and lambda_balance > 0.0:
        balance_pen = float(lambda_balance) * (((float(m) - target_size) / target_size) ** 2) * float(m)
    else:
        balance_pen = 0.0
    # 趋势惩罚：鼓励高BMI组选择更晚孕周
    if lambda_trend > 0.0 and global_bmi_std > 0.0:
        mean_bmi = float(np.nanmean(bmi_sorted[l : r + 1]))
        # 目标孕周：以12周为基准，按 z-score 线性递增 1周/σ
        z = (mean_bmi - global_bmi_mean) / global_bmi_std
        t_target = 12.0 + z * 1.0
        trend_pen = float(m) * float(lambda_trend) * ((t_grid - t_target) ** 2)
    else:
        trend_pen = 0.0
    total_costs = total_risks + alpha_penalty + balance_pen + trend_pen
    idx_best = int(np.argmin(total_costs))
    return IntervalCost(float(total_costs[idx_best]), float(t_grid[idx_best]), float(avg_probs[idx_best]), float(avg_risks[idx_best]))


def dp_optimize_segments(
    df_subj_sorted: pd.DataFrame,
    probs: np.ndarray,
    risks: np.ndarray,
    t_grid: np.ndarray,
    k_groups: int,
    alpha: float,
    min_group_size: int,
    lambda_alpha: float,
    lambda_balance: float,
    lambda_trend: float,
) -> Tuple[List[Tuple[int, int]], List[IntervalCost]]:
    """动态规划在 BMI 排序后的样本上做 k 段划分。
    返回：
    - segments: [(l0, r0), (l1, r1), ...]
    - costs: 与各段对应的 IntervalCost
    """
    n = len(df_subj_sorted)
    # 预计算所有 [l, r] 的 IntervalCost（可按需要裁剪以加速）
    target_size = n / float(k_groups)
    bmi_sorted = df_subj_sorted["bmi"].to_numpy(dtype=float)
    global_bmi_mean = float(np.nanmean(bmi_sorted))
    global_bmi_std = float(np.nanstd(bmi_sorted))
    interval_cost = [[None for _ in range(n)] for __ in range(n)]
    for l in range(n):
        # 至少满足最小组大小
        r_start = max(l + min_group_size - 1, l)
        for r in range(r_start, n):
            interval_cost[l][r] = _compute_interval_cost(
                probs, risks, bmi_sorted, l, r, t_grid, alpha, lambda_alpha, target_size, lambda_balance, lambda_trend, global_bmi_mean, global_bmi_std
            )

    # DP
    dp = np.full((k_groups + 1, n), np.inf, dtype=float)
    prev = [[-1 for _ in range(n)] for __ in range(k_groups + 1)]

    # k=1 的情况
    for i in range(min_group_size - 1, n):
        cost_lr = interval_cost[0][i]
        if cost_lr is not None:
            dp[1, i] = cost_lr.total_cost
            prev[1][i] = -1

    # k>=2
    for k in range(2, k_groups + 1):
        for i in range(k * min_group_size - 1, n):
            best_val = np.inf
            best_j = -1
            # 最后一个分段起点 j 在 [ (k-1)*min_group_size -1, i - min_group_size ]
            j_min = (k - 1) * min_group_size - 1
            j_max = i - min_group_size
            for j in range(j_min, j_max + 1):
                if j >= 0 and np.isfinite(dp[k - 1, j]):
                    cost_j1_i = interval_cost[j + 1][i]
                    if cost_j1_i is None:
                        continue
                    val = dp[k - 1, j] + cost_j1_i.total_cost
                    if val < best_val:
                        best_val = val
                        best_j = j
            dp[k, i] = best_val
            prev[k][i] = best_j

    # 回溯
    # 终点 i* = n-1
    k = k_groups
    i = n - 1
    if not np.isfinite(dp[k, i]):
        # 回退组数直到可行
        while k > 1 and not np.isfinite(dp[k, i]):
            k -= 1
    segments: List[Tuple[int, int]] = []
    costs: List[IntervalCost] = []
    while k >= 1:
        j = prev[k][i]
        l = j + 1
        segments.append((l, i))
        costs.append(interval_cost[l][i])
        i = j
        k -= 1
        if i < 0:
            break
    segments.reverse()
    costs.reverse()
    return segments, costs


# -----------------------------
# 主流程
# -----------------------------

def solve_q2(cfg: Q2Config) -> pd.DataFrame:
    df_long, df_subj = load_and_prepare_longitudinal(cfg.input_path)

    # 拟合纵向Logit模型
    model = AttainmentProbModel(random_state=cfg.random_state)
    model.fit(df_long)

    # 构造 t 网格
    t_grid = np.arange(cfg.t_min, cfg.t_max + 1e-8, cfg.t_step)

    # 为每个个体预计算在网格上的达标概率与风险
    probs, risks = _precompute_subject_risk_grid(df_subj, model, t_grid, cfg)

    # 按 BMI 排序（连续分组）
    df_sorted = df_subj.sort_values(by=["bmi", "subject_id"])  # 保留原索引以获取重排顺序
    order_idx = df_sorted.index.to_numpy()
    df_sorted = df_sorted.reset_index(drop=True)
    # 需要同步重排 probs, risks（依据原索引顺序）
    probs_sorted = probs[order_idx, :]
    risks_sorted = risks[order_idx, :]

    # DP 优化
    segments, seg_costs = dp_optimize_segments(
        df_sorted, probs_sorted, risks_sorted, t_grid, cfg.k_groups, cfg.alpha_attain, cfg.min_group_size, cfg.lambda_alpha, cfg.lambda_balance, cfg.lambda_trend
    )

    # 组装输出
    rows = []
    for g, ((l, r), c) in enumerate(zip(segments, seg_costs), start=1):
        seg_df = df_sorted.iloc[l : r + 1]
        bmi_low = float(seg_df["bmi"].min())
        bmi_high = float(seg_df["bmi"].max())
        rows.append({
            "group": g,
            "bmi_low": bmi_low,
            "bmi_high": bmi_high,
            "t_opt_weeks": round(c.best_t, 3),
            "avg_attain_prob": round(c.avg_attain, 4),
            "avg_risk": round(c.avg_risk_report, 6),
            "n_subjects": int(len(seg_df)),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(cfg.output_csv, index=False, encoding="utf-8-sig")
    return out_df


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="问题2：基于BMI的最优分组与NIPT时点（Logit+DP）")
    p.add_argument("--input", dest="input_path", default=os.path.join(os.path.dirname(__file__), "..", "附件.xlsx"),
                   help="输入Excel路径（默认项目根目录的 附件.xlsx）")
    p.add_argument("--output", dest="output_csv", default=os.path.join(os.path.dirname(__file__), "..", "q2_groups.csv"),
                   help="输出CSV路径")
    p.add_argument("--k", dest="k_groups", type=int, default=5, help="BMI分组数K")
    p.add_argument("--alpha", dest="alpha_attain", type=float, default=0.9, help="组内平均达标概率下限")
    p.add_argument("--lambda_fail", dest="lambda_fail", type=float, default=1.0, help="未达标风险权重")
    p.add_argument("--lambda_time", dest="lambda_time", type=float, default=0.1, help="时间惩罚权重")
    p.add_argument("--lambda_alpha", dest="lambda_alpha", type=float, default=0.5, help="未达标软惩罚权重（推动更晚孕周以满足alpha）")
    p.add_argument("--lambda_balance", dest="lambda_balance", type=float, default=0.0, help="组均衡惩罚权重（缓解样本量失衡）")
    p.add_argument("--lambda_trend", dest="lambda_trend", type=float, default=0.0, help="时点随BMI递增趋势惩罚权重（鼓励高BMI更晚孕周）")
    p.add_argument("--c1", dest="c1_mid", type=float, default=1.0, help="中期惩罚水平（平滑上限1）")
    p.add_argument("--c2", dest="c2_late", type=float, default=3.0, help="晚期惩罚水平（平滑上限2）")
    p.add_argument("--c0", dest="c0_early", type=float, default=1.2, help="早期惩罚基线（<w1）")
    p.add_argument("--w1", dest="w1_mid", type=float, default=12.0, help="早->中平滑过渡阈值（周）")
    p.add_argument("--w2", dest="w2_late", type=float, default=28.0, help="中->晚平滑过渡阈值（周）")
    p.add_argument("--smooth_s", dest="smooth_s", type=float, default=0.5, help="平滑强度（越大越平滑）")
    p.add_argument("--t_min", dest="t_min", type=float, default=10.0, help="孕周下限")
    p.add_argument("--t_max", dest="t_max", type=float, default=25.0, help="孕周上限")
    p.add_argument("--t_step", dest="t_step", type=float, default=0.1, help="孕周步长")
    p.add_argument("--min_group", dest="min_group_size", type=int, default=20, help="每组最小样本数")
    return p


def main():
    args = build_argparser().parse_args()
    cfg = Q2Config(
        input_path=os.path.abspath(args.input_path),
        output_csv=os.path.abspath(args.output_csv),
        k_groups=args.k_groups,
        alpha_attain=args.alpha_attain,
        lambda_fail=args.lambda_fail,
        lambda_time=args.lambda_time,
        lambda_alpha=args.lambda_alpha,
        lambda_balance=args.lambda_balance,
        lambda_trend=args.lambda_trend,
        w1_mid=args.w1_mid,
        w2_late=args.w2_late,
        smooth_s=args.smooth_s,
        c1_mid=args.c1_mid,
        c2_late=args.c2_late,
        c0_early=args.c0_early,
        t_min=args.t_min,
        t_max=args.t_max,
        t_step=args.t_step,
        min_group_size=args.min_group_size,
    )

    os.makedirs(os.path.dirname(cfg.output_csv), exist_ok=True)
    out_df = solve_q2(cfg)
    print("生成分组与时点建议：")
    print(out_df)


if __name__ == "__main__":
    main()


