import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.inspection import permutation_importance


# -------------
# 配置
# -------------


@dataclass
class Q4Config:
    input_path: str
    sheet_name: Optional[str]
    output_pred_csv: str
    output_report_txt: str
    test_size: float = 0.2
    random_state: int = 42
    hidden_layer_sizes: Tuple[int, ...] = (64, 32)
    max_iter: int = 800
    alpha: float = 1e-4
    learning_rate_init: float = 1e-3
    early_stopping: bool = True
    pos_label_name: str = "女胎异常"
    # 数据增强
    augment: bool = False
    aug_target_n: int = 2000
    aug_noise_scale: float = 1.0
    augment_stratify: bool = False
    aug_pos_multiplier: float = 3.0
    aug_lowq_highbmi_multiplier: float = 2.0
    # 模型与交叉验证
    model: str = "mlp"  # mlp | lgbm | xgb
    cv_enable: bool = False
    cv_folds: int = 5
    augment_in_cv: bool = False


# -------------
# 列检测与数据准备
# -------------


def _normalize(s: str) -> str:
    return "".join(str(s).lower().split())


def detect_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    # 依据题干：X染色体Z值、13/18/21 Z值、GC含量、读段数比例、BMI、年龄、AB列（异常）
    cols = list(df.columns)
    norm_cols = {c: _normalize(c) for c in cols}

    def find(cands: List[str]) -> Optional[str]:
        cands_n = [_normalize(c) for c in cands]
        # 1) 完全相等
        for c, n in norm_cols.items():
            for cn in cands_n:
                if n == cn:
                    return c
        # 2) 包含
        for c, n in norm_cols.items():
            for cn in cands_n:
                if len(cn) >= 2 and cn in n:
                    return c
        # 3) 回退：字母列
        for cand in cands:
            if cand in df.columns:
                return cand
        return None

    name_map = {
        "x_z": find(["X染色体的Z值", "X染色体Z值", "xz", "T", "X Z"]),
        "y_z": find(["Y染色体的Z值", "Y染色体Z值", "yz", "U", "Y Z"]),
        "z13": find(["13号染色体的Z值", "13号Z值", "Q", "chrom13_z", "chr13 z"]),
        "z18": find(["18号染色体的Z值", "18号Z值", "R", "chrom18_z", "chr18 z"]),
        "z21": find(["21号染色体的Z值", "21号Z值", "S", "chrom21_z", "chr21 z"]),
        "gc13": find(["13号染色体的GC含量", "X", "gc13", "chr13 gc"]),
        "gc18": find(["18号染色体的GC含量", "Y", "gc18", "chr18 gc"]),
        "gc21": find(["21号染色体的GC含量", "Z", "gc21", "chr21 gc"]),
        "gc_all": find(["GC含量", "P", "gc", "总体GC含量"]),
        "bmi": find(["孕妇BMI指标", "孕妇BMI", "BMI", "K"]),
        "age": find(["孕妇年龄", "年龄", "C", "age"]),
        "mapped_ratio": find(["在参考基因组上比对的比例", "M", "比对比例", "mapped ratio"]),
        "repeat_ratio": find(["重复读段的比例", "N", "repeat ratio"]),
        "filtered_ratio": find(["被过滤掉的读段数占总读段数的比例", "AA", "filtered ratio"]),
        "reads_total": find(["原始测序数据的总读段数（个）", "总读段数", "L"]),
        "abnormal": find(["检测出的13号，18号，21号染色体非整倍体", "非整倍体", "AB", "胎儿异常", "女胎异常"]),
        "x_conc": find(["X染色体浓度", "W"]),
    }

    return name_map


def load_and_prepare(input_path: str, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    if isinstance(df, dict):
        # 如果读取了多表，优先匹配包含“女胎”的表名
        key = None
        for k in df.keys():
            if "女胎" in str(k):
                key = k
                break
        if key is None:
            # 回退第一个表
            key = list(df.keys())[0]
        df = df[key]
    name_map = detect_columns(df)

    # 构造统一字段
    data = pd.DataFrame()
    def col(mkey: str) -> Optional[str]:
        val = name_map.get(mkey)
        return val if val in df.columns else None

    def to_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    # 标签：女胎异常（依据 AB 或已有列）
    y_col = col("abnormal")
    if y_col is None:
        raise ValueError("未找到异常判定列（AB/胎儿异常/女胎异常）。请先运行 tools/update_attachment_abnormality.py 生成标签或检查表头。")
    y_raw = df[y_col]
    y = (~y_raw.isna() & (y_raw.astype(str).str.strip() != "")).astype(int)
    data["label"] = y

    # 数值特征
    num_map = {
        "x_z": col("x_z"),
        "z13": col("z13"),
        "z18": col("z18"),
        "z21": col("z21"),
        "gc13": col("gc13"),
        "gc18": col("gc18"),
        "gc21": col("gc21"),
        "gc_all": col("gc_all"),
        "mapped_ratio": col("mapped_ratio"),
        "repeat_ratio": col("repeat_ratio"),
        "filtered_ratio": col("filtered_ratio"),
        "reads_total": col("reads_total"),
        "bmi": col("bmi"),
        "age": col("age"),
        "x_conc": col("x_conc"),
    }
    for k, v in num_map.items():
        if v is not None:
            data[k] = to_num(df[v])
        else:
            data[k] = np.nan

    # 派生质量特征：
    # - 过滤比例越高越差、重复比例越高越差、mapped越高越好
    # - 构造综合质量分：q = mapped - 0.5*filtered - 0.5*repeat
    data["q_score"] = data["mapped_ratio"].fillna(0.0) - 0.5 * data["filtered_ratio"].fillna(0.0) - 0.5 * data["repeat_ratio"].fillna(0.0)
    # GC接近0.5的度量
    data["gc_closeness"] = 1.0 - (data["gc_all"].fillna(0.5) - 0.5).abs()

    # 仅保留女胎：Y列为空可作为辅助，但此处直接基于标签与 X 相关特征
    # 过滤极端值（稳健性）
    def clip_series(s: pd.Series, ql: float = 0.001, qh: float = 0.999) -> pd.Series:
        lo, hi = s.quantile(ql), s.quantile(qh)
        return s.clip(lower=lo, upper=hi)

    for colname in ["z13", "z18", "z21", "x_z", "gc13", "gc18", "gc21", "gc_all", "reads_total", "bmi", "age"]:
        if colname in data.columns:
            data[colname] = clip_series(data[colname])

    return data, name_map


# -------------
# 训练与评估
# -------------


def _augment_training_set(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    target_n: int,
    rng: np.random.RandomState,
    cfg: Q4Config,
) -> Tuple[pd.DataFrame, np.ndarray]:
    n0 = len(X_train)
    if target_n <= n0:
        target_n = int(n0 * 1.2)
    need = max(0, target_n - n0)
    if need == 0:
        return X_train, y_train

    # 分层/加权采样：按 label、BMI 高分位、q_score 低分位提高权重
    if cfg.augment_stratify and ("bmi" in X_train.columns or "q_score" in X_train.columns):
        w = np.ones(n0, dtype=float)
        # label 加权（提高阳性样本被抽中的概率）
        if y_train is not None:
            w = np.where(y_train == 1, w * float(cfg.aug_pos_multiplier), w)
        # 质量与BMI分层
        try:
            bmi = X_train["bmi"].to_numpy(dtype=float) if "bmi" in X_train.columns else None
            qscore = X_train["q_score"].to_numpy(dtype=float) if "q_score" in X_train.columns else None
            hi_bmi = False
            lo_q = False
            if bmi is not None and np.isfinite(bmi).any():
                thr_bmi = np.nanquantile(bmi, 0.66)
                hi_bmi = bmi >= thr_bmi
            if qscore is not None and np.isfinite(qscore).any():
                thr_q = np.nanquantile(qscore, 0.33)
                lo_q = qscore <= thr_q
            if isinstance(hi_bmi, np.ndarray) and isinstance(lo_q, np.ndarray):
                mask_boost = hi_bmi & lo_q
                w = np.where(mask_boost, w * float(cfg.aug_lowq_highbmi_multiplier), w)
        except Exception:
            pass
        w = np.nan_to_num(w, nan=1.0, posinf=1.0, neginf=1.0)
        w = w / np.sum(w)
        idx = rng.choice(n0, size=need, replace=True, p=w)
    else:
        idx = rng.choice(n0, size=need, replace=True)
    X_base = X_train.iloc[idx].copy().reset_index(drop=True)
    y_new = y_train[idx]

    # 按列添加小噪声（经验值，保证合理范围）
    def add_noise(series: pd.Series, std: float, a: Optional[float] = None, b: Optional[float] = None, is_mult: bool = False) -> pd.Series:
        vals = series.to_numpy(dtype=float)
        if is_mult:
            noise = rng.lognormal(mean=0.0, sigma=std, size=len(vals))
            vals = vals * noise
        else:
            noise = rng.normal(loc=0.0, scale=std, size=len(vals))
            vals = vals + noise
        if a is not None:
            vals = np.maximum(a, vals)
        if b is not None:
            vals = np.minimum(b, vals)
        return pd.Series(vals, index=series.index)

    bounded01 = [c for c in ["gc_all", "gc13", "gc18", "gc21", "mapped_ratio", "repeat_ratio", "filtered_ratio"] if c in X_base.columns]
    for c in bounded01:
        X_base[c] = add_noise(X_base[c].fillna(0.5), std=0.02 * float(cfg.aug_noise_scale), a=0.0, b=1.0)

    z_cols = [c for c in ["x_z", "z13", "z18", "z21"] if c in X_base.columns]
    for c in z_cols:
        X_base[c] = add_noise(X_base[c].fillna(0.0), std=0.2 * float(cfg.aug_noise_scale), a=-8.0, b=8.0)

    if "reads_total" in X_base.columns:
        X_base["reads_total"] = add_noise(X_base["reads_total"].fillna(1e6), std=0.15 * float(cfg.aug_noise_scale), a=0.0, is_mult=True)

    if "bmi" in X_base.columns:
        X_base["bmi"] = add_noise(X_base["bmi"].fillna(25.0), std=1.0 * float(cfg.aug_noise_scale), a=12.0, b=70.0)
    if "age" in X_base.columns:
        X_base["age"] = add_noise(X_base["age"].fillna(28.0), std=1.5 * float(cfg.aug_noise_scale), a=15.0, b=55.0)
    if "x_conc" in X_base.columns:
        X_base["x_conc"] = add_noise(X_base["x_conc"].fillna(0.0), std=0.03 * float(cfg.aug_noise_scale), a=-0.2, b=1.0)

    # 重新计算派生特征
    if all(c in X_base.columns for c in ["mapped_ratio", "filtered_ratio", "repeat_ratio"]):
        X_base["q_score"] = X_base["mapped_ratio"].fillna(0.0) - 0.5 * X_base["filtered_ratio"].fillna(0.0) - 0.5 * X_base["repeat_ratio"].fillna(0.0)
    if "gc_all" in X_base.columns:
        X_base["gc_closeness"] = 1.0 - (X_base["gc_all"].fillna(0.5) - 0.5).abs()

    X_aug = pd.concat([X_train.reset_index(drop=True), X_base], axis=0).reset_index(drop=True)
    y_aug = np.concatenate([y_train, y_new], axis=0)
    return X_aug, y_aug


def _build_pipeline(model_name: str, numeric_features: List[str], cfg: Q4Config, y_train: Optional[np.ndarray] = None) -> Pipeline:
    # 预处理：树模型不需要标准化；MLP需要
    if model_name.lower() == "mlp":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), numeric_features),
            ],
            remainder="drop",
        )
        clf = MLPClassifier(
            hidden_layer_sizes=cfg.hidden_layer_sizes,
            random_state=cfg.random_state,
            max_iter=cfg.max_iter,
            alpha=cfg.alpha,
            learning_rate_init=cfg.learning_rate_init,
            early_stopping=cfg.early_stopping,
            n_iter_no_change=20,
            verbose=False,
        )
        return Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    # 树模型：LightGBM / XGBoost
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_features),
        ],
        remainder="drop",
    )
    pos = int(np.sum(y_train == 1)) if y_train is not None else 0
    neg = int(np.sum(y_train == 0)) if y_train is not None else 0
    scale_pos_weight = float(neg / max(1, pos)) if pos > 0 and neg > 0 else 1.0

    if model_name.lower() == "lgbm":
        try:
            import lightgbm as lgb  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"缺少 lightgbm 依赖或导入失败: {exc}")
        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=cfg.random_state,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )
        return Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    if model_name.lower() == "xgb":
        try:
            from xgboost import XGBClassifier  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"缺少 xgboost 依赖或导入失败: {exc}")
        clf = XGBClassifier(
            objective="binary:logistic",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=cfg.random_state,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )
        return Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    raise ValueError(f"未知模型: {model_name}")


def train_eval_nn(df: pd.DataFrame, cfg: Q4Config, file_suffix: str = "") -> Tuple[pd.DataFrame, str, Dict[str, object]]:
    # 选择特征
    feature_cols = [
        "x_z", "z13", "z18", "z21",
        "gc13", "gc18", "gc21", "gc_all",
        "mapped_ratio", "repeat_ratio", "filtered_ratio",
        "reads_total", "bmi", "age", "x_conc",
        "q_score", "gc_closeness",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df["label"].astype(int).to_numpy()

    # 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y if np.unique(y).size > 1 else None
    )

    # 丢弃在训练集中全缺失的特征，避免填充器警告/错误
    valid_features = [c for c in feature_cols if X_train[c].notna().any()]
    if len(valid_features) == 0:
        raise ValueError("训练集中所有候选特征均为空，无法训练模型。请检查数据列。")
    X_train = X_train[valid_features].copy()
    X_test = X_test[valid_features].copy()

    # 构建模型管道
    numeric_features = valid_features
    pipe = _build_pipeline(cfg.model, numeric_features, cfg, y_train)

    # 类别不平衡/数据增强
    pos = int(np.sum(y_train == 1))
    neg = int(np.sum(y_train == 0))
    if cfg.augment:
        rng = np.random.RandomState(cfg.random_state)
        X_train_bal, y_train_bal = _augment_training_set(X_train, y_train, cfg.aug_target_n, rng, cfg)
    else:
        if pos > 0 and neg > 0 and neg > pos:
            factor = int(np.floor(neg / pos)) - 1
            factor = max(0, factor)
            X_pos = X_train[y_train == 1]
            y_pos = y_train[y_train == 1]
            X_aug_list = [X_train]
            y_aug_list = [y_train]
            for _ in range(factor):
                X_aug_list.append(X_pos)
                y_aug_list.append(y_pos)
            rem = neg - (pos * (factor + 1))
            if rem > 0 and len(X_pos) > 0:
                idx = np.random.RandomState(cfg.random_state).choice(len(X_pos), size=rem, replace=True)
                X_aug_list.append(X_pos.iloc[idx])
                y_aug_list.append(y_pos[idx])
            X_train_bal = pd.concat(X_aug_list, axis=0).reset_index(drop=True)
            y_train_bal = np.concatenate(y_aug_list, axis=0)
        else:
            X_train_bal = X_train
            y_train_bal = y_train

    pipe.fit(X_train_bal, y_train_bal)

    # 预测概率
    proba_test = pipe.predict_proba(X_test)[:, 1]

    # 阈值调优：使用F1最大或Youden指数最大
    thresholds = np.linspace(0.05, 0.95, 19)
    best = {"thr": 0.5, "f1": -1.0, "youden": -1.0, "prec": 0.0, "rec": 0.0}
    thr_rows: List[Dict[str, float]] = []
    for t in thresholds:
        y_pred = (proba_test >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        youden = sens + spec - 1.0
        thr_rows.append({"threshold": float(t), "precision": float(prec), "recall": float(rec), "f1": float(f1), "sensitivity": float(sens), "specificity": float(spec), "youden": float(youden)})
        if f1 > best["f1"] + 1e-12 or (abs(f1 - best["f1"]) <= 1e-12 and youden > best["youden"]):
            best = {"thr": float(t), "f1": float(f1), "youden": float(youden), "prec": float(prec), "rec": float(rec)}

    # 以最佳阈值输出报告
    y_pred_best = (proba_test >= best["thr"]).astype(int)
    auc = roc_auc_score(y_test, proba_test) if len(np.unique(y_test)) > 1 else np.nan
    report = []
    report.append(f"==== 女胎异常判定（{cfg.model.upper()}）====")
    report.append(f"样本量: 训练 {len(X_train_bal)}, 测试 {len(X_test)}; 特征数: {len(numeric_features)}")
    report.append(f"类别计数(训练): 正={pos}, 负={neg}; 使用正类权重≈{(neg/max(1,pos)):.3f}")
    report.append(f"AUC(测试): {auc:.4f}" if not np.isnan(auc) else "AUC(测试): N/A")
    report.append(f"最佳阈值: {best['thr']:.2f}，F1={best['f1']:.4f}，Precision={best['prec']:.4f}，Recall={best['rec']:.4f}，Youden={best['youden']:.4f}")
    report.append("--- 分类报告(最佳阈值) ---")
    report.append(classification_report(y_test, y_pred_best, digits=4))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best, labels=[0, 1]).ravel()
    report.append(f"混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # 置换重要度（全局可解释性）
    try:
        pi = permutation_importance(
            pipe, X_test, y_test,
            scoring="roc_auc" if len(np.unique(y_test)) > 1 else None,
            n_repeats=20, random_state=cfg.random_state, n_jobs=None,
        )
        importances = pd.DataFrame({
            "feature": valid_features,
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std,
        }).sort_values("importance_mean", ascending=False)
        topk = importances.head(10)
        report.append("--- 置换重要度 Top10 (基于ROC-AUC) ---")
        for _, r in topk.iterrows():
            report.append(f"{r['feature']}: {r['importance_mean']:.6f} ± {r['importance_std']:.6f}")
    except Exception as exc:
        importances = None
        report.append(f"置换重要度计算失败: {exc}")

    report_txt = "\n".join(report)

    # 整表预测并导出
    proba_all = pipe.predict_proba(X)[:, 1]
    df_out = df.copy()
    df_out["prob_abnormal"] = proba_all
    df_out["pred_abnormal"] = (df_out["prob_abnormal"] >= best["thr"]).astype(int)
    df_out["thr_used"] = best["thr"]

    # 写出可解释性文件与阈值表
    out_dir = os.path.dirname(cfg.output_report_txt) or "."
    thr_df = pd.DataFrame(thr_rows)
    thr_path = os.path.join(out_dir, f"q4_threshold_metrics{file_suffix}.csv")
    thr_df.to_csv(thr_path, index=False, encoding="utf-8-sig")
    if importances is not None:
        fi_path = os.path.join(out_dir, f"q4_feature_importance{file_suffix}.csv")
        importances.to_csv(fi_path, index=False, encoding="utf-8-sig")

    stats = {
        "auc": auc,
        "best_thr": best["thr"],
        "best_f1": best["f1"],
        "best_prec": best["prec"],
        "best_rec": best["rec"],
        "best_youden": best["youden"],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "n_train": len(X_train_bal),
        "n_test": len(X_test),
        "features": numeric_features,
        "thr_csv": thr_path,
        "fi_csv": (fi_path if importances is not None else None),
    }

    return df_out, report_txt, stats


def cross_validate_model(df: pd.DataFrame, cfg: Q4Config, file_suffix: str = "") -> str:
    # 特征选择与准备
    feature_cols = [
        "x_z", "z13", "z18", "z21",
        "gc13", "gc18", "gc21", "gc_all",
        "mapped_ratio", "repeat_ratio", "filtered_ratio",
        "reads_total", "bmi", "age", "x_conc",
        "q_score", "gc_closeness",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    X_full = df[feature_cols].copy()
    y_full = df["label"].astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=max(2, int(cfg.cv_folds)), shuffle=True, random_state=cfg.random_state)
    rows: List[Dict[str, float]] = []

    fold_idx = 0
    for tr_idx, te_idx in skf.split(X_full, y_full):
        fold_idx += 1
        X_train = X_full.iloc[tr_idx].copy()
        X_test = X_full.iloc[te_idx].copy()
        y_train = y_full[tr_idx]
        y_test = y_full[te_idx]

        # 丢弃全缺失特征
        valid_features = [c for c in feature_cols if X_train[c].notna().any()]
        X_train = X_train[valid_features].copy()
        X_test = X_test[valid_features].copy()

        # 训练集增强（可选）
        if cfg.augment and cfg.augment_in_cv:
            rng = np.random.RandomState(cfg.random_state + fold_idx)
            X_train, y_train = _augment_training_set(X_train, y_train, cfg.aug_target_n, rng, cfg)

        # 构建与训练
        pipe = _build_pipeline(cfg.model, valid_features, cfg, y_train)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]

        # 阈值扫描
        thresholds = np.linspace(0.05, 0.95, 19)
        best = {"thr": 0.5, "f1": -1.0, "youden": -1.0, "prec": 0.0, "rec": 0.0}
        for t in thresholds:
            y_pred = (proba >= t).astype(int)
            prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            sens = tp / max(1, tp + fn)
            spec = tn / max(1, tn + fp)
            youden = sens + spec - 1.0
            if f1 > best["f1"] + 1e-12 or (abs(f1 - best["f1"]) <= 1e-12 and youden > best["youden"]):
                best = {"thr": float(t), "f1": float(f1), "youden": float(youden), "prec": float(prec), "rec": float(rec)}

        auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else np.nan
        rows.append({
            "fold": fold_idx,
            "auc": float(auc) if not np.isnan(auc) else np.nan,
            "best_thr": best["thr"],
            "best_f1": best["f1"],
            "best_prec": best["prec"],
            "best_rec": best["rec"],
            "best_youden": best["youden"],
        })

    df_cv = pd.DataFrame(rows)
    out_dir = os.path.dirname(cfg.output_report_txt) or "."
    cv_csv = os.path.join(out_dir, f"q4_cv_results_{cfg.model}{file_suffix}.csv")
    df_cv.to_csv(cv_csv, index=False, encoding="utf-8-sig")

    # 汇总
    def fmt(x: float) -> str:
        return "N/A" if (x is None or np.isnan(x)) else f"{x:.4f}"
    summary_lines = [
        f"==== {cfg.model.upper()} {cfg.cv_folds}折交叉验证 ====",
        f"AUC: 均值={fmt(df_cv['auc'].mean())}，标准差={fmt(df_cv['auc'].std())}",
        f"F1: 均值={fmt(df_cv['best_f1'].mean())}，标准差={fmt(df_cv['best_f1'].std())}",
        f"Precision: 均值={fmt(df_cv['best_prec'].mean())}",
        f"Recall: 均值={fmt(df_cv['best_rec'].mean())}",
        f"Youden: 均值={fmt(df_cv['best_youden'].mean())}",
    ]
    cv_txt = os.path.join(out_dir, f"q4_cv_summary_{cfg.model}{file_suffix}.txt")
    with open(cv_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(cv_csv)
    print(cv_txt)
    return cv_txt


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="问题4：基于神经网络的女胎异常判定（MLP）")
    p.add_argument("--input", dest="input_path", default=str(Path(__file__).resolve().parents[1] / "附件.xlsx"), help="输入Excel路径")
    p.add_argument("--sheet", dest="sheet_name", default=None, help="工作表名称，可选；为空则自动选择")
    p.add_argument("--pred", dest="output_pred_csv", default=str(Path(__file__).resolve().parents[0] / "q4_pred_nn.csv"), help="输出预测CSV")
    p.add_argument("--report", dest="output_report_txt", default=str(Path(__file__).resolve().parents[0] / "q4_report_nn.txt"), help="输出报告TXT")
    p.add_argument("--test_size", dest="test_size", type=float, default=0.2, help="测试集比例")
    p.add_argument("--seed", dest="random_state", type=int, default=42, help="随机种子")
    p.add_argument("--hidden", dest="hidden", type=str, default="64,32", help="隐层结构，如 128,64,32")
    p.add_argument("--max_iter", dest="max_iter", type=int, default=800, help="最大迭代步数")
    p.add_argument("--alpha", dest="alpha", type=float, default=1e-4, help="L2正则系数")
    p.add_argument("--lr", dest="learning_rate_init", type=float, default=1e-3, help="学习率")
    p.add_argument("--augment", dest="augment", action="store_true", help="启用数据增强并输出对比结果")
    p.add_argument("--aug_target_n", dest="aug_target_n", type=int, default=2000, help="增强后训练集目标样本量")
    p.add_argument("--augment_stratify", dest="augment_stratify", action="store_true", help="分层增强：按label/BMI/质量加权采样")
    p.add_argument("--aug_noise_scale", dest="aug_noise_scale", type=float, default=1.0, help="增强噪声强度缩放")
    p.add_argument("--aug_pos_multiplier", dest="aug_pos_multiplier", type=float, default=3.0, help="阳性样本采样权重倍数")
    p.add_argument("--aug_lowq_highbmi_multiplier", dest="aug_lowq_highbmi_multiplier", type=float, default=2.0, help="低质量且高BMI样本权重倍数")
    p.add_argument("--model", dest="model", type=str, default="mlp", help="模型: mlp|lgbm|xgb")
    p.add_argument("--cv", dest="cv_enable", action="store_true", help="启用K折交叉验证")
    p.add_argument("--cv_folds", dest="cv_folds", type=int, default=5, help="交叉验证折数")
    p.add_argument("--augment_in_cv", dest="augment_in_cv", action="store_true", help="在交叉验证中也应用增强")
    return p


def main():
    args = build_argparser().parse_args()

    hidden = tuple(int(x) for x in str(args.hidden).split(",") if str(x).strip())
    cfg = Q4Config(
        input_path=os.path.abspath(args.input_path),
        sheet_name=args.sheet_name,
        output_pred_csv=os.path.abspath(args.output_pred_csv),
        output_report_txt=os.path.abspath(args.output_report_txt),
        test_size=args.test_size,
        random_state=args.random_state,
        hidden_layer_sizes=hidden if len(hidden) > 0 else (64, 32),
        max_iter=args.max_iter,
        alpha=args.alpha,
        learning_rate_init=args.learning_rate_init,
        augment=bool(args.augment),
        aug_target_n=int(args.aug_target_n),
        augment_stratify=bool(args.augment_stratify),
        aug_noise_scale=float(args.aug_noise_scale),
        aug_pos_multiplier=float(args.aug_pos_multiplier),
        aug_lowq_highbmi_multiplier=float(args.aug_lowq_highbmi_multiplier),
        model=str(args.model).lower(),
        cv_enable=bool(args.cv_enable),
        cv_folds=int(args.cv_folds),
        augment_in_cv=bool(args.augment_in_cv),
    )

    os.makedirs(os.path.dirname(cfg.output_pred_csv), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.output_report_txt), exist_ok=True)

    df, name_map = load_and_prepare(cfg.input_path, cfg.sheet_name)
    # 基线训练
    pred_df, report_txt, stats_base = train_eval_nn(df, cfg, file_suffix="")

    pred_df.to_csv(cfg.output_pred_csv, index=False, encoding="utf-8-sig")
    with open(cfg.output_report_txt, "w", encoding="utf-8") as f:
        f.write(report_txt)

    print("已生成：")
    print(cfg.output_pred_csv)
    print(cfg.output_report_txt)
    print(stats_base.get("thr_csv"))
    if stats_base.get("fi_csv"):
        print(stats_base.get("fi_csv"))

    # 若启用增强：再训练并对比
    if cfg.augment:
        # 为增强版使用带后缀的输出
        pred_path_aug = os.path.splitext(cfg.output_pred_csv)
        pred_path_aug = pred_path_aug[0] + "_aug" + pred_path_aug[1]
        report_path_aug = os.path.splitext(cfg.output_report_txt)
        report_path_aug = report_path_aug[0] + "_aug" + report_path_aug[1]

        # 临时开启增强（只影响本次调用）
        cfg_aug = Q4Config(**{**cfg.__dict__})
        cfg_aug.augment = True

        pred_df_aug, report_txt_aug, stats_aug = train_eval_nn(df, cfg_aug, file_suffix="_aug")
        pred_df_aug.to_csv(pred_path_aug, index=False, encoding="utf-8-sig")
        with open(report_path_aug, "w", encoding="utf-8") as f:
            f.write(report_txt_aug)

        # 写出对比
        compare_path = os.path.join(os.path.dirname(cfg.output_report_txt) or ".", "q4_aug_compare.txt")
        lines = []
        lines.append("==== 增强前后对比 ====")
        lines.append(f"训练集(前/后): {stats_base['n_train']} / {stats_aug['n_train']}")
        lines.append(f"测试集: {stats_base['n_test']}")
        def fmt(s):
            return "N/A" if (s is None or (isinstance(s, float) and np.isnan(s))) else f"{s:.4f}"
        lines.append(f"AUC(前/后): {fmt(stats_base['auc'])} / {fmt(stats_aug['auc'])}")
        lines.append(f"最佳阈值(前/后): {fmt(stats_base['best_thr'])} / {fmt(stats_aug['best_thr'])}")
        lines.append(f"F1(前/后): {fmt(stats_base['best_f1'])} / {fmt(stats_aug['best_f1'])}")
        lines.append(f"Precision(前/后): {fmt(stats_base['best_prec'])} / {fmt(stats_aug['best_prec'])}")
        lines.append(f"Recall(前/后): {fmt(stats_base['best_rec'])} / {fmt(stats_aug['best_rec'])}")
        lines.append(f"Youden(前/后): {fmt(stats_base['best_youden'])} / {fmt(stats_aug['best_youden'])}")
        with open(compare_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(pred_path_aug)
        print(report_path_aug)
        print(stats_aug.get("thr_csv"))
        if stats_aug.get("fi_csv"):
            print(stats_aug.get("fi_csv"))
        print(compare_path)

    # 若启用CV
    if cfg.cv_enable:
        cross_validate_model(df, cfg, file_suffix="")


if __name__ == "__main__":
    main()


