import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def run(input_csv: str, output_csv: str, output_report: str, thr: float) -> None:
    df = pd.read_csv(input_csv)
    if "prob_abnormal" not in df.columns:
        raise ValueError("输入CSV缺少列 prob_abnormal。请先运行 analyze_q4_nn.py 生成概率列。")
    if "label" not in df.columns:
        raise ValueError("输入CSV缺少列 label（真值标签）。")

    probs = df["prob_abnormal"].to_numpy(dtype=float)
    y_true = df["label"].to_numpy(dtype=int)
    y_pred = (probs >= float(thr)).astype(int)

    df_out = df.copy()
    df_out["pred_abnormal"] = y_pred
    df_out["thr_used"] = float(thr)

    # 评估
    try:
        auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        auc = np.nan
    rep = []
    rep.append("==== 高召回版阈值后处理 ====")
    rep.append(f"使用阈值: {thr:.2f}")
    rep.append(f"AUC(全表): {auc:.4f}" if not np.isnan(auc) else "AUC(全表): N/A")
    rep.append("--- 分类报告 ---")
    rep.append(classification_report(y_true, y_pred, digits=4))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    rep.append(f"混淆矩阵: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(output_report) or ".", exist_ok=True)
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    with open(output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(rep))


def build_argparser():
    p = argparse.ArgumentParser(description="基于指定阈值的Q4后处理与指标评估")
    p.add_argument("--input", default="q4_pred_nn.csv", help="输入CSV，包含prob_abnormal与label")
    p.add_argument("--output", default="q4_pred_nn_highrecall.csv", help="输出CSV")
    p.add_argument("--report", default="q4_report_nn_highrecall.txt", help="输出报告TXT")
    p.add_argument("--thr", type=float, default=0.15, help="用于判定的概率阈值")
    return p


def main():
    args = build_argparser().parse_args()
    run(args.input, args.output, args.report, args.thr)


if __name__ == "__main__":
    main()


