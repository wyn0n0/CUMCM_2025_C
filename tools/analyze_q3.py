import pandas as pd
import numpy as np
from pathlib import Path
import re

# Config
EXCEL_PATH = Path(__file__).resolve().parents[1] / "附件.xlsx"
MIN_WEEK = 10.0
MAX_WEEK = 25.0
BMI_GROUPS = [
    (20.0, 28.0),
    (28.0, 32.0),
    (32.0, 36.0),
    (36.0, 40.0),
    (40.0, float("inf")),
]
PROB_THRESHOLD = 0.85
MIN_SAMPLES_PER_BIN = 5
BIN_WIDTH_WEEKS = 1.0
NOISE_LEVELS = [0.0, 0.10, 0.15]
RANDOM_SEED = 42
USE_QUANTILE_BMI_GROUPS = True
BMI_GROUP_COUNT = 5
USE_DEDUP = True
USE_RISK_MINIMIZATION = True
# Risk weights: emphasize failure vs time
LAMBDA_TIME = 0.25
LAMBDA_FAIL = 0.75
USE_CONTINUOUS_RISK = True
RISK_GRID_STEP = 0.1
RISK_MIN_WINDOW = 0.6
RISK_MAX_WINDOW = 2.0
RISK_WINDOW_STEP = 0.2
RISK_MIN_SAMPLES = 8
CF_FAIL = 1.0
CF_WAIT = 0.04

def compute_time_penalty(week: float) -> float:
    """Linear time cost from MIN_WEEK to MAX_WEEK to ensure positive waiting cost when w > MIN_WEEK.
    Normalized: 0 at MIN_WEEK; 1 at MAX_WEEK; clamped to [0,1].
    """
    if pd.isna(week):
        return 1.0
    span = max(1e-9, (MAX_WEEK - MIN_WEEK))
    val = (week - MIN_WEEK) / span
    if val < 0:
        return 0.0
    if val > 1:
        return 1.0
    return float(val)

def estimate_pass_rate_window(weeks_arr: np.ndarray, passed_arr: np.ndarray, w: float,
                              min_samples: int, min_window: float, max_window: float,
                              window_step: float) -> tuple[float, float, int]:
    if len(weeks_arr) == 0:
        return np.nan, np.nan, 0
    window = min_window
    while window <= max_window + 1e-9:
        half = window / 2.0
        mask = (weeks_arr >= w - half) & (weeks_arr <= w + half)
        count = int(mask.sum())
        if count >= min_samples:
            rate = float(np.mean(passed_arr[mask])) if count > 0 else np.nan
            return rate, window, count
        window += window_step
    # Fallback: use all available as last resort
    rate = float(np.mean(passed_arr)) if len(passed_arr) > 0 else np.nan
    return rate, np.nan, len(passed_arr)

np.random.seed(RANDOM_SEED)


def parse_gestation_weeks(value) -> float:
    """Parse 孕周（周数+天数） to decimal weeks.
    Accepts formats like '12+3', '12＋3', '12周3天', 12.5, or datetime-like.
    Returns np.nan if cannot parse.
    """
    if pd.isna(value):
        return np.nan
    # Already numeric
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    # Normalize plus sign
    s = s.replace('＋', '+').replace('周', '+').replace('天', '')
    # Remove spaces
    s = s.replace(' ', '')
    if '+' in s:
        parts = s.split('+')
        try:
            weeks = float(parts[0])
            days = float(parts[1]) if len(parts) > 1 and parts[1] != '' else 0.0
            return weeks + days / 7.0
        except Exception:
            pass
    # Regex fallback like 12周3天 / 12 3
    m = re.search(r"(\d+)[^0-9]+(\d+)", s)
    if m:
        try:
            return float(m.group(1)) + float(m.group(2)) / 7.0
        except Exception:
            pass
    # Try direct float
    try:
        return float(s)
    except Exception:
        return np.nan


def detect_columns(df: pd.DataFrame):
    """Map expected semantics to actual column names in the Excel."""
    cols = {c: str(c) for c in df.columns}
    name_map = {}
    # Try direct letter mapping A..Z, AA.. etc if present
    expected_letters = {
        'id': 'A',
        'mother_code': 'B',
        'age': 'C',
        'height': 'D',
        'weight': 'E',
        'last_menses': 'F',
        'pregnancy_mode': 'G',
        'test_time': 'H',
        'draw_count': 'I',
        'gest_weeks': 'J',
        'bmi': 'K',
        'reads_total': 'L',
        'reads_filtered_ratio': 'AA',
        'mapped_ratio': 'M',
        'repeat_ratio': 'N',
        'unique_reads': 'O',
        'gc_all': 'P',
        'chrom13_z': 'Q',
        'chrom18_z': 'R',
        'chrom21_z': 'S',
        'x_z': 'T',
        'y_z': 'U',
        'y_conc': 'V',
        'x_conc': 'W',
        'gc13': 'X',
        'gc18': 'Y',
        'gc21': 'Z',
        'gravidity': 'AC',
        'parity': 'AD',
        'infant_health': 'AE',
    }
    # If columns are exactly letters
    col_set = set(df.columns.astype(str))
    if set(expected_letters.values()).issubset(col_set):
        # simple case
        inv = {v: k for k, v in expected_letters.items()}
        for col in df.columns.astype(str):
            if col in inv:
                name_map[inv[col]] = col
        return name_map
    # Try fuzzy Chinese names
    def normalize(s: str) -> str:
        return "".join(str(s).lower().split())
    norm_cols = {col: normalize(col) for col in df.columns}

    def find_col(candidates):
        cand_norms = [normalize(c) for c in candidates]
        # 1) exact normalized equality first
        for col, ncs in norm_cols.items():
            for cn in cand_norms:
                if ncs == cn:
                    return col
        # 2) then contains, but only for tokens length >= 2 to avoid 'v' matching 'ivf'
        for col, ncs in norm_cols.items():
            for cn in cand_norms:
                if len(cn) >= 2 and cn in ncs:
                    return col
        return None

    name_map['mother_code'] = find_col(['孕妇代码', '母亲', '样本', '编号', 'ID', 'id', 'A', 'B'])
    name_map['age'] = find_col(['年龄', 'age', 'C'])
    name_map['height'] = find_col(['身高', 'height', 'D'])
    name_map['weight'] = find_col(['体重', 'weight', 'E'])
    name_map['gest_weeks'] = find_col(['检测孕周', '孕周', '周数+天数', 'J'])
    name_map['bmi'] = find_col(['孕妇BMI', 'BMI', 'K'])
    name_map['y_conc'] = find_col(['Y染色体浓度', 'Y 染色体浓度', 'Y染色体游离DNA片段的比例', 'Y浓度'])
    name_map['filtered_ratio'] = find_col(['被过滤掉读段数的比例', '过滤', 'AA'])
    name_map['mapped_ratio'] = find_col(['在参考基因组上比对的比例', '比对的比例', 'M'])
    name_map['repeat_ratio'] = find_col(['重复读段的比例', '重复', 'N'])
    name_map['unique_reads'] = find_col(['唯一比对的读段数', '唯一比对', 'O'])
    name_map['gc_all'] = find_col(['GC含量', 'gc含量', 'P'])

    return name_map


def add_noise(values: np.ndarray, noise_level: float) -> np.ndarray:
    if noise_level <= 0:
        return values
    noise = np.random.normal(loc=0.0, scale=noise_level, size=values.shape)
    noisy = values * (1.0 + noise)
    noisy = np.clip(noisy, a_min=0.0, a_max=None)
    return noisy


def select_reliable_row(group_df: pd.DataFrame, name_map: dict) -> pd.Series:
    """Select one reliable measurement for a mother based on:
    1) Prefer rows with y_conc >= 0.04, choose earliest gestational week among them
    2) If multiple tie, pick highest quality score
    3) If none passed, pick highest quality score; if tie, pick highest y_conc; if tie, earliest week
    """
    df = group_df.copy()
    # Quality fields
    mapped = name_map.get('mapped_ratio')
    filtered = name_map.get('filtered_ratio')
    repeat_r = name_map.get('repeat_ratio')
    unique_r = name_map.get('unique_reads')
    gc_all = name_map.get('gc_all')

    # Normalize numeric quality
    def to_num(s):
        return pd.to_numeric(s, errors='coerce')

    if mapped in df.columns:
        df['_mapped'] = to_num(df[mapped])
    else:
        df['_mapped'] = np.nan
    if filtered in df.columns:
        df['_filtered'] = to_num(df[filtered])
    else:
        df['_filtered'] = np.nan
    if repeat_r in df.columns:
        df['_repeat'] = to_num(df[repeat_r])
    else:
        df['_repeat'] = np.nan
    if unique_r in df.columns:
        df['_unique'] = to_num(df[unique_r])
    else:
        df['_unique'] = np.nan
    if gc_all in df.columns:
        df['_gc'] = to_num(df[gc_all])
    else:
        df['_gc'] = np.nan

    # Unique reads normalization within the group
    umax = np.nanmax(df['_unique'].to_numpy(dtype=float)) if df['_unique'].notna().any() else np.nan
    if not np.isnan(umax) and umax > 0:
        df['_unique_norm'] = df['_unique'] / umax
    else:
        df['_unique_norm'] = np.nan

    # GC closeness to 0.5
    df['_gc_closeness'] = 1.0 - (df['_gc'] - 0.5).abs()

    # Quality score (higher is better)
    df['_q'] = 0.0
    df['_q'] = df['_q'] + df['_mapped'].fillna(0.0) * 2.0
    df['_q'] = df['_q'] + (1.0 - df['_filtered'].fillna(1.0))
    df['_q'] = df['_q'] + (1.0 - df['_repeat'].fillna(1.0))
    df['_q'] = df['_q'] + df['_unique_norm'].fillna(0.0)
    df['_q'] = df['_q'] + df['_gc_closeness'].fillna(0.0)

    # Passed mask
    passed_mask = df['y_conc'] >= 0.04
    passed_df = df[passed_mask].copy()
    if not passed_df.empty:
        # earliest week first, then q desc
        passed_df = passed_df.sort_values(['weeks', '_q'], ascending=[True, False])
        return passed_df.iloc[0]

    # No passed: pick best quality; tie -> highest y_conc; tie -> earliest week
    df = df.sort_values(['_q', 'y_conc', 'weeks'], ascending=[False, False, True])
    return df.iloc[0]


def build_single_measurements(data: pd.DataFrame, name_map: dict) -> pd.DataFrame:
    """Group by mother and return one row per mother with selected reliable measurement."""
    grouped = []
    for code, grp in data.groupby('code'):
        try:
            row = select_reliable_row(grp, name_map)
            grouped.append(row)
        except Exception:
            continue
    if not grouped:
        return data.iloc[0:0]
    result = pd.DataFrame(grouped)
    # Keep only essential columns
    cols = ['code', 'age', 'height', 'weight', 'weeks', 'bmi', 'y_conc']
    extra = [c for c in cols if c in result.columns]
    return result[extra].copy()


def compute_group_recommendations(df: pd.DataFrame, name_map: dict):
    # Extract needed columns
    code_col = name_map.get('mother_code')
    age_col = name_map.get('age')
    height_col = name_map.get('height')
    weight_col = name_map.get('weight')
    weeks_col = name_map.get('gest_weeks')
    bmi_col = name_map.get('bmi')
    y_col = name_map.get('y_conc')

    if y_col is None or weeks_col is None or bmi_col is None:
        raise RuntimeError("无法识别必要列：孕周、BMI或Y染色体浓度。")

    data = pd.DataFrame({
        'code': df[code_col] if code_col in df.columns else np.arange(len(df)),
        'age': df[age_col] if age_col in df.columns else np.nan,
        'height': df[height_col] if height_col in df.columns else np.nan,
        'weight': df[weight_col] if weight_col in df.columns else np.nan,
        'weeks_raw': df[weeks_col],
        'bmi': df[bmi_col],
        'y_conc': df[y_col],
    })

    # Clean
    data['weeks'] = data['weeks_raw'].apply(parse_gestation_weeks)
    data['bmi'] = pd.to_numeric(data['bmi'], errors='coerce')
    # Clean Y浓度: strip % and convert
    def clean_percent(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().replace('%', '')
        try:
            return float(s)
        except Exception:
            return np.nan
    y_raw = data['y_conc'].apply(clean_percent)
    # If scale seems like percentage (>1 typical), convert to fraction
    if y_raw.median(skipna=True) and y_raw.median(skipna=True) > 1.0:
        y_raw = y_raw / 100.0
    data['y_conc'] = y_raw

    # Debug: show cleaned columns
    print("\n---- 清洗后样例(head) ----")
    print(data[['weeks', 'bmi', 'y_conc']].head(10))
    print("非空计数:", data[['weeks', 'bmi', 'y_conc']].notna().sum().to_dict())

    # Filter plausible ranges
    data = data[(data['weeks'] >= 6) & (data['weeks'] <= 40)]
    data = data[(data['bmi'] >= 15) & (data['bmi'] <= 60)]

    # Select male rows: y_conc not null
    male = data[~data['y_conc'].isna()].copy()
    # Keep only NIPT window
    male = male[(male['weeks'] >= MIN_WEEK) & (male['weeks'] <= MAX_WEEK)]

    # Optionally deduplicate by mother_code -> one reliable row per mother
    if USE_DEDUP:
        male_single = build_single_measurements(male, name_map)
    else:
        male_single = male.copy()

    # Define week bins
    bins = np.arange(MIN_WEEK, MAX_WEEK + BIN_WIDTH_WEEKS, BIN_WIDTH_WEEKS)
    bin_centers = bins[:-1] + BIN_WIDTH_WEEKS / 2.0

    # Define BMI groups
    results = []
    if USE_QUANTILE_BMI_GROUPS:
        try:
            male_single['bmi_group'] = pd.qcut(male_single['bmi'], q=BMI_GROUP_COUNT, duplicates='drop')
        except Exception:
            # fallback: equal-width cut into up to BMI_GROUP_COUNT bins
            male_single['bmi_group'] = pd.cut(male_single['bmi'], bins=BMI_GROUP_COUNT)
        group_keys = sorted(male_single['bmi_group'].dropna().unique(), key=lambda iv: (iv.left, iv.right))
        group_iter = [(str(iv), male_single[male_single['bmi_group'] == iv].copy()) for iv in group_keys]
    else:
        group_iter = []
        for bmin, bmax in BMI_GROUPS:
            mask = (male_single['bmi'] >= bmin) & (male_single['bmi'] < bmax)
            label = f"[{bmin},{bmax})" if np.isfinite(bmax) else f"[{bmin},+∞)"
            group_iter.append((label, male_single[mask].copy()))

    for group_name, group_df in group_iter:
        if group_df.empty:
            results.append({
                'bmi_range': group_name,
                'n_samples': 0,
                'recommendations': []
            })
            continue
        # For each noise level
        recs = []
        for nl in NOISE_LEVELS:
            yvals = add_noise(group_df['y_conc'].to_numpy(dtype=float), nl)
            passed = (yvals >= 0.04).astype(int)
            weeks_vals = group_df['weeks'].to_numpy(dtype=float)
            tmp = group_df.copy()
            tmp['passed'] = passed
            tmp['week_bin'] = pd.cut(tmp['weeks'], bins=bins, include_lowest=True, right=False)
            agg = tmp.groupby('week_bin').agg(
                n=('passed', 'size'),
                pass_rate=('passed', 'mean'),
            ).reset_index()
            # Extract bin start from interval
            agg['week_start'] = agg['week_bin'].apply(lambda x: x.left if pd.notna(x) else np.nan)
            # Stability filter
            agg = agg[agg['n'] >= MIN_SAMPLES_PER_BIN]
            agg = agg.sort_values('week_start')
            # Ensure numeric dtypes
            agg['week_start'] = pd.to_numeric(agg['week_start'], errors='coerce')
            agg['pass_rate'] = pd.to_numeric(agg['pass_rate'], errors='coerce')
            # Threshold-based earliest week
            thr_candidate = agg[agg['pass_rate'] >= PROB_THRESHOLD].head(1)
            thr_week = float(thr_candidate['week_start'].iloc[0]) if not thr_candidate.empty else np.nan
            thr_rate = float(thr_candidate['pass_rate'].iloc[0]) if not thr_candidate.empty else np.nan
            thr_nbin = int(thr_candidate['n'].iloc[0]) if not thr_candidate.empty else 0

            # Risk-minimization candidate
            if USE_RISK_MINIMIZATION:
                if USE_CONTINUOUS_RISK:
                    w_grid = np.round(np.arange(MIN_WEEK, MAX_WEEK + 1e-9, RISK_GRID_STEP), 1)
                    best = {'risk': np.inf, 'w': np.nan, 'rate': np.nan, 'n': 0}
                    for w in w_grid:
                        pr, used_win, used_n = estimate_pass_rate_window(
                            weeks_vals, passed,
                            w,
                            min_samples=max(RISK_MIN_SAMPLES, MIN_SAMPLES_PER_BIN//2),
                            min_window=RISK_MIN_WINDOW,
                            max_window=RISK_MAX_WINDOW,
                            window_step=RISK_WINDOW_STEP,
                        )
                        if np.isnan(pr):
                            continue
                        risk_val = CF_FAIL * (1.0 - pr) + CF_WAIT * max(0.0, w - MIN_WEEK)
                        if risk_val < best['risk'] - 1e-12 or (abs(risk_val - best['risk']) <= 1e-12 and w < best['w']):
                            best = {'risk': risk_val, 'w': w, 'rate': pr, 'n': used_n}
                    rk_week = float(best['w']) if not np.isnan(best['w']) else np.nan
                    rk_rate = float(best['rate']) if not np.isnan(best['rate']) else np.nan
                    rk_nbin = int(best['n']) if best['n'] else 0
                    rk_val = float(best['risk']) if np.isfinite(best['risk']) else np.nan
                else:
                    # Fallback to binned risk if continuous disabled
                    if not agg.empty:
                        agg['risk'] = CF_FAIL * (1.0 - agg['pass_rate']) + CF_WAIT * (agg['week_start'] - MIN_WEEK).clip(lower=0.0)
                        best_risk = agg.sort_values(['risk', 'week_start'], ascending=[True, True]).head(1)
                        rk_week = float(best_risk['week_start'].iloc[0])
                        rk_rate = float(best_risk['pass_rate'].iloc[0])
                        rk_nbin = int(best_risk['n'].iloc[0])
                        rk_val = float(best_risk['risk'].iloc[0])
                    else:
                        rk_week = np.nan
                        rk_rate = np.nan
                        rk_nbin = 0
                        rk_val = np.nan
            else:
                rk_week = np.nan
                rk_rate = np.nan
                rk_nbin = 0
                rk_val = np.nan

            # Prefer threshold week if exists; otherwise risk-based
            if not np.isnan(thr_week):
                recs.append({'noise': nl, 'week': thr_week, 'pass_rate': thr_rate, 'bin_n': thr_nbin, 'risk_week': rk_week, 'risk_rate': rk_rate, 'risk_bin_n': rk_nbin, 'risk_value': rk_val})
            elif not np.isnan(rk_week):
                recs.append({'noise': nl, 'week': rk_week, 'pass_rate': rk_rate, 'bin_n': rk_nbin, 'risk_week': rk_week, 'risk_rate': rk_rate, 'risk_bin_n': rk_nbin, 'risk_value': rk_val})
            else:
                recs.append({'noise': nl, 'week': np.nan, 'pass_rate': np.nan, 'bin_n': 0, 'risk_week': np.nan, 'risk_rate': np.nan, 'risk_bin_n': 0, 'risk_value': np.nan})
        results.append({
            'bmi_range': group_name,
            'n_samples': int(len(group_df)),
            'recommendations': recs,
        })
    return results, male_single


def main():
    if not EXCEL_PATH.exists():
        print(f"未找到Excel：{EXCEL_PATH}")
        return
    df = pd.read_excel(EXCEL_PATH)
    print("==== 文件信息 ====")
    print(f"路径: {EXCEL_PATH}")
    print(f"形状: {df.shape}")
    print("列名:")
    print(list(df.columns))

    name_map = detect_columns(df)
    print("\n==== 列映射 ====")
    for k, v in name_map.items():
        print(f"{k}: {v}")
    # Debug: show raw values of key columns
    gwc = name_map.get('gest_weeks')
    ycc = name_map.get('y_conc')
    print("\n---- 原始列样例(head) ----")
    with pd.option_context('display.max_colwidth', 200):
        try:
            print(df[[gwc, ycc]].head(12))
        except Exception:
            pass

    try:
        results, male = compute_group_recommendations(df, name_map)
    except Exception as e:
        print(f"\n计算失败: {e}")
        return

    print("\n==== 男胎样本统计（NIPT窗口内，按孕妇去重后） ====")
    print(f"样本量: {len(male)}，孕周范围: [{male['weeks'].min():.2f}, {male['weeks'].max():.2f}]，BMI范围: [{male['bmi'].min():.2f}, {male['bmi'].max():.2f}]")

    print("\n==== 分组建议（按BMI） ====")
    for r in results:
        print(f"BMI组 {r['bmi_range']} — 样本量: {r['n_samples']}")
        for rec in r['recommendations']:
            nl = rec['noise']
            label = "无误差" if nl == 0.0 else ("10%误差" if abs(nl-0.10) < 1e-6 else "15%误差")
            wk = rec['week']
            pr = rec['pass_rate']
            nbin = rec['bin_n']
            rw = rec.get('risk_week')
            rr = rec.get('risk_rate')
            rnb = rec.get('risk_bin_n')
            rv = rec.get('risk_value')
            if np.isnan(wk):
                print(f"  - {label}: 无足够样本得出结论")
            else:
                if not np.isnan(rw) and abs(rw - wk) > 1e-6:
                    print(f"  - {label}: 建议周数≈{wk:.1f}周（通过率≈{pr*100:.1f}%；样本数={nbin}）；风险最小≈{rw:.1f}周（通过率≈{rr*100:.1f}；样本数={rnb}；最小风险≈{rv:.3f}）")
                else:
                    print(f"  - {label}: 建议周数≈{wk:.1f}周（该周通过率≈{pr*100:.1f}%；该周样本数={nbin}）")

    # Provide a compact table-like summary for easy copying
    print("\n==== 建议时点汇总（便于粘贴） ====")
    header = ["BMI组", "样本量", "无误差周数", "无误差通过率(%)", "10%误差周数", "10%通过率(%)", "15%误差周数", "15%通过率(%)"]
    print("\t".join(header))
    for r in results:
        row = [r['bmi_range'], str(r['n_samples'])]
        rec_dict = {round(rec['noise'], 2): rec for rec in r['recommendations']}
        for nl in [0.0, 0.10, 0.15]:
            rec = rec_dict.get(round(nl, 2))
            if rec and not np.isnan(rec['week']):
                row.append(f"{rec['week']:.1f}")
                row.append(f"{rec['pass_rate']*100:.1f}")
            else:
                row.append("")
                row.append("")
        print("\t".join(row))
    # Extra: risk-based minimal weeks (for 无误差情景)
    print("\n==== 风险最小化建议（无误差情景，便于粘贴） ====")
    print("BMI组\t风险最小周数\t该周通过率(%)\t最小风险值")
    for r in results:
        rec0 = None
        for rec in r['recommendations']:
            if abs(rec['noise'] - 0.0) < 1e-6:
                rec0 = rec
                break
        if rec0 and not np.isnan(rec0.get('risk_week', np.nan)):
            rv = rec0.get('risk_value', np.nan)
            rv_str = f"{rv:.3f}" if not np.isnan(rv) else ""
            print(f"{r['bmi_range']}\t{rec0['risk_week']:.1f}\t{rec0['risk_rate']*100:.1f}\t{rv_str}")
        else:
            print(f"{r['bmi_range']}\t\t\t")

    # Extra: risk-based minimal weeks (for 10% noise)
    print("\n==== 风险最小化建议（10%误差情景，便于粘贴） ====")
    print("BMI组\t风险最小周数\t该周通过率(%)\t最小风险值")
    for r in results:
        recx = None
        for rec in r['recommendations']:
            if abs(rec['noise'] - 0.10) < 1e-6:
                recx = rec
                break
        if recx and not np.isnan(recx.get('risk_week', np.nan)):
            rv = recx.get('risk_value', np.nan)
            rv_str = f"{rv:.3f}" if not np.isnan(rv) else ""
            print(f"{r['bmi_range']}\t{recx['risk_week']:.1f}\t{recx['risk_rate']*100:.1f}\t{rv_str}")
        else:
            print(f"{r['bmi_range']}\t\t\t")

    # Extra: risk-based minimal weeks (for 15% noise)
    print("\n==== 风险最小化建议（15%误差情景，便于粘贴） ====")
    print("BMI组\t风险最小周数\t该周通过率(%)\t最小风险值")
    for r in results:
        recx = None
        for rec in r['recommendations']:
            if abs(rec['noise'] - 0.15) < 1e-6:
                recx = rec
                break
        if recx and not np.isnan(recx.get('risk_week', np.nan)):
            rv = recx.get('risk_value', np.nan)
            rv_str = f"{rv:.3f}" if not np.isnan(rv) else ""
            print(f"{r['bmi_range']}\t{recx['risk_week']:.1f}\t{recx['risk_rate']*100:.1f}\t{rv_str}")
        else:
            print(f"{r['bmi_range']}\t\t\t")


if __name__ == "__main__":
    main() 