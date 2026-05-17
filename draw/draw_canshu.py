# -*- coding: utf-8 -*-
"""
按照示例代码风格进一步统一后的敏感性分析画图脚本：
1. 默认保存 SVG / PDF / PNG（其中 SVG、PDF 为矢量图）
2. 四面封闭坐标轴，刻度线朝里
3. 图例带边框和阴影，并继续向示例代码风格对齐
4. 全部中文标签
5. 柱状图颜色、边框、网格线进一步向示例代码对齐
"""

import os
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ============================================================
# 1. 用户配置
# ============================================================
EXPERIMENTS = {
    "eav_threshold": {
        "x_label": "窃听门限",
        "legend_prefix": "窃听门限",
        "runs": {
            "5":  "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_00_26_19_0",
            "10": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_00_22_24_0",
            "15": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_20_36_37_0",
        },
    },
    "eav_penalty_coef": {
        "x_label": "窃听惩罚系数",
        "legend_prefix": "惩罚系数",
        "runs": {
            "2": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_00_23_48_0",
            "5": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_00_22_24_0",
            "8": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_18_50_10_0",
        },
    },
    "E_tot": {
        "x_label": "总能量预算",
        "legend_prefix": "总能量",
        "runs": {
            "25000": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_00_22_24_0",
            "35000": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_00_24_46_0",
            "45000": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_18_55_26_0",
        },
    },
}

OUTPUT_ROOT = "./figures_sensitivity_csv_only_revised_v2"
SAVE_SVG = True
SAVE_PDF = True
SAVE_PNG = True

# ============================================================
# 2. 平滑配置
# ============================================================
SMOOTH_METHOD = "ema"       # None / "rolling" / "ema"
ROLLING_WINDOW = 3
EMA_ALPHA = 0.1

# ============================================================
# 3. 风格参数（对齐示例代码）
# ============================================================
# 折线图主色：延续示例中更稳的橙/蓝/绿
LINE_COLORS = ["#F58518", "#4C78A8", "#54A24B"]

# 柱状图配色与边框，直接向示例代码的浅色填充 + 深色边框靠拢
BAR_FACE_COLORS = ["#F4C56A", "#84B6E3", "#8FCB81"]
BAR_EDGE_COLORS = ["#D89C2F", "#5B95CC", "#5FA653"]

# 字体
plt.rcParams["font.family"] = ["Droid Sans Fallback", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 16
plt.rcParams["axes.labelsize"] = 19
plt.rcParams["axes.titlesize"] = 19
plt.rcParams["xtick.labelsize"] = 17
plt.rcParams["ytick.labelsize"] = 17
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["text.usetex"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 600

# 线型/透明度
MAIN_LINEWIDTH = 2.8
GRID_ALPHA_LINE = 0.20
GRID_ALPHA_BAR = 0.10
BAR_EDGE_WIDTH = 2.0
SPINE_WIDTH = 1.2

# 图例样式：继续向示例对齐
LEGEND_FRAME_ALPHA = 0.90
LEGEND_EDGE_COLOR = "0.75"
LEGEND_HANDLE_LENGTH = 2.8
LEGEND_SHADOW = True

# 柱状图子图标题
FINAL_SUBPLOT_TITLES = {
    "final_eval_reward": "(a) 最终奖励",
    "final_legal_snr_db": "(b) 最终合法信噪比",
    "final_leakage_rate": "(c) 最终泄露率",
}

Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

# ============================================================
# 4. 工具函数
# ============================================================
def smooth_series(y, method=None, rolling_window=3, ema_alpha=0.3):
    y = pd.Series(np.asarray(y, dtype=float))
    if method is None:
        return y.values
    if method == "rolling":
        return y.rolling(window=rolling_window, min_periods=1).mean().values
    if method == "ema":
        return y.ewm(alpha=ema_alpha, adjust=False).mean().values
    raise ValueError(f"Unsupported smooth method: {method}")


def find_single_file(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matched: {pattern}")
    if len(files) > 1:
        print(f"[Warn] Multiple files matched {pattern}, using the first one:")
        for f in files:
            print("   ", f)
    return files[0]


def load_training_csv(run_dir):
    csv_dir = os.path.join(run_dir, "csv_logs")
    f = find_single_file(os.path.join(csv_dir, "training_metrics_*.csv"))
    df = pd.read_csv(f)
    df = df.sort_values("step").reset_index(drop=True)
    return df, f


def load_final_csv(run_dir):
    csv_dir = os.path.join(run_dir, "csv_logs")
    f = find_single_file(os.path.join(csv_dir, "final_comparison_*.csv"))
    df = pd.read_csv(f)
    return df, f


def parse_numeric_label(x):
    s = str(x).lower().strip()
    if s.endswith("k"):
        try:
            return float(s[:-1]) * 1000
        except ValueError:
            pass

    nums = re.findall(r"[-+]?\d*\.?\d+", s)
    if nums:
        try:
            return float(nums[0])
        except ValueError:
            return s
    return s


def setup_closed_axes(ax):
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(SPINE_WIDTH)

    ax.tick_params(
        axis="x",
        which="both",
        direction="in",
        bottom=True,
        top=False,
        labelbottom=True,
    )
    ax.tick_params(
        axis="y",
        which="both",
        direction="in",
        left=True,
        right=False,
        labelleft=True,
        labelright=False,
    )


def savefig_all(fig, out_dir, name):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    saved = []
    if SAVE_SVG:
        p = os.path.join(out_dir, f"{name}.svg")
        fig.savefig(p, bbox_inches="tight")
        saved.append(p)
    if SAVE_PDF:
        p = os.path.join(out_dir, f"{name}.pdf")
        fig.savefig(p, bbox_inches="tight")
        saved.append(p)
    if SAVE_PNG:
        p = os.path.join(out_dir, f"{name}.png")
        fig.savefig(p, bbox_inches="tight")
        saved.append(p)
    plt.close(fig)
    for p in saved:
        print(f"[Saved] {p}")


# ============================================================
# 5. 读取全部数据
# ============================================================
all_exp_data = {}

for exp_name, exp_cfg in EXPERIMENTS.items():
    exp_data = {}
    print(f"\n========== Loading {exp_name} ==========")

    for label, run_dir in exp_cfg["runs"].items():
        training_df, training_csv = load_training_csv(run_dir)
        final_df, final_csv = load_final_csv(run_dir)

        exp_data[label] = {
            "run_dir": run_dir,
            "training_df": training_df,
            "final_df": final_df,
            "training_csv": training_csv,
            "final_csv": final_csv,
        }

        print(f"[Loaded] {exp_name} = {label}")
        print(f"  training csv: {training_csv}")
        print(f"  final csv   : {final_csv}")

    all_exp_data[exp_name] = exp_data


# ============================================================
# 6. 画图主函数
# ============================================================
def plot_one_experiment(exp_name, exp_cfg, exp_data):
    out_dir = os.path.join(OUTPUT_ROOT, exp_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    labels_sorted = sorted(exp_data.keys(), key=parse_numeric_label)
    x_label = exp_cfg["x_label"]
    legend_prefix = exp_cfg["legend_prefix"]

    color_map = {lab: LINE_COLORS[i % len(LINE_COLORS)] for i, lab in enumerate(labels_sorted)}
    bar_face_map = {lab: BAR_FACE_COLORS[i % len(BAR_FACE_COLORS)] for i, lab in enumerate(labels_sorted)}
    bar_edge_map = {lab: BAR_EDGE_COLORS[i % len(BAR_EDGE_COLORS)] for i, lab in enumerate(labels_sorted)}

    # --------------------------------------------------------
    # Figure 1: eval reward
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    for label in labels_sorted:
        d = exp_data[label]
        df = d["training_df"]
        color = color_map[label]

        x = df["step"].values / 1e6
        y = df["eval_reward_mean"].values
        y_s = smooth_series(y, method=SMOOTH_METHOD, rolling_window=ROLLING_WINDOW, ema_alpha=EMA_ALPHA)
        ax.plot(x, y_s, color=color, linewidth=MAIN_LINEWIDTH, label=f"{legend_prefix}={label}")

    ax.set_xlabel("训练步数（×10^6）")
    ax.set_ylabel("评估奖励")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=GRID_ALPHA_LINE)
    setup_closed_axes(ax)
    ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        shadow=LEGEND_SHADOW,
        framealpha=LEGEND_FRAME_ALPHA,
        edgecolor=LEGEND_EDGE_COLOR,
        handlelength=LEGEND_HANDLE_LENGTH,
        borderpad=0.45,
        labelspacing=0.35,
    )
    fig.tight_layout()
    savefig_all(fig, out_dir, "01_eval_reward")

    # --------------------------------------------------------
    # Figure 2: eval leakage rate
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    for label in labels_sorted:
        d = exp_data[label]
        df = d["training_df"]
        color = color_map[label]

        x = df["step"].values / 1e6
        y = df["eval_leakage_rate"].values
        y_s = smooth_series(y, method=SMOOTH_METHOD, rolling_window=ROLLING_WINDOW, ema_alpha=EMA_ALPHA)
        ax.plot(x, y_s, color=color, linewidth=MAIN_LINEWIDTH, label=f"{legend_prefix}={label}")

    ax.set_xlabel("训练步数（×10^6）")
    ax.set_ylabel("评估泄露率")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=GRID_ALPHA_LINE)
    setup_closed_axes(ax)
    ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        shadow=LEGEND_SHADOW,
        framealpha=LEGEND_FRAME_ALPHA,
        edgecolor=LEGEND_EDGE_COLOR,
        handlelength=LEGEND_HANDLE_LENGTH,
        borderpad=0.45,
        labelspacing=0.35,
    )
    fig.tight_layout()
    savefig_all(fig, out_dir, "02_eval_leakage_rate")

    # --------------------------------------------------------
    # Figure 3: eval legal snr
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    for label in labels_sorted:
        d = exp_data[label]
        df = d["training_df"]
        color = color_map[label]

        x = df["step"].values / 1e6
        y = df["eval_legal_snr_db_mean"].values
        y_s = smooth_series(y, method=SMOOTH_METHOD, rolling_window=ROLLING_WINDOW, ema_alpha=EMA_ALPHA)
        ax.plot(x, y_s, color=color, linewidth=MAIN_LINEWIDTH, label=f"{legend_prefix}={label}")

    ax.set_xlabel("训练步数（×10^6）")
    ax.set_ylabel("评估合法信噪比（dB）")
    ax.grid(True, linestyle="--", alpha=GRID_ALPHA_LINE)
    setup_closed_axes(ax)
    ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        shadow=LEGEND_SHADOW,
        framealpha=LEGEND_FRAME_ALPHA,
        edgecolor=LEGEND_EDGE_COLOR,
        handlelength=LEGEND_HANDLE_LENGTH,
        borderpad=0.45,
        labelspacing=0.35,
    )
    fig.tight_layout()
    savefig_all(fig, out_dir, "03_eval_legal_snr_db_mean")

    # --------------------------------------------------------
    # Figure 4: final metrics comparison
    # --------------------------------------------------------
    final_reward = []
    final_legal_snr = []
    final_leakage = []

    for label in labels_sorted:
        row = exp_data[label]["final_df"].iloc[-1]
        final_reward.append(row["final_eval_reward"])
        final_legal_snr.append(row["final_legal_snr_db"])
        final_leakage.append(row["final_leakage_rate"])

    x = np.arange(len(labels_sorted))
    face_colors = [bar_face_map[l] for l in labels_sorted]
    edge_colors = [bar_edge_map[l] for l in labels_sorted]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
    fig.patch.set_facecolor("white")

    metric_data = [
        ("final_eval_reward", final_reward),
        ("final_legal_snr_db", final_legal_snr),
        ("final_leakage_rate", final_leakage),
    ]

    for ax, (metric_key, values) in zip(axes, metric_data):
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linestyle="--", alpha=GRID_ALPHA_BAR, zorder=0)

        for i, val in enumerate(values):
            ax.bar(
                x[i],
                val,
                width=0.68,
                color=face_colors[i],
                edgecolor=edge_colors[i],
                linewidth=BAR_EDGE_WIDTH,
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels_sorted)
        ax.set_xlabel(x_label)
        ax.set_ylabel("")  # 按要求删除三个子图的纵轴文字
        ax.set_title(FINAL_SUBPLOT_TITLES[metric_key], fontsize=19, pad=8)

        if metric_key == "final_leakage_rate":
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
            ax.set_ylim(bottom=0)

        setup_closed_axes(ax)

    fig.tight_layout()
    savefig_all(fig, out_dir, "04_final_metrics_comparison")

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------
    summary_rows = []
    for label in labels_sorted:
        row = exp_data[label]["final_df"].iloc[-1]
        summary_rows.append({
            x_label: label,
            "final_eval_reward": row["final_eval_reward"],
            "final_legal_snr_db": row["final_legal_snr_db"],
            "final_leakage_rate": row["final_leakage_rate"],
            "best_eval_reward": row.get("best_eval_reward", np.nan),
            "best_step": row.get("best_step", np.nan),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(out_dir, f"{exp_name}_final_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Saved] {summary_csv}")


# ============================================================
# 7. 执行
# ============================================================
for exp_name, exp_cfg in EXPERIMENTS.items():
    plot_one_experiment(exp_name, exp_cfg, all_exp_data[exp_name])

print("\nDone.")
print(f"All figures are saved under: {os.path.abspath(OUTPUT_ROOT)}")
