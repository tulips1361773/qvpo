# python plot_sensitivity_csv_only.py

# -*- coding: utf-8 -*-
"""
CSV-only plotting for sensitivity analysis:
1) eav_threshold
2) eav_penalty_coef
3) E_tot

For each parameter group, generate:
1. eval reward
2. eval leakage rate
3. eval_legal_snr_db_mean
4. final_eval_reward / final_legal_snr_db / final_leakage_rate

Need:
    pip install pandas matplotlib
"""

import os
import re
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. USER CONFIG
# ============================================================

EXPERIMENTS = {
    "eav_threshold": {
        "x_label": "eav_threshold",
        "runs": {
            "5":  "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_00_26_19_0",
            "10": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_00_22_24_0",
            "15": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_20_36_37_0",
        },
    },
    "eav_penalty_coef": {
        "x_label": "eav_penalty_coef",
        "runs": {
            "2": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_00_23_48_0",
            "5": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_00_22_24_0",
            "8": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_18_50_10_0",
        },
    },
    "E_tot": {
        "x_label": "E_tot",
        "runs": {
            "25000": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_00_22_24_0",
            "35000": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_00_24_46_0",
            "45000": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_18_55_26_0",
        },
    },
}

OUTPUT_ROOT = "./figures_sensitivity_csv_only"

# 配色：参考你给的风格
DEFAULT_COLORS = ["#F58518", "#4C78A8", "#54A24B"]

# 字号：小四 ≈ 12 pt
FONT_SIZE = 12
LEGEND_SIZE = 11
TICK_SIZE = 11
TITLE_SIZE = 12

# ============================================================
# 2. SMOOTHING CONFIG
# ============================================================
# 三选一：
# SMOOTH_METHOD = None       -> 不平滑
# SMOOTH_METHOD = "rolling"  -> 滑动平均
# SMOOTH_METHOD = "ema"      -> 指数滑动平均
SMOOTH_METHOD = "ema"

# rolling window
ROLLING_WINDOW = 3

# ema alpha, 越小越平滑，建议 0.2 ~ 0.4
EMA_ALPHA = 0.1

# ============================================================
# 3. GLOBAL STYLE
# ============================================================

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "lines.linewidth": 2.0,
})

Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

# ============================================================
# 4. UTILS
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


def savefig(fig, out_dir, name):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    png_path = os.path.join(out_dir, f"{name}.png")
    pdf_path = os.path.join(out_dir, f"{name}.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}")


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


def make_color_map(labels):
    labels_sorted = sorted(labels, key=parse_numeric_label)
    return {lab: DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i, lab in enumerate(labels_sorted)}


# ============================================================
# 5. LOAD ALL DATA
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
# 6. PLOTTING
# ============================================================

def plot_one_experiment(exp_name, exp_cfg, exp_data):
    out_dir = os.path.join(OUTPUT_ROOT, exp_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    labels_sorted = sorted(exp_data.keys(), key=parse_numeric_label)
    color_map = make_color_map(labels_sorted)
    x_label = exp_cfg["x_label"]

    # --------------------------------------------------------
    # Figure 1: eval reward
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for label in labels_sorted:
        d = exp_data[label]
        df = d["training_df"]
        color = color_map[label]

        x = df["step"].values
        y = df["eval_reward_mean"].values
        y_s = smooth_series(
            y,
            method=SMOOTH_METHOD,
            rolling_window=ROLLING_WINDOW,
            ema_alpha=EMA_ALPHA
        )

        ax.plot(x, y_s, color=color, label=f"{x_label}={label}")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Eval reward")
    ax.set_title(f"{exp_name}: Eval Reward")
    ax.legend(frameon=False)
    savefig(fig, out_dir, "01_eval_reward")

    # --------------------------------------------------------
    # Figure 2: eval leakage rate
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for label in labels_sorted:
        d = exp_data[label]
        df = d["training_df"]
        color = color_map[label]

        x = df["step"].values
        y = df["eval_leakage_rate"].values
        y_s = smooth_series(
            y,
            method=SMOOTH_METHOD,
            rolling_window=ROLLING_WINDOW,
            ema_alpha=EMA_ALPHA
        )

        ax.plot(x, y_s, color=color, label=f"{x_label}={label}")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Eval leakage rate")
    ax.set_title(f"{exp_name}: Eval Leakage Rate")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    savefig(fig, out_dir, "02_eval_leakage_rate")

    # --------------------------------------------------------
    # Figure 3: eval_legal_snr_db_mean
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for label in labels_sorted:
        d = exp_data[label]
        df = d["training_df"]
        color = color_map[label]

        x = df["step"].values
        y = df["eval_legal_snr_db_mean"].values
        y_s = smooth_series(
            y,
            method=SMOOTH_METHOD,
            rolling_window=ROLLING_WINDOW,
            ema_alpha=EMA_ALPHA
        )

        ax.plot(x, y_s, color=color, label=f"{x_label}={label}")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Eval legal SNR (dB)")
    ax.set_title(f"{exp_name}: Eval Legal SNR")
    ax.legend(frameon=False)
    savefig(fig, out_dir, "03_eval_legal_snr_db_mean")

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
    colors = [color_map[l] for l in labels_sorted]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    axes[0].bar(x, final_reward, width=0.6, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_sorted)
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Final eval reward")
    axes[0].set_title("Final Eval Reward")

    axes[1].bar(x, final_legal_snr, width=0.6, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_sorted)
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Final legal SNR (dB)")
    axes[1].set_title("Final Legal SNR")

    axes[2].bar(x, final_leakage, width=0.6, color=colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels_sorted)
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel("Final leakage rate")
    axes[2].set_title("Final Leakage Rate")
    axes[2].set_ylim(bottom=0)

    fig.suptitle(f"{exp_name}: Final Metrics Comparison", y=1.02, fontsize=TITLE_SIZE)
    fig.tight_layout()
    savefig(fig, out_dir, "04_final_metrics_comparison")

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
# 7. RUN ALL
# ============================================================

for exp_name, exp_cfg in EXPERIMENTS.items():
    plot_one_experiment(exp_name, exp_cfg, all_exp_data[exp_name])

print("\nDone.")
print(f"All figures are saved under: {os.path.abspath(OUTPUT_ROOT)}")