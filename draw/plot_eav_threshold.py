# -*- coding: utf-8 -*-
"""
Plot sensitivity-analysis figures for eav_threshold experiments.

Need:
    pip install pandas matplotlib tensorboard

What this script draws:
1) train/eval reward
2) train/eval leakage rate
3) eval_legal_snr_db_mean
4) final_eval_reward, final_legal_snr_db, final_leakage_rate

Author: ChatGPT
"""

import os
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# 1. User config
# ============================================================

RUNS = {
    "10": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_00_22_24_0",
    "5":  "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_00_26_19_0",
    "15": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_20_36_37_0",
}

PARAM_NAME = "eav_threshold"
OUTPUT_DIR = "./figures_eav_threshold"

# 参考你给的配色风格：orange / blue / green
COLOR_MAP = {
    "10": "#F58518",   # orange
    "5":  "#4C78A8",   # blue
    "15": "#54A24B",   # green
}

# 字号：小四 = 12 pt
FONT_SIZE = 12
LEGEND_SIZE = 11
TICK_SIZE = 11
TITLE_SIZE = 12

# 是否把 train 曲线额外做一点平滑；0 表示不做
TRAIN_SMOOTH_WINDOW = 0

# ============================================================
# 2. Global plotting style
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

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================
# 3. Utilities
# ============================================================

def maybe_smooth(y, window=0):
    y = np.asarray(y, dtype=float)
    if window is None or window <= 1 or len(y) < window:
        return y
    return pd.Series(y).rolling(window=window, min_periods=1).mean().values


def find_single_file(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matched: {pattern}")
    if len(files) > 1:
        print(f"[Warn] Multiple files matched {pattern}, use first one:\n  {files[0]}")
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


def load_tb_scalars(run_dir, tags):
    """
    Read selected scalar tags from TensorBoard event files in run_dir.
    Returns:
        dict[tag] = pd.DataFrame({"step": ..., "value": ...})
    """
    event_files = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    if not event_files:
        # 有些情况下 event 文件可能在子目录
        event_files = sorted(glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True))

    if not event_files:
        print(f"[Warn] No TensorBoard event files found in: {run_dir}")
        return {tag: pd.DataFrame(columns=["step", "value"]) for tag in tags}

    merged = {tag: [] for tag in tags}

    for ef in event_files:
        try:
            ea = event_accumulator.EventAccumulator(
                ef,
                size_guidance={
                    event_accumulator.SCALARS: 0
                }
            )
            ea.Reload()
            available = set(ea.Tags().get("scalars", []))

            for tag in tags:
                if tag in available:
                    events = ea.Scalars(tag)
                    for e in events:
                        merged[tag].append((e.step, e.value))
        except Exception as e:
            print(f"[Warn] Failed to read TB file: {ef}\n  {e}")

    out = {}
    for tag in tags:
        if len(merged[tag]) == 0:
            out[tag] = pd.DataFrame(columns=["step", "value"])
        else:
            df = pd.DataFrame(merged[tag], columns=["step", "value"])
            df = df.sort_values("step").drop_duplicates(subset=["step"], keep="last").reset_index(drop=True)
            out[tag] = df
    return out


def savefig(fig, name):
    png_path = os.path.join(OUTPUT_DIR, f"{name}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{name}.pdf")
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}")


def get_run_label(level):
    return f"{PARAM_NAME}={level}"


# ============================================================
# 4. Load all runs
# ============================================================

# TB tag choices:
# train reward -> reward/train_ma100
# train leakage -> security/train_leakage_rate_ma100
TB_TAGS = [
    "reward/train_ma100",
    "reward/train",
    "reward/eval_mean",
    "security/train_leakage_rate_ma100",
    "security/train_leakage_rate_window200",
    "security/eval_leakage_rate",
]

all_data = {}

for level, run_dir in RUNS.items():
    training_df, training_csv_path = load_training_csv(run_dir)
    final_df, final_csv_path = load_final_csv(run_dir)
    tb_data = load_tb_scalars(run_dir, TB_TAGS)

    all_data[level] = {
        "run_dir": run_dir,
        "training_df": training_df,
        "final_df": final_df,
        "tb": tb_data,
        "training_csv_path": training_csv_path,
        "final_csv_path": final_csv_path,
    }

    print(f"\n[Loaded] {level}")
    print(f"  training csv: {training_csv_path}")
    print(f"  final csv   : {final_csv_path}")


# ============================================================
# 5. Figure 1: train/eval reward
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 4.8))

for level in ["5", "10", "15"]:
    d = all_data[level]
    color = COLOR_MAP[level]

    # train reward: TensorBoard 优先用 reward/train_ma100；缺失则回退到 training_metrics 的 train_reward_ma100
    tb_train = d["tb"]["reward/train_ma100"]
    if not tb_train.empty:
        x_train = tb_train["step"].values
        y_train = maybe_smooth(tb_train["value"].values, TRAIN_SMOOTH_WINDOW)
    else:
        x_train = d["training_df"]["step"].values
        y_train = d["training_df"]["train_reward_ma100"].values

    # eval reward: 优先用 CSV 的 eval_reward_mean
    x_eval = d["training_df"]["step"].values
    y_eval = d["training_df"]["eval_reward_mean"].values

    ax.plot(x_train, y_train, color=color, linestyle="-",  label=f"{get_run_label(level)} (train)")
    ax.plot(x_eval,  y_eval,  color=color, linestyle="--", label=f"{get_run_label(level)} (eval)")

ax.set_xlabel("Training step")
ax.set_ylabel("Reward")
ax.set_title("Train / Eval Reward vs Step")
ax.legend(ncol=2, frameon=False)
savefig(fig, "01_train_eval_reward")


# ============================================================
# 6. Figure 2: train/eval leakage rate
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 4.8))

for level in ["5", "10", "15"]:
    d = all_data[level]
    color = COLOR_MAP[level]

    # train leakage: TensorBoard 优先用 MA100；没有就回退到 window200
    tb_train_leak = d["tb"]["security/train_leakage_rate_ma100"]
    if tb_train_leak.empty:
        tb_train_leak = d["tb"]["security/train_leakage_rate_window200"]

    if tb_train_leak.empty:
        print(f"[Warn] No train leakage TensorBoard scalar for {level}")
    else:
        ax.plot(
            tb_train_leak["step"].values,
            maybe_smooth(tb_train_leak["value"].values, TRAIN_SMOOTH_WINDOW),
            color=color,
            linestyle="-",
            label=f"{get_run_label(level)} (train)"
        )

    # eval leakage: CSV
    df = d["training_df"]
    ax.plot(
        df["step"].values,
        df["eval_leakage_rate"].values,
        color=color,
        linestyle="--",
        label=f"{get_run_label(level)} (eval)"
    )

ax.set_xlabel("Training step")
ax.set_ylabel("Leakage rate")
ax.set_title("Train / Eval Leakage Rate vs Step")
ax.set_ylim(bottom=0)
ax.legend(ncol=2, frameon=False)
savefig(fig, "02_train_eval_leakage_rate")


# ============================================================
# 7. Figure 3: eval_legal_snr_db_mean
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 4.8))

for level in ["5", "10", "15"]:
    d = all_data[level]
    color = COLOR_MAP[level]
    df = d["training_df"]

    ax.plot(
        df["step"].values,
        df["eval_legal_snr_db_mean"].values,
        color=color,
        linestyle="-",
        label=get_run_label(level)
    )

ax.set_xlabel("Training step")
ax.set_ylabel("Eval legal SNR (dB)")
ax.set_title("Eval Legal SNR vs Step")
ax.legend(frameon=False)
savefig(fig, "03_eval_legal_snr_db_mean")


# ============================================================
# 8. Figure 4: final metrics
# ============================================================

levels = ["5", "10", "15"]
x = np.arange(len(levels))
bar_width = 0.6

final_reward = []
6

final_reward = []
final_legal_snr = []
final_leakage = []

for level in levels:
    fdf = all_data[level]["final_df"]

    # final_comparison 每个 run 通常只有 1 行，这里取最后一行更稳妥
    row = fdf.iloc[-1]
    final_reward.append(row["final_eval_reward"])
    final_legal_snr.append(row["final_legal_snr_db"])
    final_leakage.append(row["final_leakage_rate"])

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

# 4.1 final_eval_reward
axes[0].bar(
    x,
    final_reward,
    width=bar_width,
    color=[COLOR_MAP[k] for k in levels]
)
axes[0].set_xticks(x)
axes[0].set_xticklabels(levels)
axes[0].set_xlabel(PARAM_NAME)
axes[0].set_ylabel("Final eval reward")
axes[0].set_title("Final Eval Reward")

# 4.2 final_legal_snr_db
axes[1].bar(
    x,
    final_legal_snr,
    width=bar_width,
    color=[COLOR_MAP[k] for k in levels]
)
axes[1].set_xticks(x)
axes[1].set_xticklabels(levels)
axes[1].set_xlabel(PARAM_NAME)
axes[1].set_ylabel("Final legal SNR (dB)")
axes[1].set_title("Final Legal SNR")

# 4.3 final_leakage_rate
axes[2].bar(
    x,
    final_leakage,
    width=bar_width,
    color=[COLOR_MAP[k] for k in levels]
)
axes[2].set_xticks(x)
axes[2].set_xticklabels(levels)
axes[2].set_xlabel(PARAM_NAME)
axes[2].set_ylabel("Final leakage rate")
axes[2].set_title("Final Leakage Rate")
axes[2].set_ylim(bottom=0)

fig.suptitle("Final Metrics Comparison", y=1.02, fontsize=TITLE_SIZE)
fig.tight_layout()
savefig(fig, "04_final_metrics_comparison")


# ============================================================
# 9. Optional: export a merged summary csv
# ============================================================

summary_rows = []
for level in levels:
    row = all_data[level]["final_df"].iloc[-1]
    summary_rows.append({
        PARAM_NAME: level,
        "final_eval_reward": row["final_eval_reward"],
        "final_legal_snr_db": row["final_legal_snr_db"],
        "final_leakage_rate": row["final_leakage_rate"],
        "best_eval_reward": row.get("best_eval_reward", np.nan),
        "best_step": row.get("best_step", np.nan),
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_DIR, f"{PARAM_NAME}_final_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"[Saved] {summary_path}")

print("\nDone.")
print(f"All figures are saved in: {os.path.abspath(OUTPUT_DIR)}")