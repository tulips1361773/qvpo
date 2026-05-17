import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# ============================================================
# 1. 用户可调参数（最终推荐配置）
# ============================================================
smooth_method = "ema"          # "none" / "ema" / "moving_average"
smooth_weight = 0.9            # 论文图默认推荐 0.8；不建议正文默认用 0.9

show_seed_lines = True         # 显示每个 seed 的细线（推荐 True）
show_band = True               # 显示波动阴影（推荐 True）
band_mode = "minmax"           # "minmax" / "std"
# 说明：
# - 只有 2 个 seed 时，推荐 minmax，更诚实、更稳妥
# - 如果以后补到 >=3 个 seed，可考虑改为 "std"

save_fig = True
save_dir = "./figures_ch2"

save_stem_reward = f"eval_reward_final_{smooth_method}_sw{smooth_weight}_{band_mode}"
save_stem_leakage = f"eval_leakage_rate_{smooth_method}_sw{smooth_weight}_{band_mode}"
save_stem_final = f"final_test_metrics_{band_mode}"

show_title = False
strict_missing_files = False

# 坐标轴范围
xlim = (0.0, 1.0)
ylim_reward = (-60, 50)
ylim_leakage = (0.02, 0.17)   # 改为 2% ~ 17%

# moving average 窗口设置
ma_window_map = {
    0.7: 5,
    0.8: 9,
    0.9: 15
}

# 配色：你的算法用橙色突出
color_map = {
    "PIM-DiffTD3": "#F58518",   # orange
    "TD3": "#4C78A8",           # blue
    "SAC": "#54A24B",           # green
    "PPO": "#E45756",           # red
}

final_bar_color_map = {
    "PIM-DiffTD3": "#EFCB72",
    "SAC":         "#93C47D",
    "TD3":         "#7FAFDC",
    "PPO":         "#E9A3B4",
}

# 绘图风格
main_linewidth = 2.8
seed_linewidth = 1.1
seed_line_alpha = 0.20   # 原 0.26，略淡一点
band_alpha = 0.10        # 原 0.14，略淡一点

# 字体与导出设置
plt.rcParams["font.family"] = ["Droid Sans Fallback", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 16
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["text.usetex"] = False

# final 图论文风格设置（在上一版基础上整体 +3）
final_title_fontsize = 19
final_label_fontsize = 19
final_tick_fontsize = 17

final_subplot_titles = {
    "final_eval_reward": "(a) 最终奖励",
    "final_leakage_rate": "(b) 最终泄露率",
    "final_legal_snr_db": "(c) 最终合法信噪比",
}

# TD3 不再使用占位符
final_placeholder_algorithms = set()
placeholder_facecolor = "#E6E6E6"
placeholder_edgecolor = "#888888"
placeholder_hatch = "//"
placeholder_text = "N/A"
placeholder_text_size = 17


# ============================================================
# 2. 数据路径（4个算法，每个算法2个 seed）
# ============================================================
algo_seed_paths = {
    "PIM-DiffTD3": {
        "seed42": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_00_22_24_0/csv_logs/training_metrics_26_03_30_00_22_24_0.csv",
        "seed101": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=101/run_id=26_04_01_00_35_00_0/csv_logs/training_metrics_26_04_01_00_35_00_0.csv",
    },
    "SAC": {
        "seed42": "/home/moqianyu_26/sda/qvpo/record/sac/fsac_2026-03-31_12-38-17/csv_logs/training_metrics_sac_seed42_2026-03-31_12-38-17.csv",
        "seed101": "/home/moqianyu_26/sda/qvpo/record/sac/fsac_2026-04-01_20-10-34/csv_logs/training_metrics_sac_seed101_2026-04-01_20-10-34.csv",
    },
    "TD3": {
        "seed42": "/home/moqianyu_26/sda/qvpo/record/Env/TD3/ratio=0.1/seed=42/run_id=260402_230248/csv_logs/training_metrics_260402_230248.csv",
        "seed101": "/home/moqianyu_26/sda/qvpo/record/Env/TD3/ratio=0.1/seed=101/run_id=260402_230550/csv_logs/training_metrics_260402_230550.csv",
    },
    "PPO": {
        "seed42": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=PPO/seed=42/run_id=26_04_01_21_18_15_0/csv_logs/training_metrics_26_04_01_21_18_15_0.csv",
        "seed101": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=PPO/seed=101/run_id=26_04_01_22_38_28_0/csv_logs/training_metrics_26_04_01_22_38_28_0.csv",
    },
}


# ============================================================
# 3. 平滑函数
# ============================================================
def ema_smooth(y, weight=0.8):
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y

    smoothed = np.zeros_like(y, dtype=float)
    smoothed[0] = y[0]
    for i in range(1, len(y)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * y[i]
    return smoothed


def moving_average_smooth(y, window=9):
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y
    if window <= 1:
        return y.copy()

    if window % 2 == 0:
        window += 1

    pad = window // 2
    y_pad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y_pad, kernel, mode="valid")
    return y_smooth


def smooth_curve(y, method="ema", weight=0.8):
    if method == "none":
        return np.asarray(y, dtype=float)
    elif method == "ema":
        return ema_smooth(y, weight=weight)
    elif method == "moving_average":
        window = ma_window_map.get(weight, 9)
        return moving_average_smooth(y, window=window)
    else:
        raise ValueError(f"Unsupported smooth_method: {method}")


# ============================================================
# 4. 读取 CSV
# ============================================================
def load_training_metric_csv(csv_path, metric_col):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["step", metric_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {csv_path}")

    df = df.dropna(subset=["step", metric_col]).copy()
    df = df.sort_values("step").reset_index(drop=True)
    return df[["step", metric_col]]


def load_multi_seed_runs_metric(seed_path_dict, metric_col, strict_missing_files=False):
    runs = {}

    for seed_name, csv_path in seed_path_dict.items():
        if not os.path.exists(csv_path):
            msg = f"[Missing seed file] {seed_name}: {csv_path}"
            if strict_missing_files:
                raise FileNotFoundError(msg)
            warnings.warn(msg + " -> skipped.")
            continue

        runs[seed_name] = load_training_metric_csv(csv_path, metric_col)

    if len(runs) == 0:
        raise ValueError(f"No valid seed csv files were loaded for metric={metric_col}.")

    return runs


# ============================================================
# 5. 多 seed 对齐与聚合
# ============================================================
def align_runs_on_step_generic(runs, value_col_name):
    """
    输入:
        runs: dict[seed_name] = DataFrame(step, value_col_name)

    返回:
        merged: DataFrame(step, seed1, seed2, ...)
        seed_names: list[str]
    """
    merged = None
    seed_names = list(runs.keys())

    for seed_name, df in runs.items():
        tmp = df.rename(columns={value_col_name: seed_name})
        merged = tmp if merged is None else pd.merge(merged, tmp, on="step", how="outer")

    merged = merged.sort_values("step").reset_index(drop=True)

    # 只填补内部缺口，不对序列两端外推
    for seed_name in seed_names:
        merged[seed_name] = merged[seed_name].interpolate(
            method="linear",
            limit_area="inside"
        )

    return merged, seed_names


def build_smoothed_seed_matrix(merged_df, seed_names, smooth_method="ema", smooth_weight=0.8):
    """
    对每个 seed 单独平滑，返回 shape = [num_steps, num_seeds] 的矩阵
    """
    smooth_list = []

    for seed_name in seed_names:
        y = merged_df[seed_name].to_numpy(dtype=float)
        valid_mask = ~np.isnan(y)

        y_smooth = np.full_like(y, np.nan, dtype=float)
        if valid_mask.any():
            y_valid = y[valid_mask]
            y_smooth[valid_mask] = smooth_curve(
                y_valid,
                method=smooth_method,
                weight=smooth_weight
            )

        smooth_list.append(y_smooth)

    seed_smooth_matrix = np.column_stack(smooth_list)   # [T, N]
    return seed_smooth_matrix


def aggregate_seed_curves(seed_matrix, band_mode="minmax"):
    """
    输入:
        seed_matrix: [num_steps, num_seeds]

    返回:
        mean_curve, lower_curve, upper_curve
    """
    mean_curve = np.nanmean(seed_matrix, axis=1)

    if band_mode == "minmax":
        lower_curve = np.nanmin(seed_matrix, axis=1)
        upper_curve = np.nanmax(seed_matrix, axis=1)
    elif band_mode == "std":
        std_curve = np.nanstd(seed_matrix, axis=1)
        lower_curve = mean_curve - std_curve
        upper_curve = mean_curve + std_curve
    else:
        raise ValueError(f"Unsupported band_mode: {band_mode}")

    return mean_curve, lower_curve, upper_curve


# ============================================================
# 6. 画 eval 曲线（reward / leakage）
# ============================================================
def plot_eval_metric_curves(
    algo_seed_paths,
    metric_col,
    ylabel,
    save_stem,
    smooth_method="ema",
    smooth_weight=0.8,
    show_seed_lines=True,
    show_band=True,
    band_mode="minmax",
    save_fig=True,
    save_dir="./figures",
    show_title=False,
    strict_missing_files=False,
    xlim=(0.0, 1.0),
    ylim=None,
    legend_loc="best",
):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    for algo_name, seed_path_dict in algo_seed_paths.items():
        runs = load_multi_seed_runs_metric(
            seed_path_dict,
            metric_col=metric_col,
            strict_missing_files=strict_missing_files
        )

        merged, valid_seed_names = align_runs_on_step_generic(runs, metric_col)
        x = merged["step"].to_numpy(dtype=float) / 1e6
        color = color_map[algo_name]

        seed_smooth_matrix = build_smoothed_seed_matrix(
            merged,
            valid_seed_names,
            smooth_method=smooth_method,
            smooth_weight=smooth_weight
        )

        mean_curve, lower_curve, upper_curve = aggregate_seed_curves(
            seed_smooth_matrix,
            band_mode=band_mode
        )

        # 1) seed 细线
        if show_seed_lines:
            for j, seed_name in enumerate(valid_seed_names):
                y_seed = seed_smooth_matrix[:, j]
                valid = ~np.isnan(y_seed)
                ax.plot(
                    x[valid], y_seed[valid],
                    color=color,
                    linewidth=seed_linewidth,
                    alpha=seed_line_alpha,
                    label="_nolegend_"
                )

        # 2) 波动阴影
        if show_band and len(valid_seed_names) >= 2:
            valid = ~np.isnan(mean_curve) & ~np.isnan(lower_curve) & ~np.isnan(upper_curve)
            ax.fill_between(
                x[valid],
                lower_curve[valid],
                upper_curve[valid],
                color=color,
                alpha=band_alpha,
                linewidth=0.0,
                label="_nolegend_"
            )

        # 3) 主线：多个 seed 平滑后取均值
        valid = ~np.isnan(mean_curve)
        ax.plot(
            x[valid], mean_curve[valid],
            color=color,
            linewidth=main_linewidth,
            label=algo_name
        )

        print(f"[Loaded] {algo_name}: {len(valid_seed_names)} seed(s) -> {valid_seed_names}")

    ax.set_xlabel("训练步数（×10^6）")
    ax.set_ylabel(ylabel)

    if show_title:
        ax.set_title(ylabel)

    if xlim is not None:
        ax.set_xlim(*xlim)
        ax.set_xticks(np.linspace(xlim[0], xlim[1], 6))

    if ylim is not None:
        ax.set_ylim(*ylim)

    # 刻度线朝里，数字仍在外面
    ax.tick_params(
        axis="x",
        which="both",
        direction="in",
        bottom=True,
        top=False,
        labelbottom=True
    )
    ax.tick_params(
        axis="y",
        which="both",
        direction="in",
        left=True,
        right=False,
        labelleft=True,
        labelright=False
    )

    # eval leakage rate 用百分比显示
    if metric_col == "eval_leakage_rate":
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.set_yticks([0.02, 0.05, 0.08, 0.11, 0.14, 0.17])

    ax.grid(True, linestyle="--", alpha=0.20)

    ax.legend(
        loc=legend_loc,
        frameon=True,
        framealpha=0.90,
        edgecolor="0.75",
        fontsize=14,
        handlelength=2.8
    )

    plt.tight_layout()

    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        pdf_path = os.path.join(save_dir, f"{save_stem}.svg")
        png_path = os.path.join(save_dir, f"{save_stem}.png")

        plt.savefig(pdf_path, dpi=600, bbox_inches="tight")
        plt.savefig(png_path, dpi=600, bbox_inches="tight")

        print(f"[Saved PDF] {pdf_path}")
        print(f"[Saved PNG] {png_path}")

    plt.show()


# ============================================================
# 7. final CSV 路径推断与读取
# ============================================================
def infer_final_csv_path_from_training_csv(training_csv_path):
    """
    例如：
    .../csv_logs/training_metrics_xxx.csv
    -> .../csv_logs/final_comparison_xxx.csv
    """
    dir_name = os.path.dirname(training_csv_path)
    base_name = os.path.basename(training_csv_path)

    if not base_name.startswith("training_metrics_"):
        raise ValueError(f"Unexpected training csv filename: {training_csv_path}")

    suffix = base_name[len("training_metrics_"):]
    final_name = f"final_comparison_{suffix}"
    return os.path.join(dir_name, final_name)


def load_final_csv(final_csv_path):
    if not os.path.exists(final_csv_path):
        raise FileNotFoundError(f"Final comparison CSV not found: {final_csv_path}")

    df = pd.read_csv(final_csv_path)

    required_cols = [
        "final_eval_reward",
        "final_leakage_rate",
        "final_legal_snr_db",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {final_csv_path}")

    if len(df) == 0:
        raise ValueError(f"Empty final comparison CSV: {final_csv_path}")

    return df


# ============================================================
# 8. 聚合每个算法的 final 指标（支持占位符算法）
# ============================================================
def collect_final_metrics(
    algo_seed_paths,
    strict_missing_files=False,
    placeholder_algorithms=None
):
    """
    返回:
        result[algo_name] = {
            "final_eval_reward": np.array([...]) 或 None,
            "final_leakage_rate": np.array([...]) 或 None,
            "final_legal_snr_db": np.array([...]) 或 None,
            "valid_seed_names": [...],
            "is_placeholder": bool,
        }
    """
    if placeholder_algorithms is None:
        placeholder_algorithms = set()

    result = {}

    for algo_name, seed_path_dict in algo_seed_paths.items():
        # 指定算法直接用占位符，不读取 final csv
        if algo_name in placeholder_algorithms:
            result[algo_name] = {
                "final_eval_reward": None,
                "final_leakage_rate": None,
                "final_legal_snr_db": None,
                "valid_seed_names": list(seed_path_dict.keys()),
                "is_placeholder": True,
            }
            print(f"[Final Placeholder] {algo_name}: use placeholder bars.")
            continue

        final_reward_list = []
        final_leakage_list = []
        final_legal_snr_list = []
        valid_seed_names = []

        for seed_name, training_csv_path in seed_path_dict.items():
            if not os.path.exists(training_csv_path):
                msg = f"[Missing training csv] {algo_name}-{seed_name}: {training_csv_path}"
                if strict_missing_files:
                    raise FileNotFoundError(msg)
                warnings.warn(msg + " -> skipped.")
                continue

            final_csv_path = infer_final_csv_path_from_training_csv(training_csv_path)
            if not os.path.exists(final_csv_path):
                msg = f"[Missing final csv] {algo_name}-{seed_name}: {final_csv_path}"
                if strict_missing_files:
                    raise FileNotFoundError(msg)
                warnings.warn(msg + " -> skipped.")
                continue

            df_final = load_final_csv(final_csv_path)
            row = df_final.iloc[-1]

            final_reward_list.append(float(row["final_eval_reward"]))
            final_leakage_list.append(float(row["final_leakage_rate"]))
            final_legal_snr_list.append(float(row["final_legal_snr_db"]))
            valid_seed_names.append(seed_name)

        if len(valid_seed_names) == 0:
            raise ValueError(f"No valid final csv files found for algorithm {algo_name}")

        result[algo_name] = {
            "final_eval_reward": np.asarray(final_reward_list, dtype=float),
            "final_leakage_rate": np.asarray(final_leakage_list, dtype=float),
            "final_legal_snr_db": np.asarray(final_legal_snr_list, dtype=float),
            "valid_seed_names": valid_seed_names,
            "is_placeholder": False,
        }

        print(f"[Final Loaded] {algo_name}: {len(valid_seed_names)} seed(s) -> {valid_seed_names}")

    return result


def compute_bar_center_and_error(values, band_mode="minmax"):
    """
    对柱状图做聚合：
    - center = mean
    - error:
        minmax -> 到 min/max 的半区间
        std    -> std
    """
    values = np.asarray(values, dtype=float)
    center = np.nanmean(values)

    if len(values) <= 1:
        return center, 0.0

    if band_mode == "minmax":
        lower = center - np.nanmin(values)
        upper = np.nanmax(values) - center
        err = np.array([[lower], [upper]], dtype=float)  # asymmetric
    elif band_mode == "std":
        s = np.nanstd(values)
        err = np.array([[s], [s]], dtype=float)
    else:
        raise ValueError(f"Unsupported band_mode: {band_mode}")

    return center, err


# ============================================================
# 9. 画 final 对比图
# ============================================================
def plot_final_metric_comparison(
    algo_seed_paths,
    save_fig=True,
    save_dir="./figures",
    save_stem="final_metrics_comparison",
    show_title=False,
    strict_missing_files=False,
    band_mode="minmax",
):
    final_data = collect_final_metrics(
        algo_seed_paths,
        strict_missing_files=strict_missing_files,
        placeholder_algorithms=final_placeholder_algorithms
    )

    algo_names = list(algo_seed_paths.keys())
    x = np.arange(len(algo_names))
    width = 0.68

    metrics = [
        ("final_eval_reward", "最终评估奖励"),
        ("final_leakage_rate", "最终泄露率"),
        ("final_legal_snr_db", "最终合法信噪比（dB）"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))

    for ax, (metric_key, ylabel) in zip(axes, metrics):
        centers = []
        yerr_lower = []
        yerr_upper = []
        placeholder_mask = []

        for algo_name in algo_names:
            algo_info = final_data[algo_name]

            if algo_info["is_placeholder"]:
                centers.append(0.0)
                yerr_lower.append(0.0)
                yerr_upper.append(0.0)
                placeholder_mask.append(True)
            else:
                center, err = compute_bar_center_and_error(
                    algo_info[metric_key],
                    band_mode=band_mode
                )
                centers.append(center)
                if np.isscalar(err):
                    yerr_lower.append(err)
                    yerr_upper.append(err)
                else:
                    yerr_lower.append(float(err[0, 0]))
                    yerr_upper.append(float(err[1, 0]))
                placeholder_mask.append(False)

        # 分开画：正常柱 + 占位柱
        for i, algo_name in enumerate(algo_names):
            if placeholder_mask[i]:
                ax.bar(
                    x[i],
                    centers[i],
                    width=width,
                    color=placeholder_facecolor,
                    edgecolor=placeholder_edgecolor,
                    hatch=placeholder_hatch,
                    linewidth=1.0
                )
            else:
                ax.bar(
                    x[i],
                    centers[i],
                    width=width,
                    color=final_bar_color_map[algo_name],
                    edgecolor=color_map[algo_name],
                    linewidth=0.8,
                    yerr=np.array([[yerr_lower[i]], [yerr_upper[i]]]),
                    capsize=4,
                    error_kw=dict(linewidth=1.2, alpha=0.9)
                )

        ax.set_xticks(x)
        ax.set_xticklabels(algo_names, rotation=0)
        #ax.set_xlabel("算法", fontsize=final_label_fontsize)
        ax.set_ylabel("")

        ax.tick_params(
            axis="x",
            labelsize=final_tick_fontsize,
            direction="in",
            bottom=True,
            top=False,
            labelbottom=True
        )
        ax.tick_params(
            axis="y",
            labelsize=final_tick_fontsize,
            direction="in",
            left=True,
            right=False,
            labelleft=True,
            labelright=False
        )

        ax.grid(True, axis="y", linestyle="--", alpha=0.10)

        ax.set_title(
            final_subplot_titles[metric_key],
            fontsize=final_title_fontsize,
            pad=8
        )

        if metric_key == "final_leakage_rate":
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

        ymin, ymax = ax.get_ylim()
        for i, is_placeholder in enumerate(placeholder_mask):
            if is_placeholder:
                text_y = ymin + 0.08 * (ymax - ymin)
                ax.text(
                    x[i],
                    text_y,
                    placeholder_text,
                    ha="center",
                    va="bottom",
                    fontsize=placeholder_text_size,
                    color=placeholder_edgecolor
                )

    plt.tight_layout()

    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        pdf_path = os.path.join(save_dir, f"{save_stem}.svg")
        png_path = os.path.join(save_dir, f"{save_stem}.png")

        plt.savefig(pdf_path, dpi=600, bbox_inches="tight")
        plt.savefig(png_path, dpi=600, bbox_inches="tight")

        print(f"[Saved PDF] {pdf_path}")
        print(f"[Saved PNG] {png_path}")

    plt.show()


# ============================================================
# 10. 运行
# ============================================================
if __name__ == "__main__":
    # 1) eval reward
    plot_eval_metric_curves(
        algo_seed_paths=algo_seed_paths,
        metric_col="eval_reward_mean",
        ylabel="评估奖励",
        save_stem=save_stem_reward,
        smooth_method=smooth_method,
        smooth_weight=smooth_weight,
        show_seed_lines=show_seed_lines,
        show_band=show_band,
        band_mode=band_mode,
        save_fig=save_fig,
        save_dir=save_dir,
        show_title=show_title,
        strict_missing_files=strict_missing_files,
        xlim=xlim,
        ylim=ylim_reward,
        legend_loc="lower right",
    )

    # 2) eval leakage rate
    plot_eval_metric_curves(
        algo_seed_paths=algo_seed_paths,
        metric_col="eval_leakage_rate",
        ylabel="评估泄露率",
        save_stem=save_stem_leakage,
        smooth_method=smooth_method,
        smooth_weight=smooth_weight,
        show_seed_lines=show_seed_lines,
        show_band=show_band,
        band_mode=band_mode,
        save_fig=save_fig,
        save_dir=save_dir,
        show_title=show_title,
        strict_missing_files=strict_missing_files,
        xlim=xlim,
        ylim=ylim_leakage,
        legend_loc="upper right",
    )

    # 3) final test metrics
    plot_final_metric_comparison(
        algo_seed_paths=algo_seed_paths,
        save_fig=save_fig,
        save_dir=save_dir,
        save_stem=save_stem_final,
        show_title=show_title,
        strict_missing_files=strict_missing_files,
        band_mode=band_mode,
    )