import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
save_dir = "./figures"
save_stem = f"eval_reward_final_{smooth_method}_sw{smooth_weight}_{band_mode}"

show_title = False             # 论文正文图建议 False
strict_missing_files = False   # 当前 TD3 第二个 seed 还没跑完时建议 False；最终出图可改 True

# 坐标轴范围
xlim = (0.0, 1.0)
ylim = (-60, 50)

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

# 绘图风格
main_linewidth = 2.8
seed_linewidth = 1.1
seed_line_alpha = 0.26
band_alpha = 0.14

# 字体与导出设置
plt.rcParams["font.family"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif", "STIXGeneral"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 13
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["text.usetex"] = False


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
        "seed42": "/home/moqianyu_26/sda/qvpo/record/Env/TD3/ratio=0.1/seed=42/run_id=260401_223304/csv_logs/training_metrics_260401_223304.csv",
        # TODO: TD3 第二个 seed 跑完后，把下面这一行改成正确路径
        "seed101": "/home/moqianyu_26/sda/qvpo/record/Env/TD3/ratio=0.1/seed=101/run_id=TO_BE_FILLED/csv_logs/training_metrics_TO_BE_FILLED.csv",
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
def load_training_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["step", "eval_reward_mean"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {csv_path}")

    df = df.dropna(subset=["step", "eval_reward_mean"]).copy()
    df = df.sort_values("step").reset_index(drop=True)
    return df[["step", "eval_reward_mean"]]


def load_multi_seed_runs(seed_path_dict, strict_missing_files=False):
    runs = {}

    for seed_name, csv_path in seed_path_dict.items():
        if not os.path.exists(csv_path):
            msg = f"[Missing seed file] {seed_name}: {csv_path}"
            if strict_missing_files:
                raise FileNotFoundError(msg)
            warnings.warn(msg + " -> skipped.")
            continue

        runs[seed_name] = load_training_csv(csv_path)

    if len(runs) == 0:
        raise ValueError("No valid seed csv files were loaded.")

    return runs


# ============================================================
# 5. 多 seed 对齐与聚合
# ============================================================
def align_runs_on_step(runs):
    """
    输入:
        runs: dict[seed_name] = DataFrame(step, eval_reward_mean)

    返回:
        merged: DataFrame(step, seed1, seed2, ...)
        seed_names: list[str]
    """
    merged = None
    seed_names = list(runs.keys())

    for seed_name, df in runs.items():
        tmp = df.rename(columns={"eval_reward_mean": seed_name})
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
# 6. 绘图
# ============================================================
def plot_eval_reward_curves(
    algo_seed_paths,
    smooth_method="ema",
    smooth_weight=0.8,
    show_seed_lines=True,
    show_band=True,
    band_mode="minmax",
    save_fig=True,
    save_dir="./figures",
    save_stem="eval_reward_final",
    show_title=False,
    strict_missing_files=False,
    xlim=(0.0, 1.0),
    ylim=(-60, 50),
):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    for algo_name, seed_path_dict in algo_seed_paths.items():
        runs = load_multi_seed_runs(
            seed_path_dict,
            strict_missing_files=strict_missing_files
        )

        merged, valid_seed_names = align_runs_on_step(runs)
        x = merged["step"].to_numpy(dtype=float) / 1e6
        color = color_map[algo_name]

        # 每个 seed 分别平滑
        seed_smooth_matrix = build_smoothed_seed_matrix(
            merged,
            valid_seed_names,
            smooth_method=smooth_method,
            smooth_weight=smooth_weight
        )

        # 聚合得到均值主线与阴影边界
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

    ax.set_xlabel(r"Training Steps ($\times 10^6$)")
    ax.set_ylabel("Evaluation Reward")

    if show_title:
        ax.set_title("Evaluation Reward During Training")

    if xlim is not None:
        ax.set_xlim(*xlim)
        ax.set_xticks(np.linspace(xlim[0], xlim[1], 6))

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, linestyle="--", alpha=0.20)

    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        fontsize=12,
        handlelength=2.8
    )

    plt.tight_layout()

    if save_fig:
        os.makedirs(save_dir, exist_ok=True)
        pdf_path = os.path.join(save_dir, f"{save_stem}.pdf")
        png_path = os.path.join(save_dir, f"{save_stem}.png")

        plt.savefig(pdf_path, dpi=600, bbox_inches="tight")
        plt.savefig(png_path, dpi=600, bbox_inches="tight")

        print(f"[Saved PDF] {pdf_path}")
        print(f"[Saved PNG] {png_path}")

    plt.show()


# ============================================================
# 7. 运行
# ============================================================
if __name__ == "__main__":
    plot_eval_reward_curves(
        algo_seed_paths=algo_seed_paths,
        smooth_method=smooth_method,
        smooth_weight=smooth_weight,
        show_seed_lines=show_seed_lines,
        show_band=show_band,
        band_mode=band_mode,
        save_fig=save_fig,
        save_dir=save_dir,
        save_stem=save_stem,
        show_title=show_title,
        strict_missing_files=strict_missing_files,
        xlim=xlim,
        ylim=ylim,
    )
