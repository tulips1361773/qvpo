import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. 用户可调参数
# =========================
smooth_method = "ema"          # "none" / "ema" / "moving_average"
smooth_weight = 0.9            # for ema: 0.7 / 0.8 / 0.9
show_raw = False                # True / False，是否显示每个 seed 的原始曲线
show_seed_band = True          # True / False，是否显示 seed 间标准差阴影

# moving average窗口设置
ma_window_map = {
    0.7: 5,
    0.8: 9,
    0.9: 15
}

save_fig = True
save_dir = "./figures"
save_stem = f"eval_reward_4algos_2seeds_{smooth_method}_sw{smooth_weight}_raw{show_raw}"

# 论文正文图建议不显示标题
show_title = False

# 当某个 seed 文件还不存在时：
# False -> 给出 warning，并跳过该 seed（适合你现在 TD3 第二个 seed 还没跑完）
# True  -> 直接报错（适合最终论文出图时确保数据完整）
strict_missing_files = False

# 横纵轴范围
xlim = (0.0, 1.0)
ylim = (-60, 50)

# 配色：你的算法用橙色突出
color_map = {
    "PIM-DiffTD3": "#F58518",   # orange
    "TD3": "#4C78A8",           # blue
    "SAC": "#54A24B",           # green
    "PPO": "#E45756",           # red
}

# 字体与导出设置
plt.rcParams["font.family"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif", "STIXGeneral"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 13
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["text.usetex"] = False


# =========================
# 2. 数据路径（每个算法两个随机种子）
# =========================
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
        # TODO: 这里先放占位路径；你跑完正确的第二个种子后，直接替换下面这一行
        "seed101": "/home/moqianyu_26/sda/qvpo/record/Env/TD3/ratio=0.1/seed=101/run_id=TO_BE_FILLED/csv_logs/training_metrics_TO_BE_FILLED.csv",
    },
    "PPO": {
        "seed42": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=PPO/seed=42/run_id=26_04_01_21_18_15_0/csv_logs/training_metrics_26_04_01_21_18_15_0.csv",
        "seed101": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=PPO/seed=101/run_id=26_04_01_22_38_28_0/csv_logs/training_metrics_26_04_01_22_38_28_0.csv",
    },
}


# =========================
# 3. 平滑函数
# =========================
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


# =========================
# 4. 读CSV
# =========================
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
    """
    返回:
        runs: dict[seed_name] = df(step, eval_reward_mean)
    """
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


def align_and_aggregate_runs(runs, smooth_method="ema", smooth_weight=0.8):
    """
    将多个 seed 按 step 对齐，并聚合成：
        x
        mean_raw / std_raw
        mean_smooth / std_smooth
        各 seed 的原始列与平滑列
    """
    merged = None
    seed_names = list(runs.keys())

    for seed_name, df in runs.items():
        tmp = df.rename(columns={"eval_reward_mean": seed_name})
        merged = tmp if merged is None else pd.merge(merged, tmp, on="step", how="outer")

    merged = merged.sort_values("step").reset_index(drop=True)

    # 仅对内部缺口做插值，不向两端外推
    for seed_name in seed_names:
        merged[seed_name] = merged[seed_name].interpolate(
            method="linear",
            limit_area="inside"
        )

    # 原始统计
    merged["mean_raw"] = merged[seed_names].mean(axis=1, skipna=True)
    merged["std_raw"] = merged[seed_names].std(axis=1, skipna=True).fillna(0.0)

    # 每个 seed 先各自平滑，再统计均值和标准差
    smooth_cols = []
    for seed_name in seed_names:
        smooth_col = f"{seed_name}_smooth"
        valid_mask = merged[seed_name].notna()

        merged[smooth_col] = np.nan
        if valid_mask.any():
            y_valid = merged.loc[valid_mask, seed_name].to_numpy()
            merged.loc[valid_mask, smooth_col] = smooth_curve(
                y_valid,
                method=smooth_method,
                weight=smooth_weight
            )

        smooth_cols.append(smooth_col)

    merged["mean_smooth"] = merged[smooth_cols].mean(axis=1, skipna=True)
    merged["std_smooth"] = merged[smooth_cols].std(axis=1, skipna=True).fillna(0.0)

    return merged, seed_names


# =========================
# 5. 绘图
# =========================
def plot_eval_reward_curves(
    algo_seed_paths,
    smooth_method="ema",
    smooth_weight=0.8,
    show_raw=True,
    show_seed_band=True,
    save_fig=True,
    save_dir="./figures",
    save_stem="eval_reward",
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
        merged, valid_seed_names = align_and_aggregate_runs(
            runs,
            smooth_method=smooth_method,
            smooth_weight=smooth_weight
        )

        x = merged["step"].to_numpy() / 1e6
        color = color_map[algo_name]

        # 每个 seed 的原始曲线：同色、更淡、不进图例
        if show_raw and smooth_method != "none":
            for seed_name in valid_seed_names:
                y_seed = merged[seed_name].to_numpy()
                valid = ~np.isnan(y_seed)
                ax.plot(
                    x[valid], y_seed[valid],
                    color=color,
                    linewidth=1.0,
                    alpha=0.12,
                    label="_nolegend_"
                )

        # seed 间方差阴影（基于平滑后的统计）
        if show_seed_band and len(valid_seed_names) >= 2:
            y_mean = merged["mean_smooth"].to_numpy()
            y_std = merged["std_smooth"].to_numpy()
            valid = ~np.isnan(y_mean)

            ax.fill_between(
                x[valid],
                (y_mean - y_std)[valid],
                (y_mean + y_std)[valid],
                color=color,
                alpha=0.16,
                linewidth=0.0,
                label="_nolegend_"
            )

        # 主曲线：各 seed 平滑后再求均值
        if smooth_method == "none":
            y_main = merged["mean_raw"].to_numpy()
        else:
            y_main = merged["mean_smooth"].to_numpy()

        valid = ~np.isnan(y_main)
        ax.plot(
            x[valid], y_main[valid],
            color=color,
            linewidth=2.8,
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

    # 图例放右下角空白区
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


# =========================
# 6. 运行
# =========================
if __name__ == "__main__":
    plot_eval_reward_curves(
        algo_seed_paths=algo_seed_paths,
        smooth_method=smooth_method,
        smooth_weight=smooth_weight,
        show_raw=show_raw,
        show_seed_band=show_seed_band,
        save_fig=save_fig,
        save_dir=save_dir,
        save_stem=save_stem,
        show_title=show_title,
        strict_missing_files=strict_missing_files,
        xlim=xlim,
        ylim=ylim,
    )
