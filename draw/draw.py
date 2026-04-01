import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# 1. 用户可调参数
# =========================
smooth_method = "ema"          # "none" / "ema" / "moving_average"
smooth_weight = 0.8            # for ema: 0.7 / 0.8 / 0.9
show_raw = True                # True / False

# moving average窗口设置
ma_window_map = {
    0.7: 5,
    0.8: 9,
    0.9: 15
}

save_fig = True
save_dir = "./figures"
save_stem = f"eval_reward_paper_{smooth_method}_sw{smooth_weight}_raw{show_raw}"

# 论文正文图建议不显示标题
show_title = False

# 配色：你的算法用橙色突出
color_map = {
    "PIM-DiffTD3": "#F58518",   # orange
    "TD3": "#4C78A8",           # blue
    "SAC": "#54A24B",           # green
}

# 字体与导出设置
plt.rcParams["font.family"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif", "STIXGeneral"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 13
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["text.usetex"] = False


# =========================
# 2. 数据路径
# =========================
csv_paths = {
    "PIM-DiffTD3": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_30_00_22_24_0/csv_logs/training_metrics_26_03_30_00_22_24_0.csv",
    "TD3": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_31_12_40_00_0/csv_logs/training_metrics_26_03_31_12_40_00_0.csv",
    "SAC": "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/fsac_2026-03-31_12-38-17/csv_logs/training_metrics_sac_seed42_2026-03-31_12-38-17.csv",
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
    return df


# =========================
# 5. 绘图
# =========================
def plot_eval_reward_curves(
    csv_paths,
    smooth_method="ema",
    smooth_weight=0.8,
    show_raw=True,
    save_fig=True,
    save_dir="./figures",
    save_stem="eval_reward",
    show_title=False
):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    for algo_name, csv_path in csv_paths.items():
        df = load_training_csv(csv_path)

        x = df["step"].to_numpy() / 1e6
        y = df["eval_reward_mean"].to_numpy()
        y_smooth = smooth_curve(y, method=smooth_method, weight=smooth_weight)

        color = color_map[algo_name]

        # 原始曲线：同色、更淡、不进图例
        if show_raw and smooth_method != "none":
            ax.plot(
                x, y,
                color=color,
                linewidth=1.1,
                alpha=0.20,
                label="_nolegend_"
            )

        # 主曲线：同色、更深、更粗
        ax.plot(
            x, y_smooth,
            color=color,
            linewidth=2.6,
            label=algo_name
        )

    ax.set_xlabel(r"Training Steps ($\times 10^6$)")
    ax.set_ylabel("Evaluation Reward")

    if show_title:
        ax.set_title("Evaluation Reward During Training")

    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))

    # 纵轴从 -60 开始，更紧凑
    ax.set_ylim(-60, 50)

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
        csv_paths=csv_paths,
        smooth_method=smooth_method,
        smooth_weight=smooth_weight,
        show_raw=show_raw,
        save_fig=save_fig,
        save_dir=save_dir,
        save_stem=save_stem,
        show_title=show_title
    )