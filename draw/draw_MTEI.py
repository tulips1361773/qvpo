import os
import glob
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


# ============================================================
# 1. 用户可调参数
# ============================================================
smooth_method = "ema"          # "none" / "ema" / "moving_average"
smooth_weight = 0.9

show_seed_lines = True
show_band = True
band_mode = "minmax"           # "minmax" / "std"

save_fig = True
save_dir = "./figures4"
show_title = False
strict_missing_files = False

xlim = (0.0, 1.0)
ylim_mtei = (0.0, 0.5)

# 与环境参数保持一致；若实验改过，这里同步修改
eav_penalty_clip_max = 10.0
save_stem_mtei = f"train_mtei_{smooth_method}_sw{smooth_weight}_{band_mode}"

# 候选 TensorBoard tag：按顺序匹配
candidate_tags = [
    "security/eav_penalty_clipped",
    "reward_terms/eav_penalty_clipped",
]

# moving average 窗口设置
ma_window_map = {
    0.7: 5,
    0.8: 9,
    0.85: 11,
    0.9: 15,
}

# 配色
color_map = {
    "PIM-DiffTD3": "#F58518",
    "TD3": "#4C78A8",
    "SAC": "#54A24B",
    "PPO": "#E45756",
}

# 绘图风格
main_linewidth = 2.8
seed_linewidth = 1.1
seed_line_alpha = 0.20
band_alpha = 0.10

plt.rcParams["font.family"] = ["Times New Roman", "Liberation Serif", "DejaVu Serif", "STIXGeneral"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 13
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["text.usetex"] = False


# ============================================================
# 2. training csv 路径（沿用你的原代码）
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
    if method == "ema":
        return ema_smooth(y, weight=weight)
    if method == "moving_average":
        window = ma_window_map.get(weight, 9)
        return moving_average_smooth(y, window=window)
    raise ValueError(f"Unsupported smooth_method: {method}")


# ============================================================
# 4. 多 seed 平滑与聚合
# ============================================================
def build_smoothed_seed_matrix(merged_dict, seed_names, smooth_method="ema", smooth_weight=0.8):
    smooth_list = []

    for seed_name in seed_names:
        y = np.asarray(merged_dict[seed_name], dtype=float)
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

    return np.column_stack(smooth_list)


def aggregate_seed_curves(seed_matrix, band_mode="minmax"):
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
# 5. 从 training csv 自动推断 event 文件
# ============================================================
def infer_event_file_from_training_csv(training_csv_path):
    if not os.path.exists(training_csv_path):
        raise FileNotFoundError(f"Training CSV not found: {training_csv_path}")

    csv_logs_dir = os.path.dirname(training_csv_path)
    run_dir = os.path.dirname(csv_logs_dir)

    pattern = os.path.join(run_dir, "events.out.tfevents.*")
    candidates = sorted(glob.glob(pattern))

    if len(candidates) == 0:
        raise FileNotFoundError(f"No event file found under run dir: {run_dir}")

    candidates = sorted(candidates, key=os.path.getmtime)
    return candidates[-1]


def build_algo_seed_event_paths_from_csv_paths(algo_seed_paths, strict_missing_files=False):
    algo_seed_event_paths = {}

    for algo_name, seed_path_dict in algo_seed_paths.items():
        algo_seed_event_paths[algo_name] = {}

        for seed_name, training_csv_path in seed_path_dict.items():
            if not os.path.exists(training_csv_path):
                msg = f"[Missing training csv] {algo_name}-{seed_name}: {training_csv_path}"
                if strict_missing_files:
                    raise FileNotFoundError(msg)
                warnings.warn(msg + " -> skipped.")
                continue

            try:
                event_path = infer_event_file_from_training_csv(training_csv_path)
                algo_seed_event_paths[algo_name][seed_name] = event_path
            except FileNotFoundError as e:
                if strict_missing_files:
                    raise
                warnings.warn(str(e) + " -> skipped.")

    return algo_seed_event_paths


# ============================================================
# 6. 读取 TensorBoard scalar（自动匹配候选 tag）
# ============================================================
def load_tb_scalar_any(event_path, candidate_tags):
    if not os.path.exists(event_path):
        raise FileNotFoundError(f"Event file not found: {event_path}")

    ea = event_accumulator.EventAccumulator(
        event_path,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()

    scalar_tags = ea.Tags().get("scalars", [])

    matched_tag = None
    for tag in candidate_tags:
        if tag in scalar_tags:
            matched_tag = tag
            break

    if matched_tag is None:
        raise ValueError(
            f"None of candidate tags found in {event_path}\n"
            f"Candidate tags: {candidate_tags}\n"
            f"Available scalar tags: {scalar_tags}"
        )

    events = ea.Scalars(matched_tag)
    if len(events) == 0:
        raise ValueError(f"No scalar data found for tag '{matched_tag}' in {event_path}")

    steps = np.array([e.step for e in events], dtype=float)
    values = np.array([e.value for e in events], dtype=float)

    order = np.argsort(steps)
    steps = steps[order]
    values = values[order]

    uniq_steps = []
    uniq_values = []
    last_step = None
    for s, v in zip(steps, values):
        if last_step is not None and s == last_step:
            uniq_values[-1] = v
        else:
            uniq_steps.append(s)
            uniq_values.append(v)
            last_step = s

    return {
        "step": np.asarray(uniq_steps, dtype=float),
        "value": np.asarray(uniq_values, dtype=float),
        "tag_used": matched_tag,
    }


def load_multi_seed_tb_runs(seed_event_dict, candidate_tags, strict_missing_files=False):
    runs = {}

    for seed_name, event_path in seed_event_dict.items():
        if not os.path.exists(event_path):
            msg = f"[Missing event file] {seed_name}: {event_path}"
            if strict_missing_files:
                raise FileNotFoundError(msg)
            warnings.warn(msg + " -> skipped.")
            continue

        try:
            run = load_tb_scalar_any(event_path, candidate_tags)
            runs[seed_name] = run
            print(f"[Tag matched] {seed_name}: {run['tag_used']}")
        except ValueError as e:
            if strict_missing_files:
                raise
            warnings.warn(f"[Skip seed] {seed_name}: {e}")

    return runs


# ============================================================
# 7. 多 seed 对齐
# ============================================================
def align_tb_runs_on_step(runs):
    all_steps = set()
    for _, run in runs.items():
        all_steps.update(run["step"].tolist())

    merged_steps = np.array(sorted(all_steps), dtype=float)
    merged_dict = {}

    for seed_name, run in runs.items():
        src_steps = run["step"]
        src_values = run["value"]

        y = np.full(merged_steps.shape, np.nan, dtype=float)
        step_to_value = {s: v for s, v in zip(src_steps, src_values)}

        for i, s in enumerate(merged_steps):
            if s in step_to_value:
                y[i] = step_to_value[s]

        valid = ~np.isnan(y)
        if valid.sum() >= 2:
            x_valid = merged_steps[valid]
            y_valid = y[valid]
            interp_mask = np.isnan(y) & (merged_steps >= x_valid[0]) & (merged_steps <= x_valid[-1])
            y[interp_mask] = np.interp(merged_steps[interp_mask], x_valid, y_valid)

        merged_dict[seed_name] = y

    return merged_steps, merged_dict


# ============================================================
# 8. 画 train / MTEI
# ============================================================
def plot_train_mtei_curves(
    algo_seed_paths,
    candidate_tags=None,
    eav_penalty_clip_max=200.0,
    ylabel="Maximum Threat Exposure Index",
    save_stem="train_mtei",
    smooth_method="ema",
    smooth_weight=0.8,
    show_seed_lines=True,
    show_band=True,
    band_mode="minmax",
    save_fig=True,
    save_dir="./figures3",
    show_title=False,
    strict_missing_files=False,
    xlim=(0.0, 1.0),
    ylim=None,
    legend_loc="upper right",
):
    if candidate_tags is None:
        candidate_tags = [
            "security/eav_penalty_clipped",
            "reward_terms/eav_penalty_clipped",
        ]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    plotted_algorithms = []

    algo_seed_event_paths = build_algo_seed_event_paths_from_csv_paths(
        algo_seed_paths,
        strict_missing_files=strict_missing_files
    )

    for algo_name, seed_event_dict in algo_seed_event_paths.items():
        if len(seed_event_dict) == 0:
            warnings.warn(f"[Skip algorithm] {algo_name}: no valid event files.")
            continue

        runs_raw = load_multi_seed_tb_runs(
            seed_event_dict,
            candidate_tags=candidate_tags,
            strict_missing_files=strict_missing_files
        )

        if len(runs_raw) == 0:
            warnings.warn(
                f"[Skip algorithm] {algo_name}: none of its seeds contain MTEI-related tags."
            )
            continue

        runs_mtei = {}
        valid_seed_names = []

        for seed_name, run in runs_raw.items():
            mtei = np.clip(run["value"] / eav_penalty_clip_max, 0.0, 1.0)
            runs_mtei[seed_name] = {
                "step": run["step"],
                "value": mtei,
            }
            valid_seed_names.append(seed_name)

        if len(valid_seed_names) == 0:
            warnings.warn(f"[Skip algorithm] {algo_name}: no usable seeds after filtering.")
            continue

        merged_steps, merged_dict = align_tb_runs_on_step(runs_mtei)
        x = merged_steps / 1e6
        color = color_map.get(algo_name, None)

        seed_smooth_matrix = build_smoothed_seed_matrix(
            merged_dict,
            valid_seed_names,
            smooth_method=smooth_method,
            smooth_weight=smooth_weight
        )

        mean_curve, lower_curve, upper_curve = aggregate_seed_curves(
            seed_smooth_matrix,
            band_mode=band_mode
        )

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

        valid = ~np.isnan(mean_curve)
        ax.plot(
            x[valid], mean_curve[valid],
            color=color,
            linewidth=main_linewidth,
            label=algo_name
        )

        plotted_algorithms.append(algo_name)
        print(f"[Loaded MTEI] {algo_name}: {len(valid_seed_names)} seed(s) -> {valid_seed_names}")

    if len(plotted_algorithms) == 0:
        raise RuntimeError(
            "No algorithm has usable MTEI-related TensorBoard tags. "
            "Nothing can be plotted."
        )

    ax.set_xlabel(r"Training Steps ($\times 10^6$)")
    ax.set_ylabel(ylabel)

    if show_title:
        ax.set_title(ylabel)

    if xlim is not None:
        ax.set_xlim(*xlim)
        ax.set_xticks(np.linspace(xlim[0], xlim[1], 6))

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.tick_params(axis="x", which="both", direction="in", bottom=True, top=False, labelbottom=True)
    ax.tick_params(axis="y", which="both", direction="in", left=True, right=False, labelleft=True, labelright=False)

    ax.grid(True, linestyle="--", alpha=0.20)

    ax.legend(
        loc=legend_loc,
        frameon=True,
        framealpha=0.90,
        edgecolor="0.75",
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
# 9. 运行
# ============================================================
if __name__ == "__main__":
    plot_train_mtei_curves(
        algo_seed_paths=algo_seed_paths,
        candidate_tags=candidate_tags,
        eav_penalty_clip_max=eav_penalty_clip_max,
        ylabel="Maximum Threat Exposure Index",
        save_stem=save_stem_mtei,
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
        ylim=ylim_mtei,
        legend_loc="upper right",
    )