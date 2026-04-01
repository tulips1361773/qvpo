import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- 配置区域 -----------------
# 1. 实验文件路径
csv_pim = "/home/moqianyu_26/sda/qvpo/record/Env/policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_03_27_23_51_51_0/csv_logs/training_metrics_26_03_27_23_51_51_0.csv"
csv_sac = "/home/moqianyu_26/sda/qvpo/record/sac/sac_myenv3_improved_2026-03-27_20-44-10/csv_logs/training_metrics_sac_seed42_2026-03-27_20-44-10.csv"

# 2. 轴列名
x_col = "step"
y_col = "eval_reward_mean"

# 3. 平滑权重 (与 TensorBoard 完全一致)
# 波动很大建议设为 0.9 或 0.95；0.7 适合波动中等的数据
SMOOTH_WEIGHT = 0.7 

# --------------------------------------------

def tensorboard_smoothing(scalars, weight):
    """
    复刻 TensorBoard 的 EMA 平滑算法
    :param scalars: 原始数据序列 (pandas Series 或 list)
    :param weight: 平滑权重 (0.0~1.0)，越高越平滑
    :return: 平滑后的列表
    """
    if len(scalars) == 0:
        return []
        
    last = scalars.iloc[0]  # 初始化为第一个值
    smoothed = []
    for point in scalars:
        if pd.isna(point): # 跳过 NaN
            smoothed.append(point)
            continue
        
        # TensorBoard 平滑公式
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
        
    return smoothed

def main():
    # 设置全局绘图风格，学术质感
    sns.set_theme(style="whitegrid")
    
    # 1. 读取数据
    print("正在读取数据...")
    df_pim = pd.read_csv(csv_pim)
    df_sac = pd.read_csv(csv_sac)
    
    # 2. 计算平滑数据
    pim_smooth = tensorboard_smoothing(df_pim[y_col], weight=SMOOTH_WEIGHT)
    sac_smooth = tensorboard_smoothing(df_sac[y_col], weight=SMOOTH_WEIGHT)
    
    # 3. 创建画布
    plt.figure(figsize=(10, 6), dpi=300) # dpi=300 保证图片高清
    
    # === 绘制 PIM-DiffTD3 ===
    # 背景真实波动线 (浅色、半透明)
    plt.plot(df_pim[x_col], df_pim[y_col], alpha=0.2, color='#1f77b4', linewidth=1)
    # 前景平滑线 (深色、加粗)
    plt.plot(df_pim[x_col], pim_smooth, label='PIM-DiffTD3', color='#1f77b4', linewidth=2.5)
    
    # === 绘制 SAC ===
    # 背景真实波动线
    plt.plot(df_sac[x_col], df_sac[y_col], alpha=0.2, color='#ff7f0e', linewidth=1)
    # 前景平滑线
    plt.plot(df_sac[x_col], sac_smooth, label='SAC', color='#ff7f0e', linewidth=2.5)
    
    # 4. 美化图表
    plt.title("Eval Reward Comparison: PIM-DiffTD3 vs SAC", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Environment Steps", fontsize=14)
    plt.ylabel("Eval Reward (Mean)", fontsize=14)
    
    # 设置图例位置和样式
    plt.legend(fontsize=12, loc="lower right", frameon=True, shadow=True)
    
    # 调整坐标轴的数字大小
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    
    # 让 step 的数字不用科学计数法显示（可选）
    plt.ticklabel_format(style='plain', axis='x')
    
    # 5. 保存并展示
    plt.tight_layout()
    save_path = "reward_comparison2.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"绘图完成！图片已保存至当前目录的: {save_path}")

if __name__ == "__main__":
    main()