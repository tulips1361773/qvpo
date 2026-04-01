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
SMOOTH_WEIGHT = 0.9 

# --------------------------------------------

def tensorboard_smoothing(scalars, weight):
    """
    复刻 TensorBoard 的 EMA 平滑算法
    """
    if len(scalars) == 0:
        return []
        
    last = scalars.iloc[0]  
    smoothed = []
    for point in scalars:
        if pd.isna(point):
            smoothed.append(point)
            continue
        
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
        
    return smoothed

def main():
    # 1. 全局风格与字体设置 (匹配毕业论文要求)
    # 使用 seaborn 基础白底风格
    sns.set_theme(style="white") 
    
    # 强制设置字体为 Times New Roman，字号设为 12 (对应小四)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    
    # 2. 读取数据
    print("正在读取数据...")
    df_pim = pd.read_csv(csv_pim)
    df_sac = pd.read_csv(csv_sac)
    
    # 3. 计算平滑数据
    pim_smooth = tensorboard_smoothing(df_pim[y_col], weight=SMOOTH_WEIGHT)
    sac_smooth = tensorboard_smoothing(df_sac[y_col], weight=SMOOTH_WEIGHT)
    
    # 4. 创建画布 (8x5 是较经典的论文插图比例)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    # === 绘制 PIM-DiffTD3 ===
    ax.plot(df_pim[x_col], df_pim[y_col], alpha=0.2, color='#1f77b4', linewidth=1)
    ax.plot(df_pim[x_col], pim_smooth, label='PIM-DiffTD3', color='#1f77b4', linewidth=2.5)
    
    # === 绘制 SAC ===
    ax.plot(df_sac[x_col], df_sac[y_col], alpha=0.2, color='#ff7f0e', linewidth=1)
    ax.plot(df_sac[x_col], sac_smooth, label='SAC', color='#ff7f0e', linewidth=2.5)
    
    # 5. 美化图表
    # 设置坐标轴标签 (去掉了内嵌的 Title，请在论文正文中用图注说明)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Eval Reward (Mean)")
    
    # 添加轻量级的辅助网格线 (虚线，半透明)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 设置图例：去掉阴影，使用简洁的灰色边框，半透明背景防止遮挡数据
    ax.legend(loc="lower right", frameon=True, shadow=False, framealpha=0.9, edgecolor='silver')
    
    # 边框处理：去掉顶部和右侧的边框 (符合顶刊审美)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 稍微加粗底部和左侧边框
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    
    # 让 step 的数字不用科学计数法显示
    ax.ticklabel_format(style='plain', axis='x')
    
    # 6. 保存并展示
    plt.tight_layout()
    save_path = "reward22.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"绘图完成！图片已保存至当前目录的: {save_path}")

if __name__ == "__main__":
    main()