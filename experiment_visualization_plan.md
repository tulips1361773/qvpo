# 实验结果可视化方案设计文档（建议修改版）

## 一、可视化目标

为验证所提基于生成式强化学习的感知安全策略在 AAV 使能 ISAC 系统中的有效性，需要绘制以下对比图：

1. **评估奖励收敛曲线**  
   展示不同算法在训练过程中的 `eval reward` 变化趋势，并基于多随机种子绘制均值曲线与方差带。

2. **感知泄露率对比图**  
   展示不同算法在训练完成后的最终感知泄露率。

3. **感知安全 SNR 对比图**  
   展示合法接收器感知 SNR、窃听者最大感知 SNR，以及二者形成的感知安全裕度。

---

## 二、指标设计

### 2.1 统一评估指标

所有算法在评估阶段统一输出以下指标：

- `eval_reward_mean`：评估回报均值
- `eval_reward_ep_std`：评估回报在多个 episode 上的标准差
- `eval_leakage_rate`：评估阶段平均感知泄露率
- `eval_legal_snr_db_mean`：合法接收器平均感知 SNR
- `eval_legal_snr_db_ep_std`：合法接收器感知 SNR 的 episode 标准差
- `eval_eav_snr_max_db_mean`：窃听用户中最大感知 SNR 的平均值
- `eval_eav_snr_avg_db_mean`：所有窃听用户感知 SNR 的平均值
- `eval_snr_gap_db_mean`：感知安全裕度，定义为

\[
\Delta_{\text{SNR}} = \eta_0 - \max_k \eta_{e,k}
\]

其中，`eval_leakage_rate` 建议统一定义为：

\[
\text{LeakageRate}=\frac{1}{TK}\sum_{t=1}^{T}\sum_{k=1}^{K}\mathbf{1}\{\gamma_{e,k}(t)>\gamma_{\text{th}}\}
\]

### 2.2 训练阶段辅助指标

用于诊断训练过程，但不作为主对比指标：

- `train_reward`
- `train_reward_ma100`
- `time_elapsed_sec`

不建议将算法特有指标（如某些算法专有的 EMA reward）作为统一主表字段；若保留，应允许缺失值 `NaN`，而非写 `0.0`。

---

## 三、CSV 记录方案

### 3.1 训练过程文件：`training_metrics_[run_id].csv`

建议字段如下：

```text
run_id
algorithm
seed
scenario_name
step
eval_episode_count
eval_reward_mean
eval_reward_ep_std
eval_leakage_rate
eval_legal_snr_db_mean
eval_legal_snr_db_ep_std
eval_eav_snr_max_db_mean
eval_eav_snr_avg_db_mean
eval_snr_gap_db_mean
train_reward
train_reward_ma100
time_elapsed_sec

用途：绘制训练过程中的 reward 曲线、leakage 曲线，并支持多 seed 聚合。

3.2 最终结果文件：final_comparison_[run_id].csv

建议字段如下：

run_id
algorithm
seed
scenario_name
total_train_steps
eval_episode_count
final_eval_reward
final_eval_reward_ep_std
final_leakage_rate
final_legal_snr_db
final_eav_snr_max_db
final_eav_snr_avg_db
final_snr_gap_db
best_eval_reward
best_step
training_time_sec

用途：绘制最终柱状图，并用于论文表格统计。

四、代码修改建议
4.1 统一 evaluate 接口

建议在 main.py 和 sac2.py 中统一 evaluate() 的返回格式：

{
    'mean_return': ...,
    'std_return': ...,
    'eval_leakage_rate': ...,
    'legal_snr_db_mean': ...,
    'legal_snr_db_std': ...,
    'eav_snr_max_db_mean': ...,
    'eav_snr_avg_db_mean': ...,
    'snr_gap_db_mean': ...
}

要求两套算法代码使用完全相同的指标定义与统计口径。

4.2 日志写入建议
训练过程每次评估时写入 training_metrics_[run_id].csv
训练结束后额外执行一次独立的 final eval
最终评估建议使用比训练中更大的 eval_episodes
将最终评估结果写入 final_comparison_[run_id].csv
4.3 dB 统计建议

若需更严格地统计感知 SNR，建议先转为线性域求均值，再转回 dB：

𝜂
mean
=
10
log
⁡
10
(
1
𝑀
∑
𝑖
=
1
𝑀
10
𝜂
𝑖
/
10
)
η
mean
	​

=10log
10
	​

(
M
1
	​

i=1
∑
M
	​

10
η
i
	​

/10
)

若仅用于趋势比较，也可直接对 dB 值做算术平均，但必须在全文中保持一致。

五、绘图方案
5.1 折线图：评估奖励对比
横轴：step
纵轴：eval_reward_mean
分组：algorithm
基于多 seed 计算均值与标准差带

建议补充一张：

5.2 折线图：感知泄露率收敛曲线
横轴：step
纵轴：eval_leakage_rate

这样可以更直接体现安全性能的训练趋势。

5.3 柱状图：最终泄露率对比
横轴：algorithm
纵轴：final_leakage_rate

建议使用：

柱状图 + 误差棒
叠加每个 seed 的散点
5.4 柱状图：最终感知安全 SNR 对比

推荐两种方案之一：

方案 A

分组柱状图展示：

final_legal_snr_db
final_eav_snr_max_db
方案 B（更推荐）

直接展示：

final_snr_gap_db

该指标更能体现合法感知端与潜在窃听端之间的安全优势。

六、实验实施建议
每种算法至少运行 3–5 个随机种子
所有算法必须使用相同的：
环境配置
训练步数
评估 episode 数
泄露判定阈值
训练 CSV 和 final CSV 中必须显式记录：
seed
run_id
scenario_name
绘图前先按同一实验配置筛选数据，避免不同配置实验混合
七、补充建议

如果当前目标是尽快产出论文级对比图，建议优先完成以下四项：

补充 seed、run_id、scenario_name 三列
统一 evaluate() 返回字段
将 SNR 图改为 SNR gap 图
补充一张 leakage rate 训练曲线

这样生成的结果将更符合正式论文中的实验可视化规范，也更能突出“通信用户兼潜在感知窃听者”这一系统模型下所提出安全策略的有效性。