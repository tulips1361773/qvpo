QVPO-UAV-ISAC 实验报告：Solu2 → Solu3 → Solu3(2) 迭代分析
实验日期: 2026-01-08
实验目标: 解决QVPO训练不收敛问题，优化无人机综合感知与通信(UAV-ISAC)任务的策略学习
代码仓库: https://github.com/tulips1361773/qvpo

一、实验版本对照表
版本	曲线颜色	核心修改	状态
Solu2	🟢 绿色	基线版本	参考基准
Solu3	⚫ 黑色	窃听者惩罚改为logsumexp聚合	❌ 性能退化
Solu3(2)	🔵 蓝色	在Solu3基础上进行算法和参数修正	✅ 当前最优
二、实验结果分析
2.1 训练稳定性指标
Loss/Critic (右上图)
指标	🟢 绿色(Solu2)	⚫ 黑色(Solu3)	🔵 蓝色(Solu3(2))	观察
初始值	~45	~45	~45	一致
稳定值	20-25	15-20	15-20	黑/蓝色更低
波动性	中等	高频震荡	平滑	✅ 蓝色最稳定
结论: Solu3(2)的Critic Loss收敛至15-20且波动最小，说明价值函数估计更准确。

Loss/Actor (左上图)
指标	🟢 绿色	⚫ 黑色	🔵 蓝色	观察
趋势	0.011-0.013	0.011-0.014	0.010-0.013	接近
波动	中等	高频	低频	✅ 蓝色更平滑
结论: Actor Loss三者相近，但蓝色曲线更平滑，说明扩散策略训练更稳定。

2.2 性能核心指标
Reward/Eval_Mean (左下图)
指标	🟢 绿色	⚫ 黑色	🔵 蓝色	变化
峰值	~42	~32	~40	🔵 接近绿色
稳定值	38-40	28-30	36-40	⚫ 下降25%
收敛性	100k步后稳定	持续波动	80k步后稳定	✅ 蓝色最快
关键发现:

⚫ Solu3失败原因: logsumexp聚合导致窃听者惩罚过重，eval reward大幅下降
🔵 Solu3(2)成功原因: 通过算法修正恢复至Solu2水平，且收敛更快
Reward/Train (底部大图)
指标	🟢 绿色	⚫ 黑色	🔵 蓝色	观察
波动幅度	10-55	10-45	20-50	🔵 下界更高
平均值	~40	~30	~42	✅ 蓝色最优
结论: 蓝色曲线的训练奖励最高且下界更稳定(20 vs 10)，说明策略质量提升。

2.3 Q值系统分析
Q/Current_Q1_Mean & Q/Current_Q2_Mean (左侧前两图)
指标	🟢 绿色	⚫ 黑色	🔵 蓝色	观察
峰值	30	17	30	⚫ 被严重压制
稳定值	28-30	15-16	28-30	蓝色恢复正常
Q/Running_Q_Mean (右侧上图)
指标	🟢 绿色	⚫ 黑色	🔵 蓝色	观察
趋势	升至30后稳定	升至15后停滞	升至28后稳定	⚫ 无法增长
Q/Running_Q_Std (左侧下图)
指标	🟢 绿色	⚫ 黑色	🔵 蓝色	观察
最终值	~11	~15	~7	✅ 蓝色方差最小
核心诊断:

⚫ 黑色曲线问题: Q值被系统性压低50%，说明logsumexp惩罚过重导致所有状态价值下降
🔵 蓝色曲线优势: Q值恢复正常且标准差更小(7 vs 11)，说明价值估计更精准
2.4 奖励构成分析 (底部12个子图)
Eta_0 (主奖励 - 感知SNR)
指标	🟢 绿色	⚫ 黑色	🔵 蓝色	观察
范围	8-14 dB	8-13 dB	9-14 dB	蓝色上界更高
Eav_Penalty (窃听者惩罚)
指标	🟢 绿色	⚫ 黑色	🔵 蓝色	观察
峰值	~2	~9	~2	⚫ 爆炸式增长
裁剪后	~2	~3.5	~2	⚫ 仍然过高
关键发现: 黑色曲线的eav_penalty裁剪前达到9(绿色仅2)，裁剪后仍达3.5，这是Solu3失败的直接原因！

Comm_Penalty (通信惩罚)
指标	🟢 绿色	⚫ 黑色	🔵 蓝色	观察
范围	0-0.8	0-1.0	0-0.8	接近
Action_Smooth_Penalty (动作平滑惩罚)
指标	🟢 绿色	⚫ 黑色	🔵 蓝色	观察
范围	0.8-1.5	1.0-1.5	0.8-1.4	蓝色略好
三、问题根因总结
3.1 Solu3 (⚫ 黑色) 失败原因
问题1: Logsumexp聚合放大窃听者惩罚
Python

# myenv.py (GitHub最新代码)
elif self.eav_agg == 'logsumexp':
    x = np.array(eavesdropper_snr_list, dtype=np.float32)
    kappa = float(self.eav_logsumexp_kappa)  # 命令行传入0.5
    m = float(np.max(x))
    sensing_snr_eavesdropper = m + (1.0 / kappa) * float(np.log(np.sum(np.exp(kappa * (x - m)))))
数学分析:

当kappa=0.5时，3个窃听者SNR为[12, 11, 10] dB:
Plain Text

logsumexp = 12 + (1/0.5) * log(exp(0) + exp(-0.5) + exp(-1.0))
          = 12 + 2 * log(1 + 0.606 + 0.368)
          = 12 + 2 * 0.72
          = 13.44 dB  (远高于top2的11.5 dB)
问题: logsumexp过度惩罚多个高SNR窃听者共存的场景，导致智能体学到"远离所有用户"的消极策略
问题2: 与安全定义冲突
物理意义: 无人机感知安全应关注"最危险的窃听者"(max)或"主要威胁"(top2)
Logsumexp问题: 将所有窃听者风险累加，夸大了实际威胁
3.2 GitHub代码中已存在的严重问题
问题A: Q权重被过度裁剪 (qvpo.py Line 163)
Python

q = torch.clamp(q, min=0.0, max=5.0)  # ❌ 致命裁剪
影响: Q值范围20-30被压缩到0-5，QVPO的重要性采样机制失效

问题B: 奖励被多次裁剪 (myenv.py)
Python

# 第一次裁剪
reward = np.clip(reward, -50.0, 80.0)
# 第二次裁剪
reward = np.clip(reward, -30.0, 50.0)
# 缩放
reward = reward * self.reward_scale  # 0.1
影响: 原始信号被严重压缩，智能体无法区分好坏

问题C: Q统计量更新过慢 (qvpo.py Line 155-160)
Python

self.running_q_std += self.alpha_std * (std_update - self.running_q_std)  # alpha_std=0.001
self.running_q_std = max(1.0, min(self.running_q_std, 15.0))
影响: Q值剧烈变化时，归一化统计量无法跟上，导致训练信号失真

问题D: 动作平滑惩罚过大
Bash

--action_smooth_coef 0.8  # 对3维动作空间过大
影响: 惩罚可达9.6，远超主奖励(~10 dB)，导致"不动"策略

四、下次训练修改指南
4.1 立即修改项 (Solu4核心修复)
修改1: 移除Q权重裁剪
文件: agent/qvpo.py
位置: Line 163 (train方法中)

Python

# 修改前
q = torch.clamp(q, min=0.0, max=5.0)

# 修改后 (注释掉或删除)
# q = torch.clamp(q, min=0.0, max=5.0)  # 移除裁剪，让Q值自由变化
原因: 当前Q值范围20-30，裁剪到0-5破坏了QVPO的核心机制

修改2: 优化奖励裁剪
文件: myenv.py
位置: Line ~220-230 (step方法中)

Python

# 修改前
reward = np.clip(reward, -50.0, 80.0)
info['reward_clip_1'] = float(reward)
# ... 能耗和动作平滑惩罚 ...
reward = np.clip(reward, -30.0, 50.0)
reward = reward * self.reward_scale

# 修改后
# 第一次裁剪：宽松范围防止极端值
reward = np.clip(reward, -20.0, 30.0)
info['reward_clip_1'] = float(reward)

# ... 能耗和动作平滑惩罚 ...

# 移除第二次裁剪
# reward = np.clip(reward, -30.0, 50.0)  # 删除

# 保留缩放但调大系数
reward = reward * self.reward_scale  # reward_scale将从0.1改为1.0
info['reward_final'] = float(reward)
修改3: 加速Q统计量更新
文件: agent/qvpo.py
位置: Line 155-160 (train方法中)

Python

# 修改前
std_update = min(std, self.running_q_std * 2.0)
self.running_q_std += self.alpha_std * (std_update - self.running_q_std)
self.running_q_std = max(1.0, min(self.running_q_std, 15.0))
self.running_q_mean += self.alpha_mean * (mean - self.running_q_mean)

# 修改后
# 自适应学习率：早期快速适应，后期稳定
adaptive_alpha_std = 0.01 if self.step < 50000 else 0.002
adaptive_alpha_mean = 0.01 if self.step < 50000 else 0.001

self.running_q_std += adaptive_alpha_std * (std - self.running_q_std)
# 扩大上限以适应Q值增长
self.running_q_std = np.clip(self.running_q_std, 1.0, 50.0)
self.running_q_mean += adaptive_alpha_mean * (mean - self.running_q_mean)
修改4: 降低Target Smoothing噪声
文件: agent/qvpo.py
位置: Line 123-125 (train方法中)

Python

# 修改前
target_noise = torch.randn_like(next_actions) * 0.1
target_noise = target_noise.clamp(-0.25, 0.25)

# 修改后
target_noise = torch.randn_like(next_actions) * 0.05
target_noise = target_noise.clamp(-0.1, 0.1)
原因: 扩散策略自带随机性，过大噪声导致Q值估计过于保守

4.2 窃听者聚合策略决策
建议: 保持eav_agg='top2'，不使用logsumexp

理由:

物理意义正确: top2反映"主要威胁"，符合安全定义
实验证据: 黑色曲线显示logsumexp导致性能下降25%
惩罚可控: top2的惩罚范围0-2，logsumexp达9
如果必须使用logsumexp，需满足:

Python

# 大幅降低kappa和系数
--eav_logsumexp_kappa 5.0  # 从0.5提升到5.0，减少放大效应
--eav_penalty_coef 0.5     # 从3.0降到0.5
五、Solu4 运行命令
Bash

python main.py \
  --env_name Env \
  --seed 42 \
  --num_steps 200000 \
  --batch_size 256 \
  --gamma 0.99 \
  --tau 0.005 \
  --diffusion_lr 0.0001 \
  --critic_lr 0.0001 \
  --n_timesteps 20 \
  --beta_schedule cosine \
  --entropy_alpha 0.02 \
  --train_sample 32 \
  --behavior_sample 8 \
  --target_sample 2 \
  --eval_sample 16 \
  --ac_grad_norm 1.0 \
  --q_transform qadv \
  --chosen 1 \
  --q_neg 0.001 \
  --cut 0.8 \
  --policy_freq 2 \
  --weighted \
  --aug \
  --normalize_state True \
  --action_smooth_coef 0.1 \
  --user_move_range 20.0 \
  --reward_scale 1.0 \
  --eav_agg top2 \
  --eav_threshold 10.0 \
  --eav_penalty_coef 3.0 \
  --eav_penalty_cap 20.0 \
  --comm_penalty softplus \
  --comm_threshold 10.0 \
  --comm_penalty_coef 1.5 \
  --comm_softplus_kappa 5.0 \
  --comm_penalty_cap_per_user 15.0 \
  --comm_penalty_cap_total 30.0 \
  --comm_penalty_avg_over_k True \
  --start_steps 10000 \
  --cuda cuda:0
关键参数变化对照
参数	Solu3(2)	Solu4	变化原因
reward_scale	0.1	1.0	移除过度压缩
action_smooth_coef	0.8	0.1	降低动作惩罚
eav_agg	logsumexp	top2	回退到物理合理方案
eav_logsumexp_kappa	0.5	(不使用)	-
六、预期改善效果
6.1 定量指标预测
指标	🔵 Solu3(2)	🎯 Solu4预期	提升
Eval Reward	36-40	42-46	+15%
Train Reward	~42	~48	+14%
Q Value	28-30	35-45	+40%
Q Std	~7	~5	-28% (更稳定)
Critic Loss	15-20	8-12	-47%
收敛步数	80k	60k	-25%
6.2 定性改善
✅ 训练稳定性: Q值曲线平滑上升，无剧烈震荡
✅ 策略质量: Eval reward超越Solu2基线10%
✅ 动作平滑性: action_smooth_penalty从1.2降至0.3
✅ 收敛速度: 60k步达到稳定(vs Solu2的100k)
七、调试监控要点
训练过程中重点观察以下TensorBoard指标:

7.1 必须正常的指标
q/current_q1_mean: 应在30-50范围逐步上升
q/running_q_std: 应在5-15范围稳定
loss/critic: 应在10以下收敛
7.2 警告信号
❌ 如果q/current_q1_mean < 20超过50k步 → Q值被压制，检查裁剪
❌ 如果reward/eval_mean < 30超过80k步 → 奖励设计问题
❌ 如果loss/critic > 20持续震荡 → 检查target_noise和tau
7.3 对比基准
将Solu4的绿色曲线与Solu3(2)的蓝色曲线对比:

期望: Solu4的eval_mean曲线在50k步后超越蓝色10-15%
期望: Solu4的Q值在30k步后突破35
八、后续优化方向
如果Solu4达到预期，可尝试:

自适应奖励缩放 (优先级高)

Python

# 在myenv.py中添加
class AdaptiveRewardScaler:
    def __init__(self, target=0.5):
        self.scale = 1.0
        self.ema = 0.0
    
    def update(self, reward):
        self.ema = 0.99 * self.ema + 0.01 * reward
        if self.ema < self.target * 0.8:
            self.scale *= 1.01
        elif self.ema > self.target * 1.2:
            self.scale *= 0.99
        return self.scale
状态归一化优化 (优先级中)

在random exploration阶段(前10k步)禁用统计量更新
防止噪声数据污染mean/var
熵正则化调整 (优先级低)

移除或修正随机动作的Q权重计算(qvpo.py Line 165-173)
九、给代码修改AI的检查清单
请按以下顺序确认修改:

 qvpo.py Line 163: 已注释q = torch.clamp(q, min=0.0, max=5.0)
 qvpo.py Line 155-160: 已添加自适应学习率adaptive_alpha_std和adaptive_alpha_mean
 qvpo.py Line 159: 已将running_q_std上限从15.0改为50.0
 qvpo.py Line 123-125: 已将target_noise从0.1/0.25改为0.05/0.1
 myenv.py Line ~225: 已修改第一次裁剪为np.clip(reward, -20.0, 30.0)
 myenv.py Line ~230: 已删除第二次裁剪np.clip(reward, -30.0, 50.0)
 运行命令: 已确认--reward_scale 1.0和--action_smooth_coef 0.1
 运行命令: 已确认--eav_agg top2(不使用logsumexp)
十、总结
核心发现
Solu3失败: logsumexp窃听者聚合放大惩罚4.5倍(2→9)，与物理安全定义冲突
GitHub代码缺陷: Q权重裁剪(0-5)破坏QVPO机制，奖励多次裁剪损失信息
Solu3(2)成功: 通过算法修正恢复性能，但仍受限于底层代码问题
Solu4改进策略
移除Q裁剪: 释放QVPO重要性采样能力
优化奖励处理: 单次宽松裁剪+放大缩放系数
加速统计更新: 自适应学习率快速跟踪Q值变化
降低动作惩罚: 从0.8→0.1，避免"不动"策略
回退窃听者聚合: 保持top2的物理合理性
预期成果
Solu4应在60k步收敛至eval_mean=42-46，超越所有前序版本，且训练曲线更稳定(Q_std<5, Critic_loss<10)。