# 结果分析报告：基于扩散模型的 UAV 控制策略 (Run 2)
## 核心诊断摘要
相较于之前的实验，模型在边界约束 (boundary_penalty) 和通信约束 (comm_penalty) 上表现出了更好的依从性。然而，当前实验面临三个新的严峻挑战：
评估性能崩塌 (Evaluation Collapse): reward/eval_mean 在 70k 步后出现了显著的性能骤降，表明发生了过拟合或策略退化。
动作平滑度恶化: action_smooth_penalty 随着训练进行反而上升，说明 Agent 为了追求高奖励，正在采取越来越激进、抖动越来越大的动作。
Critic 不确定性增加: Q 值的均值虽然稳定，但方差 (running_q_std) 持续上升，这意味着 Critic 对状态价值的判断越来越模糊，这对引导扩散策略（Diffusion Guidance）非常不利。
## 详细图表解读

A. 奖励与惩罚 (Reward & Penalties)
总奖励 (reward/train_ma100): 曲线在 60k 步达到峰值（~55），随后进入震荡。并没有实现持续稳健的增长。
评估奖励 (reward/eval_mean): 关键故障点。 在 60k-70k 步达到最高点后，迅速下跌至 30-40 区间。这通常意味着训练分布（Training Distribution）和评估环境之间出现了 mismatch，或者策略过度优化了某些特定的“高收益但高风险”动作。
平滑度惩罚 (action_smooth_penalty): 异常趋势。 该曲线呈现上升趋势（从 0.2 升至 0.7）。理想情况下，这应该下降或保持低位。解读： 扩散模型生成的动作通常带有高频噪声，如果 Smooth Penalty 权重不够大，或者 Actor Loss 中的重建项（Reconstruction Loss）未能有效平滑动作，Agent 就会倾向于“高频颤振”来微调位置，导致惩罚上升。
其他惩罚:
boundary_penalty & comm_penalty: 偶有尖峰，但大部分时间为 0，控制良好。
energy_penalty: 依然恒为 0（需确认是否由于无人机未达到消耗能量的阈值，或计算逻辑问题）。
eta_0 (Lagrange Multiplier / Entropy Alpha): 在 0 到 18 之间剧烈全幅震荡。这极不稳定，导致 Loss Function 的权重不断变化，阻碍了收敛。

B. Q值分析 (Critic Dynamics)
Q值均值 (q/current_q1_mean): 在 40k 步达到约 40 的峰值，随后缓慢回落。这种回落（Underestimation bias）在 TD3 风格中是由于 Target Policy Smoothing 引起的，但在扩散模型中，可能意味着 Critic 无法准确评估扩散生成的多样化动作。
Q值标准差 (q/running_q_std): 警示信号。 标准差从 4 一路飙升到 9 以上。这意味着 Critic 对同一状态下的不同动作给出的评分差异越来越大。对于扩散策略，如果 Critic 梯度（Guidance）方差过大，去噪过程（Denoising Process）就会被带偏，导致生成离谱的动作。

C. 损失函数 (Losses)
Actor Loss (loss/actor):
数值极小 (0.01 - 0.015)。由于您使用的是扩散模型，这通常代表 去噪误差 (Denoising MSE)。
趋势：先降后升。这表明策略正在“变差”。原因可能是 Critic 的梯度引导（Q-Guidance）与模仿学习/行为克隆的目标（BC Term）发生了冲突，导致 Actor 既无法完美重建数据，也无法完美最大化 Q 值。
Critic Loss (loss/critic): 震荡极其剧烈。结合 eta_0 的震荡来看，这表明 Reward Scale 或 Penalty 系数可能过大，导致 TD Error 极不稳定。

## 3. 针对“扩散模型”的特定假设与推断
由于使用了扩散策略 (Diffusion Policy)，当前的图表特征暗示了以下机制性问题：
Guidance Scale 失调: 如果您在采样时使用了 Classifier-Free Guidance (CFG) 或直接使用 Q-Gradient Guidance，当前的 Q 值方差 (q/std) 过大会导致梯度爆炸，从而生成极其抖动（不平滑）的轨迹，解释了 action_smooth_penalty 的上升。
采样步数与噪声: 扩散模型的训练是一个去噪过程。loss/actor 上升可能意味着模型在“高噪声”阶段（High Noise Level）的学习还算可以，但在“低噪声”精细控制阶段失败了，导致动作微操很差。
下一步代码修改建议 (Request to AI)

请根据上述分析，向 AI 提出以下具体的修改请求：
稳定 Lagrange 乘子 (eta_0):
"请检查 eta_0 的更新逻辑。它目前的震荡幅度太大（0-18）。建议给 eta 设定一个上限（例如 clip at 5.0），或者降低 eta 的学习率（alpha learning rate）。"
强化动作平滑约束:
"观察到 action_smooth_penalty 随着训练不降反升。请修改 Reward 函数，显著增加 Action Smoothness 的惩罚权重。或者，在扩散模型的采样阶段（Inference time）加入后处理（如移动平均滤波）来强制平滑。"
调整扩散模型的 Actor Loss:
"Actor Loss 在后期呈现上升趋势。请检查 Actor 的 Loss 计算。如果是 BC Loss + lambda * Q_Loss 的形式，尝试减小 Q-Loss 的权重 lambda，防止 Critic 的高方差误导 Actor 的去噪过程。"
解决 Critic 方差问题:
"Q 值的标准差 q/running_q_std 持续升高。建议在计算 Target Q 时，增加 Target Action 的噪声平滑度（Target Policy Smoothing），或者检查 Reward Function 中是否存在某些极端大的瞬时奖励值。"
补充信息包 (建议连同报告一起发给 AI):
Env Info: UAV Control, Continuous Action.
Observation: The oscillation of eta_0 correlates with the instability of the critic loss.
Specific Concern: The action_smooth_penalty is rising, indicating the diffusion model is generating jerky trajectories.

---

# 已实施的代码修改 (Run 3)

## 问题诊断总结

| 问题 | 根因 | 解决方案 |
|------|------|----------|
| `action_smooth_penalty` 上升 | 动作平滑惩罚权重不足 | 增大 `action_smooth_coef`: 0.3 → 1.0 |
| `running_q_std` 持续升高 | Q值方差更新无限制 | 限制 std 更新幅度和范围 [1, 20] |
| Actor Loss 后期上升 | Q-guidance 权重过大导致冲突 | 对 q 权重 clip 到 [0, 5] |
| Critic 不稳定 | 缺少 Target Policy Smoothing | 添加 TD3 风格的 target action 噪声 |

---

## 1. myenv.py 修改

### 1.1 增大动作平滑惩罚权重（第129行）

**修改前：**
```python
action_smooth_coef: float = 0.3, user_move_range: float = 20.0,
```

**修改后：**
```python
action_smooth_coef: float = 1.0, user_move_range: float = 20.0,  # 增大动作平滑惩罚权重 0.3→1.0
```

---

## 2. agent/qvpo.py 修改

### 2.1 添加 Target Policy Smoothing（第166-170行）

**修改前：**
```python
next_actions = self.actor_target(next_states, eval=False, q_func=self.critic_target)
target_q1, target_q2 = self.critic_target(next_states, next_actions)
target_q = torch.min(target_q1, target_q2)
```

**修改后：**
```python
next_actions = self.actor_target(next_states, eval=False, q_func=self.critic_target)
# Target Policy Smoothing: 添加噪声平滑 target action，稳定 Critic 训练
target_noise = torch.randn_like(next_actions) * 0.1  # 噪声标准差 0.1
target_noise = target_noise.clamp(-0.2, 0.2)  # clip 噪声范围
next_actions = (next_actions + target_noise).clamp(-1.0, 1.0)
target_q1, target_q2 = self.critic_target(next_states, next_actions)
target_q = torch.min(target_q1, target_q2)
```

### 2.2 限制 running_q_std 更新幅度和范围（第210-213行）

**修改前：**
```python
self.running_q_std += self.alpha_std * (std - self.running_q_std)
self.running_q_mean += self.alpha_mean * (mean - self.running_q_mean)
```

**修改后：**
```python
# 限制 running_q_std 的更新幅度，防止方差爆炸
std_clipped = min(std, self.running_q_std * 1.5)  # 限制单次更新不超过1.5倍
self.running_q_std += self.alpha_std * (std_clipped - self.running_q_std)
self.running_q_std = max(1.0, min(self.running_q_std, 20.0))  # clip到[1, 20]
self.running_q_mean += self.alpha_mean * (mean - self.running_q_mean)
```

### 2.3 对 q 权重进行 clip（第219行）

**新增代码：**
```python
# 对 q 值进行标准化后再 clip，防止极端值影响扩散模型训练
q = eval(self.q_transform)(q, ...)
q = torch.clamp(q, min=0.0, max=5.0)  # 限制 q 权重范围，防止极端值
```

---

## 3. main.py 修改

### 3.1 更新默认参数

| 参数 | 原值 | 新值 | 原因 |
|------|------|------|------|
| `--action_smooth_coef` | 0.3 | 1.0 | 增强动作平滑约束 |
| `--alpha_std` | 0.001 | 0.0005 | 降低 std 更新率使其更稳定 |

---

## 4. 参数修改汇总表

| 参数/代码位置 | 原值 | 新值 | 修改原因 |
|---------------|------|------|----------|
| `action_smooth_coef` | 0.3 | 1.0 | 抑制扩散模型生成的抖动动作 |
| `alpha_std` | 0.001 | 0.0005 | 降低 running_q_std 更新速度 |
| `running_q_std` 范围 | 无限制 | [1, 20] | 防止 Q 值方差爆炸 |
| q 权重范围 | 无限制 | [0, 5] | 防止极端 q 值误导 Actor |
| Target Policy Smoothing | 无 | noise_std=0.1, clip=0.2 | 稳定 Critic 训练 |

---

## 5. 推荐训练命令

```bash
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
  --action_smooth_coef 1.0 \
  --alpha_std 0.0005 \
  --user_move_range 20.0 \
  --reward_scale 0.1 \
  --comm_penalty_coef 0.5 \
  --comm_softplus_kappa 1.0 \
  --eav_penalty_coef 1.0 \
  --start_steps 10000 \
  --cuda cuda:1
```

**关键调整说明：**
- `--action_smooth_coef 1.0`：增强动作平滑约束，抑制 Bang-Bang 控制
- `--alpha_std 0.0005`：降低 Q 值方差更新速度，提高稳定性
- Target Policy Smoothing 已在代码中硬编码启用

---

## 6. 预期效果

1. **`action_smooth_penalty` 应该下降或保持低位**：更强的平滑惩罚会迫使 Agent 生成更平滑的轨迹
2. **`running_q_std` 应该稳定在 [1, 20] 范围内**：不再无限上升
3. **Actor Loss 应该更稳定**：q 权重 clip 防止极端值干扰去噪过程
4. **评估性能应该更稳定**：Target Policy Smoothing 减少 Critic 过拟合