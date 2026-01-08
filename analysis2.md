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

下面是供参考的方案，没有结合具体代码！！请你结合代码分析，再考虑是否使用下面的方案：
稳定 Lagrange 乘子 (eta_0):
"请检查 eta_0 的更新逻辑。它目前的震荡幅度太大（0-18）。可以eta 设定一个上限（例如 clip at 5.0），或者降低 eta 的学习率（alpha learning rate）。"
强化动作平滑约束:
"观察到 action_smooth_penalty 随着训练不降反升。可以改 Reward 函数，显著增加 Action Smoothness 的惩罚权重。或者，在扩散模型的采样阶段（Inference time）加入后处理（如移动平均滤波）来强制平滑。"
调整扩散模型的 Actor Loss:
"Actor Loss 在后期呈现上升趋势。请检查 Actor 的 Loss 计算。如果是 BC Loss + lambda * Q_Loss 的形式，尝试减小 Q-Loss 的权重 lambda，防止 Critic 的高方差误导 Actor 的去噪过程。"
解决 Critic 方差问题:
"Q 值的标准差 q/running_q_std 持续升高。建议在计算 Target Q 时，增加 Target Action 的噪声平滑度（Target Policy Smoothing），或者检查 Reward Function 中是否存在某些极端大的瞬时奖励值。"

补充信息包 
Env Info: UAV Control, Continuous Action.
Observation: The oscillation of eta_0 correlates with the instability of the critic loss.
Specific Concern: The action_smooth_penalty is rising, indicating the diffusion model is generating jerky trajectories.

---

# 代码深度分析与修改建议 (AI 分析)

## 问题根源分析

### 1. `eta_0` 震荡 (0-18 dB) - **不需要修改**

**代码位置**: `myenv.py:457-474`

`eta_0` 是感知 SNR（dB），由雷达方程计算：`SNR ∝ 1/(d_t² × d_r²)`

- 这是**环境物理特性**，不是 Lagrange 乘子
- SNR 随 UAV 位置变化是正常的，反映了任务目标
- **不应该 clip `eta_0`**，否则会破坏奖励信号的物理意义
- 已有的 `reward_scale=0.1` 和奖励裁剪已经在控制其影响

### 2. `action_smooth_penalty` 上升 - **需要增大权重**

**代码位置**: `myenv.py:303-307`

```python
action_smooth_penalty = self.action_smooth_coef * np.sum(action_diff ** 2)
```

**根因**：
- 当前 `action_smooth_coef=0.3`，相对于 `eta_0`（0-18）太小
- 扩散模型通过 Q-guidance 选择动作（`diffusion.py:158-163`），选 Q 值最高的
- 如果 Critic 对"抖动动作"给出高 Q 值，Agent 会倾向于抖动

**建议**：增大 `action_smooth_coef` 到 **0.8-1.0**

### 3. Actor Loss 后期上升 - **需要限制 q 权重**

**代码位置**: `qvpo.py:213-214`, `diffusion.py:226-240`

Actor Loss 计算方式：
```python
# qvpo.py:230
actor_loss = self.actor.loss(best_actions, states, weights=q)

# diffusion.py:236
loss = self.loss_fn(x_recon, noise, weights)  # Q-weighted denoising loss
```

**根因**：
- 这是 **Q-weighted BC loss**，不是 BC + λ*Q_loss
- 当 `running_q_std` 上升时，q 权重分布变得极端
- 使用 `qadv` 转换（`q_transform.py:43-50`）时，`adv = q - v`，方差大时 adv 分布极端

**建议**：对 q 权重进行 clip，例如 `q = torch.clamp(q, min=0.0, max=5.0)`

### 4. Critic 方差 (`running_q_std`) 持续上升 - **需要添加 Target Policy Smoothing**

**代码位置**: `qvpo.py:166-172`

```python
next_actions = self.actor_target(next_states, eval=False, q_func=self.critic_target)
target_q1, target_q2 = self.critic_target(next_states, next_actions)
target_q = torch.min(target_q1, target_q2)
```

**根因**：
- 当前代码**没有 Target Policy Smoothing**
- 扩散模型生成的动作虽有随机性，但总是选 Q 最高的（结构化偏差）
- 导致 Critic 对特定"高 Q"动作过拟合

**建议**：添加 TD3 风格的 target action 噪声

### 5. 评估性能崩塌 - **综合结果**

这是上述问题的综合表现：Critic 方差大 → Q-guidance 不稳定 → 动作抖动 → 评估性能差

---

## 具体代码修改

### 修改 1: `myenv.py` - 增大动作平滑惩罚

**位置**: 第 129 行

```python
# 修改前
action_smooth_coef: float = 0.3

# 修改后
action_smooth_coef: float = 0.8  # 增大权重，使平滑惩罚更显著
```

### 修改 2: `agent/qvpo.py` - 添加 Target Policy Smoothing

**位置**: 第 166-167 行之间插入

```python
next_actions = self.actor_target(next_states, eval=False, q_func=self.critic_target)
# === 新增：Target Policy Smoothing ===
target_noise = torch.randn_like(next_actions) * 0.1
target_noise = target_noise.clamp(-0.25, 0.25)
next_actions = (next_actions + target_noise).clamp(-1.0, 1.0)
# === 新增结束 ===
target_q1, target_q2 = self.critic_target(next_states, next_actions)
```

### 修改 3: `agent/qvpo.py` - 限制 q 权重范围

**位置**: 第 214 行之后插入

```python
q = eval(self.q_transform)(q, q_neg=self.q_neg, cut=self.cut, ...)
# === 新增：限制 q 权重范围 ===
q = torch.clamp(q, min=0.0, max=5.0)
# === 新增结束 ===
```

### 修改 4: `agent/qvpo.py` - 限制 running_q_std 更新

**位置**: 第 210 行

```python
# 修改前
self.running_q_std += self.alpha_std * (std - self.running_q_std)

# 修改后
std_update = min(std, self.running_q_std * 2.0)  # 限制单次更新幅度
self.running_q_std += self.alpha_std * (std_update - self.running_q_std)
self.running_q_std = max(1.0, min(self.running_q_std, 15.0))  # clip 到 [1, 15]
```

---

## 参数修改汇总

| 参数/代码 | 原值 | 建议值 | 原因 |
|-----------|------|--------|------|
| `action_smooth_coef` | 0.3 | 0.8 | 增强平滑约束，抑制抖动 |
| `running_q_std` 范围 | 无限制 | [1, 15] | 防止 Q 值方差爆炸 |
| q 权重范围 | 无限制 | [0, 5] | 防止极端权重误导去噪 |
| Target Policy Smoothing | 无 | noise_std=0.1, clip=0.25 | 稳定 Critic 训练 |

---

## 推荐训练命令

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
  --action_smooth_coef 0.8 \
  --user_move_range 20.0 \
  --reward_scale 0.1 \
  --comm_penalty_coef 0.5 \
  --start_steps 10000 \
  --cuda cuda:1
```

---

## 预期效果

1. **`action_smooth_penalty` 应下降**：更强的惩罚迫使 Agent 生成平滑轨迹
2. **`running_q_std` 应稳定**：不再无限上升，保持在 [1, 15] 范围
3. **Actor Loss 应更稳定**：q 权重 clip 防止极端值干扰去噪
4. **评估性能应更稳定**：Target Policy Smoothing 减少 Critic 过拟合