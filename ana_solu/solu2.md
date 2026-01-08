# 基于ana2做的修改

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

## 7. 实验结果
policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_01_07_22_41_29_0