# UAV-ISAC 强化学习项目：奖励与泄露率计算方法分析

## 1. 概述

本文档详细分析 QVPO 项目中训练/测试时的**奖励计算方法**和**泄露率计算方法**，并评估其合理性。

---

## 2. 奖励计算方法

### 2.1 奖励函数总体结构

奖励函数定义在 `myenv3.py` 的 `_calculate_reward()` 方法中，总体公式为：

```
reward = R_sense - R_eav - R_comm - energy_penalty - action_smooth_penalty
```

其中：
- **R_sense**: 感知收益（正向奖励）
- **R_eav**: 窃听/安全惩罚（核心优先项）
- **R_comm**: 通信惩罚（次要项）
- **energy_penalty**: 能耗惩罚
- **action_smooth_penalty**: 动作平滑惩罚

最终奖励会乘以 `reward_scale` 进行缩放。

---

### 2.2 各分项计算方法

#### 2.2.1 感知收益 (R_sense)

```python
eta_0 = self._calculate_sensing_snr_legal(uav_position, power_allocation)
R_sense = min(eta_0, 30.0)  # 软截断，防止正向奖励爆炸
```

- **计算方式**: 计算合法感知目标的信噪比 (SNR)，单位为 dB
- **截断处理**: 将感知 SNR 限制在 30 dB 以内，防止 Agent 追求极端高 SNR 而忽视其他目标

#### 2.2.2 窃听/安全惩罚 (R_eav) - 核心优先项

```python
# 计算所有窃听者的 SNR
eavesdropper_snr_list = self._calculate_sensing_snr_eavesdropper(uav_position, power_allocation)
max_eav_snr = np.max(eav_snrs)

# 计算 SNR Gap
snr_gap_eav = max_eav_snr - self.eav_threshold  # 默认阈值 10 dB

# Softplus 平滑惩罚（无硬截断）
kappa = 2.0
eav_penalty_raw = np.logaddexp(0.0, kappa * snr_gap_eav) / kappa

# 乘以惩罚系数
R_eav = eav_penalty_raw * self.eav_penalty_coef
```

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `eav_threshold` | 10.0 dB | 窃听 SNR 安全阈值 |
| `eav_penalty_coef` | 2.0 | 惩罚系数 |
| `kappa` | 2.0 | Softplus 陡峭度 |

**Softplus 函数特性**:
- 当 `snr_gap_eav < 0`（安全）时，惩罚趋近于 0
- 当 `snr_gap_eav > 0`（不安全）时，惩罚近似线性增长
- 平滑过渡，避免梯度突变

#### 2.2.3 通信惩罚 (R_comm) - 次要项

```python
for k in range(self.K):
    distance = np.linalg.norm(uav_position - self.user_positions[k])
    snr = self._calculate_communication_snr(distance, power_allocation)
    snr_gap = self.comm_threshold - snr  # 通信 SNR 低于阈值时产生惩罚
    
    # Softplus 平滑惩罚
    p_smooth = np.logaddexp(0.0, self.comm_softplus_kappa * snr_gap) / self.comm_softplus_kappa
    comm_penalties.append(p_smooth)

avg_comm_penalty = np.mean(comm_penalties)
avg_comm_penalty_clipped = min(avg_comm_penalty, self.comm_penalty_clip_total)
R_comm = avg_comm_penalty_clipped * self.comm_penalty_coef
```

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `comm_threshold` | 10.0 dB | 通信 SNR 最低要求 |
| `comm_penalty_coef` | 0.5 | 惩罚系数（较小，优先级低于安全） |
| `comm_softplus_kappa` | 2.0 | Softplus 陡峭度 |
| `comm_penalty_clip_total` | 50.0 | 总惩罚截断值 |

#### 2.2.4 能耗惩罚

```python
horizontal_speed = abs(distance) / 4.0
energy_t = calc_energy(horizontal_speed, self.delta_t)
self.total_energy += energy_t

if self.total_energy > self.E_tot:
    raw_reward -= self.energy_penalty  # 默认 5.0
```

- 当累计能耗超过总能量预算 `E_tot` (25000 J) 时，施加固定惩罚

#### 2.2.5 动作平滑惩罚

```python
action_diff = action - self.prev_action
action_smooth_penalty = self.action_smooth_coef * np.sum(action_diff ** 2)
raw_reward -= action_smooth_penalty
```

- 惩罚动作的剧烈变化，鼓励平滑轨迹
- 默认系数 `action_smooth_coef = 0.8`

#### 2.2.6 边界惩罚

```python
if new_uav_position[0] < self.X_min or new_uav_position[0] > self.X_max or \
   new_uav_position[1] < self.Y_min or new_uav_position[1] > self.Y_max:
    raw_reward = -100.0  # 越界重罚
```

- UAV 超出边界 (±400m) 时，直接返回 -100 的重罚

---

### 2.3 训练时 vs 测试时的奖励计算

| 阶段 | 奖励计算 | 区别 |
|------|----------|------|
| **训练时** | 完整奖励函数 | 用于更新策略网络 |
| **测试时** | 完整奖励函数 | 仅用于评估，不更新网络 |

**关键差异**:
- 训练时：状态归一化器持续更新统计量
- 测试时：状态归一化器冻结（`set_training(False)`）

```python
# 测试时冻结归一化器
if hasattr(env, 'state_normalizer'):
    env.state_normalizer.set_training(False)
```

---

## 3. 泄露率计算方法

### 3.1 定义

**泄露率 (Leakage Rate)** = 窃听 SNR 超过安全阈值的用户数 / 总用户数

```python
leakage_rate = leakage_count / total_users
```

### 3.2 单步泄露统计

在 `_calculate_reward()` 中计算：

```python
# 统计泄漏用户数（SNR 超过阈值的用户）
leakage_count = int(np.sum(eav_snrs > self.eav_threshold))
total_users = self.K  # 总用户数

info = {
    'leakage_count': leakage_count,
    'total_users': total_users,
}
```

### 3.3 训练时泄露率累计

在 `main.py` 的训练循环中：

```python
# 初始化
train_leakage_count = 0
train_total_users = 0

# 每步累计
train_leakage_count += info.get('leakage_count', 0)
train_total_users += info.get('total_users', 0)

# 计算训练泄露率
train_leakage_rate = train_leakage_count / train_total_users
```

### 3.4 测试时泄露率计算

在 `evaluate()` 函数中：

```python
eval_leakage_count = 0
eval_total_users = 0

for i in range(episodes):  # 10 个评估 episode
    while not (done or truncated):
        # 累计泄露率统计
        eval_leakage_count += info.get('leakage_count', 0)
        eval_total_users += info.get('total_users', 0)

# 计算评估泄露率
eval_leakage_rate = eval_leakage_count / eval_total_users if eval_total_users > 0 else 0.0
```

### 3.5 泄露率记录

通过 TensorBoard 记录：

```python
# 单步泄露率
writer.add_scalar('security/step_leakage_rate', step_leakage_rate, steps)

# 训练累计泄露率
writer.add_scalar('security/train_leakage_rate', train_leakage_rate, steps)

# 评估泄露率
writer.add_scalar('security/eval_leakage_rate', eval_leakage_rate, steps)
```

---

## 4. 合理性评估

### 4.1 优点

| 方面 | 评价 |
|------|------|
| **多目标平衡** | ✅ 奖励函数同时考虑感知、安全、通信、能耗、动作平滑，符合 UAV-ISAC 场景需求 |
| **安全优先** | ✅ `eav_penalty_coef` 可调节，使安全惩罚权重高于通信惩罚 |
| **Softplus 平滑** | ✅ 避免硬阈值导致的梯度不连续，有利于策略学习 |
| **泄露率定义** | ✅ 基于用户级别的统计，直观反映安全性能 |
| **分项记录** | ✅ 详细记录各奖励分项，便于调试和分析 |

### 4.2 潜在问题与改进建议

#### 问题 1: 感知收益截断可能限制性能

```python
R_sense = min(eta_0, 30.0)
```

**问题**: 硬截断可能导致 Agent 在达到 30 dB 后缺乏进一步优化动力

**建议**: 考虑使用 Softplus 或 Tanh 进行软截断：
```python
R_sense = 30.0 * np.tanh(eta_0 / 30.0)
```

#### 问题 2: 窃听惩罚使用 Max 策略

```python
max_eav_snr = np.max(eav_snrs)
```

**问题**: 只关注最大窃听 SNR，可能忽略多个用户同时泄露的情况

**建议**: 考虑使用 Sum 或加权策略：
```python
# 方案 A: 累加所有超标用户的惩罚
eav_penalty_raw = np.sum(np.maximum(0, eav_snrs - self.eav_threshold))

# 方案 B: 使用 LogSumExp 平滑最大值
eav_penalty_raw = np.log(np.sum(np.exp(eav_snrs))) - self.eav_threshold
```

#### 问题 3: 泄露率统计粒度

**当前**: 基于用户数统计
**建议**: 可增加基于时间步的泄露率统计，更细粒度地反映安全性能

```python
# 时间步级别泄露率
step_has_leakage = 1 if leakage_count > 0 else 0
```

#### 问题 4: 能耗惩罚为阶跃函数

```python
if self.total_energy > self.E_tot:
    raw_reward -= self.energy_penalty
```

**问题**: 阶跃惩罚可能导致策略在能量边界处行为不稳定

**建议**: 使用渐进式惩罚：
```python
energy_ratio = self.total_energy / self.E_tot
if energy_ratio > 0.8:
    energy_penalty = self.energy_penalty * (energy_ratio - 0.8) / 0.2
```

### 4.3 总体评价

| 维度 | 评分 | 说明 |
|------|------|------|
| **功能完整性** | ⭐⭐⭐⭐⭐ | 覆盖所有关键目标 |
| **数学合理性** | ⭐⭐⭐⭐ | Softplus 设计合理，但部分硬截断可优化 |
| **可调节性** | ⭐⭐⭐⭐⭐ | 参数化设计，便于超参调优 |
| **可解释性** | ⭐⭐⭐⭐⭐ | 分项记录，便于分析 |
| **训练稳定性** | ⭐⭐⭐⭐ | 平滑惩罚有助于稳定训练 |

**总结**: 当前奖励函数设计**总体合理**，符合 UAV-ISAC 场景的多目标优化需求。建议根据实际训练效果，针对上述潜在问题进行微调。

---

## 5. 附录：关键参数汇总

| 参数 | 默认值 | 所属模块 | 说明 |
|------|--------|----------|------|
| `eav_threshold` | 10.0 | 安全 | 窃听 SNR 阈值 (dB) |
| `eav_penalty_coef` | 2.0 | 安全 | 窃听惩罚系数 |
| `comm_threshold` | 10.0 | 通信 | 通信 SNR 阈值 (dB) |
| `comm_penalty_coef` | 0.5 | 通信 | 通信惩罚系数 |
| `comm_softplus_kappa` | 2.0 | 通信 | Softplus 陡峭度 |
| `action_smooth_coef` | 0.8 | 动作 | 动作平滑系数 |
| `energy_penalty` | 5.0 | 能耗 | 能耗超标惩罚 |
| `reward_scale` | 0.1 | 全局 | 奖励缩放因子 |
| `E_tot` | 25000 | 能耗 | 总能量预算 (J) |
