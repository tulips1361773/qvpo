# QVPO 项目代码审查报告

## 一、项目概述

### 1.1 项目背景
- **任务**：使用强化学习优化无人机（UAV）轨迹，实现感知与通信一体化（ISAC）
- **场景**：单个无人机配备全向天线，同时执行：
  - 感知任务：探测目标并向合法雷达接收器发送感知信息
  - 通信任务：为 K 个通信用户提供服务
  - 安全约束：防止通信用户窃听感知信息

### 1.2 核心文件结构
| 文件 | 说明 |
|------|------|
| `main.py` | 训练主程序，参数配置 |
| `myenv.py` | 自定义 UAV-ISAC 环境（QVPO 使用） |
| `myenv2.py` | SAC 基线使用的环境 |
| `myenv3.py` | 基于 myenv.py 的统一 reward 处理版本 |
| `agent/qvpo.py` | QVPO 智能体实现 |
| `agent/diffusion.py` | Diffusion Policy |
| `agent/flow_matching.py` | Flow Matching Policy |
conda环境：qvpo
### 1.3 当前问题
- **训练结果**：稳定，reward 高于基线
- **关键问题**：感知信息泄露率高达 **70%**

---

## 二、代码审查发现

### 2.1 🔴 **核心问题：奖励函数设计导致窃听惩罚权重不足**

#### 问题位置
`myenv.py` 第 362-439 行 `_calculate_reward()` 函数

#### 问题分析

**问题 1：窃听惩罚仅在超过阈值时生效**

```python
# myenv.py:418-424
snr_gap2 = sensing_snr_eavesdropper - self.eav_threshold
eav_penalty = 0.0
if snr_gap2 > 0:  # ❌ 只有超过阈值才惩罚
    eav_penalty = min(self.eav_penalty_coef * snr_gap2, self.eav_penalty_cap)
    eav_penalty_clipped = np.clip(eav_penalty, 0.0, self.eav_penalty_clip_max)
    reward -= eav_penalty_clipped
```

**后果**：
- 当窃听者 SNR 接近但未超过阈值（如 9.9 dB vs 10 dB 阈值）时，**完全没有惩罚**
- 智能体没有动力主动远离窃听者，只要"刚好不超标"即可
- 这解释了为什么泄露率高达 70%——智能体学会了"擦边球"策略

**问题 2：感知 SNR 奖励与窃听惩罚的尺度不平衡**

```python
# 默认参数
eta_clip_max = 15.0          # 感知SNR裁剪上限
eav_penalty_clip_max = 8.0   # 窃听惩罚裁剪上限（实际默认5.0）
eav_penalty_coef = 5.0       # 窃听惩罚系数
```

**分析**：
- 感知 SNR 可以贡献最多 +15.0 的奖励
- 窃听惩罚最多只扣 -5.0（`eav_penalty_clip_max`）
- **比例为 3:1**，智能体自然倾向于追求高感知 SNR 而忽视窃听风险

**问题 3：物理模型导致的固有冲突**

```python
# myenv.py:475-492 感知SNR计算
def _calculate_sensing_snr_legal(self, uav_position, power_allocation):
    d_t = np.linalg.norm(uav_position - self.target_position)  # UAV到目标距离
    d_r = np.linalg.norm(self.target_position - self.radar_receiver_position)  # 目标到接收器距离
    P_r = (power_allocation * ...) / (d_t**2 * d_r**2)  # SNR ∝ 1/(d_t² * d_r²)

# myenv.py:494-516 窃听者SNR计算
def _calculate_sensing_snr_eavesdropper(self, uav_position, power_allocation):
    d_t = np.linalg.norm(uav_position - self.target_position)  # 同样的 d_t
    d_k_r = np.linalg.norm(self.target_position - self.user_positions[k])  # 目标到用户距离
    P_r_k = (power_allocation * ...) / (d_t**2 * d_k_r**2)  # SNR ∝ 1/(d_t² * d_k_r²)
```

**关键发现**：
- 合法感知 SNR 和窃听者 SNR **共享同一个 `d_t`**（UAV 到目标距离）
- 当 UAV 靠近目标以提高感知 SNR 时，**窃听者 SNR 也同步提高**
- 唯一的差异是 `d_r`（合法接收器距离）vs `d_k_r`（用户距离）
- 如果用户恰好比合法接收器更靠近目标，窃听者 SNR 可能比合法 SNR 还高！

---

### 2.2 🟡 **次要问题：状态空间缺少关键信息**

#### 问题位置
`myenv.py` 第 250-256 行 `_get_obs()` 函数

```python
def _get_obs(self):
    obs = np.concatenate([
        self.uav_position[:2],           # UAV位置 (2)
        self.user_positions[:, :2].flatten(),  # 用户位置 (2*K=6)
        self.prev_action                  # 上一步动作 (3)
    ])
    return obs
```

**缺失信息**：
1. **目标位置** (`self.target_position`) - 智能体不知道要感知的目标在哪
2. **合法接收器位置** (`self.radar_receiver_position`) - 智能体不知道要向哪里发送感知信息
3. **当前功率分配** - 智能体不知道当前的发射功率
4. **剩余能量** (`self.total_energy / self.E_tot`) - 智能体不知道能量预算

**后果**：
- 智能体无法做出最优决策，因为缺少关键环境信息
- 但这不是泄露率高的直接原因（目标和接收器位置是固定的，智能体可以隐式学习）

---

### 2.3 🟡 **潜在问题：通信用户移动导致的不稳定性**

#### 问题位置
`myenv.py` 第 518-542 行 `_update_user_positions()` 函数

```python
def _update_user_positions(self):
    for k in range(self.K):
        move_distance = self.np_random.uniform(0, self.user_move_range)  # 默认20m
        move_angle = self.np_random.uniform(-np.pi, np.pi)
        # ... 更新位置
```

**分析**：
- 用户每步最多移动 20m，50 步后可能移动 1000m
- 用户位置的剧烈变化导致窃听者 SNR 波动大
- 智能体难以学习稳定的防窃听策略

---

### 2.4 🟢 **代码质量问题（不影响功能）**

1. **未使用的变量**：`myenv.py` 中 `self.t1`, `self.rresult`, `self.episode_rewards` 被注释但未删除
2. **硬编码值**：目标位置 `[100, 100, 50]` 和接收器位置 `[0, 0, 0]` 硬编码，不可配置
3. **环境注册路径错误**：`myenv.py:9` 中 `entry_point='doubleobservation:UAVISACEnvironment'` 应为 `'myenv:UAVISACEnvironment'`

---

## 三、根本原因分析

### 3.1 泄露率 70% 的根本原因

```
┌─────────────────────────────────────────────────────────────────┐
│                    奖励函数设计缺陷                               │
├─────────────────────────────────────────────────────────────────┤
│  1. 窃听惩罚采用"阈值触发"机制，低于阈值时完全无惩罚              │
│  2. 感知奖励(+15) vs 窃听惩罚(-5) 比例失衡                       │
│  3. 物理模型决定了感知SNR和窃听SNR正相关                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    智能体学习到的策略                             │
├─────────────────────────────────────────────────────────────────┤
│  "最大化感知SNR，只要窃听者SNR不超过阈值太多即可"                 │
│  → 智能体学会了"擦边球"策略                                      │
│  → 窃听者SNR经常接近或略超阈值                                   │
│  → 泄露率高达70%                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、改进建议

### 4.1 🔴 **高优先级：重新设计窃听惩罚机制**

#### 方案 A：使用安全裕度（Safety Margin）

```python
# 不是等到超过阈值才惩罚，而是鼓励远离阈值
safety_margin = self.eav_threshold - sensing_snr_eavesdropper
if safety_margin < 5.0:  # 当裕度小于5dB时开始惩罚
    eav_penalty = self.eav_penalty_coef * (5.0 - safety_margin)
```

#### 方案 B：使用软惩罚函数（类似通信惩罚的 softplus）

```python
# 使用softplus使惩罚平滑过渡
snr_gap = sensing_snr_eavesdropper - self.eav_threshold
eav_penalty = self.eav_penalty_coef * np.logaddexp(0, self.eav_kappa * snr_gap) / self.eav_kappa
```

#### 方案 C：使用保密容量（Secrecy Capacity）作为奖励

```python
# 直接优化保密容量 = log(1 + SNR_legal) - log(1 + SNR_eavesdropper)
secrecy_capacity = np.log2(1 + 10**(eta_0/10)) - np.log2(1 + 10**(sensing_snr_eavesdropper/10))
reward = secrecy_capacity * self.secrecy_coef
```

### 4.2 🔴 **高优先级：调整奖励尺度平衡**

```python
# 建议参数调整
eav_penalty_coef = 10.0      # 从5.0提高到10.0
eav_penalty_clip_max = 15.0  # 从5.0提高到15.0，与感知SNR上限一致
eta_clip_max = 10.0          # 从15.0降低到10.0
```

### 4.3 🟡 **中优先级：增强状态空间**

```python
def _get_obs(self):
    obs = np.concatenate([
        self.uav_position[:2],
        self.user_positions[:, :2].flatten(),
        self.prev_action,
        # 新增：
        self.target_position[:2],           # 目标位置
        self.radar_receiver_position[:2],   # 接收器位置
        [self.total_energy / self.E_tot],   # 剩余能量比例
    ])
    return obs
```

### 4.4 🟡 **中优先级：添加窃听相关指标到状态**

```python
# 让智能体直接观察到当前的窃听风险
current_eav_snr = max(self._calculate_sensing_snr_eavesdropper(...))
eav_margin = self.eav_threshold - current_eav_snr  # 安全裕度
obs = np.concatenate([..., [eav_margin]])
```

---

## 五、验证建议

### 5.1 添加泄露率监控指标

在 `main.py` 中添加：

```python
# 在 step 后记录
if info.get('snr_gap2', 0) > 0:
    leakage_count += 1
total_steps += 1

# 定期输出
if steps % 10000 == 0:
    leakage_rate = leakage_count / total_steps
    writer.add_scalar('security/leakage_rate', leakage_rate, steps)
```

### 5.2 消融实验建议

1. **实验1**：仅提高 `eav_penalty_coef` 从 5.0 到 15.0
2. **实验2**：使用 softplus 替代阈值触发机制
3. **实验3**：使用保密容量作为奖励
4. **实验4**：增强状态空间

---

## 六、总结

| 问题类型 | 问题描述 | 严重程度 | 建议优先级 |
|----------|----------|----------|------------|
| 奖励设计 | 窃听惩罚阈值触发机制 | 🔴 高 | 立即修复 |
| 奖励设计 | 感知奖励与窃听惩罚尺度失衡 | 🔴 高 | 立即修复 |
| 状态空间 | 缺少目标/接收器位置信息 | 🟡 中 | 建议修复 |
| 环境设计 | 用户移动范围过大 | 🟡 中 | 可选优化 |
| 代码质量 | 未使用变量、硬编码 | 🟢 低 | 可选清理 |

**核心结论**：泄露率高的根本原因是**奖励函数设计**，智能体学会了"擦边球"策略。建议优先修改窃听惩罚机制，使用软惩罚或安全裕度方法。
