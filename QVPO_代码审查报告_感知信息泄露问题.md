# QVPO项目代码审查报告：感知信息泄露问题分析

**审查日期**: 2026-03-06  
**项目名称**: QVPO强化学习算法用于UAV-ISAC环境  
**核心问题**: 训练结果稳定且reward高于基线，但感知信息泄露率高达70%  
**审查范围**: main.py, myenv.py, myenv3.py, agent/qvpo.py及相关配置文件  

---

## 一、项目概述

### 1.1 项目背景
- **算法**: QVPO (Quadratic Value Policy Optimization) 
- **环境**: UAV-ISAC (无人机综合感知与通信)
- **任务**: 单个无人机配备全向天线，同时执行感知任务和通信服务，需防止通信用户窃听感知信息
- **目标**: 平衡感知质量、通信服务和感知安全

### 1.2 环境配置
- **无人机**: 1架，飞行高度100m，活动范围[-400, 400]m
- **通信用户**: 3个，随机移动
- **感知目标**: 1个固定目标(100, 100, 50m)
- **合法接收器**: 雷达接收器(0, 0, 0m)
- **时隙**: 每回合50个时隙，每时隙4秒

---

## 二、核心问题诊断：70%感知信息泄露率

### 2.1 问题严重性评估
**70%泄露率**意味着在大部分时隙中，至少有一个通信用户能够成功窃听到感知信息，这严重违反了ISAC系统的安全要求。

### 2.2 泄露率计算逻辑
根据代码分析，泄露率基于以下逻辑：
```python
# myenv.py Line 418-424
snr_gap2 = sensing_snr_eavesdropper - self.eav_threshold  # eav_threshold=10.0dB
eav_penalty = 0.0
if snr_gap2 > 0:  # 窃听SNR超过阈值
    eav_penalty = min(self.eav_penalty_coef * snr_gap2, self.eav_penalty_cap)
```

**泄露判定**: 窃听SNR > 10dB 即视为泄露

---

## 三、关键问题识别

### 3.1 【严重】窃听惩罚机制失效

#### 问题1: 惩罚系数过小
```python
# main.py 默认参数
--eav_penalty_coef 3.0
--eav_penalty_cap 20.0
```

**分析**: 
- 典型窃听SNR差距: 15-25dB
- 惩罚值: 3.0 × (15-10) = 15dB
- 相对于感知收益(8-14dB)，惩罚力度不足

#### 问题2: 分项裁剪削弱惩罚效果
```python
# myenv.py Line 423
eav_penalty_clipped = np.clip(eav_penalty, 0.0, self.eav_penalty_clip_max)  # 默认5.0
```

**影响**: 即使窃听很严重，惩罚也被限制在5.0，无法形成有效约束。

### 3.2 【严重】奖励设计不平衡

#### 问题1: 感知收益权重过高
```python
# myenv.py Line 366-367
eta_0_clipped = np.clip(eta_0, 0.0, self.eta_clip_max)  # 默认15.0
reward = eta_0_clipped
```

**分析**: 
- 感知收益可达15dB
- 窃听惩罚最大仅5dB
- 净收益仍为正，缺乏安全约束

#### 问题2: 奖励缩放过小
```python
# main.py Line 162
--reward_scale 0.1
```

**影响**: 所有奖励被压缩10倍，但惩罚机制相对失效问题被放大。

### 3.3 【中等】窃听聚合策略不当

#### 问题: logsumexp过度放大威胁
```python
# myenv.py Line 405-410
elif self.eav_agg == 'logsumexp':
    x = np.array(eavesdropper_snr_list, dtype=np.float32)
    kappa = float(self.eav_logsumexp_kappa)  # 默认0.5
    m = float(np.max(x))
    sensing_snr_eavesdropper = m + (1.0 / kappa) * float(np.log(np.sum(np.exp(kappa * (x - m)))))
```

**数学分析**:
- 当3个用户SNR为[12, 11, 10]dB时
- logsumexp = 12 + 2×log(1 + 0.606 + 0.368) = 13.44dB
- 相比max策略(12dB)放大了12%

**问题**: 过度惩罚导致智能体学会"远离所有用户"的消极策略。

### 3.4 【中等】状态表示缺失关键信息

#### 问题: 缺乏窃听风险直接观测
当前状态空间:
```python
# myenv.py Line 251-256
obs = np.concatenate([
    self.uav_position[:2],           # 无人机位置 (2维)
    self.user_positions[:, :2].flatten(),  # 用户位置 (6维)
    self.prev_action                  # 历史动作 (3维)
])  # 总计11维，组合后22维
```

**缺失信息**:
- 当前窃听SNR水平
- 各用户的窃听风险评估
- 感知-窃听权衡指标

### 3.5 【轻微】训练参数配置问题

#### 问题1: 探索阶段过长
```python
--start_steps 10000  # 占总步数的0.4%
```

**影响**: 长期随机探索可能让智能体学到"窃听无关紧要"的错误模式。

#### 问题2: 批次大小可能不足
```python
--batch_size 256
```

**对于复杂安全约束任务，可能需要更大批次来学习稀疏的安全信号。**

---

## 四、代码层面具体问题

### 4.1 myenv.py 关键问题

#### Line 362-439: _calculate_reward方法
```python
def _calculate_reward(self, uav_position, power_allocation):
    eta_0 = self._calculate_sensing_snr_legal(uav_position, power_allocation)
    eta_0_clipped = np.clip(eta_0, 0.0, self.eta_clip_max)  # 15.0上限过高
    reward = eta_0_clipped  # 感知收益直接作为奖励，无安全约束
    
    # 窃听惩罚计算
    eav_penalty_clipped = np.clip(eav_penalty, 0.0, self.eav_penalty_clip_max)  # 5.0上限过低
    reward -= eav_penalty_clipped  # 惩罚力度不足
```

**问题**: 奖励函数设计天然偏向感知收益，安全约束薄弱。

#### Line 494-516: _calculate_sensing_snr_eavesdropper方法
```python
def _calculate_sensing_snr_eavesdropper(self, uav_position, power_allocation):
    # 计算每个用户的窃听SNR，但未在状态中暴露
    eavesdropper_snr_list = []
    # ... 计算逻辑 ...
    return eavesdropper_snr_list
```

**问题**: 窃听风险信息未传递给智能体，只能通过奖励信号学习。

### 4.2 agent/qvpo.py 关键问题

#### Line 240-241: Q值变换
```python
q = eval(self.q_transform)(q, q_neg=self.q_neg, cut=self.cut, 
                           running_q_std=self.running_q_std, beta=self.beta,
                           running_q_mean=self.running_q_mean, v=v, ...)
```

**问题**: Q值变换可能削弱安全约束的权重，特别是在running_q_std较大时。

#### Line 186-188: Target Policy Smoothing
```python
target_noise = torch.randn_like(next_actions) * 0.05
target_noise = target_noise.clamp(-0.1, 0.1)
next_actions = (next_actions + target_noise).clamp(-1.0, 1.0)
```

**问题**: 噪声可能让智能体学习到"偶尔违规是可以接受的"。

### 4.3 main.py 配置问题

#### Line 134-137: 窃听参数默认值
```python
parser.add_argument('--eav_penalty_coef', type=float, default=3.0, metavar='G')
parser.add_argument('--eav_penalty_cap', type=float, default=20.0, metavar='G')
parser.add_argument('--eav_penalty_clip_max', type=float, default=5.0, metavar='G')
```

**问题**: 默认参数设置偏向感知收益，安全约束不足。

---

## 五、根本原因分析

### 5.1 设计哲学问题
**当前设计**: 感知优先，安全为辅  
**应该设计**: 安全约束下的感知优化  

### 5.2 奖励工程问题
- **正反馈过强**: 感知收益(最高15dB) > 窃听惩罚(最高5dB)
- **负反馈延迟**: 窃听风险需要通过多步学习才能理解
- **信号稀疏**: 只有在窃听发生时才有惩罚信号

### 5.3 表示学习问题
- **状态不完整**: 缺乏窃听风险的直接观测
- **价值估计偏差**: Q函数可能低估安全状态的价值

---

## 六、改进建议

### 6.1 【紧急】奖励函数重新设计

#### 建议1: 安全约束优先
```python
def _calculate_reward_secure(self, uav_position, power_allocation):
    # 1. 首先检查安全约束
    eavesdropper_snr_list = self._calculate_sensing_snr_eavesdropper(uav_position, power_allocation)
    max_eav_snr = max(eavesdropper_snr_list) if eavesdropper_snr_list else 0.0
    
    # 2. 如果不安全，直接给予负奖励
    if max_eav_snr > self.eav_threshold:
        return -10.0, {'security_violation': True, 'max_eav_snr': max_eav_snr}
    
    # 3. 安全情况下才计算感知收益
    eta_0 = self._calculate_sensing_snr_legal(uav_position, power_allocation)
    return eta_0 * 0.1, {'eta_0': eta_0, 'secure': True}
```

#### 建议2: 零和博弈设计
```python
# 感知收益与窃听风险直接对冲
reward = eta_0 - 2.0 * max_eav_snr  # 窃听惩罚权重加倍
```

### 6.2 【重要】状态空间增强

#### 建议1: 添加窃听风险观测
```python
def _get_obs_enhanced(self):
    base_obs = self._get_obs()
    
    # 添加窃听风险评估
    eavesdropper_snr_list = self._calculate_sensing_snr_eavesdropper(
        self.uav_position, self.current_power)
    eav_risk = np.array(eavesdropper_snr_list, dtype=np.float32)
    
    # 添加安全指标
    max_eav_snr = max(eav_risk) if len(eav_risk) > 0 else 0.0
    security_indicator = np.array([max_eav_snr, max(0, max_eav_snr - self.eav_threshold)])
    
    return np.concatenate([base_obs, eav_risk, security_indicator])
```

#### 建议2: 相对位置信息
```python
# 添加无人机到用户的相对距离向量
relative_distances = self.user_positions[:, :2] - self.uav_position[:2]
```

### 6.3 【重要】参数重新调优

#### 建议1: 激进安全参数
```python
--eav_penalty_coef 10.0      # 从3.0提升到10.0
--eav_penalty_cap 50.0        # 从20.0提升到50.0  
--eav_penalty_clip_max 20.0   # 从5.0提升到20.0
--eta_clip_max 8.0           # 从15.0降低到8.0
--reward_scale 1.0           # 从0.1提升到1.0
```

#### 建议2: 训练参数优化
```python
--start_steps 5000           # 减少随机探索
--batch_size 512             # 增大批次
--policy_freq 1              # 更频繁策略更新
```

### 6.4 【中等】算法层面改进

#### 建议1: 安全约束Actor-Critic
```python
class SafeActorCritic:
    def __init__(self):
        self.safety_critic = Critic(state_dim, action_dim)  # 专门评估安全风险
        
    def get_safe_action(self, state):
        # 采样多个动作
        actions = self.actor.sample_multiple(state, n=32)
        
        # 筛选安全动作
        safe_actions = []
        for action in actions:
            safety_score = self.safety_critic(state, action)
            if safety_score > safety_threshold:
                safe_actions.append(action)
        
        # 从安全动作中选择最优
        if safe_actions:
            return max(safe_actions, key=lambda a: self.critic(state, a))
        else:
            return self.actor(state)  # 回退到常规策略
```

#### 建议2: 课程学习
```python
# 阶段1: 只学习安全约束 (前50k步)
reward = -max_eav_snr if max_eav_snr > threshold else 1.0

# 阶段2: 引入感知收益 (50k-150k步)  
reward = eta_0 - 5.0 * max_eav_snr

# 阶段3: 完整任务 (150k步后)
reward = eta_0 - 2.0 * max_eav_snr + comm_reward
```

---

## 七、实施优先级

### 7.1 立即实施 (P0)
1. **修改奖励函数**: 实施安全约束优先的设计
2. **调整关键参数**: eav_penalty_coef提升到10.0
3. **移除分项裁剪**: 让惩罚机制充分发挥作用

### 7.2 短期实施 (P1)
1. **增强状态空间**: 添加窃听风险直接观测
2. **优化训练参数**: 减少探索步数，增大批次
3. **改进聚合策略**: 使用max而非logsumexp

### 7.3 中期实施 (P2)
1. **算法改进**: 实施安全约束Actor-Critic
2. **课程学习**: 分阶段训练策略
3. **自适应参数**: 动态调整惩罚权重

---

## 八、预期效果

### 8.1 量化指标改善
- **泄露率**: 从70% → 目标<20%
- **感知收益**: 可能略有下降，但应在可接受范围
- **整体奖励**: 短期下降，长期稳定提升

### 8.2 训练曲线预期
- **初期**: 奖励下降，智能体学习安全约束
- **中期**: 泄露率快速下降，感知收益逐步恢复  
- **后期**: 稳定在安全且高效的平衡点

---

## 九、风险评估

### 9.1 潜在风险
1. **性能下降**: 过度安全约束可能影响感知质量
2. **训练不稳定**: 奖励函数大幅修改可能导致收敛困难
3. **计算开销**: 增强状态空间和算法复杂度

### 9.2 缓解措施
1. **渐进式改进**: 分阶段实施修改
2. **A/B测试**: 保留原版本作为对比
3. **监控指标**: 实时跟踪泄露率和感知性能

---

## 十、总结与建议

### 10.1 核心结论
当前70%的感知信息泄露率主要源于：
1. **奖励设计不平衡**: 感知收益权重远超安全惩罚
2. **参数配置不当**: 窃听惩罚系数和上限过低
3. **状态信息缺失**: 缺乏窃听风险的直接观测
4. **约束机制薄弱**: 分项裁剪削弱了惩罚效果

### 10.2 最终建议
**立即行动**:
1. 重新设计奖励函数，实施安全约束优先策略
2. 大幅提升窃听惩罚参数(eav_penalty_coef: 3.0→10.0)
3. 移除或大幅放宽分项裁剪限制

**预期效果**: 通过上述改进，感知信息泄露率应能从70%降至20%以下，同时保持可接受的感知性能。

**长期目标**: 建立安全-感知-通信的多目标优化框架，实现真正的安全ISAC系统。

---

**审查人**: AI代码审查系统  
**联系方式**: 如需进一步讨论，请查看项目文档或联系开发团队  
**文档版本**: v1.0  
**最后更新**: 2026-03-06
