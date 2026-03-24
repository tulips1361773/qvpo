# Agent-Side Observation Normalizer Design

## 当前状态数据流

### 方案 A：固定手工缩放（无 obs normalizer）

```
Environment (myenv3.py)
  ↓ _get_raw_obs() → [UAV xy, users xy, prev_action]
  ↓ _scale_obs() → 坐标 ÷400, action 不变
  ↓ _get_combined_obs() → [scaled_current, scaled_prev]
  ↓ 返回 fixed-scaled state (22-dim, 全部在 [-1,1])
  
Main.py
  ↓ state 直接写入 replay buffer (固定口径)
  
QVPO.py
  ↓ 从 replay 采样 → 直接喂给 critic/actor
```

### 方案 B：固定手工缩放 + agent-side Welford normalizer

```
Environment (myenv3.py)
  ↓ _get_raw_obs() → [UAV xy, users xy, prev_action]
  ↓ _scale_obs() → 坐标 ÷400, action 不变
  ↓ _get_combined_obs() → [scaled_current, scaled_prev]
  ↓ 返回 fixed-scaled state (22-dim, 全部在 [-1,1])
  
Main.py
  ↓ state 直接写入 replay buffer (固定口径，未 Welford 标准化)
  
QVPO.py
  ├─ sample_action(eval=False)
  │   ↓ ObsNormalizer.normalize(state, update_stats=True)  # 更新统计量
  │   ↓ 送给 actor
  │
  ├─ sample_action(eval=True)
  │   ↓ ObsNormalizer.normalize(state, update_stats=False) # 不更新
  │   ↓ 送给 actor
  │
  └─ train()
      ↓ 从 replay 采样 batch
      ↓ ObsNormalizer.normalize_batch(states)  # 不更新统计量
      ↓ ObsNormalizer.normalize_batch(next_states)
      ↓ 送给 critic/actor_target/critic_target
```

## 关键设计决策

### 1. 为什么复用 Welford 算法？

**Welford 算法本身没有问题**，问题在于旧实现的**位置错误**：

- ❌ **旧方案（错误）**：环境在 `reset()/step()` 中在线标准化 → 标准化后的状态写入 replay
  - 问题：replay buffer 中混有不同历史 mean/std 口径的状态
  - step 1000 存的状态和 step 100000 存的状态，虽然数值相同，但物理含义不同
  - off-policy critic 从 replay 采样时看到的是混合口径数据

- ✅ **新方案（正确）**：环境返回固定缩放状态 → replay 存固定口径 → agent-side 统一标准化
  - replay buffer 始终存固定口径状态（同一物理状态永远映射到同一数值）
  - 标准化只在"取出后"应用，不污染 replay
  - 所有网络输入都经过相同的标准化处理

### 2. 为什么不污染 replay？

| 对比项 | 旧方案（环境侧） | 新方案（agent 侧） |
|--------|-----------------|-------------------|
| **环境输出** | Welford 标准化后的状态 | 固定缩放状态 |
| **replay 存储** | 标准化状态（口径漂移） | 固定缩放状态（口径一致） |
| **统计量更新** | 每个 step() 都更新 | 只在在线采样时更新 |
| **replay 采样** | 直接使用（已标准化） | 采样后再标准化（不更新统计量） |
| **off-policy 一致性** | ❌ 破坏 | ✅ 保持 |

**核心区别**：
- 旧方案：`normalize → store → sample → use`（存的是标准化后的）
- 新方案：`store → sample → normalize → use`（存的是固定口径的）

### 3. 统计量更新策略

```python
# sample_action() - 在线采样时
if not eval:
    state = obs_normalizer.normalize(state, update_stats=True)  # ✅ 更新
else:
    state = obs_normalizer.normalize(state, update_stats=False) # ❌ 不更新

# train() - replay 采样时
states = obs_normalizer.normalize_batch(states)  # ❌ 永远不更新
next_states = obs_normalizer.normalize_batch(next_states)  # ❌ 永远不更新
```

**原因**：
- 统计量应该反映**真实在线访问分布**
- replay batch 被反复采样，如果每次都更新统计量会导致：
  - 统计量更新过快
  - 偏向 replay 中的旧数据分布
  - 失去对当前策略分布的追踪

### 4. 冻结策略

```python
if t >= obs_norm_freeze_after:  # 默认 50000 steps
    obs_normalizer.freeze()
```

**原因**：
- 训练后期策略趋于稳定，状态分布变化小
- 冻结统计量可以稳定训练，避免后期微小波动
- 类似于学习率退火的思想

## ObsNormalizer 实现细节

### 核心方法

```python
class ObsNormalizer:
    def __init__(self, state_dim, epsilon=1e-8, clip_range=5.0, device='cpu'):
        # Welford 统计量
        self.mean = np.zeros(state_dim, dtype=np.float64)
        self.M2 = np.zeros(state_dim, dtype=np.float64)
        self.var = np.ones(state_dim, dtype=np.float64)
        self.count = 0.0
        self.frozen = False
    
    def update(self, state):
        """Welford 在线更新"""
        if self.frozen:
            return
        self.count += 1
        delta = state - self.mean
        self.mean += delta / self.count
        delta2 = state - self.mean
        self.M2 += delta * delta2
        if self.count > 1:
            self.var = self.M2 / self.count
    
    def normalize(self, state, update_stats=None):
        """标准化（可选更新统计量）"""
        if update_stats and not is_batch:
            self.update(state)
        std = np.sqrt(self.var + self.epsilon)
        normalized = (state - self.mean) / std
        return np.clip(normalized, -self.clip_range, self.clip_range)
    
    def normalize_batch(self, states):
        """批量标准化（不更新统计量）"""
        return self.normalize(states, update_stats=False)
```

### 与旧 StateNormalizer 的对比

| 特性 | 旧 StateNormalizer | 新 ObsNormalizer |
|------|-------------------|-----------------|
| **位置** | 环境内部 | Agent 内部 |
| **算法** | Welford | Welford（相同） |
| **输入** | raw observation | fixed-scaled state |
| **输出去向** | replay buffer | critic/actor 网络 |
| **统计量更新时机** | 每个 step() | 只在在线采样 |
| **支持冻结** | ❌ | ✅ |
| **支持 batch** | ❌ | ✅ |
| **torch 兼容** | ❌ | ✅ |

## 命令行参数

```bash
# 方案 A：只用固定缩放
python main.py --use_state_scaling --no-use_obs_normalizer

# 方案 B：固定缩放 + agent-side normalizer
python main.py \
  --use_state_scaling \
  --use_obs_normalizer \
  --obs_norm_freeze_after 50000 \
  --obs_norm_clip 5.0 \
  --obs_norm_eps 1e-8
```

## A/B 测试建议

### 实验组织

1. **Baseline**：`--no-use_obs_normalizer`
   - 只用固定手工缩放
   - 最简单，最稳定

2. **Treatment**：`--use_obs_normalizer`
   - 固定缩放 + Welford 标准化
   - 理论上可能提升 critic 稳定性

### 预期结果

- 如果 **方案 B 更好**：说明 Welford 标准化确实有助于 critic 训练
- 如果 **方案 A 更好**：说明固定缩放已经足够，额外标准化引入噪声
- 如果 **差不多**：说明当前问题瓶颈不在状态预处理

## 未修改的逻辑（确认）

✅ **以下逻辑完全不变**：
- `reward_scale`、`_calculate_reward`
- `critic_loss`、TD target、Bellman equation
- `q_transform`、`qadv`
- `running_q_mean/std`（Q 值归一化）
- Diffusion actor loss
- `action_scale`/`action_bias`

❌ **唯一改变**：
- 状态输入链增加可选的 Welford 标准化层
- 但 replay buffer 存储内容不变（仍是固定缩放状态）

## 潜在问题与注意事项

### 1. prev_obs 初始值

当前 `reset()` 中 `prev_obs = zeros(11)`，缩放后仍为零，映射到坐标系原点。
- 第一个 step 的 combined_obs 后半段全为 0
- 如果 obs normalizer 启用，这些 0 会参与统计量计算
- 建议：观察训练初期 mean/std 是否合理

### 2. 统计量收敛速度

- 默认 50k steps 后冻结
- 如果环境状态分布变化快，可能需要延长
- 如果收敛慢，可以考虑更激进的初始化

### 3. 评估一致性

- 评估时使用训练阶段的 mean/std（冻结状态）
- 不会再出现旧方案中"训练和评估统计量不一致"的问题

## 总结

**旧方案的问题不是 Welford 本身，而是"环境输出端在线标准化 + 标准化后状态直接写入 replay"。**

**新方案通过"buffer 外统一归一化"解决了 off-policy 口径一致性问题，同时保留了 Welford 标准化可能带来的训练稳定性优势。**
