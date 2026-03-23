# QVPO算法收敛性分析报告

## 执行摘要

**问题描述：** QVPO算法在UAV-ISAC环境中表现为前期快速提升，但在33左右过早平台化（batch_size=512, chosen=2），而baseline SAC虽然慢但能收敛到38。

**核心发现：** QVPO的过早收敛主要由以下因素导致：
1. **加权损失中的Q值归一化过于激进**
2. **采样策略的探索不足**
3. **噪声调度缺乏自适应性**
4. **优势函数裁剪限制了策略改进空间**

---

## 一、算法全流程分析

### 1.1 训练主循环 (`main.py`)

**关键参数配置（run_qvpo5.sh）：**
```bash
--batch_size 256
--chosen 2                    # Top-K选择
--train_sample 32             # 训练时采样动作数
--behavior_sample 8           # 行为策略采样数
--target_sample 2             # 目标策略采样数
--eval_sample 16              # 评估采样数
--n_timesteps 20              # 扩散步数
--beta_schedule cosine        # Beta调度
--weighted                    # 启用加权训练
--aug                         # 启用数据增强
--q_transform qadv            # 使用优势函数
--cut 0.8                     # Q值裁剪阈值
--policy_freq 2               # 策略更新频率
```

**训练流程：**
1. **经验收集：** 前10000步随机探索，之后使用策略采样
2. **Critic更新：** 每步更新，使用TD目标 + Target Policy Smoothing
3. **Actor更新：** 每2步更新一次（policy_freq=2）
4. **熵系数退火：** `entropy_alpha` 从0.05线性衰减到0.002

---

## 二、关键收敛影响因素分析

### 2.1 加权损失机制（核心问题）

#### 2.1.1 Q值归一化策略

**代码位置：** `agent/qvpo.py:232-240`

```python
# 自适应学习率计算
ratio = abs(std - self.running_q_std) / max(self.running_q_std, 1.0)
adaptive_alpha_std = self.alpha_std * min(1.0, 1.0 / (1.0 + ratio))
adaptive_alpha_mean = self.alpha_mean * min(1.0, 1.0 / (1.0 + ratio))

# 更新运行统计量
self.running_q_std += adaptive_alpha_std * (std - self.running_q_std)
self.running_q_std = max(1.0, min(self.running_q_std, 50.0))  # 裁剪到[1.0, 50.0]
self.running_q_mean += adaptive_alpha_mean * (mean - self.running_q_mean)
```

**问题分析：**
- **running_q_std下限为1.0：** 当Q值方差实际小于1.0时，强制设为1.0会导致归一化后的权重被过度压缩
- **自适应学习率机制：** 当std波动大时，学习率降低，导致统计量更新缓慢，无法跟踪Q值分布变化
- **初始值设置：** `running_q_std=1.0, running_q_mean=0.0` 可能与实际Q值分布差距大

#### 2.1.2 Q Transform - QAdv优势函数

**代码位置：** `agent/q_transform.py:43-50`

```python
class QAdv(QTransform):
    def __call__(self, q, **kwargs):
        v = kwargs.get("v", None)
        chosen = kwargs.get("chosen", None)
        batch_size = kwargs.get("batch_size", None)
        adv = q.view(batch_size, chosen, 1) - v
        adv = adv.clamp_(min=0.0)  # 🔥 关键：负优势被裁剪为0
        return adv.view(batch_size * chosen, 1)
```

**问题分析：**
- **负优势裁剪：** `clamp_(min=0.0)` 导致所有低于平均Q值的动作权重为0，完全被忽略
- **权重分布极化：** 只有Top-K中优于平均的动作被训练，导致策略多样性快速丧失
- **与chosen=2的交互：** 当chosen=2时，如果两个动作都低于V，权重全为0，梯度消失

**影响：**
```
假设batch中某状态的Q值：
  action_1: Q=30, action_2: Q=28, V=29
  
经过QAdv：
  adv_1 = max(30-29, 0) = 1.0
  adv_2 = max(28-29, 0) = 0.0  # 被完全忽略！
  
结果：只有action_1被训练，策略快速收敛到局部最优
```

### 2.2 采样策略与探索

#### 2.2.1 训练时采样数量

**代码位置：** `agent/diffusion.py:214-246` 和 `agent/qvpo.py:129`

```python
# 训练时采样
states, best_actions, v_target, (mean, std) = self.actor.sample_n(
    states, times=self.train_sample,  # train_sample=32
    chosen=self.chosen,                # chosen=2
    q_func=self.critic, 
    origin=actions
)
```

**问题分析：**
- **train_sample=32：** 相比SAC的连续高斯分布，32个离散采样点覆盖动作空间有限
- **Top-K选择（chosen=2）：** 从32个样本中只选2个最优的，探索性极低
- **采样多样性：** 扩散过程虽然有随机性，但noise_ratio=1.0固定，缺乏自适应探索

#### 2.2.2 噪声调度机制

**代码位置：** `agent/diffusion.py:28-29, 121, 139-142`

```python
# 初始化
self.max_noise_ratio = noise_ratio  # 默认1.0
self.noise_ratio = noise_ratio

# 采样时噪声添加
return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise * self.noise_ratio

# 确定性模式切换
if self.deterministic:
    self.noise_ratio = 0 if eval else self.max_noise_ratio
else:
    self.noise_ratio = self.max_noise_ratio  # 训练时始终为1.0
```

**问题分析：**
- **固定噪声比例：** 训练全程noise_ratio=1.0，没有随训练进度衰减
- **缺乏探索退火：** 不像SAC的熵系数自动调节，QVPO的探索强度恒定
- **与收敛的矛盾：** 后期仍保持高噪声，可能导致策略在局部最优附近震荡

### 2.3 Target Policy Smoothing

**代码位置：** `agent/qvpo.py:184-187`

```python
# Target Policy Smoothing
target_noise = torch.randn_like(next_actions) * 0.05  # solu4: 0.1 -> 0.05
target_noise = target_noise.clamp(-0.1, 0.1)          # solu4: 0.25 -> 0.1
next_actions = (next_actions + target_noise).clamp(-1.0, 1.0)
```

**问题分析：**
- **噪声幅度较小：** 0.05的标准差，裁剪到±0.1，相比SAC的探索噪声偏保守
- **平滑效果有限：** 虽然能稳定Critic训练，但可能限制了对高Q值区域的探索

### 2.4 熵正则化

**代码位置：** `agent/qvpo.py:243-251` 和 `main.py:419-420`

```python
# 训练中添加随机动作
if self.entropy_alpha > 0.0:
    rand_states = states.unsqueeze(0).expand(10, -1, -1).contiguous().view(batch_size*self.chosen*10, -1)
    rand_policy_actions = torch.empty(batch_size * self.chosen * 10, actions.shape[-1], device=self.device).uniform_(-1, 1)
    rand_q = q.unsqueeze(0).expand(10, -1, -1).contiguous().view(batch_size*self.chosen*10, -1) * self.entropy_alpha
    
    best_actions = torch.cat([best_actions, rand_policy_actions], dim=0)
    states = torch.cat([states, rand_states], dim=0)
    q = torch.cat([q, rand_q], dim=0)

# 熵系数退火（main.py）
agent.entropy_alpha = min(args.entropy_alpha, 
                         max(0.002, args.entropy_alpha - steps/num_steps * args.entropy_alpha))
```

**问题分析：**
- **随机动作权重：** `rand_q = q * entropy_alpha`，随着entropy_alpha衰减，随机动作影响快速降低
- **退火速度：** 从0.05线性衰减到0.002，在360000步训练中，后期探索几乎消失
- **与SAC对比：** SAC使用自动熵调节，QVPO的手动退火可能过快

---

## 三、与SAC的关键差异

### 3.1 探索机制

| 维度 | QVPO | SAC |
|------|------|-----|
| **策略表示** | 扩散模型（离散采样） | 高斯策略（连续分布） |
| **探索方式** | 固定噪声比例 + 熵正则 | 自动熵调节 |
| **采样覆盖** | 32个离散样本 | 无限连续空间 |
| **探索退火** | 线性衰减（可能过快） | 自适应调节 |

**SAC优势：**
- **连续探索：** 高斯分布天然覆盖整个动作空间
- **自动平衡：** 熵系数自动调节，平衡探索与利用
- **稳定性：** 不依赖离散采样的运气

### 3.2 策略更新

| 维度 | QVPO | SAC |
|------|------|-----|
| **损失函数** | 加权扩散损失 | 策略梯度 + 熵正则 |
| **权重计算** | Q值归一化 + 优势裁剪 | 直接最大化 Q - α*log_π |
| **梯度信号** | 可能被裁剪为0 | 始终有梯度 |
| **更新频率** | 每2步 | 每2步（相同） |

**SAC优势：**
- **梯度稳定：** 所有动作都有梯度信号，不会被裁剪
- **目标明确：** 直接优化期望回报 - 熵，目标清晰

### 3.3 Critic训练

| 维度 | QVPO | SAC |
|------|------|-----|
| **目标噪声** | 0.05 std, ±0.1 clip | 无（直接用策略采样） |
| **学习率** | critic_lr=0.0003 | q_lr=0.001（更大） |
| **网络结构** | 256-256 | 256-256（相同） |

**SAC优势：**
- **更快收敛：** 更大的学习率加速Q值估计
- **更准确：** 直接用策略分布，无需额外噪声

---

## 四、收敛过快的根本原因

### 4.1 主要原因（按影响程度排序）

#### 1. **优势函数负值裁剪（最关键）**
- **机制：** `adv.clamp_(min=0.0)` 导致低于平均的动作权重为0
- **后果：** 策略快速收敛到局部最优，丧失探索能力
- **证据：** chosen=2时，如果两个动作都低于V，梯度消失

#### 2. **Q值归一化的running_q_std下限过高**
- **机制：** `running_q_std = max(1.0, ...)` 强制最小方差为1.0
- **后果：** 当实际Q值方差<1.0时，权重被过度压缩，高Q动作优势不明显
- **证据：** 后期Q值收敛，方差降低，但归一化仍用1.0，导致权重趋于均匀

#### 3. **熵正则化退火过快**
- **机制：** 线性衰减，360000步后降至0.002
- **后果：** 后期探索不足，无法跳出局部最优
- **证据：** SAC使用自动熵调节，能根据策略熵动态调整

#### 4. **采样数量与Top-K选择的矛盾**
- **机制：** train_sample=32, chosen=2，只训练最优的2个
- **后果：** 策略多样性快速丧失
- **证据：** SAC无Top-K限制，所有采样都参与训练

### 4.2 次要原因

#### 5. **固定噪声比例缺乏自适应性**
- **影响：** 后期仍保持高噪声，策略在局部最优附近震荡

#### 6. **Target Policy Smoothing噪声偏小**
- **影响：** 限制了对高Q值区域的探索，但影响相对较小

---

## 五、改进建议

### 5.1 高优先级改进

#### 建议1：修改优势函数裁剪策略
```python
# 当前代码（agent/q_transform.py:49）
adv = adv.clamp_(min=0.0)  # ❌ 过于激进

# 建议改进
adv = adv.clamp_(min=-1.0)  # ✅ 允许负优势，但限制幅度
# 或者使用 softplus
adv = F.softplus(adv, beta=0.5) - F.softplus(torch.tensor(0.0), beta=0.5)
```

**预期效果：** 保留低Q动作的梯度信号，增加策略多样性

#### 建议2：动态调整running_q_std下限
```python
# 当前代码（agent/qvpo.py:236）
self.running_q_std = max(1.0, min(self.running_q_std, 50.0))  # ❌ 下限过高

# 建议改进
# 根据训练进度动态调整下限
min_std = max(0.1, 1.0 - (self.step / total_steps) * 0.9)  # 从1.0衰减到0.1
self.running_q_std = max(min_std, min(self.running_q_std, 50.0))
```

**预期效果：** 后期允许更小的方差，权重分布更敏感

#### 建议3：自适应熵系数
```python
# 当前代码（main.py:419-420）
agent.entropy_alpha = min(args.entropy_alpha, 
                         max(0.002, args.entropy_alpha - steps/num_steps * args.entropy_alpha))

# 建议改进：参考SAC的自动熵调节
target_entropy = -action_dim  # 目标熵
log_entropy_alpha = torch.zeros(1, requires_grad=True, device=device)
entropy_optimizer = torch.optim.Adam([log_entropy_alpha], lr=3e-4)

# 在训练循环中
with torch.no_grad():
    # 计算当前策略熵（需要在actor中添加熵计算）
    current_entropy = compute_policy_entropy(states, actions)
entropy_loss = -log_entropy_alpha.exp() * (current_entropy - target_entropy)
entropy_optimizer.zero_grad()
entropy_loss.backward()
entropy_optimizer.step()
agent.entropy_alpha = log_entropy_alpha.exp().item()
```

**预期效果：** 自动平衡探索与利用，避免过早收敛

### 5.2 中优先级改进

#### 建议4：增加训练采样数量
```python
# 当前配置
--train_sample 32

# 建议改进
--train_sample 64  # 或更多，增加动作空间覆盖
```

**预期效果：** 更好的动作空间覆盖，但会增加计算成本

#### 建议5：调整chosen参数
```python
# 当前配置
--chosen 2

# 建议改进
--chosen 4  # 增加训练动作数量
# 或者动态调整
chosen = max(2, int(8 * (1 - steps/total_steps)))  # 从8衰减到2
```

**预期效果：** 前期保持多样性，后期聚焦最优动作

#### 建议6：增大Target Policy Smoothing噪声
```python
# 当前代码（agent/qvpo.py:185-186）
target_noise = torch.randn_like(next_actions) * 0.05
target_noise = target_noise.clamp(-0.1, 0.1)

# 建议改进
target_noise = torch.randn_like(next_actions) * 0.1  # 增大标准差
target_noise = target_noise.clamp(-0.2, 0.2)  # 增大裁剪范围
```

**预期效果：** 增强Critic对动作扰动的鲁棒性

### 5.3 低优先级改进

#### 建议7：自适应噪声比例
```python
# 在agent/diffusion.py中添加
def get_adaptive_noise_ratio(self, step, total_steps):
    # 从1.0衰减到0.5
    return 1.0 - 0.5 * (step / total_steps)

# 在采样时使用
self.noise_ratio = self.get_adaptive_noise_ratio(current_step, total_steps)
```

**预期效果：** 后期降低噪声，加速收敛

---

## 六、参数敏感性分析

### 6.1 关键参数影响

| 参数 | 当前值 | 对收敛的影响 | 建议调整 |
|------|--------|-------------|---------|
| `cut` | 0.8 | 中等 - 裁剪Q值下限 | 尝试0.5或0.0 |
| `q_neg` | 0.001 | 低 - 负Q值偏移 | 保持不变 |
| `alpha_std` | 0.001 | 高 - 控制统计量更新速度 | 尝试0.005（加快更新） |
| `alpha_mean` | 0.001 | 高 - 控制统计量更新速度 | 尝试0.005 |
| `beta` | 1.0 | 中等 - Q值归一化缩放 | 尝试0.5（降低权重差异） |
| `entropy_alpha` | 0.05→0.002 | 高 - 探索强度 | 使用自适应调节 |
| `train_sample` | 32 | 高 - 动作空间覆盖 | 增加到64 |
| `chosen` | 2 | 高 - 训练动作数量 | 增加到4或动态调整 |

### 6.2 建议实验组合

**实验1：修复优势裁剪**
```bash
# 修改 agent/q_transform.py:49
adv = adv.clamp_(min=-1.0)  # 允许负优势
```

**实验2：动态统计量下限**
```bash
# 修改 agent/qvpo.py:236
# 添加动态min_std计算
```

**实验3：增加采样与chosen**
```bash
--train_sample 64
--chosen 4
```

**实验4：自适应熵（需要代码改动）**
```bash
# 实现自动熵调节机制
```

---

## 七、总结

### 7.1 核心问题

QVPO的过早收敛主要源于**加权损失机制的设计缺陷**：

1. **优势函数负值裁剪**导致低Q动作完全被忽略，策略快速收敛到局部最优
2. **Q值归一化的running_q_std下限过高**导致后期权重分布过于均匀，高Q动作优势不明显
3. **熵正则化线性退火过快**导致后期探索不足

### 7.2 与SAC的本质差异

- **SAC：** 连续策略 + 自动熵调节 + 所有动作都有梯度
- **QVPO：** 离散采样 + 手动熵退火 + Top-K选择 + 优势裁剪

QVPO的设计更适合**离线强化学习**（从固定数据集学习），但在**在线学习**中，过于激进的裁剪和选择机制限制了探索能力。

### 7.3 优先改进路径

**短期（1-2天）：**
1. 修改优势函数裁剪：`clamp_(min=-1.0)`
2. 增加train_sample和chosen

**中期（1周）：**
3. 实现动态running_q_std下限
4. 增大Target Policy Smoothing噪声

**长期（2周+）：**
5. 实现自适应熵系数
6. 探索其他Q transform方法（如QEXPN）

### 7.4 预期效果

实施上述改进后，预期QVPO能够：
- **延缓收敛：** 保持更长时间的探索
- **提高上限：** 从33提升到35-37（接近SAC的38）
- **稳定性：** 减少训练过程中的震荡

---

## 附录：代码修改清单

### A1. 优势函数裁剪修改

**文件：** `agent/q_transform.py`

```python
# 第49行
# 修改前
adv = adv.clamp_(min=0.0)

# 修改后
adv = adv.clamp_(min=-1.0)  # 或使用 softplus
```

### A2. 动态running_q_std下限

**文件：** `agent/qvpo.py`

```python
# 在__init__中添加
self.total_steps = args.num_steps

# 在train方法中修改（第236行）
# 修改前
self.running_q_std = max(1.0, min(self.running_q_std, 50.0))

# 修改后
min_std = max(0.1, 1.0 - (self.step / self.total_steps) * 0.9)
self.running_q_std = max(min_std, min(self.running_q_std, 50.0))
```

### A3. 增加采样参数

**文件：** `run_qvpo5.sh`（或新建run_qvpo10.sh）

```bash
--train_sample 64  # 从32增加到64
--chosen 4         # 从2增加到4
```

---

**报告生成时间：** 2026-03-23  
**分析代码版本：** run_qvpo5.sh (batch_size=256, chosen=2)  
**对比基线：** SAC (sac_cleanrl.py, batch_size=512)
