# 建议2: 窃听惩罚改为 logsumexp 平滑聚合

## 实验目的

验证将窃听惩罚的聚合方式从 `top2` 改为 `logsumexp` 是否能减少 `eav_penalty` 的震荡。

## 问题分析

根据 ana3.md 的数据，`eav_penalty` 在 **0-7** 之间剧烈震荡。

**原问题**：
- 当前使用 `eav_agg='top2'`（取最大两个窃听者SNR的均值）
- 用户每步随机移动 0-20m，导致窃听者相对位置变化
- `argmax` 可能在不同窃听者之间切换，造成 reward 抖动
- 这种抖动是 Critic 难以收敛的主要原因之一

## 解决方案

使用 **logsumexp 平滑聚合**：
```python
sensing_snr_eavesdropper = m + (1/κ) * log(Σexp(κ*(x-m)))
```

**参数设计**：
- `eav_agg = 'logsumexp'`
- `eav_logsumexp_kappa = 0.5`（较小的 κ 使聚合更平滑）

**logsumexp 的数学性质**：
- 当 `κ → ∞`：logsumexp → max（硬最大）
- 当 `κ → 0`：logsumexp → mean（均值）
- `κ = 0.5` 是一个折中，既保留"关注最坏情况"的特性，又避免硬切换

## 代码修改

### logsumexp 实现代码（已存在于 solu1）

**位置**: `myenv.py:393-398`
```python
elif self.eav_agg == 'logsumexp':
    x = np.array(eavesdropper_snr_list, dtype=np.float32)
    kappa = float(self.eav_logsumexp_kappa)
    kappa = max(kappa, 1e-6)
    m = float(np.max(x))
    sensing_snr_eavesdropper = m + (1.0 / kappa) * float(np.log(np.sum(np.exp(kappa * (x - m)))))
```

### 本次修改：将默认聚合方式改为 logsumexp

1. **`myenv.py:123`** - 修改默认参数
```python
# 修改前
eav_agg: str = 'top2', eav_logsumexp_kappa: float = 1.0

# 修改后
eav_agg: str = 'logsumexp', eav_logsumexp_kappa: float = 0.5
```

2. **`main.py:128-131`** - 修改命令行参数默认值
```python
# 修改前
parser.add_argument('--eav_agg', type=str, default='top2', ...)
parser.add_argument('--eav_logsumexp_kappa', type=float, default=5.0, ...)

# 修改后
parser.add_argument('--eav_agg', type=str, default='logsumexp', ...)
parser.add_argument('--eav_logsumexp_kappa', type=float, default=0.5, ...)
```

---

## 训练命令

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
  --eav_agg logsumexp \
  --eav_logsumexp_kappa 0.5 \
  --start_steps 10000 \
  --cuda cuda:1
```

---

## 观察指标

1. **`reward_terms/eav_penalty`**：观察震荡幅度是否减小
2. **`loss/critic`**：观察 Critic Loss 是否更稳定
3. **`q/running_q_std`**：观察 Q 值标准差是否下降
4. **`reward/eval_mean`**：观察评估性能是否提升

---

## 预期效果

1. **`eav_penalty` 震荡幅度减小**：logsumexp 避免了 argmax 切换导致的跳变
2. **Critic Loss 更稳定**：reward 方差降低，Critic 更容易拟合
3. **策略更平滑**：Agent 不需要为了逃避某个特定窃听者而剧烈移动

---

## 实验结果

（待训练后填写）
