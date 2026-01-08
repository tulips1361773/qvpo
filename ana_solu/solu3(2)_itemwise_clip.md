# 建议3: 分项裁剪奖励

## 实验目的

验证对每个奖励分项分别裁剪后再相加，是否能减少 reward 方差，稳定 Critic 训练。

## 问题分析

根据 ana3.md 的数据：
- `eav_penalty` 在 **0-7** 之间剧烈震荡
- `eta_0` 在 **2-15** 之间高频大幅震荡
- Critic Loss 在 **20-55** 之间震荡，难以收敛

**原问题**：
- 当前是对**最终 reward** 进行裁剪，而不是对**每个分项**裁剪
- 某个分项的极端值可能在裁剪前就已经影响了其他分项的梯度信号
- 例如：`eta_0=18` 时，即使 `eav_penalty=7`，总 reward 仍然很高

## 解决方案

**对每个分项分别裁剪后再相加**：

```python
# 分项裁剪
eta_0_clipped = np.clip(eta_0, 0.0, eta_clip_max)  # 默认 15.0
comm_penalty_clipped = np.clip(comm_penalty, 0.0, comm_penalty_clip_max)  # 默认 5.0
eav_penalty_clipped = np.clip(eav_penalty, 0.0, eav_penalty_clip_max)  # 默认 5.0

reward = eta_0_clipped - comm_penalty_clipped - eav_penalty_clipped
```

**参数设计**：
- `eta_clip_max = 15.0`：避免极高 SNR 的边际收益过大
- `comm_penalty_clip_max = 5.0`：收紧通信惩罚上限
- `eav_penalty_clip_max = 5.0`：收紧窃听惩罚上限

## 代码修改

1. **`myenv.py:131-134`** - 添加分项裁剪参数
```python
eta_clip_max: float = 15.0,
comm_penalty_clip_max: float = 5.0,
eav_penalty_clip_max: float = 5.0
```

2. **`myenv.py:169-172`** - 保存参数到实例变量

3. **`myenv.py:367-368`** - 对 eta_0 进行裁剪
```python
eta_0_clipped = np.clip(eta_0, 0.0, self.eta_clip_max)
reward = eta_0_clipped
```

4. **`myenv.py:398-400`** - 对 comm_penalty 进行裁剪
```python
comm_penalty_clipped = np.clip(comm_penalty, 0.0, self.comm_penalty_clip_max)
reward -= comm_penalty_clipped
```

5. **`myenv.py:424-426`** - 对 eav_penalty 进行裁剪
```python
eav_penalty_clipped = np.clip(eav_penalty, 0.0, self.eav_penalty_clip_max)
reward -= eav_penalty_clipped
```

6. **`main.py:164-170`** - 添加命令行参数
7. **`main.py:266-268, 289-291`** - 传递参数给环境
8. **`main.py:363-366`** - 添加 TensorBoard 日志记录

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
  --eta_clip_max 15.0 \
  --comm_penalty_clip_max 5.0 \
  --eav_penalty_clip_max 5.0 \
  --start_steps 10000 \
  --cuda cuda:1
```

---

## 观察指标

1. **`reward_terms/eta_0`** vs **`reward_terms/eta_0_clipped`**：对比裁剪前后
2. **`reward_terms/eav_penalty`** vs **`reward_terms/eav_penalty_clipped`**：对比裁剪前后
3. **`loss/critic`**：观察 Critic Loss 是否更稳定
4. **`reward/eval_mean`**：观察评估性能是否提升

---

## 预期效果

1. **reward 方差减小**：每个分项都被限制在合理范围
2. **Critic Loss 更稳定**：target value 的噪声降低
3. **策略更稳定**：不会因为某个分项的极端值而剧烈调整

---

## 实验结果

（待训练后填写）
