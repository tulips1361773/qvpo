# 基于ana2做的修改

policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_01_05_21_53_31_0
新建分支，这个分支从solu1开始（git 版本：“问题和修改见analysis1......”），修改的内容不包含solu2

## 实验目的

验证将主奖励从线性映射改为饱和函数（tanh）是否能减少策略振荡。

## 修改内容

### 问题分析

根据 analysis2.md 的数据，`eta_0`（感知 SNR）在 **0-18 dB** 范围内震荡。

**原问题**：
- `reward = eta_0` 是线性映射，高 SNR 区域的边际收益与低 SNR 区域相同
- 策略会"疯狂追高 SNR"，因为从 15→18 dB 的收益和 5→8 dB 一样大
- 导致策略在"追高 SNR"和"被惩罚打回"之间振荡

### 解决方案

使用 **tanh 饱和函数**：
```python
r_sense = scale * tanh((eta_0 - center) / slope)
```

**参数设计**：
- `sense_reward_center = 10.0`：eta_0 的中位数附近，作为"满意点"
- `sense_reward_slope = 5.0`：控制饱和速度
- `sense_reward_scale = 10.0`：输出范围约 [-10, 10]

**预期效果**：
- 当 `eta_0 < 10`：梯度较大，策略有动力提升 SNR
- 当 `eta_0 > 10`：梯度变小，策略更愿意在可行域内微调
- 避免"追高 SNR → 被惩罚 → 回退"的振荡

### 代码修改

1. **`myenv.py:129-135`** - 添加饱和函数参数
```python
sense_reward_type: str = 'tanh',  # 'linear', 'tanh', 'sigmoid'
sense_reward_scale: float = 10.0,
sense_reward_center: float = 10.0,
sense_reward_slope: float = 5.0
```

2. **`myenv.py:366-377`** - 修改奖励计算逻辑
```python
if self.sense_reward_type == 'tanh':
    reward = self.sense_reward_scale * np.tanh((eta_0 - self.sense_reward_center) / self.sense_reward_slope)
elif self.sense_reward_type == 'sigmoid':
    reward = self.sense_reward_scale * (2.0 / (1.0 + np.exp(-(eta_0 - self.sense_reward_center) / self.sense_reward_slope)) - 1.0)
else:  # 'linear'
    reward = eta_0
```

3. **`main.py:165-172`** - 添加命令行参数
4. **`main.py:268-271, 292-295`** - 传递参数给环境
5. **`main.py:359`** - 添加 TensorBoard 日志记录 `sense_reward`

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
  --action_smooth_coef 1.0 \
  --user_move_range 20.0 \
  --reward_scale 0.1 \
  --sense_reward_type tanh \
  --sense_reward_scale 10.0 \
  --sense_reward_center 10.0 \
  --sense_reward_slope 5.0 \
  --start_steps 10000 \
  --cuda cuda:1
```

---

## 观察指标

1. **`reward_terms/eta_0`** vs **`reward_terms/sense_reward`**：对比原始 SNR 和饱和后的奖励
2. **`reward/train`** 和 **`reward/eval_mean`**：观察奖励曲线是否更平稳
3. **`reward_terms/action_smooth_penalty`**：观察动作平滑惩罚是否下降

---

## 实验结果
policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_01_08_14_04_18_0
（待训练后填写）

