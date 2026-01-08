# 基于ana1，实施的代码修改

## 1. myenv.py 修改内容

### 1.1 修改环境默认参数（第120-130行）

**修改前：**
```python
class UAVISACEnvironment(gym.Env):
    def __init__(self, N=50, K=3, H=100, H1=50, l_max=100, sigma2=1e-14, delta_t: float = 4.0,
                 E_tot: float = 600000.0, energy_penalty: float = 10.0,
                 normalize_state=True, normalize_reward=True,
                 eav_agg: str = 'top2', eav_logsumexp_kappa: float = 5.0,
                 eav_threshold: float = 10.0, eav_penalty_coef: float = 3.0, eav_penalty_cap: float = 20.0,
                 comm_penalty_type: str = 'softplus', comm_threshold: float = 10.0, comm_penalty_coef: float = 1.5,
                 comm_softplus_kappa: float = 5.0, comm_huber_delta: float = 1.0,
                 comm_penalty_cap_per_user: float = 15.0, comm_penalty_cap_total: float = 30.0,
                 comm_penalty_avg_over_k: bool = True):
```

**修改后：**
```python
class UAVISACEnvironment(gym.Env):
    def __init__(self, N=50, K=3, H=100, H1=50, l_max=100, sigma2=1e-14, delta_t: float = 4.0,
                 E_tot: float = 25000.0, energy_penalty: float = 5.0,  # 降低能量阈值使其生效
                 normalize_state=True, normalize_reward=True,
                 eav_agg: str = 'top2', eav_logsumexp_kappa: float = 1.0,  # 降低kappa使惩罚更平滑
                 eav_threshold: float = 10.0, eav_penalty_coef: float = 1.0, eav_penalty_cap: float = 10.0,  # 降低惩罚系数
                 comm_penalty_type: str = 'softplus', comm_threshold: float = 10.0, comm_penalty_coef: float = 0.5,  # 降低惩罚系数
                 comm_softplus_kappa: float = 1.0, comm_huber_delta: float = 1.0,  # 降低kappa
                 comm_penalty_cap_per_user: float = 5.0, comm_penalty_cap_total: float = 10.0,  # 降低cap
                 comm_penalty_avg_over_k: bool = True,
                 action_smooth_coef: float = 0.3, user_move_range: float = 20.0,  # 新增：动作平滑系数和用户移动范围
                 reward_scale: float = 0.1):  # 新增：奖励缩放因子
```

### 1.2 保存新增参数到实例变量（第159-163行）

**新增代码：**
```python
        # 新增参数
        self.action_smooth_coef = action_smooth_coef
        self.user_move_range = user_move_range
        self.reward_scale = reward_scale
```

### 1.3 添加动作平滑惩罚和奖励缩放（第302-314行）

**修改前：**
```python
        # 能耗计算
        ...
        # ✅ 新增：二级裁剪（能耗惩罚后的保护）
        reward = np.clip(reward, -60.0, 80.0)
        info['reward_final'] = float(reward)
```

**修改后：**
```python
        # 能耗计算
        ...
        
        # 新增：动作平滑惩罚（抑制Bang-Bang控制）
        action_diff = action - self.prev_action
        action_smooth_penalty = self.action_smooth_coef * np.sum(action_diff ** 2)
        reward -= action_smooth_penalty
        info['action_smooth_penalty'] = float(action_smooth_penalty)

        # 二级裁剪（能耗惩罚后的保护）
        reward = np.clip(reward, -30.0, 50.0)  # 缩小裁剪范围
        
        # 奖励缩放（使奖励范围更适合RL训练）
        reward = reward * self.reward_scale
        info['reward_final'] = float(reward)
```

### 1.4 使用可配置的用户移动范围（第507行）

**修改前：**
```python
move_distance = self.np_random.uniform(0, 50)
```

**修改后：**
```python
move_distance = self.np_random.uniform(0, self.user_move_range)  # 使用可配置的移动范围
```

---

## 2. main.py 修改内容

### 2.1 添加新命令行参数（第156-162行）

**新增代码：**
```python
    # 新增参数：动作平滑、用户移动范围、奖励缩放
    parser.add_argument('--action_smooth_coef', type=float, default=0.3, metavar='G',
                        help="action smoothness penalty coefficient (default: 0.3)")
    parser.add_argument('--user_move_range', type=float, default=20.0, metavar='G',
                        help="user movement range per step (default: 20.0)")
    parser.add_argument('--reward_scale', type=float, default=0.1, metavar='G',
                        help="reward scaling factor (default: 0.1)")
```

### 2.2 环境初始化时传入新参数（第255-257行和第275-277行）

**新增代码：**
```python
        action_smooth_coef=args.action_smooth_coef,
        user_move_range=args.user_move_range,
        reward_scale=args.reward_scale,
```

### 2.3 TensorBoard记录action_smooth_penalty（第345行）

**新增代码：**
```python
writer.add_scalar('reward_terms/action_smooth_penalty', float(info.get('action_smooth_penalty', 0.0)), steps)
```

---

## 3. 参数修改汇总表

| 参数 | 原值 | 新值 | 修改原因 |
|------|------|------|----------|
| `E_tot` | 600000.0 | 25000.0 | 使energy_penalty能够触发 |
| `energy_penalty` | 10.0 | 5.0 | 降低惩罚幅度 |
| `comm_penalty_coef` | 1.5 | 0.5 | 平滑惩罚函数 |
| `comm_softplus_kappa` | 5.0 | 1.0 | 降低惩罚陡峭度 |
| `eav_penalty_coef` | 3.0 | 1.0 | 平滑惩罚函数 |
| `eav_logsumexp_kappa` | 5.0 | 1.0 | 降低惩罚陡峭度 |
| `eav_penalty_cap` | 20.0 | 10.0 | 降低惩罚上限 |
| `comm_penalty_cap_per_user` | 15.0 | 5.0 | 降低惩罚上限 |
| `comm_penalty_cap_total` | 30.0 | 10.0 | 降低惩罚上限 |
| `action_smooth_coef` | 无 | 0.3 | 新增：抑制Bang-Bang控制 |
| `user_move_range` | 50 | 20.0 | 减少环境噪声 |
| `reward_scale` | 无 | 0.1 | 新增：缩放奖励范围 |
| 奖励裁剪范围 | [-60, 80] | [-30, 50] | 缩小裁剪范围 |

---

## 4. 推荐训练命令

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
  --action_smooth_coef 0.3 \
  --user_move_range 20.0 \
  --reward_scale 0.1 \
  --comm_penalty_coef 0.5 \
  --comm_softplus_kappa 1.0 \
  --eav_penalty_coef 1.0 \
  --start_steps 10000 \
  --cuda cuda:1
```

**关键调整说明：**
- 降低 `diffusion_lr` 和 `critic_lr` 到 `0.0001`（原 0.0003）- 提高训练稳定性
- 降低 `entropy_alpha` 到 `0.02`（原 0.05）- 减少探索噪声
- 使用 `cuda:1`（根据nvidia-smi显示，cuda:1显存较空闲）

## 实验效果
policy_type=Diffusion/ratio=0.1/seed=42/run_id=26_01_07_17_32_44_0