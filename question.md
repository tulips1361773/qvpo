
# 修改记录：状态归一化（State Normalization）逻辑核查与修复
## 修改时间
2026-01-03

## 涉及文件
- `main.py`
- `myenv.py`

## 背景与问题
训练环境 `env` 和评估环境 `eval_env` 是两个独立实例。

当启用状态归一化时（`normalize_state=True`），每个环境都会各自维护一套 `StateNormalizer` 的统计量（`mean/var/count`）。如果评估环境不使用训练阶段累计出来的统计量，会导致：
- 评估时的状态分布与训练时不一致
- 策略网络输入尺度不一致，从而造成评估结果失真

另外，原先命令行布尔参数解析方式存在常见陷阱：`type=bool` 会导致 `--xxx False` 仍被解析为 True。

## 已完成的修改
### 1. 修复 `--normalize_state` 的布尔参数解析
在 `main.py` 中将布尔参数解析改为“优先使用 Python 3.9+ 的 `BooleanOptionalAction`，否则退化为 `_str2bool` 解析”，从而：
- 在 Python 3.9+ 可使用 `--normalize-state / --no-normalize-state`
- 在 Python < 3.9 可使用 `--normalize_state False/True`
- 避免 `type=bool` 导致的错误解析

### 2. 统一训练/评估的归一化统计量
在 `main.py` 中调整评估调用为：

`evaluate(eval_env, agent, steps, source_env=env)`

并在 `evaluate()` 内部增加逻辑：若提供了 `source_env`，则在评估开始前将 `source_env.state_normalizer` 的 `mean/var/count` 同步到 `eval_env.state_normalizer`。

这样保证：
- 每次评估使用的归一化统计量与训练一致
- 评估环境的 reset/step 不会意外污染统计量（评估阶段会先切换 `training=False`）

## 调用链复核结论（myenv.py）
`myenv.py` 中状态归一化的调用链为：
- `reset()`：构造 `combined_obs = [current_obs, prev_obs]` 后再归一化
- `step()`：构造 `combined_obs = [current_obs, prev_obs]` 后再归一化，然后更新 `prev_obs`

评估时通过 `StateNormalizer.set_training(False)` 关闭统计量更新，因此不会在评估期间更新 running mean/std。

## 当前结果
状态归一化开关解析更可靠，训练/评估的归一化统计量一致性问题已修复，评估结果可解释性更强。


---

# 问题记录：训练震荡严重/不收敛（环境重置全局 RNG 干扰回放采样）
## 问题时间
2026-01-04

## 涉及文件
- `myenv.py`
- `agent/replay_memory.py`
- `agent/qvpo.py`

## 问题现象
训练曲线震荡严重，reward 难以稳定提升，表现为长期不收敛。

## 根因分析
`ReplayMemory.sample()` 使用全局 `numpy` 随机数生成器进行 batch 采样：

`idxs = np.random.randint(...)`

但 `myenv.py` 的 `reset(seed=...)` 里曾调用 `np.random.seed(seed)`，会在每个 episode 重置全局 RNG 状态。
这会导致经验回放采样的随机序列被周期性重置，使得采样 batch 相关性增强，从而引起 Q 学习不稳定与训练震荡。

## 解决方案
### 1. 环境内部随机数改用 Gymnasium 的 `self.np_random`
修改 `myenv.py`：
- 移除 `np.random.seed(seed)`
- 将环境内所有随机采样（初始化 UAV/用户位置、用户移动）改为 `self.np_random.uniform(...)`

这样环境随机性与回放采样随机性相互独立，避免干扰训练。

### 2. 增加 TensorBoard 诊断指标（便于定位震荡来源）
修改 `agent/qvpo.py`：在训练中记录关键指标（定期写入 TensorBoard）：
- `loss/critic`
- `q/current_q1_mean`, `q/current_q2_mean`, `q/target_q_mean`
- `q/reward_mean`
-（若启用 `weighted`）`q/running_q_mean`, `q/running_q_std`

## 当前结果
修复后可避免回放采样被环境 reset 的全局 RNG 重置影响；通过新增 TensorBoard 指标可以进一步验证 critic/Q 值是否稳定以及定位剩余不收敛原因。


---

# 调优记录：Reward 平滑化消融实验（窃听聚合 top2 + 通信惩罚 softplus）
## 修改时间
2026-01-05

## 背景
训练中 `reward/train` 波动大、`loss/critic` 波动明显，怀疑主要来自 reward 中的 `max/hinge` 结构叠加“用户每步随机移动”带来的高方差，从而影响 Q 拟合与 `qadv` 权重稳定性。

## 调优 1（消融）：窃听者聚合 max -> mean(top2)
### 涉及文件
- `myenv.py`
- `main.py`

### 修改动机
原实现使用 `max(eavesdropper_snr_list)` 计算窃听者约束，最坏用户身份可能在多个用户间频繁切换，导致窃听惩罚项抖动，进而使总 reward 抖动。

### 具体修改
- `myenv.py::_calculate_reward()`：
  - 将 `sensing_snr_eavesdropper = max(eavesdropper_snr_list)` 改为：
    - `K>=2` 时取 `mean(top2)`（最大两个窃听 SNR 的均值）
    - `K==1`/空列表时做安全退化
- `main.py`：
  - TensorBoard `log_dir` 追加 `run_id={id}`，避免多次训练覆盖同一目录导致只显示最新一次。

### 观察到的现象（对比 baseline）
- `reward/train_ema` 的收敛水平上移（策略平均回报更高）。
- 但 `loss/critic` 波动变大且均值更高，`reward/train` 波动未显著降低。

### 初步原因分析
- `mean(top2)` 仅缓解 argmax 切换噪声，但 reward 的主要抖动可能来自通信阈值惩罚 + 用户随机移动。
- `mean(top2)` 可能让窃听惩罚触发更频繁（更“严格”），使得 TD target 分布更复杂；同时回报/Q 尺度上移会使 MSE 数值变大。

## 调优 2（消融）：通信惩罚 hinge -> softplus barrier，并按 K 归一化
### 涉及文件
- `myenv.py`
- `main.py`

### 修改动机
通信惩罚对 `K` 个用户累加且为阈值型结构，用户随机移动导致频繁跨阈值，从而产生高方差；此外惩罚尺度会随 `K` 改变而改变，影响 Q 的数值稳定性。
从原来的“阈值型线性惩罚”改成了 softplus barrier，并做了 按 K 归一化,避免惩罚尺度随 K 变化.

### 具体修改
原来的实现（hinge）
```python
if snr_gap > 0:
    total_comm_penalty += min(1.5 * snr_gap, 15.0)  # 线性惩罚，每用户上限15
reward -= min(total_comm_penalty, 30.0)           # 总上限30
```

现在的实现（softplus + /K）
```python
softplus_gap = np.logaddexp(0.0, comm_softplus_kappa * snr_gap) / comm_softplus_kappa
softplus_0 = np.logaddexp(0.0, 0.0) / comm_softplus_kappa
per_user_penalty = comm_penalty_coef * (softplus_gap - softplus_0)
per_user_penalty = max(0.0, per_user_penalty)
total_comm_penalty += min(per_user_penalty, comm_penalty_cap_per_user)
if self.comm_penalty_avg_over_k and self.K > 0:
    total_comm_penalty /= float(self.K)
comm_penalty = min(total_comm_penalty, comm_penalty_cap_total)
reward -= comm_penalty

关键变化：

惩罚函数：从 max(0, 1.5*gap) 改为 1.5 * (softplus(kappa*gap) - softplus(0))
归一化：对 K 个用户的总惩罚做平均（/K），避免惩罚尺度随 K 变化
保留上限：每用户上限 15.0，总上限 30.0

### 诊断日志增强
- `myenv.py::step()`：返回 `info`，包含
  - `eta_0`, `comm_penalty`, `eav_penalty`, `energy_penalty`, `boundary_penalty`, `reward_raw`, `reward_clip_1`, `reward_final`
- `main.py`：每 `200` step 写入 TensorBoard：
  - `reward_terms/*`（上述分项），用于定位抖动来源。

## 下一步建议
- 优先观察 `reward_terms/comm_penalty` 与 `reward_terms/eav_penalty` 的方差与尖刺，确认抖动主因。
- 若 `loss/critic` 仍抖：
  - 尝试更小的 `entropy_alpha`（如 0.01/0.005）或更快退火
  - 提高 `policy_freq`（3~4）让 critic 更充分拟合再更新 actor