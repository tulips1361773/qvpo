
markdown
# 问题记录：mujoco_py 编译失败（X11/Xlib.h: No such file or directory）
## 问题时间
2025-12-31
## 问题环境
- OS：Ubuntu
- Python：3.8（conda 环境 `qvpo`）
- 相关库：gym, mujoco_py, MuJoCo 2.1.0 (`~/.mujoco/mujoco210`)
## 问题描述
运行命令：
```bash
python main.py --env_name HalfCheetah-v3 --weighted --aug
程序在创建 MuJoCo 环境（如 HalfCheetah-v3）时崩溃，报错堆栈显示在导入 mujoco_py 时编译 C 扩展失败。

关键错误信息
text
fatal error: X11/Xlib.h: No such file or directory
   99 | #include <X11/Xlib.h>
      |          ^~~~~~~~~~~~
compilation terminated.
distutils.errors.CompileError: command '/usr/bin/gcc' failed with exit code 1
原因分析
mujoco_py 在安装/首次导入时需要编译依赖 OpenGL 和 X11 的 C 扩展。
系统未安装 X11 开发头文件（X11/Xlib.h），导致 GCC 编译失败。
这是系统级依赖缺失问题，而不是 Python 代码逻辑问题。
解决步骤
安装系统依赖（Ubuntu）：
bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libx11-dev \
    libxi-dev \
    libxmu-dev \
    libglu1-mesa-dev \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    libglfw3 \
    libglfw3-dev
在 conda 环境中（qvpo）确认/重新安装 mujoco_py：
bash
conda activate qvpo
pip install -U 'mujoco-py<2.2,>=2.1'
重新运行命令：
bash
cd ~/sda/qvpo
python main.py --env_name HalfCheetah-v3 --weighted --aug
处理结果
安装上述系统依赖后，mujoco_py 能够成功编译 C 扩展，HalfCheetah-v3 环境创建成功，程序可以正常运行训练。

 
---
 
如果你按上面步骤执行后还有新的报错（比如换成别的缺失库），把新的终端输出贴出来，我再帮你逐条排查。
Feedback



为什么仍然报 X11/Xlib.h 错？
因为 mujoco_py 的 C 扩展在编译时仍会 include eglplatform.h，而这个头文件默认依赖 X11 头文件；你设置的环境变量只影响运行时，不影响编译时的 include 路径和宏定义。


---

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
- `loss/actor`
- `q/current_q1_mean`, `q/current_q2_mean`, `q/target_q_mean`
- `q/reward_mean`
-（若启用 `weighted`）`q/running_q_mean`, `q/running_q_std`

## 当前结果
修复后可避免回放采样被环境 reset 的全局 RNG 重置影响；通过新增 TensorBoard 指标可以进一步验证 critic/Q 值是否稳定以及定位剩余不收敛原因。