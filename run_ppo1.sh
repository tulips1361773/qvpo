# 3/2 11：56
python ppo.py \
  --env_name Env \
  --seed 42 \
  --num_steps 250000 \
  --gamma 0.99 \
  --normalize_state True \
  --action_smooth_coef 0.1 \
  --user_move_range 20.0 \
  --reward_scale 0.1 \
  --eav_agg top2 \
  --eav_threshold 10.0 \
  --eav_penalty_coef 0.5 \
  --eav_penalty_cap 20.0 \
  --comm_penalty softplus \
  --comm_threshold 10.0 \
  --comm_penalty_coef 1.5 \
  --comm_softplus_kappa 5.0 \
  --comm_penalty_cap_per_user 15.0 \
  --comm_penalty_cap_total 30.0 \
  --comm_penalty_avg_over_k True \
  --cuda cuda:1



# 本实验失败
# 🔴 核心问题分析
# 你遇到的三个问题（没Eval记录、负分、远差于SAC）的核心原因只有一个：对 num_steps 参数的理解偏差。
# 根本原因：--num_steps 250000 设置错误
# 在 PPO 中：num_steps 不是总训练步数，而是 “每次更新策略前收集的步数” (Batch Size)。
# 你的设置后果：你设置了 250,000。这意味着 Agent 必须先用旧策略（初始甚至是随机策略）跑完 5000 个 Episode（25万步），才进行 1次 梯度更新。
# 对比 SAC：SAC 是 Off-policy，通常每几步就更新一次网络。你的 PPO 配置相当于让 SAC 跑几个小时只更新一次参数。
# 导致没 Eval：代码逻辑是 if update % 5 == 0: evaluate。因为你收集 25万步才算 update=1，所以要跑到 250000 * 5 = 125万步 时才会出现第一次 Eval 记录。
# 导致负分：网络几乎没有更新，一直在用随机策略乱跑。
# 网络结构过小
# CleanRL 默认是 64x64 的网络。对于无人机轨迹规划（连续控制），这个容量太小。SAC 通常使用 256x256。
# 缺少 Value Normalization
# PPO 对 Value 的预测尺度很敏感，SAC 对此鲁棒性更强。