# 3/1 12：26
# 主要修改点：
# 强制修正参数：将默认 num_steps 改回 2048（PPO 标准值），将 total_timesteps 设为 250万。
# 增强网络：将 Actor 和 Critic 的隐藏层从 64 增加到 256（对齐 SAC）。
# 修复评估逻辑：不再依赖 update 次数，而是根据 global_step 每 10,000 步评估一次，确保 TensorBoard 曲线平滑。
# 参数解析优化：确保你的命令行参数都能正确传入。

python ppo.py \
  --env_name Env \
  --seed 42 \
  --total_timesteps 2500000 \
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