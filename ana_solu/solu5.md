# 修改
确定使用scale=0.1
删除 step 函数中间的 np.clip(reward, -20.0, 30.0)。
增强约束惩罚：大幅提高 boundary_penalty 和 eav_penalty_coef。
添加但未启用：越界即终止：如果 UAV 飞出边界，除了给大惩罚，建议直接 done=True（可选，视任务定义而定，通常能加速训练避免无效探索）。

# 参数
参数名	原始值	推荐新值	原因
eav_penalty_coef	3.0	5.0 ~ 8.0	增加窃听惩罚权重，迫使 Agent 哪怕牺牲一点感知性能也要避开窃听。
eav_penalty_cap	20.0	20.0	保持不变，防止极端情况下的单一负反馈过大。
boundary_penalty	(硬编码 -20)	(硬编码 -50)	让越界成为不可接受的错误。

# 命令1
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
  --entropy_alpha 0.05 \
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
  --action_smooth_coef 0.1 \
  --user_move_range 20.0 \
  --reward_scale 0.1 \
  --eav_agg top2 \
  --eav_threshold 10.0 \
  --eav_penalty_coef 5.0 \
  --eav_penalty_cap 20.0 \
  --comm_penalty softplus \
  --comm_threshold 10.0 \
  --comm_penalty_coef 1.5 \
  --comm_softplus_kappa 5.0 \
  --comm_penalty_cap_per_user 15.0 \
  --comm_penalty_cap_total 30.0 \
  --comm_penalty_avg_over_k True \
  --start_steps 10000 \
  --cuda cuda:2

  # 命令2   测试512+10k
 python main.py \
  --env_name Env \
  --seed 42 \
  --num_steps 200000 \
  --batch_size 512 \
  --gamma 0.99 \
  --tau 0.005 \
  --diffusion_lr 0.0001 \
  --critic_lr 0.0001 \
  --n_timesteps 20 \
  --beta_schedule cosine \
  --entropy_alpha 0.05 \
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
  --action_smooth_coef 0.1 \
  --user_move_range 20.0 \
  --reward_scale 0.1 \
  --eav_agg top2 \
  --eav_threshold 10.0 \
  --eav_penalty_coef 5.0 \
  --eav_penalty_cap 20.0 \
  --comm_penalty softplus \
  --comm_threshold 10.0 \
  --comm_penalty_coef 1.5 \
  --comm_softplus_kappa 5.0 \
  --comm_penalty_cap_per_user 15.0 \
  --comm_penalty_cap_total 30.0 \
  --comm_penalty_avg_over_k True \
  --start_steps 10000 \
  --cuda cuda:2
# 命令3  测试更小学习率
  python main.py \
  --env_name Env \
  --seed 42 \
  --num_steps 200000 \
  --batch_size 256 \
  --gamma 0.99 \
  --tau 0.005 \
  --diffusion_lr 0.0001 \
  --critic_lr 0.00005 \
  --n_timesteps 20 \
  --beta_schedule cosine \
  --entropy_alpha 0.05 \
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
  --action_smooth_coef 0.1 \
  --user_move_range 20.0 \
  --reward_scale 0.1 \
  --eav_agg top2 \
  --eav_threshold 10.0 \
  --eav_penalty_coef 5.0 \
  --eav_penalty_cap 20.0 \
  --comm_penalty softplus \
  --comm_threshold 10.0 \
  --comm_penalty_coef 1.5 \
  --comm_softplus_kappa 5.0 \
  --comm_penalty_cap_per_user 15.0 \
  --comm_penalty_cap_total 30.0 \
  --comm_penalty_avg_over_k True \
  --start_steps 10000 \
  --cuda cuda:1

# 命令4  更大学习率  4e-3
  python main.py \
  --env_name Env \
  --seed 42 \
  --num_steps 200000 \
  --batch_size 256 \
  --gamma 0.99 \
  --tau 0.005 \
  --diffusion_lr 0.0001 \
  --critic_lr 0.0004 \
  --n_timesteps 20 \
  --beta_schedule cosine \
  --entropy_alpha 0.05 \
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
  --action_smooth_coef 0.1 \
  --user_move_range 20.0 \
  --reward_scale 0.1 \
  --eav_agg top2 \
  --eav_threshold 10.0 \
  --eav_penalty_coef 5.0 \
  --eav_penalty_cap 20.0 \
  --comm_penalty softplus \
  --comm_threshold 10.0 \
  --comm_penalty_coef 1.5 \
  --comm_softplus_kappa 5.0 \
  --comm_penalty_cap_per_user 15.0 \
  --comm_penalty_cap_total 30.0 \
  --comm_penalty_avg_over_k True \
  --start_steps 10000 \
  --cuda cuda:1