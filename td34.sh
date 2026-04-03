#!/usr/bin/env bash
# 4/1  0；36
# 4/2 0：08  101
# (Tau) 从 0.005 降低到 0.003：让目标网络更新更缓慢，提供更稳定的学习目标。
# Policy Noise 降低至 0.1：与你现有的探索噪声水平对齐，避免在评估目标动作时引入过大的随机扰动。

python td3_2.py \
  --env_name Env \
  --seed 42 \
  --num_steps 1000000 \
  --batch_size 256 \
  --gamma 0.99 \
  --tau 0.005 \
  --actor_lr 0.0003 \
  --critic_lr 0.0003 \
  --policy_noise 0.1 \
  --noise_clip 0.5 \
  --exploration_noise 0.1 \
  --exploration_noise_min 0.03 \
  --exploration_noise_anneal_start -1 \
  --exploration_noise_anneal_end 600000 \
  --policy_freq 2 \
  --action_smooth_coef 0.1 \
  --user_move_range 20.0 \
  --reward_scale 0.1 \
  --eav_threshold 10.0 \
  --eav_penalty_coef 5 \
  --eav_penalty_clip_max 200.0 \
  --comm_threshold 10.0 \
  --comm_penalty_coef 1 \
  --comm_softplus_kappa 5.0 \
  --comm_penalty_cap_per_user 15.0 \
  --comm_penalty_cap_total 30.0 \
  --start_steps 10000 \
  --use_obs_normalizer False \
  --cuda cuda:2