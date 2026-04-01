#!/usr/bin/env bash

python td3.py \
  --env_name Env \
  --seed 42 \
  --num_steps 1000000 \
  --batch_size 256 \
  --gamma 0.99 \
  --tau 0.003 \
  --actor_lr 0.0003 \
  --critic_lr 0.0004 \
  --policy_noise 0.2 \
  --noise_clip 0.5 \
  --exploration_noise 0.1 \
  --policy_freq 2 \
  --action_smooth_coef 0.1 \
  --user_move_range 20.0 \
  --reward_scale 0.1 \
  --eav_threshold 10.0 \
  --eav_penalty_coef 4 \
  --eav_penalty_clip_max 200.0 \
  --comm_threshold 10.0 \
  --comm_penalty_coef 1 \
  --comm_softplus_kappa 5.0 \
  --comm_penalty_cap_per_user 15.0 \
  --comm_penalty_cap_total 30.0 \
  --start_steps 10000 \
  --use_obs_normalizer False \
  --cuda cuda:3
