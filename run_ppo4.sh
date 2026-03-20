#!/bin/bash
# PPO基线实验运行脚本
# 环境参数与secure.sh完全一致，用于QVPO对比实验
# 3/13 创建

python ppo.py \
    --exp_name "ppo_baseline" \
    --seed 42 \
    --total_timesteps 1000000 \
    --learning_rate 3e-4 \
    --num_steps 2048 \
    --num_minibatches 32 \
    --update_epochs 10 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_coef 0.2 \
    --ent_coef 0.01 \
    --vf_coef 0.5 \
    --max_grad_norm 0.5 \
    --anneal_lr True \
    --normalize_state True \
    --cuda "cuda:1" \
    --eav_threshold 10.0 \
    --eav_penalty_coef 5.0 \
    --eav_penalty_clip_max 200.0 \
    --comm_threshold 10.0 \
    --comm_penalty_coef 1.5 \
    --comm_softplus_kappa 5.0 \
    --comm_penalty_cap_per_user 15.0 \
    --comm_penalty_cap_total 30.0 \
    --action_smooth_coef 0.1 \
    --user_move_range 20.0 \
    --reward_scale 0.1
