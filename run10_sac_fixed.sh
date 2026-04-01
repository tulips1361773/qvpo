#!/bin/bash
# SAC baseline - 修复后的配置
# 修复内容:
# 1. 降低Q网络学习率 (1e-3 -> 3e-4)
# 2. 调整环境参数与QVPO保持一致
# 3. 明确指定所有环境参数
# 3.28 6:48
python sac2.py \
    --exp-name "sac_myenv3_fixed" \
    --seed 42 \
    --total-timesteps 1000000 \
    --learning-starts 10000 \
    --buffer-size 1000000 \
    --gamma 0.99 \
    --tau 0.005 \
    --batch-size 256 \
    --policy-lr 3e-4 \
    --q-lr 3e-4 \
    --policy-frequency 2 \
    --target-network-frequency 1 \
    --alpha 0.2 \
    --autotune True \
    --eval-frequency 10000 \
    --eval-episodes 10 \
    --cuda "cuda:1" \
    --eav_threshold 10.0 \
    --eav_penalty_coef 5.0 \
    --eav_penalty_clip_max 200.0 \
    --comm_threshold 10.0 \
    --comm_penalty_coef 1.5 \
    --comm_softplus_kappa 5.0 \
    --comm_penalty_clip_per_user 15.0 \
    --comm_penalty_clip_total 30.0 \
    --action_smooth_coef 0.8 \
    --user_move_range 20.0 \
    --reward_scale 0.1
