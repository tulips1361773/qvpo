#!/bin/bash

# 确保脚本有执行权限: chmod +x run_ddpg.sh

# 设置可见的 GPU (根据参数中的 --cuda cuda:1，这里通过环境变量强制指定，或者在脚本内由 PyTorch 处理)
# 这里我们直接传参给 python 脚本
CUDA_DEVICE="cuda:1"

echo "Starting DDPG Training..."

python ddpg.py \
    --cuda ${CUDA_DEVICE} \
    --env_name "DDPG_Baseline" \
    --total_timesteps 2500000 \
    --learning_starts 10000 \
    --batch_size 256 \
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
    --normalize_state True

echo "Training finished!"