#!/bin/bash

# 确保脚本有执行权限: chmod +x run.sh

# 这里的关键参数是 thresholds，请根据你的物理场景调整
# comm_threshold: 通信最低SNR要求 (dB)
# eav_threshold: 允许的最大窃听SNR (dB)

python aoto_arg.py \
    --env_name "UAV_ISAC_Secure" \
    --seed 101 \
    --cuda "cuda:0" \
    --num_steps 2000000 \
    --eval_sample 64 \
    --comm_threshold 10.0 \
    --eav_threshold 5.0 \
    --batch_size 256 \
    --diffusion_lr 0.0001 \
    --action_lr 0.0003 \
    --policy_type "Diffusion" \
    --normalize_state True\
    --gamma 0.99