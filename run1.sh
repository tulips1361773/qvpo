#!/bin/bash

# 3/1 20:14

# 脚本功能：运行 UAV SAC 算法并调整惩罚系数
# 修改点：
# 1. learning-starts 设置为 4500
# 2. eav-penalty-coef 设置为 2.0 (降低窃听惩罚权重)
# 3. comm-penalty-coef 设置为 0.2 (降低通信干扰惩罚权重)

python sac_cleanrl.py \
    --exp-name "uav_sac_tuned_params" \
    --seed 42 \
    --total-timesteps 1000000 \
    --learning-starts 4500 \
    --eav-penalty-coef 2.0 \
    --comm-penalty-coef 0.2 \
    --buffer-size 1000000 \
    --gamma 0.99 \
    --tau 0.005 \
    --batch-size 512 \
    --policy-lr 3e-4 \
    --q-lr 1e-3 \
    --reward_scale 0.1 \
    --normalize_state True \
    --cuda "cuda:0" \
    # 如果想开启 wandb 记录，请取消下面一行的注释
    # --track True