#!/bin/bash

# 设置可见的 GPU
CUDA_DEVICE="cuda:1"

echo "Starting Improved DDPG Training (LayerNorm + GradClip)..."

python ddpg.py \
    --cuda ${CUDA_DEVICE} \
    --env_name "DDPG_Baseline_Imp" \
    --total_timesteps 1000000 \
    --learning_starts 10000 \
    --batch_size 256 \
    --actor_lr 0.0001 \
    --critic_lr 0.001 \
    --weight_decay 0.01 \
    --normalize_state False \
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
    --comm_penalty_avg_over_k True

echo "Training finished!"

# run_ddpg1.sh的问题和ddpg2做的改进
# 从你提供的 TensorBoard 曲线来看，这是一个典型的 DDPG 算法崩溃（Collapse）和 Q 值过估计（Overestimation） 的现象。具体分析如下：
# losses/qf1_loss 爆炸（高达 4000+）：
    # 原因：Critic 网络（Q网络）无法收敛。这通常是因为 Q 值过估计。DDPG 倾向于高估 Q 值，随着训练进行，Q 值目标越来越大，导致 MSE Loss 呈指数级增长。
    # 关联问题：你的环境中有 StateNormalizer。这是 DDPG 崩溃的核心原因之一。DDPG 是 Off-policy 算法，经验回放池（Replay Buffer）中存储的是过去的状态。如果 myenv.py 中的 StateNormalizer 在不断更新均值和方差，那么 Buffer 中存储的旧状态数据的分布与当前网络看到的分布完全不一致（非平稳性）。这导致 Critic 面对的是一个分布不断输入的“移动靶”，根本学不会。
# losses/actor_loss 持续下降（负值越来越大）：
    # Actor Loss 定义为 −Q(s,a)# 。曲线持续下降说明 Q 值在持续变大。结合 Q Loss 爆炸，说明 Critic 认为“无论干什么由于 Q 值都很大”，这是一种病态的假象。
# reward 先升后降（崩溃）：
    # 初期（0-20k步）Agent 学到了一些东西，奖励到了 50 左右。
    # 中期开始，由于 Critic 估值崩坏，Actor 被误导去执行极端的错误动作（例如一直往某个方向飞直到越界），导致奖励暴跌回 0 或负数。\

# 核心修改点：
# 禁用环境内部的 StateNormalizer：在 DDPG 中，不能使用随时间变化的在线归一化。我们将改用网络内部的 Layer Normalization，它对输入尺度不敏感，且不会破坏 Replay Buffer 的一致性。
# 引入 Layer Normalization：在 Actor 和 Critic 网络的第一层加入 nn.LayerNorm。
# 梯度裁剪 (Gradient Clipping)：防止 Loss 爆炸时梯度更新过大摧毁权重。
# 权重衰减 (Weight Decay)：在优化器中加入 L2 正则化，防止过拟合和参数过大。
# 调整学习率：Critic 的学习率通常需要比 Actor 大，或者整体降低以求稳定。