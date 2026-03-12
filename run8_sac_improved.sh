#!/bin/bash
# 改进的SAC训练脚本，解决奖励下降问题
# 主要修改：
# 1. 降低窃听惩罚系数从5.0到2.0（与默认值对齐）
# 2. 增加学习开始时间从4500到10000
# 3. 调整批次大小从512到256
# 4. 添加完整的SAC参数设置
# 5. 对齐tensorboard记录格式
# 6. 支持GPU选择（可以修改为cuda:0, cuda:1, cuda:2, cuda:3）

# 26/3/12 10:17
python sac2.py \
    --exp-name "sac_myenv3_improved" \
    --seed 42 \
    --total-timesteps 1000000 \
    --learning-starts 10000 \
    --buffer-size 1000000 \
    --gamma 0.99 \
    --tau 0.005 \
    --batch-size 256 \
    --policy-lr 3e-4 \
    --q-lr 1e-3 \
    --policy-frequency 2 \
    --target-network-frequency 1 \
    --alpha 0.2 \
    --autotune True \
    --eval-frequency 10000 \
    --eval-episodes 10 \
    --cuda "cuda:1" \
    --eav_penalty_coef 2.0 \
    --eav_penalty_clip_max 1000.0 \
    --reward_scale 0.1

# 使用其他GPU的示例：
# GPU 0: --cuda "cuda:0"
# GPU 1: --cuda "cuda:1" 
# GPU 2: --cuda "cuda:2"
# GPU 3: --cuda "cuda:3"

# 我已经完成了完整的tensorboard记录对齐工作，现在sac2.py与main.py在关键指标上完全一致：

# 完成的对齐工作：


# 1. 环境修复

# 修复了myenv3.py中缺失的裁剪字段：eta0clipped、commpenaltyclipped、eavpenaltyclipped

# 添加了eavpenalty字段以对齐main.py的命名


# 2. 奖励项完全对齐

# rewardterms/eta0 - 感知SNR

# rewardterms/eta0clipped - 裁剪后的感知SNR

# rewardterms/commpenalty - 通信惩罚

# rewardterms/commpenaltyclipped - 裁剪后的通信惩罚

# rewardterms/eavpenalty - 窃听惩罚（原始值）

# rewardterms/eavpenaltyclipped - 裁剪后的窃听惩罚

# rewardterms/rewardraw、rewardterms/rewardclip1、rewardterms/rewardfinal


# 3. 安全指标完全对齐

# security/stepleakagerate - 单步泄漏率

# security/trainleakagerate - 训练累计泄漏率

# security/evalleakagerate - 评估泄漏率

# security/stepleakagecount - 泄漏用户数

# security/eavpenaltyraw、security/eavpenaltyweighted


# 4. 训练指标完全对齐

# reward/train、reward/trainma100、reward/trainema、reward/evalmean

# 记录频率完全一致（每200步环境指标，每episode结束奖励）


# 5. SAC算法特定指标

# losses/qfloss、losses/actorloss、losses/alphaloss

# alpha/value - 熵系数值


# 关于目标感知SNR的说明：

# 经过仔细检查，main.py中没有记录"目标感知SNR"相关指标，只有实际的eta0值。因此这个指标无法对齐是正常的。

# 参数优化：

# 创建了run8sacimproved.sh脚本，解决了原脚本的关键问题：
# 窃听惩罚系数从5.0降到2.0

# 学习开始时间从4500增到10000

# 批次大小从512调到256

# 添加了完整的SAC参数设置


# 现在sac2.py的tensorboard记录与main.py完全对齐，可以公平比较算法性能。
