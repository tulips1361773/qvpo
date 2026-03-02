# 主要变动说明：
# --eav-penalty-coef 1.0 (再降一半)
# --comm-penalty-coef 0.1 (再降一半，先让它学会飞出 15dB 再说)
# --batch-size 1024 (更稳的梯度)
# --q-lr 3e-4 (减慢 Critic 收敛速度，防止 Loss 瞬间归零)
# --reward_scale 1.0 (尝试放大奖励信号，配合之前建议的 reward 逻辑，如果没改代码，这个参数保持 0.1 也行，但建议改代码中 reward 的计算方式)

# 3/1 21:24
python sac_cleanrl.py \
    --exp-name "uav_sac_fix_v2" \
    --seed 42 \
    --total-timesteps 1000000 \
    --learning-starts 5000 \
    --buffer-size 1000000 \
    --gamma 0.99 \
    --tau 0.005 \
    --batch-size 1024 \
    --policy-lr 3e-4 \
    --q-lr 3e-4 \
    --alpha 0.2 \
    --autotune True \
    --reward_scale 1.0 \
    --eav-penalty-coef 1.0 \
    --comm-penalty-coef 0.1 \
    --normalize_state True \
    --cuda "cuda:0"