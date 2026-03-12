# 3/10 0:50 适配myenv3环境  eav coaf:2-->5
# SAC基线实验，使用Agent端归一化（不启用myenv3的StateNormalizer）


python sac2.py \
    --exp-name "sac_myenv3" \
    --seed 42 \
    --total-timesteps 1000000 \
    --learning-starts 4500 \
    --buffer-size 1000000 \
    --gamma 0.99 \
    --tau 0.005 \
    --batch-size 512 \
    --policy-lr 3e-4 \
    --q-lr 1e-3 \
    --reward_scale 0.1 \
    --action_smooth_coef 0.8 \
    --user_move_range 20.0 \
    --eav_threshold 10.0 \
    --eav_penalty_coef 5.0 \
    --eav_penalty_clip_max 1000.0 \
    --comm_threshold 10.0 \
    --comm_penalty_coef 0.5 \
    --comm_softplus_kappa 2.0 \
    --comm_penalty_clip_per_user 20.0 \
    --comm_penalty_clip_total 50.0 \
    --cuda "cuda:1"