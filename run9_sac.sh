# 3/12 11:08
# 参数和sucure.sh对齐
python sac2.py \
    --exp-name "sac_myenv3_improved" \
    --seed 42 \
    --total-timesteps 1000000 \
    --learning-starts 10000 \
    --buffer-size 1000000 \
    --gamma 0.99 \
    --tau 0.005 \
    --batch-size 512 \
    --policy-lr 3e-4 \
    --q-lr 1e-3 \
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
    --comm_penalty_coef 1.5 \
    --comm_softplus_kappa 5.0 \
    --action_smooth_coef 0.1 \
    --user_move_range 20.0 \
    --reward_scale 0.1